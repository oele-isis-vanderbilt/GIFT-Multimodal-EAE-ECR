"""TensorRT backend — uses ``tensorrt`` Python API directly with torch CUDA
buffers (no pycuda dep).

The detector engine has a fixed input shape (1, 3, 640, 640) and emits
(boxes, scores) with dynamic ``num_dets`` axis. The pose engine uses dynamic
batch dim ``[1..32]`` for stacked detection crops.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from libs.giftpose.codecs.simcc import decode_simcc
from libs.giftpose.meta import get_meta
from libs.giftpose.registry import DEFAULT_DET_TAG, DEFAULT_POSE_TAG, resolve_det, resolve_pose
from libs.giftpose.postprocess.nms import multiclass_nms_torch
from libs.giftpose.preprocess.letterbox import letterbox, undo_letterbox_xyxy
from libs.giftpose.preprocess.normalize import (
    normalize_det_input,
    normalize_pose_batch,
)
from libs.giftpose.preprocess.topdown_affine import apply_inverse_warps_batched, warp_crop
from libs.giftpose.runtime.base import Backend, Detection


_trt = None  # module-level cache for the tensorrt module


def _get_trt():
    """Import tensorrt lazily and cache it. Avoids re-importing on every call."""
    global _trt
    if _trt is None:
        import tensorrt as trt
        _trt = trt
    return _trt


def _load_engine(engine_path: str):
    trt = _get_trt()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        data = f.read()
    try:
        engine = runtime.deserialize_cuda_engine(data)
    except Exception as e:
        import torch
        cap = torch.cuda.get_device_capability()
        raise RuntimeError(
            f"Failed to deserialize {engine_path} with TensorRT {trt.__version__} "
            f"on CUDA sm_{cap[0]}{cap[1]}. Engines are bound to TRT version + GPU "
            f"compute capability — re-run "
            f"`python -m libs.giftpose.export.trt_build`."
        ) from e
    if engine is None:
        raise RuntimeError(
            f"deserialize_cuda_engine returned None for {engine_path} "
            f"(TRT {trt.__version__}). The engine may be corrupt or built for a "
            f"different TRT/GPU combination — re-run "
            f"`python -m libs.giftpose.export.trt_build`."
        )
    return engine


class TRTBackend(Backend):
    def __init__(
        self,
        det_engine: str,
        pose_engine: str,
        det_input_size: Tuple[int, int] | None = None,
        pose_input_size: Tuple[int, int] | None = None,
        flip_test: bool = True,
        det_score_thr: float = 0.05,
        det_iou_threshold: float = 0.6,
        det_max_per_img: int = 100,
        warmup: bool = True,
        det_spec=None,
        pose_spec=None,
    ) -> None:
        import torch
        _get_trt()  # fail-fast on missing TRT before any engine load

        self.torch = torch
        self.det_spec = det_spec or resolve_det(DEFAULT_DET_TAG)
        self.pose_spec = pose_spec or resolve_pose(DEFAULT_POSE_TAG)
        self.det_input_size = tuple(det_input_size or self.det_spec.input_size)
        self.pose_input_size = tuple(pose_input_size or self.pose_spec.input_size)
        self.num_keypoints = int(self.pose_spec.num_keypoints)
        self.simcc_split_ratio = float(self.pose_spec.simcc_split_ratio)
        self.flip_indices = list(get_meta(self.pose_spec.meta).FLIP_INDICES)
        self.flip_test = flip_test
        # Detector NMS knobs — wired through (the PyTorch backend already exposes
        # these; the TRT/ONNX paths previously hardcoded them, so the detector
        # couldn't be tuned when running on an engine).
        self.det_score_thr = det_score_thr
        self.det_iou_threshold = det_iou_threshold
        self.det_max_per_img = det_max_per_img

        self._det_engine = _load_engine(det_engine)
        self._pose_engine = _load_engine(pose_engine)
        self._det_ctx = self._det_engine.create_execution_context()
        self._pose_ctx = self._pose_engine.create_execution_context()
        self._stream = torch.cuda.Stream()

        # Reused output buffers keyed by (engine id, binding name, shape, dtype).
        # The previous code called ``torch.empty`` for every output on every
        # ``_run_engine`` invocation; with the pose batch varying frame-to-frame
        # (and chunked into up to ~7 calls/frame) that churned the CUDA caching
        # allocator and fragmented it over a long video — a steady per-frame
        # slowdown. A handful of distinct shapes recur, so caching the device
        # buffers and reusing them keeps allocation flat. Safe because every
        # caller copies the result off the buffer (NMS produces new tensors;
        # pose does ``.cpu().numpy()``) before the next call can overwrite it.
        self._out_bufs: dict = {}

        # Read the pose engine's batch profile so predict_pose can chunk
        # any oversized batch instead of hitting TRT's "Set dimension
        # [N,3,384,288] does not satisfy any optimization profiles" error.
        # (Engine was built with min=1, opt=8, max=32; querying lets us
        # adapt automatically if it's ever rebuilt with a different cap.)
        self._pose_max_batch = self._query_max_batch(self._pose_engine)

        if warmup:
            # TRT first-call latency comes from kernel init / shape resolution,
            # not autotuning — a single pass per common batch size is enough.
            self.warmup()
            torch.cuda.synchronize()

    @staticmethod
    def _query_max_batch(engine) -> int:
        trt = _get_trt()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                _, _, max_shape = engine.get_tensor_profile_shape(name, 0)
                return int(max_shape[0])
        return 32  # safe default matching trt_build.py

    def _torch_dtype(self, engine, name):
        """Map a binding's TRT dtype to torch. A weak-typed fp16 engine still
        has fp32 I/O; a strongly-typed engine (TRT 10.12+) may carry fp16 — so
        read the ACTUAL binding dtype instead of assuming fp32."""
        trt = _get_trt()
        torch = self.torch
        dt = engine.get_tensor_dtype(name)
        return {
            trt.DataType.HALF: torch.float16,
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.INT32: torch.int32,
        }.get(dt, torch.float32)

    def _run_engine(self, ctx, in_tensor, out_shapes: dict[str, tuple[int, ...]]):
        """Allocate output torch buffers, set tensor addresses, execute, return outputs.

        Uses the TensorRT 10+ tensor-named API (``num_io_tensors`` /
        ``get_tensor_*`` / ``set_input_shape`` / ``set_tensor_address`` /
        ``execute_async_v3``). The legacy ``num_bindings`` / ``execute_async_v2``
        path was removed in TRT 10. Buffers honor each binding's real dtype so
        the same code drives both fp32-I/O and strongly-typed fp16-I/O engines;
        outputs are returned as fp32 for the postprocess (NMS / SimCC decode).
        """
        trt = _get_trt()
        torch = self.torch
        engine = ctx.engine
        outputs: dict[str, torch.Tensor] = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                in_dt = self._torch_dtype(engine, name)
                if in_tensor.dtype != in_dt:
                    in_tensor = in_tensor.to(in_dt).contiguous()
                ctx.set_input_shape(name, tuple(in_tensor.shape))
                ctx.set_tensor_address(name, int(in_tensor.data_ptr()))
            else:
                shape = tuple(ctx.get_tensor_shape(name))
                # Fall back to the per-binding hint when TRT can't fully
                # infer the dynamic output shape (any -1 axis).
                if -1 in shape:
                    shape = out_shapes[name]
                dtype = self._torch_dtype(engine, name)
                # Reuse a cached device buffer for this (engine, name, shape,
                # dtype) instead of allocating every call — avoids allocator
                # churn/fragmentation across a long run. The buffer is written
                # fresh by ``execute_async_v3`` below and fully read by the
                # caller before the next call, so reuse is safe.
                key = (id(engine), name, shape, dtype)
                buf = self._out_bufs.get(key)
                if buf is None:
                    buf = torch.empty(shape, device="cuda", dtype=dtype)
                    self._out_bufs[key] = buf
                outputs[name] = buf
                ctx.set_tensor_address(name, int(buf.data_ptr()))
        ctx.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        # Normalize to fp32 — downstream NMS / SimCC decode expect float32.
        return {k: v.float() for k, v in outputs.items()}

    def predict_detector(self, frame_bgr: np.ndarray) -> Detection:
        torch = self.torch
        h, w = frame_bgr.shape[:2]
        padded, scale, pad = letterbox(frame_bgr, self.det_input_size)
        # Transfer uint8 to GPU then normalize on-device — matches the
        # PyTorch / TorchScript backends. Saves ~5ms vs the prior CPU
        # ``astype(float32)`` + ``(x-mean)/std`` on a 4.7MB array followed
        # by an fp32 H2D transfer; uint8 H2D is 4x smaller.
        x = normalize_det_input(padded, torch.device("cuda")).contiguous()

        # Detector graph emits dense pre-NMS outputs: (M, 4) boxes + (M,)
        # scores where M = total grid points across FPN levels (~8400 for
        # 640x640). Fully static shapes — TRT's enqueueV3 has nothing to
        # over/underflow on. NMS runs on the **GPU** torch buffers via
        # ``multiclass_nms_torch`` — we only sync the small post-NMS result
        # (~K boxes) to CPU instead of the dense (8400, 4) tensor, keeping
        # the CUDA pipeline saturated frame-to-frame.
        outs = self._run_engine(
            self._det_ctx,
            x,
            out_shapes={},  # all outputs are static, fallback never fires
        )
        boxes_t, scores_t = multiclass_nms_torch(
            outs["boxes"], outs["scores"],
            score_thr=self.det_score_thr,
            iou_threshold=self.det_iou_threshold,
            max_per_img=self.det_max_per_img,
        )
        if boxes_t.numel() == 0:
            return Detection(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )
        boxes = boxes_t.cpu().numpy()
        scores = scores_t.cpu().numpy()
        boxes_orig = undo_letterbox_xyxy(boxes, scale, pad, (h, w))
        return Detection(
            boxes_xyxy=boxes_orig.astype(np.float32),
            scores=scores.astype(np.float32),
        )

    def predict_pose(
        self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        n = len(boxes_xyxy)
        if n == 0:
            return (
                np.zeros((0, self.num_keypoints, 2), dtype=np.float32),
                np.zeros((0, self.num_keypoints), dtype=np.float32),
            )

        crops: list[np.ndarray] = []
        warp_mats: list[np.ndarray] = []
        for box in boxes_xyxy:
            crop, warp_mat = warp_crop(frame_bgr, box, output_size=self.pose_input_size)
            crops.append(crop)
            warp_mats.append(warp_mat)
        crops_arr = np.stack(crops, axis=0)  # (N, H, W, 3) BGR uint8
        # Stack uint8 crops, transfer once to GPU, normalize on-device.
        # Same helper used by PyTorch / TorchScript backends — handles
        # BGR->RGB swap and the per-channel mean/std on the device.
        x = normalize_pose_batch(crops_arr, torch.device("cuda")).contiguous()

        if self.flip_test:
            # On-device horizontal flip; concat the original + flipped batch
            # so we run a single (2N, ...) forward.
            x = torch.cat([x, x.flip(dims=(3,))], dim=0)
        B = x.shape[0]
        K = self.num_keypoints
        Wx = int(self.pose_input_size[0] * self.simcc_split_ratio)
        Hy = int(self.pose_input_size[1] * self.simcc_split_ratio)

        # Chunk by the engine's max profile (typically 32). A single frame
        # with many detections — e.g. 100 people × flip_test=True = 200 —
        # would exceed the profile and TRT would silently return empty
        # buffers. Chunking keeps every detection while staying inside the
        # profile bounds.
        max_b = self._pose_max_batch
        if B <= max_b:
            outs = self._run_engine(
                self._pose_ctx, x,
                out_shapes={"pred_x": (B, K, Wx), "pred_y": (B, K, Hy)},
            )
            px = outs["pred_x"].cpu().numpy()
            py = outs["pred_y"].cpu().numpy()
        else:
            px_chunks, py_chunks = [], []
            for s in range(0, B, max_b):
                xc = x[s : s + max_b].contiguous()
                bc = xc.shape[0]
                outs = self._run_engine(
                    self._pose_ctx, xc,
                    out_shapes={"pred_x": (bc, K, Wx), "pred_y": (bc, K, Hy)},
                )
                px_chunks.append(outs["pred_x"].cpu().numpy())
                py_chunks.append(outs["pred_y"].cpu().numpy())
            px = np.concatenate(px_chunks, axis=0)
            py = np.concatenate(py_chunks, axis=0)

        if self.flip_test:
            px_orig, px_flip = px[:n], px[n:]
            py_orig, py_flip = py[:n], py[n:]
            flip_idx = np.asarray(self.flip_indices, dtype=np.int64)
            px_flip = px_flip[:, :, ::-1][:, flip_idx, :]
            py_flip = py_flip[:, flip_idx, :]
            px = (px_orig + px_flip) * 0.5
            py = (py_orig + py_flip) * 0.5

        kpts_in_input, scores = decode_simcc(px, py, simcc_split_ratio=self.simcc_split_ratio)
        kpts_in_image = apply_inverse_warps_batched(warp_mats, kpts_in_input)
        return kpts_in_image, scores.astype(np.float32)
