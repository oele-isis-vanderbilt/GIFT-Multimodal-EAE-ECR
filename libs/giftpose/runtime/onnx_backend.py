"""ONNX Runtime backend — picks an execution provider matching the device.

Order of EP fallback (whichever is available + matches the requested device):
  - cuda  -> Tensorrt (if ORT was built with it), CUDA, then CPU
  - mps   -> CoreMLExecutionProvider (macOS), then CPU
  - cpu   -> CPUExecutionProvider
"""
from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from libs.giftpose.codecs.simcc import decode_simcc
from libs.giftpose.meta import get_meta
from libs.giftpose.registry import DEFAULT_DET_TAG, DEFAULT_POSE_TAG, resolve_det, resolve_pose
from libs.giftpose.postprocess.nms import multiclass_nms_numpy
from libs.giftpose.preprocess.letterbox import letterbox, undo_letterbox_xyxy
from libs.giftpose.preprocess.normalize import (
    DET_MEAN_BGR, DET_STD_BGR, POSE_MEAN_BGR, POSE_STD_BGR,
)
from libs.giftpose.preprocess.topdown_affine import apply_inverse_warps_batched, warp_crop
from libs.giftpose.runtime.base import Backend, Detection


def _pick_eps(device: str) -> List[str]:
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    if device == "cuda":
        order: List[str] = []
        # TRT EP only present when ORT was built with TensorRT support; gating
        # on availability lets the same code work on CUDA-only ORT builds.
        if "TensorrtExecutionProvider" in available:
            order.append("TensorrtExecutionProvider")
        order.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        return order
    if device == "mps":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _normalize_pose_np(crops_bgr: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) BGR uint8 -> (N, 3, H, W) RGB float32 normalized.

    Pure-numpy duplicate of preprocess.normalize.normalize_pose_batch — kept
    on the ONNX path to avoid a per-frame numpy↔torch round-trip on CPU.
    """
    rgb = crops_bgr[..., ::-1].astype(np.float32)
    mean = np.array([POSE_MEAN_BGR[2], POSE_MEAN_BGR[1], POSE_MEAN_BGR[0]], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([POSE_STD_BGR[2], POSE_STD_BGR[1], POSE_STD_BGR[0]], dtype=np.float32).reshape(1, 1, 1, 3)
    rgb = (rgb - mean) / std
    return np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))


def _normalize_det_np(image_bgr: np.ndarray) -> np.ndarray:
    f = image_bgr.astype(np.float32)
    f = (f - DET_MEAN_BGR) / DET_STD_BGR
    return np.ascontiguousarray(f.transpose(2, 0, 1))[None]


class ONNXBackend(Backend):
    def __init__(
        self,
        det_onnx: str,
        pose_onnx: str,
        device: str = "cpu",
        det_input_size: Tuple[int, int] | None = None,
        pose_input_size: Tuple[int, int] | None = None,
        flip_test: bool = True,
        det_score_thr: float = 0.05,
        det_iou_threshold: float = 0.6,
        det_max_per_img: int = 100,
        det_spec=None,
        pose_spec=None,
    ) -> None:
        import onnxruntime as ort

        self.device = device
        self.det_spec = det_spec or resolve_det(DEFAULT_DET_TAG)
        self.pose_spec = pose_spec or resolve_pose(DEFAULT_POSE_TAG)
        self.det_input_size = tuple(det_input_size or self.det_spec.input_size)
        self.pose_input_size = tuple(pose_input_size or self.pose_spec.input_size)
        self.num_keypoints = int(self.pose_spec.num_keypoints)
        self.simcc_split_ratio = float(self.pose_spec.simcc_split_ratio)
        self.flip_indices = list(get_meta(self.pose_spec.meta).FLIP_INDICES)
        self.flip_test = flip_test
        # Detector NMS knobs — wired through for parity with the PyTorch backend
        # (previously hardcoded on this path).
        self.det_score_thr = det_score_thr
        self.det_iou_threshold = det_iou_threshold
        self.det_max_per_img = det_max_per_img

        eps = _pick_eps(device)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Reuse buffer allocations across runs (intermediate tensors + arenas)
        # to amortize malloc/free.
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_cpu_mem_arena = True
        if device == "cpu":
            sess_opts.intra_op_num_threads = int(
                os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 4
            )
        self.det_sess = ort.InferenceSession(det_onnx, sess_options=sess_opts, providers=eps)
        self.pose_sess = ort.InferenceSession(pose_onnx, sess_options=sess_opts, providers=eps)
        # ORT silently drops unavailable EPs at session creation (e.g. CUDA EP
        # listed but its libs fail to load) — log what actually got used.
        logger.info(
            "ONNX EPs — det: %s, pose: %s",
            self.det_sess.get_providers()[0],
            self.pose_sess.get_providers()[0],
        )

    def predict_detector(self, frame_bgr: np.ndarray) -> Detection:
        h, w = frame_bgr.shape[:2]
        padded, scale, pad = letterbox(frame_bgr, self.det_input_size)
        x = _normalize_det_np(padded)
        # Detector graph emits dense pre-NMS outputs: (M, 4) boxes + (M,)
        # scores where M = total grid points across FPN levels (~8400 for
        # 640x640). NMS runs in Python here — graph stays fully static so
        # TRT/ONNX shape inference doesn't choke.
        boxes_dense, scores_dense = self.det_sess.run(None, {"images": x})
        boxes, scores = multiclass_nms_numpy(
            boxes_dense, scores_dense,
            score_thr=self.det_score_thr,
            iou_threshold=self.det_iou_threshold,
            max_per_img=self.det_max_per_img,
        )
        if scores.size == 0:
            return Detection(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )
        boxes_orig = undo_letterbox_xyxy(boxes, scale, pad, (h, w))
        return Detection(
            boxes_xyxy=boxes_orig.astype(np.float32),
            scores=scores.astype(np.float32),
        )

    def predict_pose(
        self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
        crops_arr = np.stack(crops, axis=0)
        x = _normalize_pose_np(crops_arr)  # (N, 3, H, W)

        if self.flip_test:
            x = np.concatenate([x, x[..., ::-1]], axis=0)  # (2N, 3, H, W)

        outs = self.pose_sess.run(None, {"images": x})  # numpy outputs
        px, py = outs[0], outs[1]
        pz = outs[2] if len(outs) == 3 else None  # RTMW3D z branch

        if self.flip_test:
            flip_idx = np.asarray(self.flip_indices, dtype=np.int64)
            px_orig, px_flip = px[:n], px[n:]
            py_orig, py_flip = py[:n], py[n:]
            px_flip = px_flip[:, :, ::-1][:, flip_idx, :]
            py_flip = py_flip[:, flip_idx, :]
            px = (px_orig + px_flip) * 0.5
            py = (py_orig + py_flip) * 0.5
            if pz is not None:
                pz = (pz[:n] + pz[n:][:, flip_idx, :]) * 0.5

        if pz is not None:
            from libs.giftpose.codecs.simcc3d import decode_simcc3d

            kpts_in_input, scores, z_metric = decode_simcc3d(
                px, py, pz,
                simcc_split_ratio=self.simcc_split_ratio,
                z_input_size=getattr(self.pose_spec, "z_input_size", 0) or 288,
            )
            kpts_in_image = apply_inverse_warps_batched(warp_mats, kpts_in_input)
            return kpts_in_image, scores.astype(np.float32), z_metric

        kpts_in_input, scores = decode_simcc(px, py, simcc_split_ratio=self.simcc_split_ratio)
        kpts_in_image = apply_inverse_warps_batched(warp_mats, kpts_in_input)
        return kpts_in_image, scores.astype(np.float32)
