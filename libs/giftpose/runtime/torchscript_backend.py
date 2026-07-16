"""TorchScript backend — runs ``*.torchscript.pt`` graphs from
``libs.giftpose.export.torchscript_export``.

Same preprocessing + decode pipeline as ``PyTorchBackend``; only the model
forward is swapped for the loaded ``torch.jit.ScriptModule``.

The detector graph emits dense pre-NMS outputs; NMS runs in this module's
``predict_detector`` so the graph stays static for TRT/ONNX shape inference.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
# Importing torchvision registers the ``torchvision::nms`` op with the
# TorchScript op registry; the saved detector graph references it.
import torchvision  # noqa: F401

logger = logging.getLogger(__name__)

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
from libs.giftpose.runtime.base import Backend, Detection, to_numpy


class TorchScriptBackend(Backend):
    def __init__(
        self,
        det_torchscript: str,
        pose_torchscript: str,
        device: str = "cpu",
        det_input_size: Tuple[int, int] | None = None,
        pose_input_size: Tuple[int, int] | None = None,
        flip_test: bool = True,
        det_score_thr: float = 0.05,
        det_iou_threshold: float = 0.6,
        det_max_per_img: int = 100,
        warmup: bool = True,
        compile_for_inference: bool = False,
        det_spec=None,
        pose_spec=None,
    ) -> None:
        self.device = torch.device(device)
        self.det_spec = det_spec or resolve_det(DEFAULT_DET_TAG)
        self.pose_spec = pose_spec or resolve_pose(DEFAULT_POSE_TAG)
        self.det_input_size = tuple(det_input_size or self.det_spec.input_size)
        self.pose_input_size = tuple(pose_input_size or self.pose_spec.input_size)
        self.num_keypoints = int(self.pose_spec.num_keypoints)
        self.simcc_split_ratio = float(self.pose_spec.simcc_split_ratio)
        self.flip_indices = list(get_meta(self.pose_spec.meta).FLIP_INDICES)
        self.flip_test = flip_test
        # Detector NMS knobs — wired through for parity with the PyTorch / ONNX /
        # TRT backends (previously hardcoded on this path).
        self.det_score_thr = det_score_thr
        self.det_iou_threshold = det_iou_threshold
        self.det_max_per_img = det_max_per_img

        # MPS requires a CPU round-trip — Apple's Metal backend rejects
        # float64 constants baked into traced graphs by the tracer. For
        # CPU / CUDA, load directly on the target device so frozen
        # constants stay where the runtime expects them (``.to(device)``
        # after freeze is a no-op since freeze removes parameters from
        # ``module.parameters()``).
        if self.device.type == "mps":
            self.detector = torch.jit.load(det_torchscript, map_location="cpu").to(self.device).eval()
            self.pose = torch.jit.load(pose_torchscript, map_location="cpu").to(self.device).eval()
        else:
            self.detector = torch.jit.load(det_torchscript, map_location=self.device).eval()
            self.pose = torch.jit.load(pose_torchscript, map_location=self.device).eval()

        # Strip dropout / fold constants for inference. Adds 60-120s of
        # startup on MPS but pays back in steady-state throughput. Gated
        # behind ``compile_for_inference`` so users opt in when they care
        # more about steady-state speed than process startup. Can fail on
        # graphs containing custom ops (e.g. ``torchvision::nms``) — fall
        # back to the unoptimized graph on error.
        if compile_for_inference:
            try:
                self.detector = torch.jit.optimize_for_inference(self.detector)
                self.pose = torch.jit.optimize_for_inference(self.pose)
            except RuntimeError as e:
                logger.warning(f"optimize_for_inference unavailable: {e}; using unoptimized graph")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark_limit = 16

        if warmup and self.device.type == "cuda":
            self.warmup()
            torch.cuda.synchronize()

    def predict_detector(self, frame_bgr: np.ndarray) -> Detection:
        h, w = frame_bgr.shape[:2]
        padded, scale, pad = letterbox(frame_bgr, self.det_input_size)
        x = normalize_det_input(padded, self.device)
        with torch.inference_mode():
            boxes_dense, scores_dense = self.detector(x)
        # The traced detector emits dense pre-NMS outputs (matches the ONNX
        # path); NMS runs on the source-device torch tensors so we only sync
        # the small post-NMS (~K boxes) result to CPU, not the dense tensor.
        boxes_t, scores_t = multiclass_nms_torch(
            boxes_dense, scores_dense,
            score_thr=self.det_score_thr,
            iou_threshold=self.det_iou_threshold,
            max_per_img=self.det_max_per_img,
        )
        if boxes_t.numel() == 0:
            return Detection(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )
        boxes_np = to_numpy(boxes_t)
        scores_np = to_numpy(scores_t)
        boxes_orig = undo_letterbox_xyxy(boxes_np, scale, pad, (h, w))
        return Detection(
            boxes_xyxy=boxes_orig.astype(np.float32),
            scores=scores_np.astype(np.float32),
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
        x = normalize_pose_batch(crops_arr, self.device)

        if self.flip_test:
            x = torch.cat([x, x.flip(dims=(3,))], dim=0)

        with torch.inference_mode():
            px, py = self.pose(x)

        if self.flip_test:
            px_orig, px_flip = px[:n], px[n:]
            py_orig, py_flip = py[:n], py[n:]
            flip_idx = torch.as_tensor(self.flip_indices, device=px.device, dtype=torch.long)
            px_flip = px_flip.flip(dims=(2,))[:, flip_idx, :]
            py_flip = py_flip[:, flip_idx, :]
            px = (px_orig + px_flip) * 0.5
            py = (py_orig + py_flip) * 0.5

        px_np = to_numpy(px)
        py_np = to_numpy(py)
        kpts_in_input, scores = decode_simcc(px_np, py_np, simcc_split_ratio=self.simcc_split_ratio)
        kpts_in_image = apply_inverse_warps_batched(warp_mats, kpts_in_input)
        return kpts_in_image, scores.astype(np.float32)
