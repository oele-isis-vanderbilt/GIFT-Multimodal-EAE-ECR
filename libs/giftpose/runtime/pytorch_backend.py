"""PyTorch backend — universal floor across CPU / MPS / CUDA.

Performance:
  - On CUDA: ``cudnn.benchmark`` autotunes conv kernels for the fixed
    detector / pose input shapes; ``channels_last`` memory format unlocks
    Tensor Core throughput on Ampere+; ``torch.autocast`` runs matmul/conv
    in fp16 while keeping params fp32 for numerical stability.
  - Batched pose forward: all detections in one frame are stacked into a
    single ``(N, 3, 384, 288)`` tensor. With ``flip_test=True``, the original
    and horizontally-flipped crops are concatenated into a ``(2N, ...)`` batch
    so the flip test costs one forward, not two.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

from libs.giftpose.codecs.simcc import decode_simcc
from libs.giftpose.meta import get_meta
from libs.giftpose.models.detector import build_detector
from libs.giftpose.models.pose_estimator import build_pose
from libs.giftpose.registry import DEFAULT_DET_TAG, DEFAULT_POSE_TAG, resolve_det, resolve_pose
from libs.giftpose.preprocess.letterbox import letterbox, undo_letterbox_xyxy
from libs.giftpose.preprocess.normalize import (
    normalize_det_input,
    normalize_pose_batch,
)
from libs.giftpose.preprocess.topdown_affine import apply_inverse_warps_batched, warp_crop
from libs.giftpose.runtime.base import Backend, Detection, to_numpy
from libs.giftpose.weights.loader import load_state_dict_from_pth, strict_load


class PyTorchBackend(Backend):
    def __init__(
        self,
        det_weights: str,
        pose_weights: str,
        device: str = "cpu",
        fp16: bool | None = None,
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
        if fp16 is None:
            fp16 = self.device.type == "cuda"
        self.fp16 = fp16
        self.det_spec = det_spec or resolve_det(DEFAULT_DET_TAG)
        self.pose_spec = pose_spec or resolve_pose(DEFAULT_POSE_TAG)
        self.det_input_size = tuple(det_input_size or self.det_spec.input_size)
        self.pose_input_size = tuple(pose_input_size or self.pose_spec.input_size)
        self.num_keypoints = int(self.pose_spec.num_keypoints)
        self.simcc_split_ratio = float(self.pose_spec.simcc_split_ratio)
        self.flip_indices = list(get_meta(self.pose_spec.meta).FLIP_INDICES)
        self.flip_test = flip_test
        self.det_score_thr = det_score_thr
        self.det_iou_threshold = det_iou_threshold
        self.det_max_per_img = det_max_per_img

        self.detector = build_detector(self.det_spec).to(self.device).eval()
        strict_load(self.detector, load_state_dict_from_pth(det_weights, "detector"))
        self.pose = build_pose(self.pose_spec).to(self.device).eval()
        strict_load(self.pose, load_state_dict_from_pth(pose_weights, "pose"))

        if self.device.type == "cuda":
            # Detector input is constant (640x640); pose input batch dim varies
            # but ``benchmark_limit`` caps cache growth and ``warmup`` below
            # primes the common batch sizes so first-encounter autotune stalls
            # don't surface mid-pipeline.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark_limit = 16
            self.detector = self.detector.to(memory_format=torch.channels_last)
            self.pose = self.pose.to(memory_format=torch.channels_last)

        if compile_for_inference and hasattr(torch, "compile"):
            # ``torch.compile`` traces the model into a fused graph at
            # first-call time (~30-60s warmup). Best speedup on CUDA
            # (Inductor backend) and CPU; MPS support is partial in
            # PyTorch 2.x and may fall back silently. Failures are caught
            # and the eager modules are kept.
            try:
                self.detector = torch.compile(self.detector, mode="reduce-overhead")
                self.pose = torch.compile(self.pose, mode="reduce-overhead")
            except Exception as e:
                logger.warning(
                    f"torch.compile unavailable on {self.device.type}: {e}; using eager modules"
                )

        if warmup and self.device.type == "cuda":
            self.warmup()
            torch.cuda.synchronize()

    def _amp_ctx(self):
        # Selective fp16 via autocast: matmul/conv run fp16 while params stay
        # fp32 (better numerical stability than full ``.half()``). MPS autocast
        # is slower than fp32 on small inputs so we keep it CUDA-only.
        if self.fp16 and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    # ---- detector ----------------------------------------------------------
    def predict_detector(self, frame_bgr: np.ndarray) -> Detection:
        h, w = frame_bgr.shape[:2]
        padded, scale, pad = letterbox(frame_bgr, self.det_input_size)
        x = normalize_det_input(padded, self.device, dtype=torch.float32)
        if self.device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        # ``model.detect()`` runs forward + sigmoid + grid decode + NMS all
        # on the source device. NMS uses ``multiclass_nms_torch`` which is
        # device-aware: full GPU on CUDA, per-op CPU bounce on MPS (only the
        # post-thresholded ~hundreds-of-boxes tensor moves, not the dense
        # pre-NMS output). Avoids per-frame GPU->CPU sync of dense outputs.
        with torch.inference_mode(), self._amp_ctx():
            boxes, scores = self.detector.detect(
                x,
                score_thr=self.det_score_thr,
                iou_threshold=self.det_iou_threshold,
                max_per_img=self.det_max_per_img,
            )
        boxes_np = to_numpy(boxes)
        scores_np = to_numpy(scores)
        if boxes_np.size == 0:
            return Detection(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )
        boxes_orig = undo_letterbox_xyxy(boxes_np, scale, pad, (h, w))
        return Detection(boxes_xyxy=boxes_orig.astype(np.float32), scores=scores_np.astype(np.float32))

    # ---- pose --------------------------------------------------------------
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
        crops_arr = np.stack(crops, axis=0)  # (N, H, W, 3) BGR uint8

        x = normalize_pose_batch(crops_arr, self.device, dtype=torch.float32)  # (N, 3, H, W)

        if self.flip_test:
            x_in = torch.cat([x, x.flip(dims=(3,))], dim=0)  # (2N, ...)
        else:
            x_in = x

        if self.device.type == "cuda":
            x_in = x_in.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode(), self._amp_ctx():
            out = self.pose(x_in)
        px, py = out[0], out[1]
        pz = out[2] if len(out) == 3 else None  # RTMW3D z branch

        if self.flip_test:
            flip_idx = torch.as_tensor(self.flip_indices, device=px.device, dtype=torch.long)
            px_orig, px_flip = px[:n], px[n:]
            py_orig, py_flip = py[:n], py[n:]
            # Reverse the flipped output along its SimCC axis to undo the
            # horizontal flip; then permute keypoint indices via FLIP_INDICES.
            px_flip = px_flip.flip(dims=(2,))[:, flip_idx, :]
            py_flip = py_flip[:, flip_idx, :]  # y-axis indices unchanged by horizontal flip
            px = (px_orig + px_flip) * 0.5
            py = (py_orig + py_flip) * 0.5
            if pz is not None:
                # Depth is invariant under a horizontal mirror — only the
                # left/right keypoint identities swap.
                pz_orig, pz_flip = pz[:n], pz[n:]
                pz = (pz_orig + pz_flip[:, flip_idx, :]) * 0.5

        px_np = to_numpy(px)
        py_np = to_numpy(py)
        if pz is not None:
            from libs.giftpose.codecs.simcc3d import decode_simcc3d

            kpts_in_input, scores, z_metric = decode_simcc3d(
                px_np, py_np, to_numpy(pz),
                simcc_split_ratio=self.simcc_split_ratio,
                z_input_size=self.pose_spec.z_input_size or 288,
            )
            kpts_in_image = apply_inverse_warps_batched(warp_mats, kpts_in_input)
            return kpts_in_image, scores.astype(np.float32), z_metric

        kpts_in_input, scores = decode_simcc(px_np, py_np, simcc_split_ratio=self.simcc_split_ratio)
        kpts_in_image = apply_inverse_warps_batched(warp_mats, kpts_in_input)
        return kpts_in_image, scores.astype(np.float32)
