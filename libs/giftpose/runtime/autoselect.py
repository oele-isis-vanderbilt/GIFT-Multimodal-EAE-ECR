"""Pick the best available runtime backend.

Order (per device):
  cuda: TRT (.engine) > ONNX (CUDA EP) > TorchScript > PyTorch
  cpu:  ONNX (CPU EP) > TorchScript > PyTorch
  mps:  TorchScript (.torchscript.mps.pt) > PyTorch  [ONNX/CoreML skipped —
        the graph splits into 9-12 partitions and runs slower than native MPS]

On CUDA without TRT engines, note that TorchScript artifacts are traced fp32
(no autocast baked in) and may run slower than the eager PyTorch backend's
fp16 autocast path; pass ``MMPoseInferencer(prefer_backend="pytorch")`` to
override the autoselect order in that case.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Optional

from libs.giftpose.runtime.base import Backend

logger = logging.getLogger(__name__)


def _try_import(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _onnx_supports_device(device: str) -> bool:
    """True when onnxruntime can actually execute on ``device``.

    A CPU-only ORT build still imports fine on a CUDA box; without this gate
    autoselect would pick ONNX and ORT would silently fall back to the CPU EP,
    leaving the GPU idle. (mps never reaches here — that branch is excluded.)
    """
    if device != "cuda":
        return True  # CPUExecutionProvider is always present
    try:
        import onnxruntime as ort
        return bool(
            set(ort.get_available_providers())
            & {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        )
    except Exception:
        return False


def _resolve_artifact(weights_path: str, suffix: str) -> Optional[str]:
    """Look for ``<weights>.<suffix>`` next to the supplied weight file —
    e.g. ``models/pose.pth`` -> ``models/pose.onnx``.
    """
    base = os.path.splitext(weights_path)[0]
    path = base + suffix
    return path if os.path.isfile(path) else None


def _resolve_torchscript(weights_path: str, device: str) -> Optional[str]:
    """TorchScript graphs bake in the trace device for intermediate tensors,
    so we look for a per-device variant first
    (``models/pose.torchscript.<device>.pt``) before falling back to the
    generic ``.torchscript.pt`` that was traced on CPU.
    """
    per_device = _resolve_artifact(weights_path, f".torchscript.{device}.pt")
    if per_device:
        return per_device
    return _resolve_artifact(weights_path, ".torchscript.pt")


def select_backend(
    det_weights: str,
    pose_weights: str,
    device: str = "cpu",
    prefer: str | None = None,
    fp16: bool | None = None,
    warmup: bool = True,
    compile_for_inference: bool = False,
    det_score_thr: float = 0.05,
    det_iou_threshold: float = 0.6,
    det_max_per_img: int = 100,
    det_spec=None,
    pose_spec=None,
) -> Backend:
    """Construct and return a ready-to-use backend.

    Args:
        det_weights / pose_weights: paths to legacy mmengine ``.pth`` files.
        device: target compute device for the PyTorch / ORT-CUDA path.
        prefer: force a specific backend (``"pytorch"`` / ``"torchscript"`` /
            ``"onnx"`` / ``"trt"``) for testing. ``None`` runs autoselect.
        fp16: PyTorch backend FP16 toggle. Defaults to True on CUDA.
        warmup: if True (default), prime cuDNN / TRT / kernel caches at
            construction time. Adds a few hundred ms to startup but eliminates
            first-frame stalls in the per-frame pipeline. Pass False for
            short-lived test scripts where startup overhead matters more
            than per-frame jitter.
        compile_for_inference: if True, run a one-time graph optimizer at
            backend construction (PyTorch: ``torch.compile``; TorchScript:
            ``torch.jit.optimize_for_inference``). Adds 30-120s of startup
            cost in exchange for faster steady-state inference. No-op for
            ONNX / TRT (those graphs are already optimized at session/engine
            build time). Default False.
    """
    # Architecture specs default to the registry's production tags so
    # spec-unaware callers (export scripts, tests) keep working unchanged.
    if det_spec is None or pose_spec is None:
        from libs.giftpose.registry import (
            DEFAULT_DET_TAG, DEFAULT_POSE_TAG, resolve_det, resolve_pose,
        )
        det_spec = det_spec or resolve_det(DEFAULT_DET_TAG)
        pose_spec = pose_spec or resolve_pose(DEFAULT_POSE_TAG)

    det_engine = _resolve_artifact(det_weights, ".engine")
    pose_engine = _resolve_artifact(pose_weights, ".engine")
    det_onnx = _resolve_artifact(det_weights, ".onnx")
    pose_onnx = _resolve_artifact(pose_weights, ".onnx")
    det_ts = _resolve_torchscript(det_weights, device)
    pose_ts = _resolve_torchscript(pose_weights, device)

    want = prefer
    if want is None:
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except (ImportError, AttributeError, RuntimeError):
            # RuntimeError covers broken CUDA installs / driver mismatch.
            cuda_ok = False
        if (
            device == "cuda" and cuda_ok and det_engine and pose_engine
            and _try_import("tensorrt")
        ):
            want = "trt"
        elif (
            device != "mps"  # CoreMLEP fragments the graph; skip ONNX on MPS
            and det_onnx and pose_onnx and _try_import("onnxruntime")
            and _onnx_supports_device(device)
        ):
            want = "onnx"
        elif det_ts and pose_ts:
            want = "torchscript"
        else:
            want = "pytorch"
    logger.info("giftpose backend: %s (device=%s)", want, device)

    det_nms = dict(
        det_score_thr=det_score_thr,
        det_iou_threshold=det_iou_threshold,
        det_max_per_img=det_max_per_img,
    )
    specs = dict(det_spec=det_spec, pose_spec=pose_spec)
    if want == "trt":
        from libs.giftpose.runtime.trt_backend import TRTBackend
        assert det_engine and pose_engine
        return TRTBackend(det_engine, pose_engine, warmup=warmup, **det_nms, **specs)
    if want == "onnx":
        from libs.giftpose.runtime.onnx_backend import ONNXBackend
        assert det_onnx and pose_onnx
        # ONNX EP graph init already happens at session-construction time —
        # no per-shape autotune layer to prime, so no warmup pass.
        return ONNXBackend(det_onnx, pose_onnx, device=device, **det_nms, **specs)
    if want == "torchscript":
        from libs.giftpose.runtime.torchscript_backend import TorchScriptBackend
        assert det_ts and pose_ts
        return TorchScriptBackend(
            det_ts, pose_ts, device=device, warmup=warmup,
            compile_for_inference=compile_for_inference, **det_nms, **specs,
        )
    from libs.giftpose.runtime.pytorch_backend import PyTorchBackend
    return PyTorchBackend(
        det_weights, pose_weights, device=device, fp16=fp16, warmup=warmup,
        compile_for_inference=compile_for_inference, **det_nms, **specs,
    )
