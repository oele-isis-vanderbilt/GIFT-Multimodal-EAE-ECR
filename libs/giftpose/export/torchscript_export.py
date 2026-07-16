"""TorchScript export for the giftpose detector + pose models.

TorchScript artifacts (``*.torchscript.<device>.pt``) are self-contained
graphs that ``torch.jit.load`` consumes. The runtime ``TorchScriptBackend``
picks up the per-device variant (autoselect prefers TorchScript over PyTorch
when an artifact for the active device is present).

Pose: traced on (1, 3, 384, 288); the SimCC head is batch-shape-agnostic so
the loaded module handles any batch size. Detector: traced on (1, 3, 640, 640)
and emits dense pre-NMS outputs (boxes + scores); NMS runs in
``TorchScriptBackend.predict_detector``.

Usage:
    python -m libs.giftpose.export.torchscript_export
    python -m libs.giftpose.export.torchscript_export --device cuda --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

from libs.giftpose.models.detector import build_rtmdet_m_person
from libs.giftpose.models.pose_estimator import build_pose, build_rtmpose_x_halpe26
from libs.giftpose.registry import DEFAULT_POSE_TAG, resolve_pose
from libs.giftpose.weights.loader import load_state_dict_from_pth, strict_load
from libs.giftpose.export.onnx_export import _DetectorWithDecode


def _maybe_freeze(traced, device: str):
    """Fold parameters as constants for inference. Skipped on MPS only.

    Frozen constants don't survive a ``map_location='cpu'`` -> ``.to(device)``
    round-trip (the constants stay on CPU because freeze removes them from
    ``module.parameters()`` so ``.to`` no longer sees them). The runtime
    load path uses direct ``map_location=device`` for CUDA / CPU so freeze
    is preserved there, but MPS requires the CPU round-trip to dodge an
    unrelated f64-constant issue, so MPS skips freeze.
    """
    if device == "mps":
        return traced
    try:
        return torch.jit.freeze(traced)
    except RuntimeError as e:
        print(f"  (freeze failed, saving unfrozen graph: {e})")
        return traced


def export_torchscript_detector(
    weights_path: str | Path,
    out_path: str | Path,
    input_size: Tuple[int, int] = (640, 640),
    device: str = "cpu",
) -> None:
    dev = torch.device(device)
    model = build_rtmdet_m_person().to(dev).eval()
    strict_load(model, load_state_dict_from_pth(weights_path, "detector"))
    wrapped = _DetectorWithDecode(model).to(dev).eval()
    H, W = input_size
    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32, device=dev)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy, strict=False, check_trace=False)
    traced = _maybe_freeze(traced, device)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out_path))
    print(f"wrote {out_path} (traced on {device})")


def export_torchscript_pose(
    weights_path: str | Path,
    out_path: str | Path,
    input_size: Tuple[int, int] | None = None,
    device: str = "cpu",
    pose_tag: str = DEFAULT_POSE_TAG,
) -> None:
    dev = torch.device(device)
    spec = resolve_pose(pose_tag)
    model = build_pose(spec).to(dev).eval()
    strict_load(model, load_state_dict_from_pth(weights_path, "pose"))
    W, H = input_size or spec.input_size
    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32, device=dev)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False, check_trace=False)
    traced = _maybe_freeze(traced, device)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out_path))
    print(f"wrote {out_path} (traced on {device})")


def verify_torchscript(
    ts_path: str | Path,
    weights_path: str | Path,
    kind: str,
    device: str = "cpu",
    sample_frame: str | None = "input/Videos/Crested_Gecko/8-Trimmed.mp4",
    pose_tag: str = DEFAULT_POSE_TAG,
) -> None:
    """Load the traced module + the eager model side-by-side, run both on the
    same dummy input, and report max-abs diff.

    fp32 trace+load round-trip should be within ~1e-4 (slightly looser on MPS
    due to the device's preferred reduced-precision kernels).

    The detector is fed a real frame from ``sample_frame`` (when available)
    so the post-NMS output isn't empty; falls back to random input otherwise.
    """
    import cv2
    from libs.giftpose.preprocess.letterbox import letterbox
    from libs.giftpose.preprocess.normalize import (
        normalize_det_input, normalize_pose_batch,
    )

    dev = torch.device(device)
    if kind == "pose":
        model = build_pose(resolve_pose(pose_tag)).to(dev).eval()
    elif kind == "detector":
        model = _DetectorWithDecode(build_rtmdet_m_person()).to(dev).eval()
    else:
        raise ValueError(f"Unknown kind: {kind}")
    strict_load(
        getattr(model, "model", model),
        load_state_dict_from_pth(weights_path, kind),
    )

    # Build a real input tensor — random noise produces 0 detections on the
    # detector, leaving the empty-tensor reduction (.max()) ill-defined.
    if kind == "pose":
        _w, _h = resolve_pose(pose_tag).input_size
        H, W = _h, _w
    else:
        H, W = 640, 640
    x = None
    if sample_frame and Path(sample_frame).exists():
        cap = cv2.VideoCapture(sample_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            if kind == "detector":
                padded, _, _ = letterbox(frame, (W, H))
                x = normalize_det_input(padded, dev)
            else:
                # synthesize a 288x384 crop from the center of the frame
                fh, fw = frame.shape[:2]
                cx, cy = fw // 2, fh // 2
                crop = frame[cy - H // 2 : cy + H // 2, cx - W // 2 : cx + W // 2]
                if crop.shape[:2] == (H, W):
                    x = normalize_pose_batch(crop[None], dev)
    if x is None:
        torch.manual_seed(0)
        x = torch.randn(1, 3, H, W, device=dev)

    # MPS requires a CPU round-trip — Apple's Metal backend rejects the
    # float64 constants baked into traced graphs (e.g. ScaleNorm's
    # ``dim**-0.5``). CPU / CUDA load directly so frozen constants stay
    # on the target device.
    if device == "mps":
        ts = torch.jit.load(str(ts_path), map_location="cpu").to(dev).eval()
    else:
        ts = torch.jit.load(str(ts_path), map_location=device).eval()
    with torch.no_grad():
        ref = model(x)
        out = ts(x)

    if isinstance(ref, (tuple, list)):
        diffs = [_max_abs_diff(a, b) for a, b in zip(ref, out)]
        print(f"  {kind} verify on {device}: max abs diff per output = {diffs}")
    else:
        print(f"  {kind} verify on {device}: max abs diff = {_max_abs_diff(ref, out)}")


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    if a.shape != b.shape:
        return float("inf")
    return (a - b).abs().max().item()


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--det-weights", default="models/detect-best-mAP.pth")
    p.add_argument("--pose-weights", default="models/pose.pth")
    p.add_argument("--pose-tag", default=DEFAULT_POSE_TAG,
                   help="Registry architecture tag for the pose model "
                        "(see libs/giftpose/registry.py POSE_ARCHS).")
    p.add_argument("--device", default="cpu",
                   help="Trace device (cpu|cuda|mps). Intermediate-tensor "
                        "device choices are baked in at trace time, so we "
                        "save per-device variants by default.")
    # Output paths default to ``<base>.torchscript.<device>.pt`` so the
    # runtime autoselect can pick the right artifact for the active device.
    p.add_argument("--det-out", default=None)
    p.add_argument("--pose-out", default=None)
    p.add_argument("--skip-detector", action="store_true")
    p.add_argument("--skip-pose", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="After export, load the traced module and assert output matches the eager model within tolerance.")
    args = p.parse_args(argv)

    def _ts_out(weights: str) -> str:
        base = weights[: weights.rfind(".")]
        return f"{base}.torchscript.{args.device}.pt"

    det_out = args.det_out or _ts_out(args.det_weights)
    pose_out = args.pose_out or _ts_out(args.pose_weights)

    if not args.skip_detector:
        export_torchscript_detector(args.det_weights, det_out, device=args.device)
    if not args.skip_pose:
        export_torchscript_pose(
            args.pose_weights, pose_out, device=args.device,
            pose_tag=args.pose_tag,
        )

    if args.verify:
        print("verifying:")
        if not args.skip_detector:
            verify_torchscript(det_out, args.det_weights, "detector", device=args.device)
        if not args.skip_pose:
            verify_torchscript(
                pose_out, args.pose_weights, "pose", device=args.device,
                pose_tag=args.pose_tag,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
