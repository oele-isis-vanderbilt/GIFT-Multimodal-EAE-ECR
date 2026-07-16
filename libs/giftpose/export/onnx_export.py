"""ONNX export for the giftpose detector + pose models.

Detector ONNX wraps the post-processing decode + NMS inside the graph so
``onnxruntime`` consumers receive ``(boxes_xyxy, scores)`` directly. Pose ONNX
returns the raw SimCC ``(pred_x, pred_y)`` tensors; flip-test averaging stays
in the runtime backend (cheaper than baking it into the graph).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from libs.giftpose.models.detector import build_rtmdet_m_person
from libs.giftpose.models.pose_estimator import build_pose, build_rtmpose_x_halpe26
from libs.giftpose.registry import DEFAULT_POSE_TAG, resolve_pose
from libs.giftpose.models.rtmdet_head import decode_rtmdet_dense
from libs.giftpose.weights.loader import load_state_dict_from_pth, strict_load


class _DetectorWithDecode(nn.Module):
    """Wrap the detector so ``forward`` returns the **dense pre-NMS** boxes +
    scores. NMS is intentionally kept out of the ONNX graph and run in Python
    by the runtime backends.

    Output shapes (for 640x640 input):
        boxes:  (M, 4) float32 — M = sum of FPN-level grid points
                                (80x80 + 40x40 + 20x20 = 8400). Static.
        scores: (M,)  float32 — sigmoid-applied class scores. Static.

    Why dense pre-NMS (vs. graph-internal NMS): TRT 10's ``enqueueV3`` cannot
    resolve data-dependent output shapes from ONNX's ``NonMaxSuppression`` op
    (the symbolic dim propagates and surfaces as ``shape calculation overflow
    -36028…``). Emitting dense outputs keeps the graph fully static; the
    runtime backends filter by ``score_thr`` and run NMS via
    ``torchvision.ops.nms`` in Python (~ms on 8400 boxes).
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        cls_scores, bbox_preds = self.model.forward_features(x)
        return decode_rtmdet_dense(
            cls_scores, bbox_preds, strides=self.model.bbox_head.strides,
        )


def export_onnx_detector(
    weights_path: str | Path,
    out_path: str | Path,
    input_size: Tuple[int, int] = (640, 640),
    opset: int = 17,
) -> None:
    model = build_rtmdet_m_person().eval()
    strict_load(model, load_state_dict_from_pth(weights_path, "detector"))
    wrapped = _DetectorWithDecode(model).eval()
    H, W = input_size
    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)
    # No dynamic_axes — the wrapped detector pads to fixed (max_per_img, 4)
    # and (max_per_img,), so both ONNX outputs have fully static shapes.
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"wrote {out_path}")


def export_onnx_pose(
    weights_path: str | Path,
    out_path: str | Path,
    input_size: Tuple[int, int] | None = None,
    opset: int = 17,
    pose_tag: str = DEFAULT_POSE_TAG,
) -> None:
    spec = resolve_pose(pose_tag)
    model = build_pose(spec).eval()
    strict_load(model, load_state_dict_from_pth(weights_path, "pose"))
    W, H = input_size or spec.input_size
    dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)
    # RTMW3D heads emit a third (z) SimCC output.
    out_names = ["pred_x", "pred_y"] + (["pred_z"] if spec.has_z else [])
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=out_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"images": {0: "batch"},
                      **{n: {0: "batch"} for n in out_names}},
    )
    print(f"wrote {out_path}")


def verify_onnx(
    onnx_path: str | Path,
    weights_path: str | Path,
    kind: str,
    tol: float = 1e-3,
) -> float:
    """Run the eager model and the exported ONNX on the same input under
    onnxruntime; return the max-abs output diff and raise if it exceeds ``tol``.

    This is the ONNX analogue of ``torchscript_export.verify_torchscript`` —
    which is why the TorchScript path has never shipped a silent export bug and
    the ONNX path (previously unverified) could. A stale ``.onnx`` (re-exported
    from an older checkpoint than the ``.pth``) or a genuine op-translation bug
    both surface here as a large diff, so callers can refuse to build a TRT
    engine on top of it.

    ``kind`` is ``"detector"`` (compares dense ``boxes``/``scores``) or
    ``"pose"`` (compares ``pred_x``/``pred_y``). Uses random input — sufficient
    to catch op-translation / stale-weight divergence; the dedicated
    ``tools/parity.py`` harness covers real-frame, post-NMS recall.
    """
    import numpy as np
    import onnxruntime as ort

    if kind == "detector":
        model = _DetectorWithDecode(build_rtmdet_m_person()).eval()
        strict_load(model.model, load_state_dict_from_pth(weights_path, "detector"))
        H, W = 640, 640
    elif kind == "pose":
        model = build_rtmpose_x_halpe26().eval()
        strict_load(model, load_state_dict_from_pth(weights_path, "pose"))
        H, W = 384, 288
    else:
        raise ValueError(f"unknown kind: {kind}")

    torch.manual_seed(0)
    x = torch.randn(1, 3, H, W, dtype=torch.float32)
    with torch.inference_mode():
        ref = model(x)
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    outs = sess.run(None, {"images": x.numpy()})

    # Relative tolerance per output: the detector emits box COORDINATES (pixel
    # magnitudes up to ~700) alongside [0,1] scores, so a flat absolute tol
    # would either false-flag sub-pixel rounding on the boxes or miss score
    # drift. Scale by each output's own magnitude — a stale checkpoint shifts
    # scores by O(0.1) and boxes by many px (both caught), while fp rounding
    # (~1e-3 relative) passes.
    worst_rel = 0.0
    for r, o in zip(ref, outs):
        r = r.detach().cpu().numpy()
        if r.shape != o.shape:
            raise RuntimeError(
                f"{kind} ONNX verify: shape mismatch {r.shape} vs {o.shape} "
                f"for {onnx_path}"
            )
        max_abs = float(np.abs(r.astype(np.float64) - o.astype(np.float64)).max())
        scale = max(1.0, float(np.abs(r).max()))
        worst_rel = max(worst_rel, max_abs / scale)
    print(f"  {kind} ONNX verify: worst relative diff = {worst_rel:.3e} (tol {tol:.0e})")
    if worst_rel > tol:
        raise RuntimeError(
            f"{kind} ONNX export diverges from eager ({worst_rel:.3e} > {tol:.0e} "
            f"relative) for {onnx_path} — refusing to treat it as valid. Re-export "
            f"from the current {weights_path}; if it still fails the export itself "
            f"is buggy."
        )
    return worst_rel


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--det-weights", default="models/detect-best-mAP.pth")
    p.add_argument("--pose-weights", default="models/pose.pth")
    p.add_argument("--pose-tag", default=DEFAULT_POSE_TAG,
                   help="Registry architecture tag for the pose model.")
    p.add_argument("--det-out", default="models/detect-best-mAP.onnx")
    p.add_argument("--pose-out", default="models/pose.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--skip-detector", action="store_true")
    p.add_argument("--skip-pose", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="After export, load the .onnx under onnxruntime and "
                        "assert it matches the eager model within tolerance.")
    args = p.parse_args(argv)

    if not args.skip_detector:
        export_onnx_detector(args.det_weights, args.det_out, opset=args.opset)
    if not args.skip_pose:
        export_onnx_pose(args.pose_weights, args.pose_out, opset=args.opset,
                         pose_tag=args.pose_tag)

    if args.verify:
        print("verifying:")
        if not args.skip_detector:
            verify_onnx(args.det_out, args.det_weights, "detector")
        if not args.skip_pose:
            verify_onnx(args.pose_out, args.pose_weights, "pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
