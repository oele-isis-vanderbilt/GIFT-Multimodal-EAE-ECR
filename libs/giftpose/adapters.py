"""Canonical keypoint adapter — every pose backend's output is converted to
the Halpe-26 layout before anything downstream (tracker, metrics, gaze,
overlays, caches) sees it.

This is the downstream-safety linchpin of the multi-backend design: the
tracker's anchor indices (Halpe 17-25), ankle positioning ([15, 16]), the
gaze face indices (0-4), skeleton edges, wrist/shoulder/elbow metric indices
and every ``== 26`` gate keep working identically for all backends. Extra
information (the full 133-keypoint set, z-values) is returned separately and
carried as optional metadata — never inside the canonical block, whose
column-2 score convention the tracker depends on.

Index maps (verified against ``meta/halpe26.py`` and the generated
``meta/wholebody133.py``):
  - COCO body 0-16 is index-identical in both layouts.
  - Halpe 17 head / 18 neck / 19 hip do not exist in COCO-WholeBody and are
    synthesized (face-center, shoulder midpoint, hip midpoint), with the min
    of the parent scores.
  - Feet: WholeBody 17 L-big-toe, 18 L-small-toe, 19 L-heel, 20 R-big-toe,
    21 R-small-toe, 22 R-heel  ->  Halpe 20, 22, 24, 21, 23, 25.
"""
from __future__ import annotations

import numpy as np

# (wholebody_index, halpe_index) for the feet block.
_WB_FOOT_TO_HALPE = ((17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25))


def wholebody_to_halpe26(
    kpts: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert COCO-WholeBody-133 keypoints to canonical Halpe-26.

    Args:
        kpts: ``(N, 133, 2)`` image coordinates.
        scores: ``(N, 133)``.
    Returns:
        ``(kpts26, scores26)`` with shapes ``(N, 26, 2)`` / ``(N, 26)``.
    """
    n = kpts.shape[0]
    out_k = np.zeros((n, 26, 2), dtype=np.float32)
    out_s = np.zeros((n, 26), dtype=np.float32)

    # COCO body block is index-identical.
    out_k[:, :17] = kpts[:, :17]
    out_s[:, :17] = scores[:, :17]

    # Synthesized Halpe head (17): face center from eyes + ears (1-4).
    face = kpts[:, 1:5]
    face_s = scores[:, 1:5]
    out_k[:, 17] = face.mean(axis=1)
    out_s[:, 17] = face_s.min(axis=1)

    # Neck (18) = shoulder midpoint; hip (19) = hip midpoint.
    out_k[:, 18] = (kpts[:, 5] + kpts[:, 6]) * 0.5
    out_s[:, 18] = np.minimum(scores[:, 5], scores[:, 6])
    out_k[:, 19] = (kpts[:, 11] + kpts[:, 12]) * 0.5
    out_s[:, 19] = np.minimum(scores[:, 11], scores[:, 12])

    for wb_i, hp_i in _WB_FOOT_TO_HALPE:
        out_k[:, hp_i] = kpts[:, wb_i]
        out_s[:, hp_i] = scores[:, wb_i]
    return out_k, out_s


def to_canonical(
    kpts: np.ndarray,
    scores: np.ndarray,
    meta_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Adapt one frame's pose output to the canonical Halpe-26 layout.

    Args:
        kpts: ``(N, K, 2)`` keypoints from the pose backend.
        scores: ``(N, K)``.
        meta_name: the backend spec's ``meta`` (``"halpe26"`` |
            ``"wholebody133"``).
    Returns:
        ``(kpts26, scores26, extras)`` — ``extras`` is ``{}`` for the
        default backend; for wholebody it carries the full raw set as
        ``{"wb_keypoints": (N,133,2), "wb_scores": (N,133)}``.
    """
    if meta_name == "halpe26":
        # Identity short-circuit: the default path is byte-identical to the
        # pre-adapter engine.
        return kpts, scores, {}
    if meta_name == "wholebody133":
        kpts26, scores26 = wholebody_to_halpe26(kpts, scores)
        return kpts26, scores26, {"wb_keypoints": kpts, "wb_scores": scores}
    raise ValueError(f"No canonical adapter for meta {meta_name!r}")
