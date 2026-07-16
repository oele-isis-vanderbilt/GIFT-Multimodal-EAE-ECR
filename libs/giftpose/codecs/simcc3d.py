"""SimCC-3D decode — port of the RTMPose3D ``SimCC3DLabel`` inference path.

x/y decode exactly like 2D SimCC (argmax / split-ratio, in input-crop
coordinates, ready for the inverse top-down warp). z decodes to a
root-normalized metric value:

    z_metric = (z_bins / split / (D / 2) - 1) * z_range

with ``D`` the codec's z input size (288 for the released RTMW3D models) and
``z_range`` the fixed scale constant from upstream (2.1744869).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

Z_RANGE_DEFAULT = 2.1744869


def decode_simcc3d(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_z: np.ndarray,
    simcc_split_ratio: float = 2.0,
    z_input_size: int = 288,
    z_range: float = Z_RANGE_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode 3-axis SimCC vectors.

    Args:
        simcc_x/y/z: ``(N, K, Wx/Wy/Wz)`` raw logits.
    Returns:
        kpts_xy: ``(N, K, 2)`` float32, input-crop pixel coordinates.
        scores: ``(N, K)`` float32 — min of the three per-axis maxima.
        z_metric: ``(N, K)`` float32 — root-normalized metric z.
    """
    assert simcc_x.ndim == 3 and simcc_y.ndim == 3 and simcc_z.ndim == 3
    x_locs = np.argmax(simcc_x, axis=2).astype(np.float32)
    y_locs = np.argmax(simcc_y, axis=2).astype(np.float32)
    z_locs = np.argmax(simcc_z, axis=2).astype(np.float32)

    # Score convention mirrors upstream rtmpose3d.utils.get_simcc_maximum:
    # min(max_x, max_y), with z EXCLUDED — the z branch is trained with a
    # softmax label (label_beta=10) so its raw logit maxima are an order of
    # magnitude smaller than x/y and would tank every keypoint score
    # (observed: ~0.35 vs ~7, silently failing downstream conf thresholds).
    max_x = np.amax(simcc_x, axis=2)
    max_y = np.amax(simcc_y, axis=2)
    vals = np.minimum(max_x, max_y).astype(np.float32)

    kpts_xy = np.stack((x_locs, y_locs), axis=-1) / float(simcc_split_ratio)
    kpts_xy[vals <= 0.0] = -1

    z_bins = z_locs / float(simcc_split_ratio)
    z_metric = ((z_bins / (float(z_input_size) / 2.0)) - 1.0) * float(z_range)
    return kpts_xy.astype(np.float32), vals, z_metric.astype(np.float32)
