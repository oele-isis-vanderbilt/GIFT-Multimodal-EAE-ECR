"""3D skeleton plot video for the ``pose3d`` backend.

Renders one matplotlib 3D panel per frame with every tracked person's
whole-body skeleton, color-coded by track id, using the 133-kp image
coordinates (``keypoints_wb``) and the per-keypoint metric depth
(``keypoints_z``). Saved as ``{basename}_Pose3D_Skeletons.mp4`` next to the
other run artifacts. Rendering is a no-op when the run carries no z data
(non-3D backends).

Axes: X = image x (px), Y = metric z (depth, root-normalized), Z = image y
flipped (so up is up). This is a qualitative inspection artifact — not an
AAR overlay.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO-WholeBody body skeleton (0-16) + feet links — face/hand points are
# drawn as dots only to keep the plot readable.
_BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 19), (19, 17), (19, 18), (16, 22), (22, 20), (22, 21),
]
_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray",
]


def render_pose3d_plot_video(
    tracker_output,
    frame_rate: float,
    output_directory: str,
    video_basename: str,
    start_frame: int = 1,
    end_frame: Optional[int] = None,
    conf_thr: float = 0.3,
    figsize: int = 720,
) -> Optional[str]:
    frames = [
        fr for fr in tracker_output
        if fr["frame"] >= start_frame
        and (end_frame is None or fr["frame"] <= end_frame)
    ]
    has_z = any(
        "keypoints_z" in o and "keypoints_wb" in o
        for fr in frames for o in fr.get("objects", [])
    )
    if not has_z:
        return None

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fixed axis bounds across the video so the camera doesn't jump around.
    xs, ys, zs = [], [], []
    for fr in frames:
        for o in fr.get("objects", []):
            if "keypoints_wb" not in o or "keypoints_z" not in o:
                continue
            wb = np.asarray(o["keypoints_wb"], dtype=float)
            sc = np.asarray(o.get("keypoint_scores_wb", [1.0] * len(wb)), dtype=float)
            zz = np.asarray(o["keypoints_z"], dtype=float)
            m = sc[:17] >= conf_thr
            if m.any():
                xs.extend(wb[:17][m, 0]); ys.extend(wb[:17][m, 1]); zs.extend(zz[:17][m])
    if not xs:
        return None
    x_lo, x_hi = np.percentile(xs, [1, 99])
    y_lo, y_hi = np.percentile(ys, [1, 99])
    z_lo, z_hi = np.percentile(zs, [1, 99])
    pad = lambda lo, hi: ((lo - 0.05 * (hi - lo)), (hi + 0.05 * (hi - lo)))
    x_lo, x_hi = pad(x_lo, x_hi)
    y_lo, y_hi = pad(y_lo, y_hi)
    z_lo, z_hi = pad(z_lo, z_hi) if z_hi > z_lo else (z_lo - 0.5, z_hi + 0.5)

    out_path = os.path.join(output_directory, f"{video_basename}_Pose3D_Skeletons.mp4")
    dpi = 100
    fig = plt.figure(figsize=(figsize / dpi, figsize / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    writer = None

    try:
        for fr in frames:
            ax.cla()
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(z_lo, z_hi)
            ax.set_zlim(-y_hi, -y_lo)
            ax.set_xlabel("image x (px)")
            ax.set_ylabel("depth z (root-norm)")
            ax.set_zlabel("height (px)")
            ax.set_title(f"frame {fr['frame']}")
            for o in fr.get("objects", []):
                if "keypoints_wb" not in o or "keypoints_z" not in o:
                    continue
                tid = int(o["id"])
                col = _COLORS[tid % len(_COLORS)]
                wb = np.asarray(o["keypoints_wb"], dtype=float)
                sc = np.asarray(o.get("keypoint_scores_wb", [1.0] * len(wb)), dtype=float)
                zz = np.asarray(o["keypoints_z"], dtype=float)
                X, Y, Z = wb[:, 0], zz, -wb[:, 1]
                ok = sc >= conf_thr
                for a, b in _BODY_EDGES:
                    if ok[a] and ok[b]:
                        ax.plot([X[a], X[b]], [Y[a], Y[b]], [Z[a], Z[b]],
                                color=col, linewidth=1.5)
                body = ok[:23]
                ax.scatter(X[:23][body], Y[:23][body], Z[:23][body],
                           color=col, s=8)
                rest = ok[23:]
                ax.scatter(X[23:][rest], Y[23:][rest], Z[23:][rest],
                           color=col, s=1, alpha=0.5)
                if ok[:17].any():
                    ax.text(float(X[0]), float(Y[0]), float(Z[0]) + 20,
                            f"id {tid}", color=col, fontsize=8)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            if writer is None:
                h, w = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (w, h)
                )
            writer.write(frame_bgr)
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)
    logger.info("Saved 3D skeleton plot video to %s", out_path)
    return out_path
