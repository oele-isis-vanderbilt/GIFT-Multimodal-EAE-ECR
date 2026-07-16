"""3D skeleton inspection video for the ``pose3d`` backend.

Composite layout per frame (1600x720):
  LEFT  — the actual camera frame with the standard 2D skeleton, boxes and
          track-id labels, color-coded per id (grounds the viewer: who is
          who, where they really are).
  RIGHT — a decluttered 3D view of the same people: body-only skeletons
          (no face/hand point clouds), one color per track id matching the
          left panel, vertical drop-lines onto a floor grid (depth
          anchoring), and a gently oscillating camera for motion parallax.

Axes on the right: X = image x, depth = per-keypoint metric z from the 3D
head (root-normalized), height = image y (up). Saved as
``{basename}_Pose3D_Skeletons.mp4``. No-op for runs without z data.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Body + feet skeleton in COCO-WholeBody indexing (face/hands intentionally
# not drawn — they turn the plot into an unreadable point cloud).
# Mirrors the artifact video's _CAMERA_SKELETON structure in COCO-WholeBody
# indexing: face links, full limbs, heel/toe fans. WholeBody has no neck
# keypoint — the head-to-torso link is drawn via a synthetic neck (shoulder
# midpoint) inside the render loops.
_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),                       # nose-eyes-ears
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),              # shoulders/arms
    (5, 11), (6, 12), (11, 12),                           # torso
    (11, 13), (13, 15), (12, 14), (14, 16),               # legs
    (15, 19), (19, 17), (19, 18),                         # L ankle-heel-toes
    (16, 22), (22, 20), (22, 21),                         # R ankle-heel-toes
]
_HEAD_IDX = 0            # nose — drawn as an enlarged head marker
_BGR_COLORS = [          # cv2 (BGR) — kept in sync with _MPL_COLORS
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
    (189, 103, 148), (75, 86, 140), (194, 119, 227), (34, 189, 188),
    (207, 190, 23), (127, 127, 127),
]
_MPL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#946ba5",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f",
]


def render_pose3d_plot_video(
    tracker_output,
    frame_rate: float,
    output_directory: str,
    video_basename: str,
    video_path: Optional[str] = None,
    start_frame: int = 1,
    end_frame: Optional[int] = None,
    conf_thr: float = 0.3,
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

    # ---- global bounds (body keypoints only) so the 3D camera is stable ----
    xs, ys, zs = [], [], []
    for fr in frames:
        for o in fr.get("objects", []):
            if "keypoints_wb" not in o or "keypoints_z" not in o:
                continue
            wb = np.asarray(o["keypoints_wb"], dtype=float)
            sc = np.asarray(o.get("keypoint_scores_wb", [1.0] * len(wb)), dtype=float)
            zz = np.asarray(o["keypoints_z"], dtype=float)
            m = sc[:23] >= conf_thr
            if m.any():
                xs.extend(wb[:23][m, 0]); ys.extend(wb[:23][m, 1]); zs.extend(zz[:23][m])
    if not xs:
        return None
    x_lo, x_hi = np.percentile(xs, [2, 98])
    y_lo, y_hi = np.percentile(ys, [2, 98])
    z_lo, z_hi = np.percentile(zs, [2, 98])
    if z_hi - z_lo < 0.2:
        mid = (z_hi + z_lo) / 2
        z_lo, z_hi = mid - 0.1, mid + 0.1
    floor = -y_hi  # height axis is -image_y; floor = lowest visible point

    # ---- panels ----
    PH = 720
    LEFT_W = 880
    RIGHT_W = 720
    dpi = 100
    fig = plt.figure(figsize=(RIGHT_W / dpi, PH / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    out_path = os.path.join(output_directory, f"{video_basename}_Pose3D_Skeletons.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (LEFT_W + RIGHT_W, PH)
    )
    cap = cv2.VideoCapture(video_path) if video_path and os.path.exists(video_path) else None
    if cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0]["frame"] - 1)

    # floor grid lines (precomputed)
    gx = np.linspace(x_lo, x_hi, 7)
    gz = np.linspace(z_lo, z_hi, 7)

    trail_len = max(2, int(1.5 * frame_rate))
    trails: dict = {}   # tid -> list of (x, depth) floor footprints

    try:
        for idx, fr in enumerate(frames):
            # ---------------- left panel: real frame + 2D skeleton ---------
            left = np.zeros((PH, LEFT_W, 3), dtype=np.uint8)
            if cap is not None:
                ok, img = cap.read()
                if ok and img is not None:
                    for o in fr.get("objects", []):
                        col = _BGR_COLORS[int(o["id"]) % len(_BGR_COLORS)]
                        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
                        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
                        cv2.putText(img, f"id {o['id']}", (x1, max(24, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                        wb = o.get("keypoints_wb")
                        sc = o.get("keypoint_scores_wb")
                        if wb and sc:
                            for a, b in _EDGES:
                                if sc[a] >= conf_thr and sc[b] >= conf_thr:
                                    cv2.line(img, (int(wb[a][0]), int(wb[a][1])),
                                             (int(wb[b][0]), int(wb[b][1])), col, 2)
                            if (sc[0] >= conf_thr and sc[5] >= conf_thr
                                    and sc[6] >= conf_thr):
                                nx = int((wb[5][0] + wb[6][0]) / 2)
                                ny = int((wb[5][1] + wb[6][1]) / 2)
                                cv2.line(img, (int(wb[0][0]), int(wb[0][1])),
                                         (nx, ny), col, 2)
                    h, w = img.shape[:2]
                    s = min(LEFT_W / w, PH / h)
                    rs = cv2.resize(img, (int(w * s), int(h * s)))
                    oy, ox = (PH - rs.shape[0]) // 2, (LEFT_W - rs.shape[1]) // 2
                    left[oy:oy + rs.shape[0], ox:ox + rs.shape[1]] = rs

            # ---------------- right panel: clean 3D ------------------------
            ax.cla()
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(z_lo, z_hi)
            ax.set_zlim(floor, -y_lo)
            ax.set_box_aspect((2.2, 1.2, 1.6))
            # gentle oscillation for motion parallax
            ax.view_init(elev=16, azim=-78 + 18 * math.sin(2 * math.pi * idx / (15 * frame_rate)))
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.set_pane_color((0.97, 0.97, 0.97, 1.0))
                pane.label.set_size(8)
            ax.set_xticks([]); ax.set_zticks([])
            ax.set_yticks(np.round(np.linspace(z_lo, z_hi, 4), 2))
            ax.tick_params(labelsize=7)
            ax.set_ylabel("depth (near ⇄ far)", fontsize=8)
            t_sec = fr["frame"] / frame_rate if frame_rate > 0 else 0.0
            ax.set_title(f"3D pose — t = {t_sec:6.2f} s  (frame {fr['frame']})",
                         fontsize=10)
            for gxx in gx:
                ax.plot([gxx, gxx], [z_lo, z_hi], [floor, floor], color="0.85", lw=0.6)
            for gzz in gz:
                ax.plot([x_lo, x_hi], [gzz, gzz], [floor, floor], color="0.85", lw=0.6)

            for o in fr.get("objects", []):
                if "keypoints_wb" not in o or "keypoints_z" not in o:
                    continue
                tid = int(o["id"])
                col = _MPL_COLORS[tid % len(_MPL_COLORS)]
                wb = np.asarray(o["keypoints_wb"], dtype=float)
                sc = np.asarray(o.get("keypoint_scores_wb", [1.0] * len(wb)), dtype=float)
                zz = np.asarray(o["keypoints_z"], dtype=float)
                X, D, H = wb[:, 0], zz, -wb[:, 1]
                ok = sc >= conf_thr
                for a, b in _EDGES:
                    if ok[a] and ok[b]:
                        ax.plot([X[a], X[b]], [D[a], D[b]], [H[a], H[b]],
                                color=col, lw=2.2, solid_capstyle="round")
                if ok[0] and ok[5] and ok[6]:
                    ax.plot([X[0], (X[5] + X[6]) / 2],
                            [D[0], (D[5] + D[6]) / 2],
                            [H[0], (H[5] + H[6]) / 2],
                            color=col, lw=2.2, solid_capstyle="round")
                body = np.where(ok[:23])[0]
                ax.scatter(X[body], D[body], H[body], color=col, s=10,
                           depthshade=False)
                if ok[_HEAD_IDX]:
                    ax.scatter([X[_HEAD_IDX]], [D[_HEAD_IDX]], [H[_HEAD_IDX]],
                               color=col, s=60, depthshade=False)
                # drop-line from pelvis to the floor grid — depth anchor
                if ok[11] and ok[12]:
                    px = (X[11] + X[12]) / 2
                    pd = (D[11] + D[12]) / 2
                    ph = (H[11] + H[12]) / 2
                    ax.plot([px, px], [pd, pd], [floor, ph],
                            color=col, lw=0.9, linestyle=":", alpha=0.8)
                    ax.scatter([px], [pd], [floor], color=col, s=18, marker="x")
                    ax.text(px, pd, floor - 18, f"id {tid}", color=col,
                            fontsize=8, ha="center")
                    # fading floor trail — movement through depth
                    tr = trails.setdefault(tid, [])
                    tr.append((px, pd))
                    del tr[:-trail_len]
                    if len(tr) >= 2:
                        txs = [q[0] for q in tr]
                        tds = [q[1] for q in tr]
                        ax.plot(txs, tds, [floor] * len(tr),
                                color=col, lw=1.4, alpha=0.45)
                    # facing arrow on the floor (KGF metadata: camera-frame
                    # yaw; 0 = toward camera = decreasing depth)
                    facing = o.get("facing")
                    if facing and facing[0] is not None:
                        yaw = math.radians(float(facing[0]))
                        ax_len = 0.10 * (x_hi - x_lo)
                        ad_len = 0.14 * (z_hi - z_lo)
                        ax.quiver(px, pd, floor,
                                  math.sin(yaw) * ax_len,
                                  -math.cos(yaw) * ad_len, 0.0,
                                  color=col, lw=1.6, alpha=0.9,
                                  arrow_length_ratio=0.35)

            present = sorted({int(o["id"]) for o in fr.get("objects", [])
                              if "keypoints_wb" in o and "keypoints_z" in o})
            if present:
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], color=_MPL_COLORS[t % len(_MPL_COLORS)],
                                  lw=2.5, label=f"id {t}") for t in present]
                ax.legend(handles=handles, loc="upper left", fontsize=7,
                          framealpha=0.6, borderpad=0.4)

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            right = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            if right.shape[:2] != (PH, RIGHT_W):
                right = cv2.resize(right, (RIGHT_W, PH))
            writer.write(np.hstack([left, right]))
    finally:
        writer.release()
        if cap is not None:
            cap.release()
        plt.close(fig)
    logger.info("Saved 3D skeleton plot video to %s", out_path)
    return out_path
