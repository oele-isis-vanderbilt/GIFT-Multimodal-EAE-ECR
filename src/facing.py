"""Keypoint-arrangement facing estimation (KGF v2) — metadata only.

Estimates a per-frame, per-track facing direction from the canonical
Halpe-26 keypoints the pipeline already produces, so it works identically
for every pose backend (body2d / wholebody / pose3d). Validated empirically
against motion direction on three drill videos (July 2026 study): the
arrangement yaw beats or ties the raw eyes-ears vector on all of them and,
unlike it, resolves the toward/away-from-camera hemisphere.

Outputs are strictly metadata: a ``{basename}_FacingCache.txt`` sidecar and a
compact per-object ``facing`` field in TrackerOutput. Nothing downstream
(renderings, metrics, analysis payload) consumes them yet.

Conventions:
  - ``yaw_cam``: degrees in the camera-aligned ground frame; 0 = facing the
    camera, +90 = facing image-right, +/-180 = facing away. Camera-relative
    by construction (the mirror-order cues are camera cues).
  - ``map_bearing``: degrees in MAP coordinates (atan2(dy, dx) of the facing
    direction on the floor plane), converted through the homography's local
    axes at the person's foot point — invariant to camera placement.
  - ``door_rel``: cosine of the angle between the map-frame facing direction
    and the door's inward normal (+1 facing straight into the room, -1
    facing back out the door) — invariant to camera-vs-door orientation.
"""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Halpe-26 canonical indices.
_NOSE, _LEYE, _REYE, _LEAR, _REAR, _LSHO, _RSHO = 0, 1, 2, 3, 4, 5, 6
_CONF = 0.3
_D_RATIO = 0.65          # nose protrusion relative to half ear-span
_MAX_SLEW_DEG = 9.0      # per-frame yaw slew limit
_HEMI_FLIP_SEC = 0.25    # sustained evidence needed to flip hemisphere
_SMOOTH_SEC = 0.15       # half-width of the circular median window


def _ang_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _frame_cues(kps, scs) -> Optional[dict]:
    """Per-frame facing cues from one canonical (26,·) keypoint set."""
    k = lambda i: np.asarray(kps[i], dtype=float)
    v = lambda i: scs[i] > _CONF

    pts = [k(i) for i in (_NOSE, _LEYE, _REYE, _LEAR, _REAR) if v(i)]
    if len(pts) < 2:
        return None
    head = max(np.linalg.norm(a - b) for i, a in enumerate(pts) for b in pts[i + 1:])
    if head <= 1e-6:
        return None

    c: dict = {}

    # Eyes-ears vector — geometrically strongest in profile views.
    eyes = [k(i) for i in (_LEYE, _REYE) if v(i)]
    if len(eyes) == 2:
        origin = (eyes[0] + eyes[1]) / 2
    elif v(_NOSE):
        origin = k(_NOSE)
    elif len(eyes) == 1:
        origin = eyes[0]
    else:
        origin = None
    ears = [k(i) for i in (_LEAR, _REAR) if v(i)]
    if origin is not None and ears:
        raw = origin - np.mean(ears, axis=0)
        n = float(np.linalg.norm(raw))
        if n > 1e-6:
            c["ee_dir"] = raw / n
            c["ee_len"] = n / head

    # Arrangement yaw: signed ear span (mirror order) + nose offset.
    S = None
    if v(_LEAR) and v(_REAR):
        S = (k(_LEAR)[0] - k(_REAR)[0]) / head
    Nx = None
    if v(_NOSE) and ears:
        Nx = (k(_NOSE)[0] - float(np.mean([e[0] for e in ears]))) / head
    elif len(eyes) == 2 and ears:
        Nx = ((eyes[0][0] + eyes[1][0]) / 2 - float(np.mean([e[0] for e in ears]))) / head
    if S is not None:
        c["arr_yaw"] = math.degrees(math.atan2((Nx or 0.0) / _D_RATIO, S))
        c["arr_w"] = min(1.0, abs(S) * 2.0 + 0.1) * float(min(scs[_LEAR], scs[_REAR]))
        c["S"] = S

    # Profile side from ear-score asymmetry (the occluded ear scores lower).
    asym = float(scs[_LEAR]) - float(scs[_REAR])
    if abs(asym) > 0.15 and v(_NOSE) and ears:
        vis = k(_LEAR) if asym > 0 else k(_REAR)
        side = 1.0 if (k(_NOSE)[0] - vis[0]) > 0 else -1.0
        c["prof_yaw"] = 90.0 * side
        c["prof_w"] = min(1.0, abs(asym) * 2.0) * 0.8

    # Hemisphere votes (positive = facing camera): ear order, eye order,
    # weak shoulder tie-break (the neck rotates independently of the torso,
    # so shoulders never set the angle).
    votes = 0.0
    if S is not None:
        votes += 2.0 * math.tanh(4.0 * S)
    if len(eyes) == 2:
        E = (k(_LEYE)[0] - k(_REYE)[0]) / head
        votes += 1.0 * math.tanh(6.0 * E)
    if v(_LSHO) and v(_RSHO):
        sw = float(np.linalg.norm(k(_LSHO) - k(_RSHO)))
        if sw > 1e-6:
            votes += 0.5 * math.tanh(3.0 * (k(_LSHO)[0] - k(_RSHO)[0]) / sw)
    c["hemi_vote"] = votes
    return c


def _fuse_track(series: Dict[int, dict], fps: float) -> Dict[int, dict]:
    """Cue fusion for one track: sticky hemisphere, conditioning-weighted
    circular blend, slew limit, circular median. Returns
    ``{frame: {yaw_cam, back, weight}}``.
    """
    frames = sorted(series)
    flip_n = max(1, int(_HEMI_FLIP_SEC * fps))

    hemi: Dict[int, int] = {}
    state: Optional[int] = None
    pending = 0
    for f in frames:
        vote = series[f].get("hemi_vote", 0.0)
        s = 1 if vote >= 0 else -1
        if state is None:
            state, pending = s, 0
        elif s != state and abs(vote) > 0.3:
            pending += 1
            if pending >= flip_n:
                state, pending = s, 0
        else:
            pending = max(0, pending - 1)
        hemi[f] = state

    fused: Dict[int, float] = {}
    weight: Dict[int, float] = {}
    for f in frames:
        c = series[f]
        angs, ws = [], []
        if "arr_yaw" in c:
            y = c["arr_yaw"]
            if hemi[f] < 0 and abs(y) < 90:
                y = math.copysign(180.0 - abs(y), y if y else 1.0)
            elif hemi[f] > 0 and abs(y) > 90:
                y = math.copysign(180.0 - abs(y), y)
            angs.append(y)
            ws.append(c["arr_w"])
        if "prof_yaw" in c:
            angs.append(c["prof_yaw"])
            ws.append(c["prof_w"])
        if "ee_dir" in c and hemi[f] > 0:
            dx, dy = c["ee_dir"]
            angs.append(math.degrees(math.atan2(dx, dy)))
            ws.append(1.6 * min(1.0, c["ee_len"]))
        if not angs:
            continue
        x = sum(w * math.cos(math.radians(a)) for a, w in zip(angs, ws))
        yv = sum(w * math.sin(math.radians(a)) for a, w in zip(angs, ws))
        if x == 0 and yv == 0:
            continue
        fused[f] = math.degrees(math.atan2(yv, x))
        weight[f] = float(sum(ws))

    out: Dict[int, dict] = {}
    slew: Dict[int, float] = {}
    prev = None
    for f in frames:
        if f not in fused:
            continue
        y = fused[f]
        if prev is not None:
            gap = f - prev[0]
            lim = _MAX_SLEW_DEG * max(1, gap)
            lim *= 0.4 + 0.6 * min(1.0, weight[f] / 2.0)
            y = prev[1] + max(-lim, min(lim, _ang_diff(y, prev[1])))
        slew[f] = y
        prev = (f, y)
    half = max(1, int(_SMOOTH_SEC * fps))
    for f in slew:
        win = [slew[g] for g in range(f - half, f + half + 1) if g in slew]
        a = np.radians(win)
        ys = math.degrees(math.atan2(float(np.median(np.sin(a))),
                                     float(np.median(np.cos(a)))))
        out[f] = dict(
            yaw_cam=ys,
            back=abs(ys) > 90.0,
            weight=weight.get(f, 0.0),
        )
    return out


def compute_facing(
    tracker_output,
    fps: float,
    pixel_mapper=None,
    door_axes=None,
) -> Dict[tuple, dict]:
    """Compute facing metadata for every (frame, track) in ``tracker_output``.

    Returns ``{(frame, id): {yaw_cam, back, map_bearing, door_rel, conf}}``.
    ``map_bearing``/``door_rel`` are ``None`` when no homography / door
    geometry is available.
    """
    series: Dict[int, Dict[int, dict]] = defaultdict(dict)
    anchors: Dict[tuple, np.ndarray] = {}
    for entry in tracker_output:
        f = entry["frame"]
        for obj in entry.get("objects", []):
            kps = obj.get("keypoints")
            scs = obj.get("keypoint_scores")
            if not kps or not scs or len(kps) != 26:
                continue
            cues = _frame_cues(kps, scs)
            if cues is not None:
                series[obj["id"]][f] = cues
            anchor = obj.get("ankle_based_point")
            if anchor is None:
                bb = obj.get("bbox")
                if bb:
                    anchor = [(bb[0] + bb[2]) / 2.0, bb[3]]
            if anchor is not None:
                anchors[(f, obj["id"])] = np.asarray(anchor, dtype=float)

    n_in = None
    if door_axes:
        try:
            n_in = np.asarray(door_axes[0].n_in, dtype=float)
        except Exception:
            n_in = None

    def _px2map(pt: np.ndarray) -> Optional[np.ndarray]:
        try:
            return np.asarray(pixel_mapper.pixel_to_map(pt), dtype=float).ravel()
        except Exception:
            return None

    result: Dict[tuple, dict] = {}
    for tid, per_frame in series.items():
        fused = _fuse_track(per_frame, fps)
        for f, rec in fused.items():
            yaw = rec["yaw_cam"]
            map_bearing = None
            door_rel = None
            anchor = anchors.get((f, tid))
            if pixel_mapper is not None and anchor is not None:
                m0 = _px2map(anchor)
                mr = _px2map(anchor + np.array([20.0, 0.0]))
                mu = _px2map(anchor - np.array([0.0, 20.0]))
                if m0 is not None and mr is not None and mu is not None:
                    right = mr - m0
                    up = mu - m0
                    nr, nu = np.linalg.norm(right), np.linalg.norm(up)
                    if nr > 1e-9 and nu > 1e-9:
                        a = math.radians(yaw)
                        g = math.sin(a) * (right / nr) + (-math.cos(a)) * (up / nu)
                        ng = np.linalg.norm(g)
                        if ng > 1e-9:
                            g = g / ng
                            map_bearing = math.degrees(math.atan2(g[1], g[0]))
                            if n_in is not None:
                                nn = np.linalg.norm(n_in)
                                if nn > 1e-9:
                                    door_rel = float(np.dot(g, n_in / nn))
            result[(f, tid)] = dict(
                yaw_cam=round(yaw, 1),
                back=bool(rec["back"]),
                map_bearing=None if map_bearing is None else round(map_bearing, 1),
                door_rel=None if door_rel is None else round(door_rel, 3),
                conf=round(min(1.0, rec["weight"] / 3.0), 3),
            )
    return result


def attach_facing_metadata(tracker_output, facing: Dict[tuple, dict]) -> None:
    """Attach compact ``facing`` fields to tracker objects, in place.

    ``facing`` value layout: ``[yaw_cam, back, map_bearing, door_rel, conf]``
    with ``None`` for unavailable map/door terms.
    """
    for entry in tracker_output:
        f = entry["frame"]
        for obj in entry.get("objects", []):
            rec = facing.get((f, obj["id"]))
            if rec is not None:
                obj["facing"] = [
                    rec["yaw_cam"],
                    1 if rec["back"] else 0,
                    rec["map_bearing"],
                    rec["door_rel"],
                    rec["conf"],
                ]


def save_facing_cache(
    facing: Dict[tuple, dict], output_directory: str, video_basename: str
) -> str:
    path = os.path.join(output_directory, f"{video_basename}_FacingCache.txt")
    with open(path, "w") as fh:
        fh.write("frame,id,yaw_cam,back,map_bearing,door_rel,conf\n")
        for (f, tid) in sorted(facing):
            r = facing[(f, tid)]
            fh.write(
                f"{f},{tid},{r['yaw_cam']},{1 if r['back'] else 0},"
                f"{'' if r['map_bearing'] is None else r['map_bearing']},"
                f"{'' if r['door_rel'] is None else r['door_rel']},{r['conf']}\n"
            )
    logger.debug("Saved FacingCache to %s", path)
    return path
