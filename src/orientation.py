"""Experimental POD-orientation replay tool.

Computes per-frame body/attention orientation for every tracked entrant from
the canonical Halpe-26 keypoints saved in ``{base}_TrackerOutput.json``,
detects the team's first collective pause after full entry (the candidate
"point of dominance" establishment), and scores two experimental metrics:

- ``POD_SECTOR_COVERAGE``  — area(union of member facing sectors ∩ room) / area(room)
- ``POD_MUTUAL_FACING``    — 1 − (members whose facing sector contains at least
                             one teammate) / team size (1.0 = nobody flags a
                             teammate; 0.75 = 1 of 4 does)

Everything replays from an existing run folder (no video, no pose model):

    python -m src.orientation output/<run_folder> [--sector-angle 20]

Orientation approach (pose-model invariant — reads only the canonical 26-kp
block every backend emits):

Two bearings are estimated per member per frame:

- BODY facing — paired keypoints at similar height (shoulders 5/6, hips
  11/12) are each projected through the floor homography and *differenced*;
  the per-point height error is nearly identical for the pair, so the
  difference is a good top-down body axis even though the individual mapped
  points are displaced. Facing = that left→right axis rotated 90° with fixed
  anatomical chirality (the pose model's left/right labels resolve front vs
  back). Heel→toe vectors (24→20, 25→21) lie on the floor plane, so they map
  exactly and are strongest exactly when the shoulder pair foreshortens.
- MUZZLE direction (what the metrics and arrows use) — from the arms/wrists
  relative to the body: the wrist-to-wrist line (a gripped rifle lies along
  the hands; the rear hand is the one nearer the chest, so muzzle =
  rear→front — pure geometry, no left/right labels needed), both
  elbow→wrist forearm vectors, and the chest→hands vector (shoulder midpoint
  → wrist midpoint). Falls back to the body facing when the arm keypoints
  are unusable or the fused result is physically implausible (>100° off the
  torso — a two-handed grip cannot point behind the shooter).

Cues combine as confidence×separation weighted unit vectors; the resultant
length is the per-frame confidence. A centered 0.5 s circular mean smooths
each track.

POD marking is a single frame: the detector finds the first moment the whole
team settles (every member's smoothed speed under an Otsu threshold over the
team's own speed distribution, sustained briefly, with a speed-contrast gate
so a never-pausing run reports uncertain instead of a fabricated mark) and
reports only that settle START frame. Positions and muzzle bearings are read
at that frame (the 0.5 s smoothing absorbs per-frame pose noise). The only
exposed knob is the sector angle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ----------------------------------------------------------------------
# Halpe-26 indices (canonical block — identical for every pose backend)
# ----------------------------------------------------------------------
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_BIG_TOE, R_BIG_TOE = 20, 21
L_HEEL, R_HEEL = 24, 25

# Internal tuning (deliberately not config — see module docstring).
KP_CONF_THRESHOLD = 0.3          # same gate the gaze code uses
SMOOTH_HALF_SEC = 0.25           # centered smoothing window = 2x this
SPEED_SMOOTH_SEC = 0.5           # moving-average window for map speed
PAUSE_MIN_SEC = 0.8              # a hold shorter than this is not a POD pause
PAUSE_EXIT_RATIO = 1.5           # hysteresis: exit only above thr * ratio
PAUSE_EXIT_SUSTAIN_SEC = 0.25    # ...sustained for this long
PAUSE_ENTER_SUSTAIN_SEC = 0.25   # ...and entered only after this long below it
PAUSE_MAX_NAN_SEC = 1.0          # a longer team-tracking gap breaks a candidate pause
PAUSE_SPEED_CONTRAST = 0.5       # median speed inside a pause must be < this x the moving phase
POS_MEDIAN_SEC = 1.0             # running-median window that de-noises held positions
MIN_BEARING_RESULTANT = 0.3      # member excluded from metrics below this cue agreement
MUZZLE_OFF_TORSO_MAX_DEG = 100.0  # fused muzzle further off the torso than this is implausible
POD_BANNER_SEC = 1.5             # how long the map video highlights the POD mark
DEFAULT_SECTOR_ANGLE_DEG = 20.0  # full sector angle (config: pod_sector_angle_degrees)

POD_JSON_SCHEMA_VERSION = "0.1"


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-9:
        return None
    return v / n


def _facing_from_lr(lr_vec: np.ndarray) -> np.ndarray:
    """Rotate an anatomical left→right axis into the facing direction.

    Map pixels are y-down. Seen top-down with y-down, a person whose
    left→right shoulder axis is +x faces −y, i.e. facing = (vy, −vx).
    Verified empirically against walking direction (see chirality diagnostic).
    """
    return np.array([lr_vec[1], -lr_vec[0]], dtype=np.float64)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(c))


# ----------------------------------------------------------------------
# Orientation estimation
# ----------------------------------------------------------------------

def _pair_cue(kps, scores, mapper, idx_l, idx_r) -> Optional[Tuple[np.ndarray, float]]:
    """Map-space facing from one same-height L/R pair. Returns (unit, weight)."""
    cl, cr = scores[idx_l], scores[idx_r]
    if cl <= KP_CONF_THRESHOLD or cr <= KP_CONF_THRESHOLD:
        return None
    pts = mapper.pixel_to_map(np.array([kps[idx_l], kps[idx_r]], dtype=np.float64))
    lr = np.asarray(pts[1], dtype=np.float64) - np.asarray(pts[0], dtype=np.float64)
    sep = float(np.linalg.norm(lr))
    u = _unit(_facing_from_lr(lr))
    if u is None:
        return None
    return u, min(cl, cr) * sep


def _foot_cue(kps, scores, mapper, idx_heel, idx_toe) -> Optional[Tuple[np.ndarray, float]]:
    """Heel→toe direction: both points on the floor plane, maps exactly."""
    ch, ct = scores[idx_heel], scores[idx_toe]
    if ch <= KP_CONF_THRESHOLD or ct <= KP_CONF_THRESHOLD:
        return None
    pts = mapper.pixel_to_map(np.array([kps[idx_heel], kps[idx_toe]], dtype=np.float64))
    d = np.asarray(pts[1], dtype=np.float64) - np.asarray(pts[0], dtype=np.float64)
    sep = float(np.linalg.norm(d))
    u = _unit(d)
    if u is None:
        return None
    return u, min(ch, ct) * sep


def _muzzle_cues(kps, scores, mapper,
                 body_dir: Optional[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
    """Map-space weapon-direction cues from the arms/wrists.

    Four cues are fused: the wrist↔wrist line (the grip axis), the two
    forearms (elbow→wrist), and chest→hands (shoulder-midpoint to
    wrist-midpoint). A wrist-line-only, torso-signed variant was trialled for
    full camera invariance (mixed-height cues carry a POV-dependent
    parallax), but an A/B against both real rooms — two near-opposite-corner
    mounts — showed this fusion is the stronger real-world estimator: each
    single cue carries an anatomical bias of ~30° (support hand offset from
    the barrel axis, forearms angled across the grip) that the fusion
    averages out, and the torso-signed wrist-line destabilises outright in a
    bladed stance (its sign flip-flops when the grip axis sits
    near-perpendicular to the torso, per-frame spread ~90°) while this cue
    set stayed within a few degrees frame to frame.

    The supported camera envelope is therefore specified rather than
    unlimited: elevated room-corner/wall mounts (~3 m and up), the standard
    GIFT room setup, validated on two opposite-corner POVs. The body facing
    is fully POV-invariant regardless (paired same-height keypoints), and a
    parallax-flipped muzzle fusion points behind the shooter, where the
    >MUZZLE_OFF_TORSO_MAX_DEG guard in compute_orientation_series discards
    it for the body facing — see test_orientation_invariance.py.

    The wrist-line's muzzle end is the wrist farther from the chest (the
    support hand rides the handguard toward the muzzle); the torso facing is
    only a fallback tiebreaker when a shoulder is unseen.
    """
    def ok(i):
        return scores[i] > KP_CONF_THRESHOLD

    def to_map(i):
        pts = mapper.pixel_to_map(np.array([kps[i]], dtype=np.float64))
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)[0]

    cues: List[Tuple[np.ndarray, float]] = []

    # Grip axis: wrist <-> wrist, muzzle end resolved by chest distance.
    if ok(L_WRIST) and ok(R_WRIST):
        wl, wr = to_map(L_WRIST), to_map(R_WRIST)
        d = wr - wl
        axis = _unit(d)
        u = None
        if axis is not None and ok(L_SHOULDER) and ok(R_SHOULDER):
            chest = 0.5 * (to_map(L_SHOULDER) + to_map(R_SHOULDER))
            far_is_r = (float(np.linalg.norm(wr - chest))
                        >= float(np.linalg.norm(wl - chest)))
            u = axis if far_is_r else -axis
        elif axis is not None and body_dir is not None:
            u = axis if float(np.dot(axis, body_dir)) >= 0 else -axis
        if u is not None:
            cues.append((u, min(scores[L_WRIST], scores[R_WRIST])
                         * float(np.linalg.norm(d))))

    # Forearms: elbow -> wrist.
    for i_elbow, i_wrist in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
        if ok(i_elbow) and ok(i_wrist):
            d = to_map(i_wrist) - to_map(i_elbow)
            u = _unit(d)
            if u is not None:
                cues.append((u, min(scores[i_elbow], scores[i_wrist])
                             * float(np.linalg.norm(d))))

    # Chest -> hands midpoint.
    if ok(L_SHOULDER) and ok(R_SHOULDER) and ok(L_WRIST) and ok(R_WRIST):
        chest = 0.5 * (to_map(L_SHOULDER) + to_map(R_SHOULDER))
        hands = 0.5 * (to_map(L_WRIST) + to_map(R_WRIST))
        d = hands - chest
        u = _unit(d)
        if u is not None:
            conf = min(scores[L_SHOULDER], scores[R_SHOULDER],
                       scores[L_WRIST], scores[R_WRIST])
            cues.append((u, conf * float(np.linalg.norm(d))))

    return cues


def compute_orientation_series(
    tracker_output: List[dict],
    mapper,
    fps: float,
) -> Dict[int, dict]:
    """Per-track orientation series over the whole video.

    Returns ``{track_id: {"body", "muzzle", "conf"}}`` where body/muzzle are
    (T, 2) unit-vector arrays aligned to 1-indexed frames 1..T (NaN where
    unavailable) and conf is the (T,) fused-cue circular resultant in [0, 1].
    """
    total = max(f["frame"] for f in tracker_output) if tracker_output else 0
    per_track: Dict[int, dict] = {}

    def _tk(tid):
        if tid not in per_track:
            per_track[tid] = {
                "body": np.full((total, 2), np.nan),
                "muzzle": np.full((total, 2), np.nan),
                "conf": np.zeros(total),
            }
        return per_track[tid]

    cos_max_off = math.cos(math.radians(MUZZLE_OFF_TORSO_MAX_DEG))

    for fr in tracker_output:
        t = fr["frame"] - 1
        for obj in fr.get("objects", []):
            kps = obj.get("keypoints") or []
            scores = obj.get("keypoint_scores") or []
            if len(kps) != 26 or len(scores) != 26:
                continue

            body_cues = []
            for c in (
                _pair_cue(kps, scores, mapper, L_SHOULDER, R_SHOULDER),
                _pair_cue(kps, scores, mapper, L_HIP, R_HIP),
                _foot_cue(kps, scores, mapper, L_HEEL, L_BIG_TOE),
                _foot_cue(kps, scores, mapper, R_HEEL, R_BIG_TOE),
            ):
                if c is not None:
                    body_cues.append(c)
            if not body_cues:
                continue

            b_wsum = sum(w for _, w in body_cues)
            b_fused = np.sum([u * w for u, w in body_cues], axis=0)
            body = _unit(b_fused)
            if body is None or b_wsum <= 0:
                continue
            body_conf = float(np.linalg.norm(b_fused) / b_wsum)

            m_cues = _muzzle_cues(kps, scores, mapper, body)
            muzzle, conf = body, body_conf * 0.7  # arms unusable -> body fallback
            if m_cues:
                m_wsum = sum(w for _, w in m_cues)
                m_fused = np.sum([u * w for u, w in m_cues], axis=0)
                m_u = _unit(m_fused)
                if m_u is not None and m_wsum > 0:
                    if float(np.dot(m_u, body)) < cos_max_off:
                        # A gripped weapon cannot point behind the shooter —
                        # arm keypoints are unreliable this frame.
                        muzzle, conf = body, body_conf * 0.5
                    else:
                        muzzle = m_u
                        conf = float(np.linalg.norm(m_fused) / m_wsum)

            rec = _tk(obj["id"])
            rec["body"][t] = body
            rec["muzzle"][t] = muzzle
            rec["conf"][t] = conf

    # Centered circular smoothing, confidence-weighted. The kernel is clamped
    # to the series length: np.convolve(mode='same') returns the LONGER of the
    # two inputs, so an over-long kernel would break the boolean masking below.
    half = max(1, int(round(SMOOTH_HALF_SEC * fps)))
    half = min(half, max(0, (total - 1) // 2))
    for rec in per_track.values():
        for key in ("body", "muzzle"):
            raw = rec[key]
            w = np.where(np.isfinite(raw[:, 0]), rec["conf"], 0.0)
            v = np.where(np.isfinite(raw), raw, 0.0) * w[:, None]
            kernel = np.ones(2 * half + 1)
            sx = np.convolve(v[:, 0], kernel, mode="same")
            sy = np.convolve(v[:, 1], kernel, mode="same")
            sw = np.convolve(w, kernel, mode="same")
            sm = np.full_like(raw, np.nan)
            good = sw > 1e-9
            norms = np.hypot(sx, sy)
            good &= norms > 1e-9
            sm[good, 0] = sx[good] / norms[good]
            sm[good, 1] = sy[good] / norms[good]
            rec[key + "_smooth"] = sm

    return per_track


# ----------------------------------------------------------------------
# Team pause detection (parameter-free: Otsu on the speed distribution)
# ----------------------------------------------------------------------

def _member_speed(traj: Sequence, fps: float, total: int) -> np.ndarray:
    """Displacement-based map speed (map px / s). NaN where unknown.

    Speed at frame t = ||pos[t] − pos[t − 0.5 s]|| / 0.5 s — net progress
    through the room, NOT accumulated path length. A member shuffling in
    place (small local adjustments while holding a position) scores near
    zero, while a walker scores their true walking speed. Frame-to-frame
    path-length speed was tried first and postponed pause detection whenever
    one member fidgeted at their post.
    """
    if total <= 0:
        return np.full(0, np.nan)
    pos = np.full((total, 2), np.nan)
    for i in range(min(total, len(traj))):
        if traj[i] is not None:
            pos[i] = np.asarray(traj[i], dtype=np.float64)

    # forward-fill positions across gaps up to 0.5 s so brief dropouts don't
    # punch holes in the speed series
    ffill = max(1, int(round(0.5 * fps)))
    filled = pos.copy()
    last, age = None, 0
    for i in range(total):
        if np.isfinite(pos[i, 0]):
            last, age = pos[i], 0
        elif last is not None and age < ffill:
            filled[i] = last
            age += 1
        else:
            age += 1

    # 1 s running median per coordinate: tracker wobble at a held position
    # (e.g. the map anchor hopping while a nearby person occludes the feet)
    # oscillates around the true spot — the median rides the true spot while
    # genuine walking passes straight through.
    med = max(1, min(int(round(POS_MEDIAN_SEC * fps)), total))
    if med > 2:
        sm = np.full_like(filled, np.nan)
        half_m = med // 2
        for i in range(total):
            seg = filled[max(0, i - half_m): i + half_m + 1]
            good = np.isfinite(seg[:, 0])
            if good.any():
                sm[i] = np.median(seg[good], axis=0)
        filled = sm

    win = max(1, min(int(round(SPEED_SMOOTH_SEC * fps)), total - 1)) if total > 1 else 1
    speed = np.full(total, np.nan)
    a, b = filled[:-win] if win < total else filled[:0], filled[win:]
    disp = np.linalg.norm(b - a, axis=1) if len(b) else np.array([])
    ok = np.isfinite(disp)
    speed[win:][ok] = disp[ok] * fps / win
    return speed


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu split on log1p(speed) — separates 'holding' from 'moving'."""
    x = np.log1p(values[np.isfinite(values)])
    if x.size < 8:
        return float("nan")
    hist, edges = np.histogram(x, bins=128)
    hist = hist.astype(np.float64)
    total = hist.sum()
    csum = np.cumsum(hist)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cmean = np.cumsum(hist * centers)
    gmean = cmean[-1] / total
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (gmean * csum - cmean) ** 2 / (csum * (total - csum))
    between[~np.isfinite(between)] = -1.0
    k = int(np.argmax(between))
    return float(np.expm1(centers[k]))


def detect_team_pause(
    team_trajs: List[Tuple[int, Sequence]],
    fps: float,
    search_start: int,
    search_end: int,
    total: int,
) -> dict:
    """First collective hold inside [search_start, search_end] (1-indexed).

    A candidate opens after every member's smoothed speed sits under the Otsu
    threshold for PAUSE_ENTER_SUSTAIN_SEC, closes on speed > threshold*1.5
    sustained PAUSE_EXIT_SUSTAIN_SEC (or a team-tracking gap longer than
    PAUSE_MAX_NAN_SEC), and its end is trimmed to the last well-tracked
    below-band frame. Because Otsu always produces *some* split even when the
    team never stops, a candidate only counts as a pause if it also shows real
    speed contrast: median team speed inside < PAUSE_SPEED_CONTRAST x median
    outside (within the search window). Runs without such a hold report
    found=False rather than fabricating one.
    """
    result = {"speed_threshold": float("nan"), "found": False}
    if not team_trajs or total <= 0 or search_end < search_start:
        result["reason"] = "no_team_tracks"
        return result

    speeds = {tid: _member_speed(traj, fps, total) for tid, traj in team_trajs}
    # forward-fill each member's speed up to 0.5 s so brief dropouts don't break the team series
    ffill = max(1, int(round(0.5 * fps)))
    team = np.full(total, np.nan)
    for t in range(search_start - 1, min(search_end, total)):
        vals = []
        for s in speeds.values():
            v = np.nan
            for back in range(min(ffill, t) + 1):
                if np.isfinite(s[t - back]):
                    v = s[t - back]
                    break
            vals.append(v)
        if vals and all(np.isfinite(v) for v in vals):
            team[t] = max(vals)

    window = team[search_start - 1 : search_end]
    thr = _otsu_threshold(window)
    result.update(speed_threshold=float(thr), team_speed=team, member_speeds=speeds)
    if not np.isfinite(thr):
        result["reason"] = "not_enough_speed_samples"
        return result

    min_len = int(round(PAUSE_MIN_SEC * fps))
    enter_len = max(1, int(round(PAUSE_ENTER_SUSTAIN_SEC * fps)))
    exit_thr = thr * PAUSE_EXIT_RATIO
    exit_len = max(1, int(round(PAUSE_EXIT_SUSTAIN_SEC * fps)))
    max_nan = max(1, int(round(PAUSE_MAX_NAN_SEC * fps)))
    lo, hi = search_start - 1, min(search_end, total)

    candidates: List[Tuple[int, int]] = []
    start = None
    below_run = above = nan_run = 0
    last_ok = None  # last finite frame not in the exit band — pauses end here
    for t in range(lo, hi):
        v = team[t]
        finite = np.isfinite(v)
        if start is None:
            if finite and v < thr:
                below_run += 1
                if below_run >= enter_len:
                    start = t - below_run + 1
                    above = nan_run = 0
                    last_ok = t
            else:
                below_run = 0
        else:
            if not finite:
                nan_run += 1
                if nan_run > max_nan:  # lost the team — candidate ends at last good frame
                    candidates.append((start, last_ok))
                    start, below_run = None, 0
            else:
                nan_run = 0
                if v > exit_thr:
                    above += 1
                    if above >= exit_len:  # sustained break — they moved off
                        candidates.append((start, last_ok))
                        start, below_run, above = None, 0, 0
                else:
                    above = 0
                    last_ok = t
    if start is not None:
        candidates.append((start, last_ok))

    contrast_rejected = False
    for s, e in candidates:
        if e is None or e - s + 1 < min_len:
            continue
        inside = team[s : e + 1]
        inside = inside[np.isfinite(inside)]
        outside = np.concatenate([team[lo:s], team[e + 1 : hi]])
        # Real hold = clearly slower than how the team MOVES. Compare against
        # the moving phase only (outside samples above the threshold — the
        # last entrant's approach guarantees some), not against other holds:
        # on a never-pausing run the below/above split is mid-mode so this
        # ratio fails, while a genuine hold is far below true movement.
        moving = outside[np.isfinite(outside) & (outside > thr)]
        if inside.size == 0 or moving.size < max(2, int(round(0.25 * fps))):
            contrast_rejected = True
            continue
        if float(np.median(inside)) > PAUSE_SPEED_CONTRAST * float(np.median(moving)):
            contrast_rejected = True
            continue
        result.update(
            found=True,
            start_frame=s + 1,
            end_frame=e + 1,
            mid_frame=(s + e) // 2 + 1,
        )
        return result

    result["reason"] = "no_speed_contrast" if contrast_rejected else "no_sustained_pause"
    return result


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _clip_polys(poly: Polygon, room: Polygon) -> List[Polygon]:
    """Intersect and keep only polygonal pieces; buffer(0) retry on invalid
    geometry (same defense helper_functions.safe_intersection uses for the
    identical gaze-cone-vs-boundary clip)."""
    try:
        inter = poly.intersection(room)
    except Exception:
        try:
            inter = poly.buffer(0).intersection(room.buffer(0))
        except Exception:
            return []
    return [g for g in getattr(inter, "geoms", [inter])
            if g.geom_type == "Polygon" and not g.is_empty]


def compute_pod_metrics(
    members: List[dict],
    room_poly: Polygon,
    sector_angle_deg: float,
) -> dict:
    """members: [{id, pos (2,), bearing (2,) unit}] averaged over the pause."""
    from src.metrics._shared import gaze_cone_triangle

    half = sector_angle_deg / 2.0
    minx, miny, maxx, maxy = room_poly.bounds
    length = 1.5 * math.hypot(maxx - minx, maxy - miny)

    sectors: Dict[int, List[Polygon]] = {}
    for m in members:
        tri = gaze_cone_triangle(m["pos"], m["bearing"], half, length=length)
        poly = Polygon([(float(x), float(y)) for x, y in tri])
        sectors[m["id"]] = _clip_polys(poly, room_poly)

    union = unary_union([p for polys in sectors.values() for p in polys])
    coverage = float(union.area / room_poly.area) if room_poly.area > 0 else 0.0

    flagged = []
    n = len(members)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vec = _unit(members[j]["pos"] - members[i]["pos"])
            if vec is None:
                continue
            off = _angle_between(members[i]["bearing"], vec)
            if off <= half:
                flagged.append(
                    {"from": members[i]["id"], "to": members[j]["id"],
                     "angle_off_deg": round(off, 1)}
                )
    # Person-based score: a violator is a member whose sector contains at
    # least one teammate, regardless of how many. 1 of 4 violating -> 0.75.
    violators = sorted({p["from"] for p in flagged})
    score = 1.0 - (len(violators) / n) if n else 1.0

    return {
        "POD_SECTOR_COVERAGE": round(coverage, 2),
        "POD_MUTUAL_FACING": round(score, 2),
        "violators": violators,
        "flagged_pairs": flagged,
        "sectors": sectors,
    }


def serialize_sectors(sectors: Dict[int, List[Polygon]]) -> Dict[str, list]:
    """Shapely sector polygons -> JSON-safe {track_id: [[[x, y], ...], ...]}."""
    out: Dict[str, list] = {}
    for tid, polys in sectors.items():
        out[str(tid)] = [
            [[round(float(x), 1), round(float(y), 1)] for x, y in g.exterior.coords]
            for g in polys
        ]
    return out


def members_at_frame(
    team: List[Tuple[int, Sequence]],
    orient: Dict[int, dict],
    frame: int,
    fps: float,
) -> Tuple[List[dict], List[dict]]:
    """Per-member position + smoothed muzzle bearing at one (1-indexed) frame.

    Falls back to the nearest valid sample within ±0.5 s; members with no
    data or an unreliable bearing are excluded (and reported)."""
    near = max(1, int(round(0.5 * fps)))
    p0 = frame - 1

    def _at_or_near(series, is_valid):
        for off in range(near + 1):
            for t in (p0 - off, p0 + off):
                if 0 <= t < len(series) and is_valid(series[t]):
                    return series[t], t
        return None, None

    members: List[dict] = []
    excluded: List[dict] = []
    for tid, traj in team:
        rec = orient.get(tid)
        pos, _ = _at_or_near(traj, lambda v: v is not None)
        bearing, bt = (None, None)
        if rec is not None:
            bearing, bt = _at_or_near(rec["muzzle_smooth"], lambda v: np.isfinite(v[0]))
        if pos is None or bearing is None:
            excluded.append({"id": int(tid), "reason": "no_data_at_pod_frame"})
            continue
        conf = float(rec["conf"][bt]) if rec["conf"][bt] > 0 else 0.0
        if conf < MIN_BEARING_RESULTANT:
            excluded.append({"id": int(tid), "reason": "unreliable_bearing",
                             "cue_agreement": round(conf, 2)})
            continue
        members.append({
            "id": int(tid),
            "pos": np.asarray(pos, dtype=np.float64),
            "bearing": np.asarray(bearing, dtype=np.float64),
            "confidence": round(conf, 2),
        })
    return members, excluded


def _serialize_members(members: List[dict]) -> List[dict]:
    return [
        {
            "id": m["id"],
            "pos": [round(float(x), 1) for x in m["pos"]],
            "bearing": [round(float(x), 4) for x in m["bearing"]],
            "bearing_deg": round(math.degrees(math.atan2(-m["bearing"][1], m["bearing"][0])), 1),
            "confidence": m["confidence"],
        }
        for m in members
    ]


def compute_pod_data(
    tracker_output: List[dict],
    mapper,
    fps: float,
    config: dict,
    tracks_by_id: Dict[int, list],
    inroom_ids,
    drill_start: int,
    drill_end: int,
    room_poly: Polygon,
    sector_angle: Optional[float] = None,
    pod_frame_override: Optional[int] = None,
) -> dict:
    """Full POD-orientation computation shared by the engine, the replay CLI,
    and (via ``pod_frame_override``) instructor adjustment.

    Returns a dict whose non-underscore keys are JSON-serializable; the
    ``_orient`` / ``_sectors`` / ``_members`` keys carry the numpy/shapely
    working objects for rendering and caching.
    """
    from src.metrics._shared import select_entry_tracks, team_size

    angle = float(sector_angle if sector_angle is not None
                  else config.get("pod_sector_angle_degrees", DEFAULT_SECTOR_ANGLE_DEG))
    if not (0.0 < angle < 180.0):
        raise ValueError(f"sector angle must be in (0, 180) degrees, got {angle:g}")
    if not room_poly.is_valid:
        room_poly = room_poly.buffer(0)

    total = max((f["frame"] for f in tracker_output), default=0)
    team = select_entry_tracks(tracks_by_id, inroom_ids=inroom_ids,
                               num_tracks=team_size(config))
    team_ids = [int(tid) for tid, _ in team]

    orient = compute_orientation_series(tracker_output, mapper, fps)

    firsts = [next((i + 1 for i, p in enumerate(traj) if p is not None), None)
              for _, traj in team]
    firsts = [f for f in firsts if f is not None]
    search_start = max([int(drill_start)] + firsts) if firsts else int(drill_start)

    if pod_frame_override is not None:
        pod_frame, source = int(pod_frame_override), "instructor"
        pause = {"found": True, "speed_threshold": float("nan")}
    else:
        pause = detect_team_pause(team, fps, search_start, int(drill_end), total)
        pod_frame = pause["start_frame"] if pause["found"] else None
        source = "auto"

    data: dict = {
        "sector_angle_degrees": angle,
        "team_ids": team_ids,
        "fps": fps,
        "source": source,
        "search_start_frame": int(search_start),
        "speed_threshold_px_s": (round(float(pause["speed_threshold"]), 2)
                                 if np.isfinite(pause["speed_threshold"]) else None),
        "_orient": orient,
        "_team": team,
    }

    members: List[dict] = []
    excluded: List[dict] = []
    if pod_frame is not None:
        members, excluded = members_at_frame(team, orient, pod_frame, fps)

    if pod_frame is not None and len(members) >= 2:
        metrics = compute_pod_metrics(members, room_poly, angle)
        data.update(
            status="ok",
            pod_frame=int(pod_frame),
            pod_sec=round((pod_frame - 1) / fps, 2),
            members=_serialize_members(members),
            metrics={
                "POD_SECTOR_COVERAGE": metrics["POD_SECTOR_COVERAGE"],
                "POD_MUTUAL_FACING": metrics["POD_MUTUAL_FACING"],
            },
            violators=metrics["violators"],
            flagged_pairs=metrics["flagged_pairs"],
            excluded_members=excluded,
            sectors=serialize_sectors(metrics["sectors"]),
            _sectors=metrics["sectors"],
            _members=members,
        )
    else:
        reason = (pause.get("reason", "insufficient_members_at_pod_frame")
                  if pod_frame is None else "insufficient_members_at_pod_frame")
        data.update(
            status="uncertain",
            reason=reason,
            pod_frame=int(pod_frame) if pod_frame is not None else None,
            pod_sec=round((pod_frame - 1) / fps, 2) if pod_frame is not None else None,
            members=_serialize_members(members),
            metrics={"POD_SECTOR_COVERAGE": -1, "POD_MUTUAL_FACING": -1},
            violators=[],
            flagged_pairs=[],
            excluded_members=excluded,
            sectors={},
        )
    return data


def pod_data_public(data: dict) -> dict:
    """The JSON-serializable view of a compute_pod_data result."""
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_orientation_cache(run_folder: str, total: int) -> Dict[int, dict]:
    """Rebuild per-track smoothed orientation series from the
    ``*_OrientationCache.txt`` sidecar (columns:
    frame, id, body_dx, body_dy, muzzle_dx, muzzle_dy, confidence)."""
    from src.metrics._shared import pick_latest

    path = pick_latest(run_folder, "*_OrientationCache.txt")
    if not path:
        return {}
    per_track: Dict[int, dict] = {}
    raw = np.genfromtxt(path, delimiter=",", comments="#")
    if raw.size == 0:
        return {}
    raw = raw.reshape(-1, 7)
    for frame, tid, bdx, bdy, mdx, mdy, conf in raw:
        t = int(frame) - 1
        if not (0 <= t < total):
            continue
        rec = per_track.setdefault(int(tid), {
            "body_smooth": np.full((total, 2), np.nan),
            "muzzle_smooth": np.full((total, 2), np.nan),
            "conf": np.zeros(total),
        })
        rec["body_smooth"][t] = (bdx, bdy)
        rec["muzzle_smooth"][t] = (mdx, mdy)
        rec["conf"][t] = conf
    return per_track


def load_run_pod_inputs(run_folder: str) -> dict:
    """Everything the cache-based POD paths need, loaded from a run folder.
    Raises ValueError on missing prerequisites."""
    import pandas as pd

    from src.metrics._shared import load_inroom_ids, pick_latest, select_entry_tracks, team_size
    from src.utils.config import load_config, load_vmeta
    from src.utils.run_info import load_run_info
    from src.utils.run_metadata import resolve_fps_from_metadata

    run_folder = os.path.abspath(run_folder)
    info = load_run_info(run_folder)
    if not info:
        raise ValueError(f"No RunInfo.json in {run_folder}")
    cfg_file, *_rest = load_vmeta(info["vmeta_path"])
    cfg_file, config = _resolve_replay_config(run_folder, cfg_file, load_config)

    room_poly = config["Boundary"]
    if not room_poly.is_valid:
        room_poly = room_poly.buffer(0)
    fps = resolve_fps_from_metadata(run_folder, fallback=float(config.get("frame_rate", 30.0)))
    pos_path = pick_latest(run_folder, "*_PositionCache.txt")
    if not pos_path:
        raise ValueError("No *_PositionCache.txt in run folder")
    df = pd.read_csv(pos_path, header=None, names=["frame", "id", "x", "y"],
                     comment="#", skipinitialspace=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    total = int(df["frame"].max())
    tracks: Dict[int, list] = {}
    for row in df.itertuples(index=False):
        traj = tracks.setdefault(int(row.id), [None] * total)
        traj[int(row.frame) - 1] = (float(row.x), float(row.y))

    inroom = load_inroom_ids(run_folder)
    team = select_entry_tracks(tracks, inroom_ids=inroom, num_tracks=team_size(config))
    orient = load_orientation_cache(run_folder, total)
    return {"config": config, "cfg_file": cfg_file, "room_poly": room_poly,
            "fps": fps, "total": total, "tracks": tracks, "inroom": inroom,
            "team": team, "orient": orient}


def auto_pod_from_caches(run_folder: str, drill_start: int, drill_end: int,
                         sector_angle: Optional[float] = None) -> dict:
    """Re-run automatic POD detection purely from run-folder caches with an
    arbitrary drill window (used when the instructor moves the drill end)."""
    inputs = load_run_pod_inputs(run_folder)
    config, fps, total = inputs["config"], inputs["fps"], inputs["total"]
    team, orient, room_poly = inputs["team"], inputs["orient"], inputs["room_poly"]

    angle = float(sector_angle if sector_angle is not None
                  else config.get("pod_sector_angle_degrees", DEFAULT_SECTOR_ANGLE_DEG))
    firsts = [next((i + 1 for i, p in enumerate(traj) if p is not None), None)
              for _, traj in team]
    firsts = [f for f in firsts if f is not None]
    search_start = max([int(drill_start)] + firsts) if firsts else int(drill_start)
    pause = detect_team_pause(team, fps, search_start, int(drill_end), total)
    pod_frame = pause["start_frame"] if pause["found"] else None

    data: dict = {
        "sector_angle_degrees": angle,
        "team_ids": [int(tid) for tid, _ in team],
        "fps": fps,
        "source": "auto",
        "search_start_frame": int(search_start),
        "speed_threshold_px_s": (round(float(pause["speed_threshold"]), 2)
                                 if np.isfinite(pause["speed_threshold"]) else None),
    }
    members, excluded = ([], [])
    if pod_frame is not None:
        members, excluded = members_at_frame(team, orient, pod_frame, fps)
    if pod_frame is not None and len(members) >= 2:
        metrics = compute_pod_metrics(members, room_poly, angle)
        data.update(
            status="ok",
            pod_frame=int(pod_frame),
            pod_sec=round((pod_frame - 1) / fps, 2),
            members=_serialize_members(members),
            metrics={
                "POD_SECTOR_COVERAGE": metrics["POD_SECTOR_COVERAGE"],
                "POD_MUTUAL_FACING": metrics["POD_MUTUAL_FACING"],
            },
            violators=metrics["violators"],
            flagged_pairs=metrics["flagged_pairs"],
            excluded_members=excluded,
            sectors=serialize_sectors(metrics["sectors"]),
        )
    else:
        # Preserve a detected pause frame even when too few members had
        # reliable bearings there — same convention as compute_pod_data, so
        # the viewer can seed the instructor's Adjust marker with it.
        data.update(
            status="uncertain",
            reason=(pause.get("reason", "insufficient_members_at_pod_frame")
                    if pod_frame is None else "insufficient_members_at_pod_frame"),
            pod_frame=int(pod_frame) if pod_frame is not None else None,
            pod_sec=(round((pod_frame - 1) / fps, 2)
                     if pod_frame is not None else None),
            members=_serialize_members(members),
            metrics={"POD_SECTOR_COVERAGE": -1, "POD_MUTUAL_FACING": -1},
            violators=[], flagged_pairs=[], excluded_members=excluded, sectors={},
        )
    return data


def recompute_pod_for_run(run_folder: str, pod_frame: int,
                          sector_angle: Optional[float] = None) -> dict:
    """Instructor-adjustment path: recompute POD members/sectors/scores at an
    arbitrary frame purely from run-folder caches (OrientationCache +
    PositionCache + TrackerOutput roles) — no video, no pose model."""
    inputs = load_run_pod_inputs(run_folder)
    config, fps, total = inputs["config"], inputs["fps"], inputs["total"]
    team, orient, room_poly = inputs["team"], inputs["orient"], inputs["room_poly"]

    angle = float(sector_angle if sector_angle is not None
                  else config.get("pod_sector_angle_degrees", DEFAULT_SECTOR_ANGLE_DEG))

    pod_frame = int(max(1, min(total, pod_frame)))
    members, excluded = members_at_frame(team, orient, pod_frame, fps)
    data: dict = {
        "sector_angle_degrees": angle,
        "team_ids": [int(tid) for tid, _ in team],
        "fps": fps,
        "source": "instructor",
        "pod_frame": pod_frame,
        "pod_sec": round((pod_frame - 1) / fps, 2),
        "excluded_members": excluded,
    }
    if len(members) >= 2:
        metrics = compute_pod_metrics(members, room_poly, angle)
        data.update(
            status="ok",
            members=_serialize_members(members),
            metrics={
                "POD_SECTOR_COVERAGE": metrics["POD_SECTOR_COVERAGE"],
                "POD_MUTUAL_FACING": metrics["POD_MUTUAL_FACING"],
            },
            violators=metrics["violators"],
            flagged_pairs=metrics["flagged_pairs"],
            sectors=serialize_sectors(metrics["sectors"]),
        )
    else:
        data.update(
            status="uncertain",
            reason="insufficient_members_at_pod_frame",
            members=_serialize_members(members),
            metrics={"POD_SECTOR_COVERAGE": -1, "POD_MUTUAL_FACING": -1},
            violators=[], flagged_pairs=[], sectors={},
        )
    return data


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def draw_orientation_arrow(img, pos, direction, color, size=26.0, alpha=0.55, border=2):
    """Translucent track-colored arrowhead with a black border.

    Alpha-blends only the arrow's bounding ROI, not the full frame — the map
    video stamps thousands of these over a long run.
    """
    u = np.asarray(direction, dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64)
    perp = np.array([-u[1], u[0]])
    tip = p + u * size
    left = p - u * size * 0.35 + perp * size * 0.42
    right = p - u * size * 0.35 - perp * size * 0.42
    back = p - u * size * 0.12
    pts = np.array([tip, left, back, right], dtype=np.int32)

    pad = border + 2
    x0 = max(0, int(pts[:, 0].min()) - pad)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    x1 = min(img.shape[1], int(pts[:, 0].max()) + pad)
    y1 = min(img.shape[0], int(pts[:, 1].max()) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.fillPoly(overlay, [pts - np.array([x0, y0])], color)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)
    if border > 0:
        cv2.polylines(img, [pts], True, (0, 0, 0), border, cv2.LINE_AA)


def render_orientation_map_video(
    out_path: str,
    map_image: np.ndarray,
    tracker_output: List[dict],
    orient: Dict[int, dict],
    team_ids: List[int],
    colors: Dict[int, tuple],
    fps: float,
    pod_frame: Optional[int],
    sectors: Optional[dict],
    start_frame: int,
    end_frame: int,
) -> None:
    h, w = map_image.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    trail = map_image.copy()

    pos_by_frame: Dict[int, list] = {}
    for fr in tracker_output:
        items = []
        for obj in fr.get("objects", []):
            if obj.get("current_map_pos") and obj["id"] in team_ids:
                items.append((obj["id"], np.asarray(obj["current_map_pos"])))
        pos_by_frame[fr["frame"]] = items

    banner_len = int(round(POD_BANNER_SEC * fps))

    sector_layer = None
    if pod_frame is not None and sectors:
        sector_layer = map_image.copy()
        for tid, polys in sectors.items():
            for g in polys:
                pts = np.array(list(g.exterior.coords), dtype=np.int32)
                cv2.fillPoly(sector_layer, [pts], colors.get(tid, (200, 200, 200)))

    for fnum in range(start_frame, end_frame + 1):
        for tid, pos in pos_by_frame.get(fnum, []):
            cv2.circle(trail, (int(pos[0]), int(pos[1])), 2, colors.get(tid, (255, 255, 255)), -1)

        frame = trail.copy()
        at_pod = (pod_frame is not None
                  and pod_frame <= fnum < pod_frame + banner_len)
        if at_pod and sector_layer is not None:
            cv2.addWeighted(sector_layer, 0.18, frame, 0.82, 0, dst=frame)

        for tid, pos in pos_by_frame.get(fnum, []):
            rec = orient.get(tid)
            if rec is None:
                continue
            d = rec["muzzle_smooth"][fnum - 1]
            if not np.isfinite(d[0]):
                continue
            draw_orientation_arrow(frame, pos, d, colors.get(tid, (255, 255, 255)),
                        size=26.0, alpha=0.55, border=2)

        cv2.putText(frame, f"frame {fnum}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, f"frame {fnum}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if at_pod:
            label = f"POD ESTABLISHED @ {pod_frame}"
            cv2.putText(frame, label, (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(frame, label, (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        writer.write(frame)
    writer.release()


def render_pod_sectors_png(
    out_path: str,
    map_image: np.ndarray,
    members: List[dict],
    metrics: dict,
    colors: Dict[int, tuple],
    pod: dict,
    fps: float,
    sector_angle_deg: float,
) -> None:
    img = map_image.copy()
    overlay = img.copy()
    for m in members:
        for g in metrics["sectors"].get(m["id"], []):
            pts = np.array(list(g.exterior.coords), dtype=np.int32)
            cv2.fillPoly(overlay, [pts], colors.get(m["id"], (200, 200, 200)))
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, dst=img)
    for m in members:
        for g in metrics["sectors"].get(m["id"], []):
            pts = np.array(list(g.exterior.coords), dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

    for pair in metrics["flagged_pairs"]:
        a = next(m for m in members if m["id"] == pair["from"])
        b = next(m for m in members if m["id"] == pair["to"])
        cv2.arrowedLine(img, tuple(a["pos"].astype(int)), tuple(b["pos"].astype(int)),
                        (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.06)

    for m in members:
        p = tuple(m["pos"].astype(int))
        draw_orientation_arrow(img, m["pos"], m["bearing"], colors.get(m["id"], (255, 255, 255)),
                    size=30.0, alpha=0.8, border=2)
        cv2.circle(img, p, 7, (255, 255, 255), -1)
        cv2.circle(img, p, 7, (0, 0, 0), 2)
        cv2.putText(img, str(m["id"]), (p[0] + 10, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, str(m["id"]), (p[0] + 10, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    lines = [
        f"POD established: frame {pod['frame']} (t={pod['sec']:.1f}s)"
        + "   muzzle bearings at that frame",
        f"Sector angle: {sector_angle_deg:.0f} deg",
        f"POD_SECTOR_COVERAGE: {metrics['POD_SECTOR_COVERAGE']:.2f}"
        f"  (team sectors cover {metrics['POD_SECTOR_COVERAGE']:.0%} of the room)",
        f"POD_MUTUAL_FACING:   {metrics['POD_MUTUAL_FACING']:.2f}"
        + (f"  ({len(metrics['violators'])} of {len(members)} flag a teammate: "
           + ", ".join(f"{p['from']}->{p['to']}" for p in metrics["flagged_pairs"]) + ")"
           if metrics["flagged_pairs"] else f"  (0 of {len(members)} flag a teammate)"),
    ]
    y = img.shape[0] - 14 - 22 * (len(lines) - 1)
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22
    cv2.imwrite(out_path, img)


def save_orientation_cache(path: str, tracker_output: List[dict],
                           orient: Dict[int, dict]) -> None:
    """Write the smoothed per-frame orientation sidecar."""
    with open(path, "w") as f:
        f.write("# frame, id, body_dx, body_dy, muzzle_dx, muzzle_dy, confidence\n")
        for fr in tracker_output:
            t = fr["frame"] - 1
            for obj in fr.get("objects", []):
                rec = orient.get(obj["id"])
                if rec is None:
                    continue
                b, m, c = rec["body_smooth"][t], rec["muzzle_smooth"][t], rec["conf"][t]
                if np.isfinite(b[0]) and np.isfinite(m[0]):
                    f.write(f"{fr['frame']}, {obj['id']}, {b[0]:.4f}, {b[1]:.4f}, "
                            f"{m[0]:.4f}, {m[1]:.4f}, {c:.3f}\n")


def save_pod_camera_frame(overlay_video: str, out_path: str, pod_frame: int,
                          drill_start: int, pod_sec: float) -> Optional[str]:
    """Grab the POD frame from the drill-trimmed tracking-overlay video
    (boxes/IDs/skeletons visible) and stamp the POD label on it."""
    cap = cv2.VideoCapture(overlay_video)
    n_ov = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = max(0, min(n_ov - 1, pod_frame - drill_start))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, cam = cap.read()
    cap.release()
    if not ok:
        return None
    label = f"POD established  frame {pod_frame}  (t={pod_sec:.1f}s)"
    cv2.putText(cam, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(cam, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(out_path, cam)
    print(f"[orientation] wrote {out_path}")
    return out_path


# ----------------------------------------------------------------------
# Replay driver
# ----------------------------------------------------------------------

def _chirality_diagnostic(orient, team_trajs, fps, total) -> Optional[float]:
    """Mean cosine between body bearing and walking direction while moving.

    Positive ≈ facing matches motion (people walk forward) — validates the
    anatomical rotation constant. Uses only clearly-moving samples.
    """
    cos_vals = []
    for tid, traj in team_trajs:
        rec = orient.get(tid)
        if rec is None:
            continue
        speeds = _member_speed(traj, fps, total)
        moving = np.nanpercentile(speeds[np.isfinite(speeds)], 75) if np.isfinite(speeds).any() else None
        if moving is None or moving <= 0:
            continue
        prev = None
        for i in range(total):
            p = traj[i] if i < len(traj) else None
            if p is None:
                continue
            if prev is not None and np.isfinite(speeds[i]) and speeds[i] > moving:
                step = _unit(np.asarray(p) - np.asarray(prev[1]))
                b = rec["body_smooth"][i]
                if step is not None and np.isfinite(b[0]):
                    cos_vals.append(float(np.dot(step, b)))
            prev = (i, p)
    return float(np.mean(cos_vals)) if cos_vals else None


def _boundary_signature(coords) -> Optional[frozenset]:
    try:
        return frozenset((int(round(float(x))), int(round(float(y)))) for x, y in coords)
    except (TypeError, ValueError):
        return None


def _resolve_replay_config(run_folder: str, cfg_file: str, load_config) -> Tuple[str, dict]:
    """Guard against a re-edited vmeta: the run's Analysis.json remembers the
    room boundary the run actually used. If the vmeta's config disagrees, scan
    the config's directory for the config whose Boundary matches and use that
    instead (raising when the room cannot be identified)."""
    import glob as _glob

    config = load_config(cfg_file)
    apath = None
    for p in _glob.glob(os.path.join(run_folder, "*_Analysis.json")):
        apath = p
        break
    if apath is None:
        return cfg_file, config
    try:
        with open(apath) as f:
            analysis = json.load(f)
        b = analysis.get("boundary")
        coords = b.get("coords") if isinstance(b, dict) else b
    except (OSError, ValueError):
        coords = None
    if not coords:
        return cfg_file, config

    want = _boundary_signature(coords[:-1] if coords[0] == coords[-1] else coords)
    have = _boundary_signature(list(config["Boundary"].exterior.coords)[:-1])
    if want is None or have is None or want == have:
        return cfg_file, config

    print(f"[orientation] WARNING: vmeta config {os.path.basename(cfg_file)} boundary does not "
          f"match this run's Analysis.json — the vmeta was likely re-pointed since this run.")
    matches = []
    for cand in sorted(_glob.glob(os.path.join(os.path.dirname(cfg_file), "*.json"))):
        try:
            c = load_config(cand)
        except Exception:
            continue
        sig = _boundary_signature(list(c["Boundary"].exterior.coords)[:-1])
        if sig == want:
            matches.append((cand, c))

    # Several configs can share a room shape (same physical room, different
    # camera calibration) — disambiguate by the room/scenario name in the
    # run's own video path before trusting a match.
    if len(matches) > 1:
        video_path = str((analysis.get("video") or {}).get("video_path") or "").lower()
        named = [(cand, c) for cand, c in matches
                 if os.path.splitext(os.path.basename(cand))[0].lower() in video_path]
        if len(named) == 1:
            matches = named
    if len(matches) == 1:
        cand, c = matches[0]
        print(f"[orientation] auto-resolved room config: {os.path.basename(cand)}")
        return cand, c
    if matches:
        names = ", ".join(os.path.basename(cand) for cand, _ in matches)
        raise ValueError(
            f"Ambiguous room config: {names} all match this run's boundary but may carry "
            "different camera calibrations. Pass --config explicitly."
        )
    raise ValueError(
        "Could not identify this run's room config (no config in "
        f"{os.path.dirname(cfg_file)} matches the run's boundary). Pass --config explicitly."
    )


def run_replay(run_folder: str, sector_angle: Optional[float] = None,
               config_path: Optional[str] = None) -> dict:
    from libs.Track.processing.utils import load_pixel_mapper
    from src.helper_functions import _build_track_color_cache, _get_track_color
    from src.metrics._shared import load_inroom_ids, pick_latest
    from src.utils.config import load_config, load_vmeta
    from src.utils.run_info import load_run_info
    from src.utils.run_metadata import resolve_fps_from_metadata

    run_folder = os.path.abspath(run_folder)
    info = load_run_info(run_folder)
    if not info:
        raise SystemExit(f"No RunInfo.json in {run_folder}")
    if config_path:
        cfg_file = os.path.abspath(config_path)
        config = load_config(cfg_file)
    else:
        cfg_file, _video, _st, _role, _title = load_vmeta(info["vmeta_path"])
        cfg_file, config = _resolve_replay_config(run_folder, cfg_file, load_config)

    angle = float(sector_angle if sector_angle is not None
                  else config.get("pod_sector_angle_degrees", DEFAULT_SECTOR_ANGLE_DEG))
    if not (0.0 < angle < 180.0):
        raise SystemExit(f"sector angle must be in (0, 180) degrees, got {angle:g}")

    tracker_path = pick_latest(run_folder, "*_TrackerOutput.json")
    if not tracker_path:
        raise SystemExit("No *_TrackerOutput.json in run folder")
    base = os.path.basename(tracker_path)[: -len("_TrackerOutput.json")]
    with open(tracker_path) as f:
        tracker_output = json.load(f)
    total = max(fr["frame"] for fr in tracker_output)

    fps = resolve_fps_from_metadata(run_folder, fallback=float(config.get("frame_rate", 30.0)))
    mapper = load_pixel_mapper(config["point_mapping_path"])
    room_poly = config["Boundary"]
    if not room_poly.is_valid:  # e.g. bowtie vertex order in a hand-authored config
        print("[orientation] WARNING: room Boundary polygon is invalid — repairing with buffer(0)")
        room_poly = room_poly.buffer(0)
    map_image = config["Map Image"]
    inroom = load_inroom_ids(run_folder)

    # tracks_by_id in the engine's convention: index = frame-1, None gaps
    tracks_by_id: Dict[int, list] = {}
    for fr in tracker_output:
        for obj in fr.get("objects", []):
            traj = tracks_by_id.setdefault(obj["id"], [None] * total)
            if obj.get("current_map_pos"):
                traj[fr["frame"] - 1] = obj["current_map_pos"]

    dw_path = pick_latest(run_folder, "*_DrillWindow.json")
    drill_start, drill_end = 1, total
    if dw_path:
        with open(dw_path) as f:
            dw = json.load(f)
        dw = dw.get("drill_window", dw)
        drill_start = int(dw.get("start_frame") or 1)
        drill_end = int(dw.get("end_frame") or total)

    print(f"[orientation] run={base} frames={total} fps={fps:g} "
          f"drill=[{drill_start},{drill_end}] sector_angle={angle:g}")

    data = compute_pod_data(
        tracker_output, mapper, fps, config, tracks_by_id, inroom,
        drill_start, drill_end, room_poly, sector_angle=angle,
    )
    orient = data["_orient"]
    team = data["_team"]
    team_ids = data["team_ids"]
    pod_frame = data.get("pod_frame")
    metrics = ({"sectors": data["_sectors"], **data["metrics"],
                "violators": data["violators"], "flagged_pairs": data["flagged_pairs"]}
               if data["status"] == "ok" else None)
    members = data.get("_members", [])

    chir = _chirality_diagnostic(orient, team, fps, total)
    if chir is not None:
        print(f"[orientation] chirality diagnostic (facing·motion while moving): {chir:+.2f} "
              f"({'OK — facing tracks walking direction' if chir > 0 else 'INVERTED?'})")

    result = {"schema_version": POD_JSON_SCHEMA_VERSION, **pod_data_public(data)}

    predefined, cache = _build_track_color_cache()
    colors = {tid: _get_track_color(tid, set(inroom), cache, predefined) for tid in team_ids}

    if data["status"] == "ok":
        print(f"[orientation] POD established at frame {pod_frame} (t={data['pod_sec']:.1f}s)")
        for m in data["members"]:
            print(f"[orientation]  member {m['id']}: pos={m['pos']} "
                  f"muzzle={m['bearing_deg']:+.0f}deg conf={m['confidence']}")
        print(f"[orientation] POD_SECTOR_COVERAGE = {data['metrics']['POD_SECTOR_COVERAGE']}")
        print(f"[orientation] POD_MUTUAL_FACING   = {data['metrics']['POD_MUTUAL_FACING']}"
              + (f"  violators: {data['violators']} "
                 f"pairs: {[(p['from'], p['to']) for p in data['flagged_pairs']]}"
                 if data["flagged_pairs"] else "  (no one flags a teammate)"))
    else:
        print(f"[orientation] POD establishment UNCERTAIN ({data.get('reason')}) — metrics = -1")

    # ---- artifacts ----
    cache_path = os.path.join(run_folder, f"{base}_OrientationCache.txt")
    save_orientation_cache(cache_path, tracker_output, orient)

    json_path = os.path.join(run_folder, f"{base}_PodOrientation.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    video_path = os.path.join(run_folder, f"{base}_Tracking_Map_Orientation.mp4")
    render_orientation_map_video(
        video_path, map_image, tracker_output, orient, team_ids, colors, fps,
        pod_frame if metrics else None,
        metrics["sectors"] if metrics else None,
        drill_start, min(drill_end, total),
    )

    if metrics:
        png_path = os.path.join(run_folder, f"{base}_POD_Sectors.png")
        render_pod_sectors_png(png_path, map_image, members, metrics, colors,
                               {"frame": pod_frame, "sec": data["pod_sec"]}, fps, angle)
        print(f"[orientation] wrote {png_path}")

        # POD-frame camera snapshot (from the run's own drill-trimmed overlay
        # video, so boxes/IDs/skeletons are visible) — lets the reviewer check
        # identification and orientation against the map-view artifacts.
        overlay_video = pick_latest(run_folder, f"{base}_Tracking_Overlays.mp4")
        if overlay_video:
            save_pod_camera_frame(
                overlay_video,
                os.path.join(run_folder, f"{base}_POD_CameraFrame.png"),
                pod_frame, drill_start, data["pod_sec"],
            )
    print(f"[orientation] wrote {cache_path}")
    print(f"[orientation] wrote {json_path}")
    print(f"[orientation] wrote {video_path}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="POD orientation replay (experimental)")
    ap.add_argument("run_folder", help="engine run folder (contains RunInfo.json)")
    ap.add_argument("--sector-angle", type=float, default=None,
                    help=f"full sector angle in degrees "
                         f"(default: config pod_sector_angle_degrees or {DEFAULT_SECTOR_ANGLE_DEG:g})")
    ap.add_argument("--config", default=None,
                    help="explicit room config JSON (overrides the vmeta's config)")
    args = ap.parse_args(argv)
    try:
        run_replay(args.run_folder, args.sector_angle, args.config)
    except ValueError as exc:  # library-level errors -> clean CLI exit
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main(sys.argv[1:])
