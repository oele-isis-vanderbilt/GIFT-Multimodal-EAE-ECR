import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon as _ShPolygon

from src.metrics._shared import (
    DoorAxes,
    classify_entry_side,
    door_for_entry,
    first_entry_frame,
    fit_entry_velocity,
    load_door_axes,
    load_entry_polygon_points,
    select_entry_tracks,
    team_size,
)
from src.metrics.entrance_vectors import ENTRY_VECTOR_WINDOW_SEC


PHASE_ID = "entrance"
PHASE_LABEL = "Entrance Phase"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _round_or_none(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _first_valid_index(traj: List[Optional[Tuple[float, float]]]) -> Optional[int]:
    for i, pt in enumerate(traj):
        if pt is not None:
            return i
    return None


def _valid_sample_count(traj: List[Optional[Tuple[float, float]]]) -> int:
    return sum(1 for pt in traj if pt is not None)


def _normalize_point(pt: Any) -> Optional[Tuple[float, float]]:
    try:
        arr = np.asarray(pt, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        if not np.isfinite(arr[:2]).all():
            return None
        return float(arr[0]), float(arr[1])
    except Exception:
        return None


def _normalize_pods_cfg(pods_cfg: Any) -> List[Tuple[float, float]]:
    if pods_cfg is None:
        return []

    try:
        arr = np.asarray(pods_cfg, dtype=float)
    except Exception:
        arr = None

    if arr is not None:
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            if arr.shape[0] < 2:
                return []
            return [(float(arr[0]), float(arr[1]))]

        out: List[Tuple[float, float]] = []
        for row in arr:
            row = np.asarray(row, dtype=float).reshape(-1)
            if row.size < 2:
                continue
            out.append((float(row[0]), float(row[1])))
        return out

    out: List[Tuple[float, float]] = []
    for row in pods_cfg:
        try:
            if row is None or len(row) < 2:
                continue
            out.append((float(row[0]), float(row[1])))
        except Exception:
            continue
    return out


def _frame_to_time_sec(frame_1based: Optional[int], frame_rate: float) -> Optional[float]:
    if frame_1based is None:
        return None
    if frame_rate <= 0:
        frame_rate = 30.0
    return _round_or_none((int(frame_1based) - 1) / frame_rate, 3)


# ``_select_entry_tracks`` previously lived here as a private duplicate of the
# selection rule used by every team-entry metric. The canonical implementation
# is now ``src.metrics._shared.select_entry_tracks``; this thin wrapper is
# kept for the existing call site below.
def _select_entry_tracks(
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]],
    inroom_ids: Optional[List[int]],
    num_tracks: int,
) -> List[Tuple[int, List[Optional[Tuple[float, float]]]]]:
    return select_entry_tracks(
        tracks_by_id, inroom_ids=inroom_ids, num_tracks=num_tracks
    )


def _compute_entry_vector_details(
    traj: List[Optional[Tuple[float, float]]],
    *,
    frame_rate: float,
    window_sec: float,
    doors: List[DoorAxes],
) -> Dict[str, Any]:
    """Per-entrant entry record using door-axis side classification.

    Direction is the least-squares velocity fit over ``window_sec`` of valid
    samples starting at the door-crossing frame; side is decided by which of
    the door's two reference path axes (``p̂_A``, ``p̂_B``) the movement is
    more aligned with. Output schema is preserved for the viewer; the legacy
    ``z_cross`` field is kept and set to ``None`` (no longer part of the
    model).
    """
    fps = float(frame_rate or 30.0)
    win_sec = float(window_sec)
    win_frames = max(1, int(round(win_sec * fps)))

    base = {
        "start_frame": None,
        "start_time_sec": None,
        "start_xy": None,
        "end_xy": None,
        "dx": None,
        "dy": None,
        "z_cross": None,
        "sign": 0,
        "direction_label": "UNKNOWN",
        "valid": False,
        "sample_count": 0,
        "window_frames": win_frames,
        "window_sec": win_sec,
        "door_index": -1,
        "door_type": None,
    }

    if not traj:
        return base

    start_idx = first_entry_frame(traj, doors)
    if start_idx is None:
        return base

    entry_pt = traj[start_idx]
    if entry_pt is None:
        for k in range(start_idx, len(traj)):
            if traj[k] is not None:
                start_idx = k
                entry_pt = traj[k]
                break
    if entry_pt is None:
        return base

    door = door_for_entry(entry_pt, doors)
    door_index = -1
    door_type = None
    if door is not None:
        try:
            door_index = doors.index(door)
        except ValueError:
            door_index = -1
        door_type = door.door_type

    sample_count = sum(
        1 for k in range(int(start_idx), min(len(traj), int(start_idx) + win_frames))
        if traj[k] is not None
    )

    v = fit_entry_velocity(traj, start_idx, fps, win_sec)
    start_xy = _normalize_point(entry_pt)
    if v is None:
        return {
            **base,
            "start_frame": int(start_idx) + 1,
            "start_time_sec": _frame_to_time_sec(int(start_idx) + 1, fps),
            "start_xy": list(start_xy) if start_xy is not None else None,
            "sample_count": int(sample_count),
            "door_index": int(door_index),
            "door_type": door_type,
        }

    m_vec = v * win_sec
    sign, side, _, _ = classify_entry_side(m_vec, door)
    valid = sign != 0
    end_xy = (float(entry_pt[0]) + float(m_vec[0]), float(entry_pt[1]) + float(m_vec[1]))
    direction_label = side if side in ("A", "B") else "UNKNOWN"

    return {
        "start_frame": int(start_idx) + 1,
        "start_time_sec": _frame_to_time_sec(int(start_idx) + 1, fps),
        "start_xy": [float(entry_pt[0]), float(entry_pt[1])],
        "end_xy": [float(end_xy[0]), float(end_xy[1])],
        "dx": float(m_vec[0]),
        "dy": float(m_vec[1]),
        "z_cross": None,
        "sign": int(sign),
        "direction_label": direction_label,
        "valid": bool(valid),
        "sample_count": int(sample_count),
        "window_frames": win_frames,
        "window_sec": win_sec,
        "door_index": int(door_index),
        "door_type": door_type,
    }


def _get_allowed_pair_gap_seconds(config: Dict[str, Any], pair_number: int) -> float:
    base_thr = float(config.get("HESITATION_THRESHOLD", 1.0))
    second_thr = float(config.get("HESITATION_THRESHOLD_SECOND", base_thr * 2.0))
    return second_thr if int(pair_number) == 2 else base_thr


def _hesitation_pair_score(gap_sec: float, threshold_sec: float) -> float:
    bottom_clamp = max(0.0, float(gap_sec) - float(threshold_sec))
    return float(np.exp(-(bottom_clamp ** 2) / 0.5))


def _total_entry_penalty(overrun: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    if overrun >= limit:
        return 0.0
    return float(np.exp(-overrun / (limit - overrun)))


def _build_artifact_entry(path: Optional[str], label: str, artifact_type: str) -> Dict[str, Any]:
    exists = bool(path and os.path.exists(path))
    return {
        "path": path,
        "label": label,
        "type": artifact_type,
        "exists": exists,
    }


# Per viewer-slot priority lists (best → fallback). The first suffix whose
# file exists on disk wins. Adding a new annotator: append its suffix to the
# slot it should populate — no other code needs to change. The viewer treats
# each slot independently (a "Motion" or "Gaze" mode appears only when its
# camera or map artifact resolves).
_MOTION_CAMERA_SUFFIX_PRIORITY: List[str] = [
    "_Tracking_WithClearance",  # annotate_camera_tracking_with_clearance
    "_Tracking_Overlays",       # annotate_camera_video
]
_MOTION_MAP_SUFFIX_PRIORITY: List[str] = [
    "_Tracking_PodAreasWithTrails",  # annotate_map_pod_with_paths_video
    "_Tracking_PodAreas",            # annotate_map_pod_video
    "_Tracking_Map",                 # annotate_map_video
]
_GAZE_CAMERA_SUFFIX_PRIORITY: List[str] = [
    "_Gaze_Triangles",  # annotate_camera_with_gaze_triangle
]
_GAZE_MAP_SUFFIX_PRIORITY: List[str] = [
    "_Gaze_MapCleared",  # annotate_map_with_gaze
]


def _resolve_first_existing(
    output_directory: str, video_basename: str, suffixes: List[str]
) -> Optional[str]:
    """Return the first ``{basename}{suffix}.mp4`` that exists on disk."""
    for suffix in suffixes:
        path = os.path.join(output_directory, f"{video_basename}{suffix}.mp4")
        if os.path.exists(path):
            return path
    return None


def _build_artifacts_section(
    *,
    output_directory: str,
    video_basename: str,
    original_video_path: Optional[str],
    tracker_output_json_path: Optional[str] = None,
    position_cache_path: Optional[str] = None,
    gaze_cache_path: Optional[str] = None,
    empty_map_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Artifact keys are explicit so the GUI can switch display modes cleanly.
    Each visualization slot resolves to the highest-priority annotator output
    that actually exists on disk, so disabled annotators degrade gracefully
    (e.g. omitting POD overlays falls through to the plain map track video).
    """
    motion_camera = _resolve_first_existing(
        output_directory, video_basename, _MOTION_CAMERA_SUFFIX_PRIORITY
    )
    motion_map = _resolve_first_existing(
        output_directory, video_basename, _MOTION_MAP_SUFFIX_PRIORITY
    )
    gaze_camera = _resolve_first_existing(
        output_directory, video_basename, _GAZE_CAMERA_SUFFIX_PRIORITY
    )
    gaze_map = _resolve_first_existing(
        output_directory, video_basename, _GAZE_MAP_SUFFIX_PRIORITY
    )

    return {
        "original_video": _build_artifact_entry(original_video_path, "Original Video", "video"),
        "motion": {
            "camera_video": _build_artifact_entry(motion_camera, "Motion Camera Video", "video"),
            "map_video": _build_artifact_entry(motion_map, "Motion Map Video", "video"),
        },
        "gaze": {
            "camera_video": _build_artifact_entry(gaze_camera, "Gaze Camera Video", "video"),
            "map_video": _build_artifact_entry(gaze_map, "Gaze Map Video", "video"),
        },
        "tracker_output_json": _build_artifact_entry(
            tracker_output_json_path, "Tracker Output JSON", "json"
        ),
        "position_cache": _build_artifact_entry(
            position_cache_path, "Position Cache", "text"
        ),
        "gaze_cache": _build_artifact_entry(
            gaze_cache_path, "Gaze Cache", "text"
        ),
        "empty_map_image": _build_artifact_entry(
            empty_map_path, "Empty Map", "image"
        ),
    }


def build_analysis_session(
    *,
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]],
    inroom_ids: Optional[List[int]],
    frame_rate: float,
    config: Dict[str, Any],
    video_basename: str,
    video_path: Optional[str],
    total_frames: int = 0,
    duration_sec: Optional[float] = None,
    start_time: Any = None,
    map_img: Any = None,
    output_directory: Optional[str] = None,
    tracker_output_json_path: Optional[str] = None,
    position_cache_path: Optional[str] = None,
    gaze_cache_path: Optional[str] = None,
    empty_map_path: Optional[str] = None,
    drill_window: Any = None,
    metric_flags: Optional[List[Dict[str, Any]]] = None,
    move_along_wall: Optional[Dict[str, Any]] = None,
    pod_orientation: Optional[Dict[str, Any]] = None,
    pod_sectors_path: Optional[str] = None,
    pod_camera_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the analysis-session payload (schema v2.0, metric-centric).

    Internally this still computes entrance-only data (the only phase the engine
    emits today), but the output shape replaces the legacy ``phases[]`` block
    with a flat ``metrics[]`` list so the new viewer can toggle each metric's
    artifacts independently. See ``_convert_to_v2_schema`` for the mapping rules.

    Output shape:
      {
        "schema_version": "2.0",
        "session": {...},
        "video": {...},
        "timeline": {"items": [...]},   # entries are baseline (metric_id=null);
                                        # vectors / pair_gaps / duration carry metric_id
        "metrics": [...],               # one record per produced metric
        "flags": [...],                 # each tagged with metric_id
        "entities": {...},
        "artifacts": {...}
      }
    """
    frame_rate = float(frame_rate or config.get("frame_rate", 30.0) or 30.0)
    pods_cfg = _normalize_pods_cfg(config.get("POD", []))
    num_tracks = team_size(config)

    selected_tracks = _select_entry_tracks(
        tracks_by_id=tracks_by_id,
        inroom_ids=inroom_ids,
        num_tracks=num_tracks,
    )

    # POD centroid is retained only as informational metadata in the output
    # payload; it is no longer part of the entry-direction calculation.
    if pods_cfg:
        _pod_centroid = np.mean(np.asarray(pods_cfg, dtype=float), axis=0)
        centroid_xy = [float(_pod_centroid[0]), float(_pod_centroid[1])]
    elif map_img is not None:
        try:
            h, w = map_img.shape[:2]
            centroid_xy = [float(w / 2.0), float(h / 2.0)]
        except Exception:
            centroid_xy = [0.0, 0.0]
    else:
        centroid_xy = [0.0, 0.0]

    # Door axes (per-door (p̂_A, p̂_B, n_in, type)) drive entry-side classification.
    boundary_cfg = config.get("Boundary")
    if boundary_cfg is None:
        boundary_pts: List[Any] = []
    elif hasattr(boundary_cfg, "exterior"):
        try:
            boundary_pts = list(boundary_cfg.exterior.coords)
        except Exception:
            boundary_pts = []
    else:
        boundary_pts = list(boundary_cfg)
    door_polys = load_entry_polygon_points(config.get("entry_polys_path"))
    doors: List[DoorAxes] = (
        load_door_axes(boundary_pts, door_polys) if boundary_pts and door_polys else []
    )

    vector_window_sec = float(ENTRY_VECTOR_WINDOW_SEC)
    entries: List[Dict[str, Any]] = []
    vectors: List[Dict[str, Any]] = []
    pair_gaps: List[Dict[str, Any]] = []
    vector_transitions: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []
    timeline_items: List[Dict[str, Any]] = []

    # Build ordered entries + vector details
    for entry_number, (track_id, traj) in enumerate(selected_tracks, start=1):
        first_idx = _first_valid_index(traj)
        if first_idx is None:
            continue

        start_frame = int(first_idx) + 1
        start_xy = _normalize_point(traj[first_idx])

        entry_record = {
            "entry_number": int(entry_number),
            "entry_name": f"Entry {entry_number}",
            "track_id": int(track_id),
            "start_frame": int(start_frame),
            "start_time_sec": _frame_to_time_sec(start_frame, frame_rate),
            "start_xy": list(start_xy) if start_xy is not None else None,
            "sample_count": int(_valid_sample_count(traj)),
        }
        entries.append(entry_record)

        vec = _compute_entry_vector_details(
            traj,
            frame_rate=frame_rate,
            window_sec=vector_window_sec,
            doors=doors,
        )
        vec_record = {
            "entry_number": int(entry_number),
            "entry_name": f"Entry {entry_number}",
            "track_id": int(track_id),
            **vec,
        }
        vectors.append(vec_record)

        if not bool(vec_record.get("valid", False)):
            flags.append(
                {
                    "flag_id": f"entrance_vector_unknown_{entry_number}",
                    "phase_id": PHASE_ID,
                    "type": "vector_unknown",
                    "severity": "warning",
                    "frame": int(start_frame),
                    "time_sec": _frame_to_time_sec(start_frame, frame_rate),
                    "entry_number": int(entry_number),
                    "track_id": int(track_id),
                    "title": f"Entry {entry_number} vector is unknown",
                    "message": (
                        f"Entry {entry_number} does not have enough valid movement to "
                        f"determine an entrance vector over the configured analysis window."
                    ),
                }
            )

    # Entrance vector transitions + score
    valid_vectors = [vector for vector in vectors if int(vector.get("sign", 0)) != 0]
    if len(valid_vectors) < 2:
        entrance_vectors_score = -1.0
        vector_alternation_count = 0
        vector_transition_count = max(0, len(valid_vectors) - 1)
    else:
        vector_alternation_count = 0
        for i in range(1, len(valid_vectors)):
            prev_vec = valid_vectors[i - 1]
            cur_vec = valid_vectors[i]
            prev_sign = int(prev_vec.get("sign", 0) or 0)
            cur_sign = int(cur_vec.get("sign", 0) or 0)
            actual_alternation = bool(np.sign(cur_sign) != np.sign(prev_sign))
            if actual_alternation:
                vector_alternation_count += 1

            transition_number = i
            transition = {
                "transition_number": int(transition_number),
                "from_entry_number": int(prev_vec["entry_number"]),
                "to_entry_number": int(cur_vec["entry_number"]),
                "from_track_id": int(prev_vec["track_id"]),
                "to_track_id": int(cur_vec["track_id"]),
                "frame": int(cur_vec.get("start_frame", 0) or 0),
                "time_sec": cur_vec.get("start_time_sec"),
                "start_frame": int(prev_vec.get("start_frame", 0) or 0),
                "end_frame": int(cur_vec.get("start_frame", 0) or 0),
                "start_time_sec": prev_vec.get("start_time_sec"),
                "end_time_sec": cur_vec.get("start_time_sec"),
                "from_sign": int(prev_sign),
                "to_sign": int(cur_sign),
                "expected_alternation": True,
                "actual_alternation": bool(actual_alternation),
                "violates_vector_rule": bool(not actual_alternation),
            }
            vector_transitions.append(transition)

            if not actual_alternation:
                flags.append(
                    {
                        "flag_id": f"entrance_vector_direction_violation_{transition_number}",
                        "phase_id": PHASE_ID,
                        "type": "vector_direction_violation",
                        "severity": "warning",
                        "frame": int(prev_vec.get("start_frame", 0) or 0),
                        "time_sec": prev_vec.get("start_time_sec"),
                        "start_frame": int(prev_vec.get("start_frame", 0) or 0),
                        "end_frame": int(cur_vec.get("start_frame", 0) or 0),
                        "start_time_sec": prev_vec.get("start_time_sec"),
                        "end_time_sec": cur_vec.get("start_time_sec"),
                        "transition_number": int(transition_number),
                        "from_entry_number": int(prev_vec["entry_number"]),
                        "to_entry_number": int(cur_vec["entry_number"]),
                        "title": f"Entries {prev_vec['entry_number']} and {cur_vec['entry_number']} did not alternate direction",
                        "message": (
                            f"Entry {cur_vec['entry_number']} did not alternate entrance direction relative to Entry {prev_vec['entry_number']}."
                        ),
                    }
                )

        vector_transition_count = len(valid_vectors) - 1
        entrance_vectors_score = round(
            vector_alternation_count / max(1, vector_transition_count),
            2,
        )

    # Hesitation details + score
    hesitation_pair_scores: List[float] = []
    for i in range(1, len(entries)):
        prev_entry = entries[i - 1]
        cur_entry = entries[i]
        pair_number = i

        start_prev = int(prev_entry["start_frame"])
        start_cur = int(cur_entry["start_frame"])
        delta_frames = int(start_cur - start_prev)
        gap_sec = float(delta_frames / frame_rate)
        allowed_gap_sec = float(_get_allowed_pair_gap_seconds(config, pair_number))
        pair_score = float(_hesitation_pair_score(gap_sec, allowed_gap_sec))
        hesitation_pair_scores.append(pair_score)

        violates_time_limit = bool(gap_sec > allowed_gap_sec)

        pair_record = {
            "pair_number": int(pair_number),
            "pair_name": f"Pair {pair_number}",
            "from_entry_number": int(prev_entry["entry_number"]),
            "to_entry_number": int(cur_entry["entry_number"]),
            "from_track_id": int(prev_entry["track_id"]),
            "to_track_id": int(cur_entry["track_id"]),
            "from_start_frame": int(start_prev),
            "to_start_frame": int(start_cur),
            "from_start_time_sec": _frame_to_time_sec(start_prev, frame_rate),
            "to_start_time_sec": _frame_to_time_sec(start_cur, frame_rate),
            "gap_frames": int(delta_frames),
            "gap_sec": _round_or_none(gap_sec, 3),
            "allowed_gap_sec": _round_or_none(allowed_gap_sec, 3),
            "score": _round_or_none(pair_score, 3),
            "violates_time_limit": violates_time_limit,
        }
        pair_gaps.append(pair_record)

        if violates_time_limit:
            flags.append(
                {
                    "flag_id": f"entrance_pair_time_violation_{pair_number}",
                    "phase_id": PHASE_ID,
                    "type": "pair_time_violation",
                    "severity": "warning",
                    "frame": int(start_prev),
                    "time_sec": _frame_to_time_sec(start_prev, frame_rate),
                    "start_frame": int(start_prev),
                    "end_frame": int(start_cur),
                    "start_time_sec": _frame_to_time_sec(start_prev, frame_rate),
                    "end_time_sec": _frame_to_time_sec(start_cur, frame_rate),
                    "pair_number": int(pair_number),
                    "from_entry_number": int(prev_entry["entry_number"]),
                    "to_entry_number": int(cur_entry["entry_number"]),
                    "title": f"Pair {pair_number} exceeded timing allowance",
                    "message": (
                        f"Gap between Entry {prev_entry['entry_number']} and Entry {cur_entry['entry_number']} "
                        f"was {gap_sec:.2f}s, exceeding the allowed {allowed_gap_sec:.2f}s."
                    ),
                }
            )

    if hesitation_pair_scores:
        entrance_hesitation_score = round(float(np.mean(hesitation_pair_scores)), 2)
    else:
        entrance_hesitation_score = -1.0

    # Total entry details + score
    if len(entries) < 2:
        total_entry_score = 0.0
        total_entry = {
            "entry_count": int(len(entries)),
            "pair_count": max(0, len(entries) - 1),
            "start_frame": int(entries[0]["start_frame"]) if entries else None,
            "end_frame": int(entries[-1]["start_frame"]) if entries else None,
            "start_time_sec": entries[0]["start_time_sec"] if entries else None,
            "end_time_sec": entries[-1]["start_time_sec"] if entries else None,
            "duration_frames": 0 if len(entries) == 1 else None,
            "duration_sec": 0.0 if len(entries) == 1 else None,
            "score": 0.0,
            "violates_total_entry_limit": False,
            "derived_allowed_duration_sec": _round_or_none(
                sum(_get_allowed_pair_gap_seconds(config, i) for i in range(1, max(1, len(entries)))),
                3,
            ),
        }
    else:
        first_frame = int(entries[0]["start_frame"])
        last_frame = int(entries[-1]["start_frame"])
        delta_frames = int(last_frame - first_frame)
        delta_secs = float(delta_frames / frame_rate)
        total_entry_limit_sec = float(sum(_get_allowed_pair_gap_seconds(config, i) for i in range(1, len(entries))))

        if delta_secs <= total_entry_limit_sec:
            total_entry_score = 1.0
        else:
            overrun = delta_secs - total_entry_limit_sec
            total_entry_score = round(_total_entry_penalty(overrun, total_entry_limit_sec), 2)

        total_entry = {
            "entry_count": int(len(entries)),
            "pair_count": int(len(entries) - 1),
            "start_frame": int(first_frame),
            "end_frame": int(last_frame),
            "start_time_sec": _frame_to_time_sec(first_frame, frame_rate),
            "end_time_sec": _frame_to_time_sec(last_frame, frame_rate),
            "duration_frames": int(delta_frames),
            "duration_sec": _round_or_none(delta_secs, 3),
            "score": float(total_entry_score),
            "violates_total_entry_limit": bool(delta_secs > total_entry_limit_sec),
            "derived_allowed_duration_sec": _round_or_none(total_entry_limit_sec, 3),
        }

        if delta_secs > total_entry_limit_sec:
            flags.append(
                {
                    "flag_id": "entrance_total_entry_time_violation",
                    "phase_id": PHASE_ID,
                    "type": "total_entry_time_violation",
                    "severity": "warning",
                    "frame": int(first_frame),
                    "time_sec": _frame_to_time_sec(first_frame, frame_rate),
                    "start_frame": int(first_frame),
                    "end_frame": int(last_frame),
                    "start_time_sec": _frame_to_time_sec(first_frame, frame_rate),
                    "end_time_sec": _frame_to_time_sec(last_frame, frame_rate),
                    "title": "Total entry time exceeded allowed limit",
                    "message": (
                        f"Total entry span was {delta_secs:.2f}s, exceeding the derived "
                        f"allowed limit of {total_entry_limit_sec:.2f}s."
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Move-along-wall: per-track excursion flags + timeline bars.
    # The metric runs in the engine and pushes its excursion records into
    # ``ctx.metric_flags``; the engine forwards them here. We also emit one
    # ``wall_excursion`` timeline item per excursion so the viewer can draw
    # a colored bar on the move_along_wall row.
    # ------------------------------------------------------------------
    wall_flags_in: List[Dict[str, Any]] = list(metric_flags or [])
    wall_excursion_items: List[Dict[str, Any]] = []
    track_to_entry_number = {
        int(entry["track_id"]): int(entry["entry_number"])
        for entry in entries
        if entry.get("track_id") is not None and entry.get("entry_number") is not None
    }
    for raw_flag in wall_flags_in:
        if str(raw_flag.get("type", "")) not in ("wall_too_close", "wall_too_far"):
            continue
        rec = raw_flag.pop("_wall_excursion_record", None) or {}
        tid = int(raw_flag.get("track_id", -1))
        entry_number = track_to_entry_number.get(tid)
        if entry_number is not None:
            raw_flag["entry_number"] = int(entry_number)
            raw_flag["phase_id"] = PHASE_ID

        flags.append(raw_flag)
        item_id = f"wall_excursion_{tid}_{int(rec.get('n', len(wall_excursion_items) + 1))}"
        # ``raw`` here matches the legacy v1 timeline-item shape; the v2
        # converter drains it via _v2_prune_item_data → "data". Carrying the
        # data this way means we don't have to special-case the converter.
        wall_excursion_items.append(
            {
                "item_id": item_id,
                "phase_id": PHASE_ID,
                "kind": "wall_excursion",
                "label": (
                    f"Track {tid}: {('too close' if raw_flag['type'] == 'wall_too_close' else 'too far')}"
                ),
                "frame": int(raw_flag.get("start_frame", 0) or 0),
                "time_sec": raw_flag.get("start_time_sec"),
                "start_frame": int(raw_flag.get("start_frame", 0) or 0),
                "end_frame": int(raw_flag.get("end_frame", 0) or 0),
                "start_time_sec": raw_flag.get("start_time_sec"),
                "end_time_sec": raw_flag.get("end_time_sec"),
                "flag_ids": [str(raw_flag["flag_id"])],
                "raw": {
                    "track_id": tid,
                    "entry_number": entry_number,
                    "label_kind": (
                        "too_close" if raw_flag["type"] == "wall_too_close" else "too_far"
                    ),
                    "duration_sec": float(raw_flag.get("duration_sec") or 0.0),
                    "L_map": float(rec.get("L_map") or 0.0),
                },
            }
        )

    # Master phase timeline item
    phase_flag_ids = [str(flag["flag_id"]) for flag in flags if flag.get("flag_id")]
    if total_entry.get("start_frame") is not None and total_entry.get("end_frame") is not None:
        timeline_items.append(
            {
                "item_id": f"timeline_{PHASE_ID}",
                "phase_id": PHASE_ID,
                "kind": "phase",
                "label": PHASE_LABEL,
                "start_frame": int(total_entry["start_frame"]),
                "end_frame": int(total_entry["end_frame"]),
                "start_time_sec": total_entry.get("start_time_sec"),
                "end_time_sec": total_entry.get("end_time_sec"),
                "flag_ids": list(phase_flag_ids),
            }
        )

    # Entry point timeline items
    for entry in entries:
        timeline_items.append(
            {
                "item_id": f"{PHASE_ID}_entry_{int(entry['entry_number'])}",
                "phase_id": PHASE_ID,
                "kind": "entry",
                "label": entry.get("entry_name", f"Entry {int(entry['entry_number'])}"),
                "frame": int(entry["start_frame"]),
                "time_sec": entry.get("start_time_sec"),
                "raw": dict(entry),
                "flag_ids": [],
            }
        )

    # Pair-gap timeline items
    for pair in pair_gaps:
        pair_flag_ids = [
            flag["flag_id"]
            for flag in flags
            if flag.get("type") == "pair_time_violation"
            and int(flag.get("pair_number", -1)) == int(pair["pair_number"])
        ]
        timeline_items.append(
            {
                "item_id": f"{PHASE_ID}_pair_{int(pair['pair_number'])}",
                "phase_id": PHASE_ID,
                "kind": "pair_gap",
                "label": pair.get("pair_name", f"Pair {int(pair['pair_number'])}"),
                "start_frame": int(pair["from_start_frame"]),
                "end_frame": int(pair["to_start_frame"]),
                "start_time_sec": pair.get("from_start_time_sec"),
                "end_time_sec": pair.get("to_start_time_sec"),
                "raw": dict(pair),
                "flag_ids": pair_flag_ids,
            }
        )

    # Vector point timeline items
    for vector in vectors:
        entry_number = int(vector["entry_number"])
        direction_label = str(vector.get("direction_label", "UNKNOWN") or "UNKNOWN")
        sign = int(vector.get("sign", 0) or 0)
        sign_label = f"{direction_label} ({sign:+d})" if sign != 0 else direction_label
        item_flag_ids = []
        if not bool(vector.get("valid", False)):
            item_flag_ids.append(f"entrance_vector_unknown_{entry_number}")
        item_flag_ids.extend(
            flag["flag_id"]
            for flag in flags
            if flag.get("type") == "vector_direction_violation"
            and int(flag.get("to_entry_number", -1)) == entry_number
        )

        timeline_items.append(
            {
                "item_id": f"{PHASE_ID}_vector_{entry_number}",
                "phase_id": PHASE_ID,
                "kind": "vector",
                "label": f"Vector {entry_number}: {sign_label}",
                "frame": int(vector.get("start_frame", 0) or 0),
                "time_sec": vector.get("start_time_sec"),
                "raw": dict(vector),
                "flag_ids": item_flag_ids,
            }
        )

    # Append the move_along_wall excursion bars onto the timeline. Each
    # item already carries its own flag_ids; the entrant's track color is
    # resolved on the frontend at render time.
    timeline_items.extend(wall_excursion_items)

    # Evaluation records for GUI metrics section
    evaluations = [
        {
            "evaluation_id": "entrance_vectors",
            "metric_name": "ENTRANCE_VECTORS",
            "score": float(entrance_vectors_score),
            "raw": {
                "valid_vector_count": int(len(valid_vectors)),
                "transition_count": int(vector_transition_count),
                "alternation_count": int(vector_alternation_count),
                "window_sec": float(vector_window_sec),
                "window_frames": max(1, int(round(vector_window_sec * frame_rate))),
                "centroid_xy": centroid_xy,
                "vectors": vectors,
                "vector_transitions": vector_transitions,
            },
        },
        {
            "evaluation_id": "entrance_hesitation",
            "metric_name": "ENTRANCE_HESITATION",
            "score": float(entrance_hesitation_score),
            "raw": {
                "pair_count": int(len(pair_gaps)),
                "base_threshold_sec": _round_or_none(float(config.get("HESITATION_THRESHOLD", 1.0)), 3),
                "second_threshold_sec": _round_or_none(
                    float(config.get("HESITATION_THRESHOLD_SECOND", float(config.get("HESITATION_THRESHOLD", 1.0)) * 2.0)),
                    3,
                ),
                "pairs": pair_gaps,
            },
        },
        {
            "evaluation_id": "total_time_of_entry",
            "metric_name": "TOTAL_TIME_OF_ENTRY",
            "score": float(total_entry_score),
            "raw": {
                **total_entry,
            },
        },
    ]

    if move_along_wall is not None:
        wall_score = float(move_along_wall.get("score") or 0.0)
        wall_per_entrant = [
            dict(e) for e in (move_along_wall.get("per_entrant") or [])
        ]
        # Per-entrant safe-band geometry. Computed once here so the frontend
        # overlay component (MapBandOverlay) doesn't need to do polygon
        # offsetting in JS — it just renders the polygons we ship.
        # ``band_inner_polygon`` is the inward buffer at the entrant's L;
        # the frontend draws "boundary minus band_inner_polygon" with SVG
        # even-odd fill rule to highlight the safe-band annulus.
        _attach_wall_band_polygons(wall_per_entrant, config.get("Boundary"))

        wall_total_close = sum(float(e.get("too_close_time_sec") or 0.0) for e in wall_per_entrant)
        wall_total_far = sum(float(e.get("too_far_time_sec") or 0.0) for e in wall_per_entrant)
        wall_excursion_count = len(wall_excursion_items)
        wall_timeline_ids = [it["item_id"] for it in wall_excursion_items]
        evaluations.append(
            {
                "evaluation_id": "move_along_wall",
                "metric_name": "STAY_ALONG_WALL",
                "score": wall_score,
                "raw": {
                    "per_entrant": wall_per_entrant,
                    "excursion_count": int(wall_excursion_count),
                    "total_too_close_time_sec": round(wall_total_close, 3),
                    "total_too_far_time_sec": round(wall_total_far, 3),
                    "timeline_item_ids": wall_timeline_ids,
                },
            }
        )

    # Phase record
    phase = {
        "phase_id": PHASE_ID,
        "phase_type": "entrance",
        "label": PHASE_LABEL,
        "summary": {
            "entry_count": int(len(entries)),
            "pair_count": int(len(pair_gaps)),
            "start_frame": total_entry.get("start_frame"),
            "end_frame": total_entry.get("end_frame"),
            "start_time_sec": total_entry.get("start_time_sec"),
            "end_time_sec": total_entry.get("end_time_sec"),
            "duration_frames": total_entry.get("duration_frames"),
            "duration_sec": total_entry.get("duration_sec"),
            "vector_score": float(entrance_vectors_score),
            "hesitation_score": float(entrance_hesitation_score),
            "total_entry_score": float(total_entry_score),
            "violates_total_entry_limit": bool(total_entry.get("violates_total_entry_limit", False)),
        },
        "segments": [
            {
                "segment_id": "entrance_total_entry",
                "label": "Total Entry",
                "kind": "aggregate",
                "start_frame": total_entry.get("start_frame"),
                "end_frame": total_entry.get("end_frame"),
                "start_time_sec": total_entry.get("start_time_sec"),
                "end_time_sec": total_entry.get("end_time_sec"),
                "flag_ids": [
                    flag_id
                    for flag_id in phase_flag_ids
                    if flag_id == "entrance_total_entry_time_violation"
                ],
            }
        ]
        + [
            {
                "segment_id": f"entrance_pair_{int(pair['pair_number'])}",
                "label": pair.get("pair_name", f"Pair {int(pair['pair_number'])}"),
                "kind": "pair_gap",
                "start_frame": int(pair["from_start_frame"]),
                "end_frame": int(pair["to_start_frame"]),
                "start_time_sec": pair.get("from_start_time_sec"),
                "end_time_sec": pair.get("to_start_time_sec"),
                "pair_number": int(pair["pair_number"]),
                "score": pair.get("score"),
                "allowed_gap_sec": pair.get("allowed_gap_sec"),
                "flag_ids": [
                    flag["flag_id"]
                    for flag in flags
                    if flag.get("type") == "pair_time_violation"
                    and int(flag.get("pair_number", -1)) == int(pair["pair_number"])
                ],
            }
            for pair in pair_gaps
        ],
        "events": [
            {
                "event_id": f"entrance_entry_{int(entry['entry_number'])}",
                "label": entry.get("entry_name", f"Entry {int(entry['entry_number'])}"),
                "kind": "entry",
                "frame": int(entry["start_frame"]),
                "time_sec": entry.get("start_time_sec"),
                "entry_number": int(entry["entry_number"]),
                "track_id": int(entry["track_id"]),
            }
            for entry in entries
        ]
        + [
            {
                "event_id": f"entrance_vector_{int(vector['entry_number'])}",
                "label": f"Vector {int(vector['entry_number'])}: {str(vector.get('direction_label', 'UNKNOWN') or 'UNKNOWN')}",
                "kind": "vector",
                "frame": int(vector.get("start_frame", 0) or 0),
                "time_sec": vector.get("start_time_sec"),
                "entry_number": int(vector["entry_number"]),
                "track_id": int(vector["track_id"]),
                "sign": int(vector.get("sign", 0) or 0),
                "direction_label": str(vector.get("direction_label", "UNKNOWN") or "UNKNOWN"),
                "valid": bool(vector.get("valid", False)),
            }
            for vector in vectors
        ]
        + [
            {
                "event_id": f"entrance_vector_transition_{int(transition['transition_number'])}",
                "label": f"Vector Transition {int(transition['transition_number'])}",
                "kind": "vector_transition",
                "frame": int(transition.get("frame", 0) or 0),
                "time_sec": transition.get("time_sec"),
                "transition_number": int(transition["transition_number"]),
                "from_entry_number": int(transition["from_entry_number"]),
                "to_entry_number": int(transition["to_entry_number"]),
                "from_sign": int(transition.get("from_sign", 0) or 0),
                "to_sign": int(transition.get("to_sign", 0) or 0),
                "actual_alternation": bool(transition.get("actual_alternation", False)),
                "violates_vector_rule": bool(transition.get("violates_vector_rule", False)),
            }
            for transition in vector_transitions
        ],
        "evaluations": evaluations,
        "flag_ids": list(phase_flag_ids),
        "data": {
            "video_basename": video_basename,
            "frame_rate": float(frame_rate),
            "num_expected_entries": int(num_tracks),
            "selected_track_ids_in_order": [int(track_id) for track_id, _traj in selected_tracks],
            "centroid_xy": centroid_xy,
            "entries": entries,
            "vectors": vectors,
            "vector_transitions": vector_transitions,
            "pair_gaps": pair_gaps,
            "total_entry": total_entry,
        },
    }

    artifacts = _build_artifacts_section(
        output_directory=output_directory or "",
        video_basename=video_basename,
        original_video_path=video_path,
        tracker_output_json_path=tracker_output_json_path,
        position_cache_path=position_cache_path,
        gaze_cache_path=gaze_cache_path,
        empty_map_path=empty_map_path,
    )

    result = {
        "schema_version": "1.0",
        "session": {
            "video_basename": video_basename,
            "analysis_mode": "entry",
            "start_time": start_time,
            "role": config.get("role"),
            "title": config.get("title"),
            "run_id": config.get("run_id"),
        },
        "video": {
            "video_path": video_path,
            "frame_rate": float(frame_rate),
            "total_frames": int(total_frames or 0),
            "duration_sec": _safe_float(duration_sec),
        },
        "timeline": {
            "items": timeline_items,
        },
        "phases": [phase],
        "flags": sorted(
            flags,
            key=lambda x: (
                _safe_int(x.get("frame"), 10**9),
                str(x.get("type", "")),
                str(x.get("flag_id", "")),
            ),
        ),
        "entities": {
            "friends": [
                {
                    "track_id": int(entry["track_id"]),
                    "entry_number": int(entry["entry_number"]),
                    "label": entry.get("entry_name", f"Entry {int(entry['entry_number'])}"),
                }
                for entry in entries
            ],
            "enemies": [
                {
                    "track_id": int(x),
                    "label": "In-room",
                }
                for x in (inroom_ids or [])
            ],
        },
        "artifacts": artifacts,
    }

    # Room boundary in map-image pixel coordinates. Frontend overlays
    # (e.g. the wall-band annulus rendered on the aux video when a wall
    # flag is selected) need this; it isn't surfaced anywhere else in
    # the session payload. Optional — if Boundary isn't a recognisable
    # polygon we just omit the field.
    boundary_coords = _serialize_boundary(config.get("Boundary"))
    if boundary_coords is not None:
        result["boundary"] = boundary_coords

    if drill_window is not None:
        if hasattr(drill_window, "to_dict"):
            result["drill_window"] = drill_window.to_dict()
        elif isinstance(drill_window, dict):
            result["drill_window"] = drill_window

    v2 = _convert_to_v2_schema(result)
    if pod_orientation is not None:
        attach_pod_block(v2, pod_orientation)
        images = v2.setdefault("artifacts", {}).setdefault("images", {})
        if pod_sectors_path and os.path.exists(pod_sectors_path):
            images["pod_sectors"] = pod_sectors_path
        if pod_camera_path and os.path.exists(pod_camera_path):
            images["pod_camera"] = pod_camera_path
    return v2


def build_pod_block(pod_data: Dict[str, Any]) -> Dict[str, Any]:
    """Schema-v2 POD slices (metric records / timeline item / flags) from a
    ``compute_pod_data`` / ``recompute_pod_for_run`` result (all times come
    precomputed as ``pod_sec`` in the pod_data itself).

    Shared by the engine's session build and the analysis-viewer backend's
    instructor-adjustment recompute so both always emit the same shape.
    """
    status_ok = pod_data.get("status") == "ok"
    pod_frame = pod_data.get("pod_frame")
    pod_sec = pod_data.get("pod_sec")
    members = pod_data.get("members") or []
    pairs = pod_data.get("flagged_pairs") or []
    violators = pod_data.get("violators") or []
    source = pod_data.get("source", "auto")

    flags: List[Dict[str, Any]] = []
    item = None
    if status_ok and pod_frame is not None:
        pair_flag_ids = []
        for p in pairs:
            # Frame-scoped id: a deletion recorded at one POD frame must never
            # silently suppress the (semantically different) flag recomputed
            # for the same member pair at another instructor-set frame.
            fid = f"pod_flagging_{p['from']}_{p['to']}_f{int(pod_frame)}"
            pair_flag_ids.append(fid)
            flags.append({
                "flag_id": fid,
                "metric_id": "pod_mutual_facing",
                "linked_item_id": "pod_establishment",
                "type": "pod_flagging",
                "severity": "warning",
                "frame": int(pod_frame),
                "time_sec": pod_sec,
                "track_id": int(p["from"]),
                "target_id": int(p["to"]),
                "angle_off_deg": p.get("angle_off_deg"),
                "title": f"Teammate in sector of fire: member {p['to']} (member {p['from']}'s sector)",
                "message": (
                    f"At the POD frame, member {p['to']} is inside member "
                    f"{p['from']}'s sector of fire — member {p['from']}'s muzzle "
                    f"bearing passes within {p.get('angle_off_deg', '?')}° of "
                    f"teammate {p['to']}."
                ),
            })
        establish_flag = {
            # Distinct from the timeline item's id ("pod_establishment") —
            # the viewer keeps items and flags in one selection registry.
            # Session-level (like the drill marks): the POD establishment is
            # an event of the run, not a property of one metric.
            "flag_id": "pod_establishment_flag",
            "metric_id": None,
            "linked_item_id": "pod_establishment",
            "type": "pod_establishment",
            "severity": "info",
            "frame": int(pod_frame),
            "time_sec": pod_sec,
            "title": ("POD established (instructor-set)" if source == "instructor"
                      else "POD established"),
            "message": (
                f"Team point-of-dominance mark at frame {pod_frame} "
                f"({'auto-detected first collective hold' if source == 'auto' else 'instructor-set'})."
            ),
            "source": source,
        }
        flags.insert(0, establish_flag)
        coverage_score = (pod_data.get("metrics") or {}).get("POD_SECTOR_COVERAGE")
        try:
            pct = f"{float(coverage_score):.0%}"
        except (TypeError, ValueError):
            pct = "?"
        flags.insert(1, {
            # Mandatory metric flag: sector coverage has no violation notion,
            # so its one flag simply carries the result (area ratio).
            "flag_id": "pod_coverage_flag",
            "metric_id": "pod_sector_coverage",
            "linked_item_id": "pod_establishment",
            "type": "pod_coverage",
            "severity": "info",
            "frame": int(pod_frame),
            "time_sec": pod_sec,
            "title": f"Sector coverage: {pct} of the room",
            "message": (
                f"At the POD frame the union of the team's sectors of fire "
                f"covers {pct} of the room area (sector angle "
                f"{pod_data.get('sector_angle_degrees')}°)."
            ),
        })
    else:
        # Uncertain runs still get an info flag so the viewer's flag list
        # carries the POD status and its Set/Adjust affordance. The wording
        # distinguishes an instructor-set frame that could not be scored from
        # a failed automatic detection (with or without a candidate pause).
        if source == "instructor":
            title = "POD set, but not scoreable"
            message = (
                f"The instructor-set POD frame {pod_frame} has fewer than two "
                "members with a reliable orientation there, so sector coverage "
                "and mutual facing are unavailable. Adjust the frame to a "
                "moment where more members are tracked."
            )
            flag_frame, flag_sec = pod_frame, pod_sec
        elif pod_frame is not None:
            title = "POD not established"
            message = (
                f"A collective pause was detected at frame {pod_frame}, but "
                "fewer than two members had a reliable orientation there, so "
                "the POD was not established. Use Adjust to set the frame "
                "manually — orientation data for the whole video is available."
            )
            flag_frame, flag_sec = None, None
        else:
            title = "POD not identified"
            message = (
                f"No POD establishment was detected automatically "
                f"({pod_data.get('reason') or 'no collective pause found'}). "
                "Use Set to mark the frame manually — orientation data for the "
                "whole video is available."
            )
            flag_frame, flag_sec = None, None
        flags.append({
            "flag_id": "pod_establishment_flag",
            "metric_id": None,
            "linked_item_id": None,
            "type": "pod_establishment",
            "severity": "info",
            "frame": int(flag_frame) if flag_frame is not None else None,
            "time_sec": flag_sec,
            "title": title,
            "message": message,
            "source": source,
        })
    if status_ok and pod_frame is not None:
        item = {
            "item_id": "pod_establishment",
            "metric_id": "pod_sector_coverage",
            "kind": "pod_establishment",
            "label": "POD",
            "frame": int(pod_frame),
            "time_sec": pod_sec,
            "flag_ids": ["pod_establishment_flag", "pod_coverage_flag"] + pair_flag_ids,
            "data": {
                "source": source,
                "member_count": len(members),
                "sector_angle_degrees": pod_data.get("sector_angle_degrees"),
            },
        }

    coverage_summary = {
        "pod_frame": int(pod_frame) if pod_frame is not None else None,
        "pod_time_sec": pod_sec,
        "source": source,
        "sector_angle_degrees": pod_data.get("sector_angle_degrees"),
        "members": members,
        "sectors": pod_data.get("sectors") or {},
        "excluded_members": pod_data.get("excluded_members") or [],
    }
    if not status_ok:
        coverage_summary["reason"] = pod_data.get("reason")

    scores = pod_data.get("metrics") or {}
    metrics = [
        {
            "metric_id": "pod_sector_coverage",
            "label": "POD Sector Coverage",
            "score": scores.get("POD_SECTOR_COVERAGE", -1),
            "uncertain": not status_ok,
            "summary": coverage_summary,
            "timeline_item_ids": ["pod_establishment"] if item else [],
            "flag_ids": ["pod_coverage_flag"] if item else [],
        },
        {
            "metric_id": "pod_mutual_facing",
            "label": "POD Mutual Facing",
            "score": scores.get("POD_MUTUAL_FACING", -1),
            "uncertain": not status_ok,
            "summary": {
                "violators": violators,
                "pairs": pairs,
                "member_count": len(members),
                "sector_angle_degrees": pod_data.get("sector_angle_degrees"),
            },
            "timeline_item_ids": ["pod_establishment"] if item else [],
            "flag_ids": [f["flag_id"] for f in flags if f["type"] == "pod_flagging"],
        },
    ]
    return {"metrics": metrics, "timeline_item": item, "flags": flags}


POD_METRIC_IDS = ("pod_sector_coverage", "pod_mutual_facing")
POD_FLAG_TYPES = ("pod_establishment", "pod_coverage", "pod_flagging")


def attach_pod_block(v2: Dict[str, Any], pod_data: Dict[str, Any]) -> None:
    """Splice (or replace) the POD slices inside a schema-v2 payload."""
    block = build_pod_block(pod_data)
    v2["metrics"] = [
        m for m in (v2.get("metrics") or []) if m.get("metric_id") not in POD_METRIC_IDS
    ] + block["metrics"]
    v2["flags"] = [
        f for f in (v2.get("flags") or []) if f.get("type") not in POD_FLAG_TYPES
    ] + block["flags"]
    items = [
        it for it in (v2.get("timeline") or {}).get("items", [])
        if it.get("item_id") != "pod_establishment"
    ]
    if block["timeline_item"] is not None:
        items.append(block["timeline_item"])
    v2.setdefault("timeline", {})["items"] = items


def apply_flag_deletion_recalcs(v2: Dict[str, Any],
                                binned: List[Dict[str, Any]]) -> None:
    """Recalculate every affected metric after instructor flag deletions.

    An instructor deletes a flag to say "this detected violation is wrong —
    ignore it when scoring". Each metric has its own ignore semantics:

    - pod_flagging            -> POD_MUTUAL_FACING from remaining pairs.
    - wall_too_close/too_far  -> credit the excursion's frames back to the
                                 entrant (per-frame safe/observed score) and
                                 drop the linked wall_excursion timeline item.
    - pair_time_violation     -> that entry pair scores 1.0 in the
                                 ENTRANCE_HESITATION mean.
    - vector_direction_violation -> that transition counts as an alternation
                                 in ENTRANCE_VECTORS (alternations/transitions).
    - total_entry_time_violation -> TOTAL_TIME_OF_ENTRY becomes 1.0.
    - vector_unknown          -> display-only (no direction can be invented),
                                 no score change.

    Deletions are non-destructive: they are applied to a freshly-loaded
    payload on every load, so restoring a flag simply stops applying them.
    """
    if not binned:
        return
    metrics = {m.get("metric_id"): m for m in v2.get("metrics") or []}
    items = (v2.get("timeline") or {}).get("items", [])
    item_by_id = {it.get("item_id"): it for it in items}
    fps = float((v2.get("video") or {}).get("frame_rate") or 30.0)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for f in binned:
        by_type.setdefault(str(f.get("type")), []).append(f)

    if "pod_flagging" in by_type:
        recalc_mutual_facing_from_flags(v2)

    wall_flags = by_type.get("wall_too_close", []) + by_type.get("wall_too_far", [])
    metric = metrics.get("move_along_wall")
    if wall_flags and metric is not None:
        summary = metric.setdefault("summary", {})
        per_entrant = summary.get("per_entrant") or []
        by_track = {int(e["track_id"]): e for e in per_entrant if e.get("track_id") is not None}
        removed_item_ids = set()
        for f in wall_flags:
            tid = f.get("track_id")
            e = by_track.get(int(tid)) if tid is not None else None
            if e is None or f.get("start_frame") is None or f.get("end_frame") is None:
                continue
            frames = max(0, int(f["end_frame"]) - int(f["start_frame"]) + 1)
            key = "too_close_frames" if f["type"] == "wall_too_close" else "too_far_frames"
            tkey = "too_close_time_sec" if f["type"] == "wall_too_close" else "too_far_time_sec"
            e[key] = max(0, int(e.get(key) or 0) - frames)
            e[tkey] = round(e[key] / fps, 3)
            observed = int(e.get("observed_frames") or 0)
            if observed > 0:
                e["score"] = (observed - int(e.get("too_close_frames") or 0)
                              - int(e.get("too_far_frames") or 0)) / observed
            if f.get("linked_item_id"):
                removed_item_ids.add(f["linked_item_id"])
        scores = [float(e["score"]) for e in per_entrant
                  if int(e.get("observed_frames") or 0) > 0]
        if scores:
            metric["score"] = round(sum(scores) / len(scores), 2)
        summary["excursion_count"] = max(
            0, int(summary.get("excursion_count") or 0) - len(wall_flags))
        summary["total_too_close_time_sec"] = round(
            sum(float(e.get("too_close_time_sec") or 0.0) for e in per_entrant), 3)
        summary["total_too_far_time_sec"] = round(
            sum(float(e.get("too_far_time_sec") or 0.0) for e in per_entrant), 3)
        if removed_item_ids:
            v2.setdefault("timeline", {})["items"] = [
                it for it in items if it.get("item_id") not in removed_item_ids
            ]
            items = v2["timeline"]["items"]
            item_by_id = {it.get("item_id"): it for it in items}
            # Keep the metric's cross-refs consistent with the pruned items.
            metric["timeline_item_ids"] = [
                iid for iid in (metric.get("timeline_item_ids") or [])
                if iid not in removed_item_ids
            ]

    pair_flags = by_type.get("pair_time_violation", [])
    metric = metrics.get("entrance_hesitation")
    if pair_flags and metric is not None:
        ignored_items = {f.get("linked_item_id") for f in pair_flags}
        pair_scores: List[float] = []
        for it in items:
            if it.get("kind") != "pair_gap":
                continue
            data = it.get("data") or {}
            if it.get("item_id") in ignored_items:
                pair_scores.append(1.0)
                data["violates_time_limit"] = False
                continue
            gap = float(data.get("gap_sec") or 0.0)
            allowed = float(data.get("allowed_gap_sec") or 0.0)
            over = max(0.0, gap - allowed)
            pair_scores.append(math.exp(-(over ** 2) / 0.5))
        if pair_scores:
            metric["score"] = round(sum(pair_scores) / len(pair_scores), 2)
        summary = metric.setdefault("summary", {})
        summary["violation_count"] = max(
            0, int(summary.get("violation_count") or 0) - len(pair_flags))

    dir_flags = by_type.get("vector_direction_violation", [])
    metric = metrics.get("entrance_vectors")
    if dir_flags and metric is not None:
        summary = metric.setdefault("summary", {})
        transitions = int(summary.get("transition_count") or 0)
        if transitions > 0:
            alt = min(transitions,
                      int(summary.get("alternation_count") or 0) + len(dir_flags))
            summary["alternation_count"] = alt
            metric["score"] = round(alt / transitions, 2)

    total_flags = by_type.get("total_entry_time_violation", [])
    metric = metrics.get("total_time_of_entry")
    if total_flags and metric is not None:
        metric["score"] = 1.0
        summary = metric.setdefault("summary", {})
        summary["violates_total_entry_limit"] = False
        duration_item = item_by_id.get("total_entry_duration")
        if duration_item is not None:
            (duration_item.get("data") or {})["violates_total_entry_limit"] = False


def recalc_mutual_facing_from_flags(v2: Dict[str, Any]) -> None:
    """Re-derive the POD_MUTUAL_FACING score from the pod_flagging flags
    still present in the payload (used after an instructor deletes or
    restores a flag): score = 1 − violators / member count."""
    metric = next(
        (m for m in v2.get("metrics", []) if m.get("metric_id") == "pod_mutual_facing"),
        None,
    )
    if metric is None or metric.get("uncertain"):
        return
    coverage = next(
        (m for m in v2.get("metrics", []) if m.get("metric_id") == "pod_sector_coverage"),
        None,
    )
    member_count = len((coverage or {}).get("summary", {}).get("members") or []) or int(
        metric.get("summary", {}).get("member_count") or 0
    )
    remaining = [
        f for f in v2.get("flags", []) if f.get("type") == "pod_flagging"
    ]
    violators = sorted({int(f["track_id"]) for f in remaining if f.get("track_id") is not None})
    pairs = [
        {
            "from": int(f["track_id"]),
            "to": int(f.get("target_id", -1)),
            "angle_off_deg": f.get("angle_off_deg"),
        }
        for f in remaining
        if f.get("track_id") is not None
    ]
    metric["score"] = round(1.0 - (len(violators) / member_count), 2) if member_count else 1.0
    metric.setdefault("summary", {})["violators"] = violators
    metric["summary"]["pairs"] = pairs
    metric["flag_ids"] = [f["flag_id"] for f in remaining]


def _attach_wall_band_polygons(
    per_entrant: List[Dict[str, Any]],
    boundary: Any,
) -> None:
    """Compute and attach per-entrant ``band_inner_polygon`` in-place.

    The inner polygon is the inward buffer of the room boundary at the
    entrant's resolved shoulder-elbow length ``L_map``. The frontend
    overlay renders the safe band by combining ``boundary`` (top-level)
    and this inner polygon with SVG even-odd fill rule.

    No-op when the boundary is missing, when shapely operations fail, or
    when ``L_map`` is non-positive. Old sessions without these fields
    just skip the overlay client-side.
    """
    if not per_entrant or boundary is None:
        return

    # ``config["Boundary"]`` is a shapely ``Polygon`` after ``load_config``;
    # the list-of-points fallback exists only for unit tests / synthetic
    # callers that bypass ``load_config``.
    if hasattr(boundary, "buffer"):
        poly = boundary
    else:
        try:
            poly = _ShPolygon(boundary)
        except Exception:
            return
    if poly.is_empty:
        return

    for entry in per_entrant:
        try:
            L = float(entry.get("L_map") or 0.0)
        except (TypeError, ValueError):
            continue
        if L <= 0:
            continue
        try:
            inner = poly.buffer(-L)
        except Exception:
            continue
        if inner is None or inner.is_empty:
            continue
        # ``buffer(-L)`` on a thin / concave room can collapse to a
        # MultiPolygon; the frontend only needs the largest piece.
        if getattr(inner, "geom_type", "") == "MultiPolygon":
            try:
                inner = max(list(inner.geoms), key=lambda g: g.area)
            except Exception:
                continue
        if not hasattr(inner, "exterior") or inner.exterior is None:
            continue
        try:
            coords: List[List[float]] = [
                [float(pt[0]), float(pt[1])] for pt in inner.exterior.coords
            ]
        except Exception:
            continue
        if len(coords) >= 3:
            entry["band_inner_polygon"] = coords


def _serialize_boundary(boundary: Any) -> Optional[List[List[float]]]:
    """Coerce a boundary polygon into a JSON-friendly list of [x, y] pairs.

    Accepts a shapely ``Polygon`` (the engine's resolved type), a list of
    points, or anything with an ``exterior.coords`` attribute. Returns
    ``None`` when the input isn't usable.
    """
    if boundary is None:
        return None
    coords_iter = None
    try:
        if hasattr(boundary, "exterior") and boundary.exterior is not None:
            coords_iter = list(boundary.exterior.coords)
        elif isinstance(boundary, (list, tuple)):
            coords_iter = list(boundary)
        else:
            arr = np.asarray(boundary, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                coords_iter = arr.tolist()
    except Exception:
        return None
    if not coords_iter:
        return None
    out: List[List[float]] = []
    for pt in coords_iter:
        try:
            out.append([float(pt[0]), float(pt[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return out if len(out) >= 3 else None


# ---------------------------------------------------------------------------
# v2 schema conversion: phase-centric -> metric-centric, with redundant fields
# pruned. The viewer (src/utils/analysis_viewer/) consumes this shape and the
# TypeScript types mirror it 1:1. Future metrics that gain timeline artifacts
# plug in by adding entries to the small dictionaries below.
#
# Cleanup vs the legacy v1 shape:
#  * "raw" subfield renamed to "data" (it's structured, not a raw dump)
#  * Inside each item's data: redundant duplicates of item-level fields
#    (frame/time_sec/start_frame/etc.) are dropped, plus *_name == label dupes
#  * Metrics: drop "metric_name" (engine identifier; metric_id + label suffice),
#    drop "summary.score" (parent already has score), and drop the redundant
#    "raw.vectors[]" list (every vector is already in timeline.items)
#  * Flags: replace pair_number/from_entry_number/to_entry_number with a single
#    linked_item_id pointing at the timeline item the flag was raised against
#  * Artifacts: collapse {path,label,type,exists} wrappers to plain path
#    strings, grouped by purpose (videos/images/data); UI labels are a frontend
#    concern, "exists" is recomputed at load time
#  * drill_window stays at top level; transcription joins it at top level too
#    (both are analysis metadata, not file artifacts)
# ---------------------------------------------------------------------------

_V2_METRIC_LABELS: Dict[str, str] = {
    "entrance_vectors": "Entrance Vectors",
    "entrance_hesitation": "Entrance Hesitation",
    "total_time_of_entry": "Total Entry Time",
    "move_along_wall": "Move Along Wall",
}

_V2_KIND_TO_METRIC: Dict[str, Optional[str]] = {
    "entry": None,  # baseline context — always shown
    "vector": "entrance_vectors",
    "pair_gap": "entrance_hesitation",
    "duration": "total_time_of_entry",
    "wall_excursion": "move_along_wall",
}

_V2_FLAG_TYPE_TO_METRIC: Dict[str, str] = {
    "pair_time_violation": "entrance_hesitation",
    "vector_unknown": "entrance_vectors",
    "vector_direction_violation": "entrance_vectors",
    "total_entry_time_violation": "total_time_of_entry",
    "wall_too_close": "move_along_wall",
    "wall_too_far": "move_along_wall",
}

# Per-kind: keys to strip from "data" because they duplicate item-level fields.
_V2_DATA_DROP: Dict[str, set] = {
    "entry": {"entry_name", "start_frame", "start_time_sec"},
    "vector": {"entry_name", "start_frame", "start_time_sec", "window_frames"},
    "pair_gap": {
        "pair_name",
        "from_start_frame",
        "to_start_frame",
        "from_start_time_sec",
        "to_start_time_sec",
        "score",  # only the aggregate metric.score is meaningful for the GUI
    },
    "duration": {"start_frame", "end_frame", "start_time_sec", "end_time_sec", "score"},
}

# Map flag.type -> the timeline item ID pattern it points at. Used to compute
# linked_item_id from flag fields without needing the upstream engine to add
# a new field.
_V2_FLAG_LINKED_ITEM: Dict[str, callable] = {  # type: ignore[type-arg]
    "pair_time_violation": lambda f: (
        f"entrance_pair_{f.get('pair_number')}" if f.get("pair_number") else None
    ),
    "vector_unknown": lambda f: (
        f"entrance_vector_{f.get('entry_number')}" if f.get("entry_number") else None
    ),
    "vector_direction_violation": lambda f: (
        f"entrance_vector_{f.get('entry_number')}" if f.get("entry_number") else None
    ),
    "total_entry_time_violation": lambda f: "total_entry_duration",
    "wall_too_close": lambda f: (
        f["flag_id"].replace("wall_too_close_", "wall_excursion_")
        if isinstance(f.get("flag_id"), str) else None
    ),
    "wall_too_far": lambda f: (
        f["flag_id"].replace("wall_too_far_", "wall_excursion_")
        if isinstance(f.get("flag_id"), str) else None
    ),
}


def _convert_to_v2_schema(v1: Dict[str, Any]) -> Dict[str, Any]:
    """Transform the legacy phase-centric shape into the cleaned metric-centric v2 shape."""
    base: Dict[str, Any] = {
        "schema_version": "2.0",
        "session": v1.get("session", {}),
        "video": v1.get("video", {}),
        "timeline": {"items": []},
        "metrics": [],
        "flags": [],
        "entities": v1.get("entities", {"friends": [], "enemies": []}),
        "artifacts": _v2_restructure_artifacts(v1.get("artifacts", {})),
    }
    if "drill_window" in v1:
        base["drill_window"] = v1["drill_window"]
    if "boundary" in v1:
        base["boundary"] = v1["boundary"]

    phases = v1.get("phases") or []
    if not phases:
        return base
    phase = phases[0]

    new_flags = _v2_build_flags(v1.get("flags") or [])
    new_items = _v2_build_timeline_items(
        (v1.get("timeline") or {}).get("items", []),
        phase,
        new_flags,
    )

    base["timeline"]["items"] = new_items
    base["flags"] = new_flags
    base["metrics"] = _v2_build_metrics(phase, new_items, new_flags)
    return base


def _v2_build_flags(v1_flags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for flag in v1_flags:
        ftype = flag.get("type")
        linked = _V2_FLAG_LINKED_ITEM.get(ftype, lambda _f: None)(flag)
        new_flag = {
            "flag_id": flag.get("flag_id"),
            "metric_id": _V2_FLAG_TYPE_TO_METRIC.get(ftype),
            "linked_item_id": linked,
            "type": ftype,
            "severity": flag.get("severity"),
            "frame": flag.get("frame"),
            "time_sec": flag.get("time_sec"),
            "title": flag.get("title"),
            "message": flag.get("message"),
        }
        # Per-track flags (wall_too_close, wall_too_far) carry track_id from
        # the metric. Preserve it so consumers like the move-along-wall
        # compare visualization can group flag windows by entrant.
        if "track_id" in flag and flag["track_id"] is not None:
            new_flag["track_id"] = flag["track_id"]
        for opt in ("start_frame", "end_frame", "start_time_sec", "end_time_sec"):
            if opt in flag and flag[opt] is not None:
                new_flag[opt] = flag[opt]
        out.append(new_flag)
    return out


def _v2_build_timeline_items(
    v1_items: List[Dict[str, Any]],
    phase: Dict[str, Any],
    new_flags: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in v1_items:
        kind = item.get("kind")
        if kind == "phase":
            continue
        new_item = {
            "item_id": item.get("item_id"),
            "metric_id": _V2_KIND_TO_METRIC.get(kind),
            "kind": kind,
            "label": item.get("label"),
            "flag_ids": list(item.get("flag_ids") or []),
        }
        for time_key in ("frame", "time_sec", "start_frame", "end_frame", "start_time_sec", "end_time_sec"):
            if time_key in item:
                new_item[time_key] = item[time_key]
        new_item["data"] = _v2_prune_item_data(item.get("raw") or {}, kind)
        out.append(new_item)

    total_entry = (phase.get("data") or {}).get("total_entry") or {}
    if (
        total_entry.get("start_frame") is not None
        and total_entry.get("end_frame") is not None
    ):
        # Keep the label terse: numeric details (duration / allowed) belong in
        # the tooltip + detail panel, not embedded in the display label
        # (otherwise the tooltip ends up showing the duration twice).
        duration_label = "Total Entry"
        flag_ids = [
            f["flag_id"]
            for f in new_flags
            if f.get("metric_id") == "total_time_of_entry"
        ]
        out.append(
            {
                "item_id": "total_entry_duration",
                "metric_id": "total_time_of_entry",
                "kind": "duration",
                "label": duration_label,
                "start_frame": total_entry.get("start_frame"),
                "end_frame": total_entry.get("end_frame"),
                "start_time_sec": total_entry.get("start_time_sec"),
                "end_time_sec": total_entry.get("end_time_sec"),
                "data": _v2_prune_item_data(total_entry, "duration"),
                "flag_ids": flag_ids,
            }
        )
    return out


def _v2_prune_item_data(data: Dict[str, Any], kind: Optional[str]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    drop = _V2_DATA_DROP.get(kind or "", set())
    return {k: v for k, v in data.items() if k not in drop}


def _v2_build_metrics(
    phase: Dict[str, Any],
    new_items: List[Dict[str, Any]],
    new_flags: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summary_block = phase.get("summary") or {}
    data_block = phase.get("data") or {}
    out: List[Dict[str, Any]] = []
    for ev in phase.get("evaluations") or []:
        metric_id = ev.get("evaluation_id")
        if not metric_id:
            continue
        ev_raw = ev.get("raw") or {}
        out.append(
            {
                "metric_id": metric_id,
                "label": _V2_METRIC_LABELS.get(metric_id, metric_id),
                "score": ev.get("score"),
                "summary": _v2_metric_summary(metric_id, summary_block, data_block, ev_raw),
                "timeline_item_ids": [
                    item["item_id"]
                    for item in new_items
                    if item.get("metric_id") == metric_id and "item_id" in item
                ],
                "flag_ids": [
                    flag["flag_id"]
                    for flag in new_flags
                    if flag.get("metric_id") == metric_id
                ],
            }
        )
    return out


def _v2_metric_summary(
    metric_id: str,
    phase_summary: Dict[str, Any],
    phase_data: Dict[str, Any],
    ev_raw: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-metric overview shown in the detail panel.

    Pulls a small set of useful aggregate numbers — no large lists, no
    duplication of timeline-item content.
    """
    if metric_id == "entrance_vectors":
        vectors = phase_data.get("vectors") or []
        return {
            "vector_count": len(vectors),
            "valid_vector_count": sum(1 for v in vectors if v.get("valid")),
            "alternation_count": ev_raw.get("alternation_count"),
            "transition_count": ev_raw.get("transition_count"),
            "window_sec": ev_raw.get("window_sec"),
            "centroid_xy": ev_raw.get("centroid_xy"),
        }
    if metric_id == "entrance_hesitation":
        pairs = phase_data.get("pair_gaps") or []
        return {
            "pair_count": len(pairs),
            "violation_count": sum(1 for p in pairs if p.get("violates_time_limit")),
        }
    if metric_id == "total_time_of_entry":
        total = phase_data.get("total_entry") or {}
        return {
            "duration_sec": total.get("duration_sec"),
            "limit_sec": total.get("derived_allowed_duration_sec")
                         or total.get("allowed_total_entry_sec")
                         or total.get("limit_sec"),
            "violates_total_entry_limit": total.get("violates_total_entry_limit"),
        }
    if metric_id == "move_along_wall":
        # Keep the per-entrant block (track_id, L_map, score breakdown,
        # band_inner_polygon) in the summary so the frontend's map-band
        # overlay can look up an entrant's band by track_id without
        # cross-referencing other parts of the payload.
        per_entrant_pruned = []
        for entry in ev_raw.get("per_entrant") or []:
            if not isinstance(entry, dict):
                continue
            pe = {k: v for k, v in entry.items() if not str(k).startswith("_")}
            per_entrant_pruned.append(pe)
        return {
            "excursion_count": int(ev_raw.get("excursion_count") or 0),
            "total_too_close_time_sec": float(
                ev_raw.get("total_too_close_time_sec") or 0.0
            ),
            "total_too_far_time_sec": float(
                ev_raw.get("total_too_far_time_sec") or 0.0
            ),
            "per_entrant": per_entrant_pruned,
        }
    return {}


def _v2_restructure_artifacts(v1_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse {path,label,type,exists} wrappers and group by purpose.

    Output:
      {
        "videos": { "original": "...", "tracking": "...", ... },
        "images": { "empty_map": "..." },
        "data":   { "tracker_output": "...", "position_cache": "...", ... }
      }
    """
    def path(node: Any) -> Optional[str]:
        # Honor the entry's ``exists`` flag so disabled / never-produced
        # artifacts don't leak into the v2 videos/images/data maps. That
        # keeps VideoStage's ``availableModes`` accurate even when an
        # annotator was skipped this run.
        if isinstance(node, dict):
            if node.get("exists") is False:
                return None
            value = node.get("path")
            return value if isinstance(value, str) and value else None
        if isinstance(node, str) and node:
            return node
        return None

    videos: Dict[str, str] = {}
    images: Dict[str, str] = {}
    data: Dict[str, str] = {}

    if not isinstance(v1_artifacts, dict):
        return {"videos": {}, "images": {}, "data": {}}

    if (p := path(v1_artifacts.get("original_video"))):
        videos["original"] = p

    motion = v1_artifacts.get("motion") or {}
    if isinstance(motion, dict):
        if (p := path(motion.get("camera_video"))):
            videos["motion_camera"] = p
        if (p := path(motion.get("map_video"))):
            videos["motion_map"] = p

    gaze = v1_artifacts.get("gaze") or {}
    if isinstance(gaze, dict):
        if (p := path(gaze.get("camera_video"))):
            videos["gaze_camera"] = p
        if (p := path(gaze.get("map_video"))):
            videos["gaze_map"] = p

    if (p := path(v1_artifacts.get("empty_map_image"))):
        images["empty_map"] = p

    if (p := path(v1_artifacts.get("tracker_output_json"))):
        data["tracker_output"] = p
    if (p := path(v1_artifacts.get("position_cache"))):
        data["position_cache"] = p
    if (p := path(v1_artifacts.get("gaze_cache"))):
        data["gaze_cache"] = p

    return {"videos": videos, "images": images, "data": data}
