from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MetricContext:
    tracker_output: List[Dict[str, Any]] = field(default_factory=list)
    all_frames: List[List[Tuple[int, float, float]]] = field(default_factory=list)
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]] = field(default_factory=dict)
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]] = field(default_factory=dict)
    bbox_details: Dict[Tuple[int, int], Tuple[float, float, float, float]] = field(default_factory=dict)
    keypoint_details: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    fall_frames: Dict[int, Optional[int]] = field(default_factory=dict)
    map_points: List[Tuple[int, int, float, float]] = field(default_factory=list)

    # Semantic track groups resolved by the tracker / processing engine.
    entry_ids: List[int] = field(default_factory=list)
    inroom_ids: List[int] = field(default_factory=list)

    # room_coverage: a dict containing coverage info:
    #   "coverage_per_frame"    -> List[Tuple[int, float]] (frame index, fraction covered)
    #   "time_to_full"          -> float (seconds from first non-inroom frame to full coverage) or None
    #   "final_fraction"        -> float (coverage fraction at last frame)
    #   "first_non_enemy_frame" -> int (legacy cache key; first frame index with non-inroom gaze) or None
    room_coverage: Optional[Dict[str, Any]] = None

    # Mapping: inroom_id -> (first_clear_frame, last_clear_frame, clearing_friend_id)
    threat_clearance: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]] = field(default_factory=dict)

    # Mapping of POD index to capture info:
    #   "assigned_id"      -> int or None
    #   "capture_time_sec" -> float or None
    #   "capture_frame"    -> int or None
    pod_capture: Dict[int, Dict[str, Optional[float]]] = field(default_factory=dict)

    # Drill window. Both ``drill_start_frame`` and ``drill_end_frame`` are
    # absolute, 1-indexed against the source video. ``drill_end_frame`` is
    # inclusive. Defaults keep behavior backward-compatible (no trim).
    drill_start_frame: int = 1
    drill_end_frame: Optional[int] = None
    drill_window_meta: Optional[Dict[str, Any]] = None

    # Pixel→map transform constructed by the processing engine from
    # ``config["point_mapping_path"]``. Metrics that need to convert
    # per-frame keypoint pixel coordinates into map units (e.g. the
    # shoulder-derived band in ``MoveAlongWall_Metric``) read it here.
    # ``Any`` typing avoids a hard dep on ``libs.Track`` from the metrics
    # package import graph. ``None`` when unavailable (e.g. unit tests
    # that don't construct a real engine).
    pixel_mapper: Optional[Any] = None

    # POD-orientation compute-family result (``src.orientation
    # .compute_pod_data``): POD-establishment frame, per-member muzzle
    # bearings/positions/sectors, and the two POD metric scores. ``None``
    # when the family did not run. Consumed by the pod_orientation metrics.
    pod_orientation: Optional[Dict[str, Any]] = None

    # Side-channel for metrics that produce flag records during
    # ``process(ctx)`` rather than only a final score. ``analysis.py``
    # drains this list when building the session payload. Each entry
    # follows the standard flag schema (flag_id / type / start_frame /
    # end_frame / track_id / message / ...). Entries are deduped by
    # flag_id at consumer time.
    metric_flags: List[Dict[str, Any]] = field(default_factory=list)
