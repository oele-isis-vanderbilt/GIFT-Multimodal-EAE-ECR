from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

@dataclass
class MetricContext:
    raw_frames: List[np.ndarray] = field(default_factory=list)
    tracker_output: List[Dict[str, Any]] = field(default_factory=list)
    all_frames: List[List[Tuple[int, float, float]]] = field(default_factory=list)
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]] = field(default_factory=dict)
    gaze_info: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    bbox_details: Dict[Tuple[int, int], Tuple[float, float, float, float]] = field(default_factory=dict)
    keypoint_details: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    fall_frames: Dict[int, Optional[int]] = field(default_factory=dict)
    map_points: List[Tuple[int, int, float, float]] = field(default_factory=list)
    
    # room_coverage: a dict containing coverage info:
    #   "coverage_per_frame"       -> List[Tuple[int, float]] (frame index, fraction covered)
    #   "time_to_full"             -> float (seconds from first non-enemy frame to full coverage) or None
    #   "final_fraction"           -> float (coverage fraction at last frame)
    #   "first_non_enemy_frame"    -> int (first frame index with non-enemy gaze) or None
    room_coverage: Optional[Dict[str, Any]] = None

    # Mapping: enemy_id -> (first_clear_frame, last_clear_frame, clearing_friend_id)
    threat_clearance: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]] = field(default_factory=dict)

    # Mapping of POD index to capture info:
    #   "assigned_id"      -> int or None
    #   "capture_time_sec" -> float or None
    #   "capture_frame"    -> int or None
    pod_capture: Dict[int, Dict[str, Optional[float]]] = field(default_factory=dict)
