# Motion Trackers for Video Analysis - Surya
# Extended from Mikel Broström's work on boxmot (10.0.81)
"""
This script is adopted from the SORT script by Alex Bewley.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shapely.geometry as geo
from shapely.affinity import scale

from ...motion.kalman_filters.xysr_kf import KalmanFilterXYSR
from ...utils import PerClassDecorator
from ...utils.association import associate, linear_assignment
from ...utils.iou import get_asso_func, run_asso_func
from ...utils.ops import xyxy2tlwh, xyxy2xywh, xyxy2xysr
from ...utils.pose_association import (
    POSE_ANCHOR_NAMES,
    compute_detection_track_pose_metrics,
    extract_pose_anchors,
)
from ..basetracker import BaseTracker


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def k_previous_obs(observations: Dict[int, np.ndarray], current_age: int, lookback: int):
    """
    Return the most recent observation within the requested lookback window.
    If none exists, return the latest available observation.
    """
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]

    for step in range(lookback):
        delta = lookback - step
        if current_age - delta in observations:
            return observations[current_age - delta]

    max_age = max(observations.keys())
    return observations[max_age]


def convert_x_to_bbox(state_vector: np.ndarray, score: Optional[float] = None):
    """
    Convert bbox state from [x, y, s, r] to [x1, y1, x2, y2].
    """
    width = np.sqrt(state_vector[2] * state_vector[3])
    height = state_vector[2] / width

    if score is None:
        return np.array(
            [
                state_vector[0] - width / 2.0,
                state_vector[1] - height / 2.0,
                state_vector[0] + width / 2.0,
                state_vector[1] + height / 2.0,
            ]
        ).reshape((1, 4))

    return np.array(
        [
            state_vector[0] - width / 2.0,
            state_vector[1] - height / 2.0,
            state_vector[0] + width / 2.0,
            state_vector[1] + height / 2.0,
            score,
        ]
    ).reshape((1, 5))


def speed_direction(previous_bbox: np.ndarray, current_bbox: np.ndarray):
    """
    Compute normalized motion direction from previous box center to current box center.
    """
    prev_center_x = (previous_bbox[0] + previous_bbox[2]) / 2.0
    prev_center_y = (previous_bbox[1] + previous_bbox[3]) / 2.0
    curr_center_x = (current_bbox[0] + current_bbox[2]) / 2.0
    curr_center_y = (current_bbox[1] + current_bbox[3]) / 2.0

    direction = np.array([curr_center_y - prev_center_y, curr_center_x - prev_center_x])
    norm = np.sqrt((curr_center_y - prev_center_y) ** 2 + (curr_center_x - prev_center_x) ** 2) + 1e-6
    return direction / norm


def normalize(vector: np.ndarray):
    """
    Normalize a vector safely.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        norm = np.finfo(vector.dtype).eps
    return vector / norm


# ---------------------------------------------------------------------
# Pose association config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PoseAssociationGateConfig:
    hybrid_threshold: float
    box_evidence_floor: float
    pose_evidence_floor: float


@dataclass(frozen=True)
class PoseAssociationConfig:
    enabled: bool
    weight: float
    min_affinity: float
    detection_mean_conf_threshold: float
    primary: PoseAssociationGateConfig
    rematch: PoseAssociationGateConfig


# ---------------------------------------------------------------------
# Small Kalman filter for 2D point smoothing
# ---------------------------------------------------------------------
class KalmanFilterPoint:
    """
    Constant-velocity Kalman filter for a 2D point.
    State: [x, y, vx, vy]
    """

    def __init__(self):
        self.state = np.zeros((4, 1))
        self.P = np.eye(4) * 1000.0
        self.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )
        self.R = np.eye(2) * 10.0
        self.Q = np.eye(4) * 0.01

    def initiate(self, measurement: np.ndarray):
        measurement = np.asarray(measurement, dtype=float).reshape(2, 1)
        self.state[:2] = measurement
        self.state[2:] = 0.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].flatten()

    def update(self, measurement: np.ndarray):
        measurement = np.asarray(measurement, dtype=float).reshape(2, 1)
        residual = measurement - (self.H @ self.state)
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + (kalman_gain @ residual)
        identity = np.eye(4)
        self.P = (identity - kalman_gain @ self.H) @ self.P


# ---------------------------------------------------------------------
# Track object
# ---------------------------------------------------------------------
class KalmanBoxTracker:
    """
    Internal state of an individual tracked object.
    """

    count = 0

    def __init__(
        self,
        bbox: np.ndarray,
        cls: int,
        det_ind: int,
        delta_t: int = 3,
        max_obs: int = 50,
        keypoints: Optional[np.ndarray] = None,
        keypoint_confidence_threshold: float = 0.2,
    ):
        self.det_ind = det_ind

        # -------------------------------------------------------------
        # Bounding-box Kalman filter
        # -------------------------------------------------------------
        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = xyxy2xysr(bbox)

        # -------------------------------------------------------------
        # General tracking state
        # -------------------------------------------------------------
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.conf = float(bbox[-1])
        self.cls = int(cls)

        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=float)
        self.observations: Dict[int, np.ndarray] = {}
        self.history_observations = deque([], maxlen=self.max_obs)

        self.velocity = None
        self.delta_t = delta_t

        # -------------------------------------------------------------
        # Mapping and identity state
        # -------------------------------------------------------------
        self.current_map_pos: Optional[np.ndarray] = None
        self.last_map_pos: Optional[np.ndarray] = None
        self.final_id: Optional[int] = None

        self.identity_role: str = "unknown"
        self.role_locked: bool = False
        self.birth_frame: Optional[int] = None
        self.birth_location: str = "unknown"
        self.frames_since_birth: int = 0
        self.frames_in_entry: int = 0
        self.frames_in_entry_buffer: int = 0
        self.frames_in_interior: int = 0
        self.consecutive_interior_frames: int = 0

        # -------------------------------------------------------------
        # Keypoints and pose-anchor state
        # -------------------------------------------------------------
        self.keypoints = keypoints
        self.keypoint_confidence_threshold = float(keypoint_confidence_threshold)
        self.filtered_keypoints = self.filter_keypoints(keypoints, self.keypoint_confidence_threshold)

        self.pose_anchors = extract_pose_anchors(
            self.filtered_keypoints,
            self.keypoint_confidence_threshold,
        )
        self.predicted_pose_anchors = {
            anchor_name: dict(anchor_info)
            for anchor_name, anchor_info in self.pose_anchors.items()
        }

        self.pose_anchor_filters: Dict[str, Optional[KalmanFilterPoint]] = {
            anchor_name: None for anchor_name in POSE_ANCHOR_NAMES
        }
        self.pose_anchor_missing_counts: Dict[str, int] = {
            anchor_name: 0 for anchor_name in POSE_ANCHOR_NAMES
        }
        self.pose_anchor_reliability: Dict[str, float] = {
            anchor_name: (
                float(self.pose_anchors[anchor_name].get("confidence", 0.0))
                if self.pose_anchors[anchor_name].get("valid", False)
                else 0.0
            )
            for anchor_name in POSE_ANCHOR_NAMES
        }

        # -------------------------------------------------------------
        # Map-position smoothing from ankle / foot evidence
        # -------------------------------------------------------------
        self.ankle_based_point: Optional[np.ndarray] = None
        self.keypoint_kalman_filter: Optional[KalmanFilterPoint] = None

        self.prev_top_center: Optional[np.ndarray] = None
        self.missing_keypoints_frames = 0

        self.joint_histories = {
            "lhip": deque([], maxlen=8),
            "rhip": deque([], maxlen=8),
            "lknee": deque([], maxlen=8),
            "rknee": deque([], maxlen=8),
            "bbox": deque([], maxlen=8),
        }

    @staticmethod
    def filter_keypoints(
        keypoints: Optional[np.ndarray],
        threshold: float,
    ) -> Optional[np.ndarray]:
        """
        Keep x and y unchanged; invalidate low-confidence keypoints by zeroing only
        the confidence channel.
        """
        if keypoints is None:
            return None

        filtered_keypoints = np.array(keypoints, copy=True)
        low_confidence_mask = filtered_keypoints[:, 2] < threshold
        filtered_keypoints[low_confidence_mask, 2] = 0.0
        return filtered_keypoints

    def _init_or_update_pose_anchor_filters(
        self,
        anchors: Dict[str, Dict[str, Any]],
        reappearance_threshold: int = 5,
    ):
        """
        Update per-anchor Kalman filters from current anchor observations.
        """
        for anchor_name in POSE_ANCHOR_NAMES:
            anchor_info = anchors.get(anchor_name, {})
            is_valid = bool(anchor_info.get("valid", False))

            if is_valid:
                point = np.asarray(anchor_info["point"], dtype=float)

                if (
                    self.pose_anchor_missing_counts[anchor_name] >= reappearance_threshold
                    or self.pose_anchor_filters[anchor_name] is None
                ):
                    self.pose_anchor_filters[anchor_name] = KalmanFilterPoint()
                    self.pose_anchor_filters[anchor_name].initiate(point)
                else:
                    self.pose_anchor_filters[anchor_name].update(point)

                anchor_filter = self.pose_anchor_filters[anchor_name]
                filtered_point = anchor_filter.state[:2].flatten() if anchor_filter is not None else point

                self.pose_anchor_missing_counts[anchor_name] = 0
                self.pose_anchor_reliability[anchor_name] = float(anchor_info.get("confidence", 0.0))
                self.pose_anchors[anchor_name] = {
                    "point": np.asarray(filtered_point, dtype=float),
                    "confidence": float(anchor_info.get("confidence", 0.0)),
                    "support": int(anchor_info.get("support", 0)),
                    "valid": True,
                }
            else:
                self.pose_anchor_missing_counts[anchor_name] += 1

                decay = 0.85 if self.pose_anchor_missing_counts[anchor_name] <= 3 else 0.65
                self.pose_anchor_reliability[anchor_name] *= decay

                predicted_info = (
                    self.predicted_pose_anchors.get(anchor_name, {})
                    if self.predicted_pose_anchors is not None
                    else {}
                )
                predicted_point = predicted_info.get("point", None)
                if predicted_point is not None:
                    predicted_point = np.asarray(predicted_point, dtype=float)

                self.pose_anchors[anchor_name] = {
                    "point": predicted_point,
                    "confidence": float(self.pose_anchor_reliability[anchor_name]),
                    "support": int(predicted_info.get("support", 0)),
                    "valid": (
                        self.pose_anchor_missing_counts[anchor_name] <= reappearance_threshold
                        and predicted_point is not None
                    ),
                }

    def _predict_pose_anchors(self, stale_after: int = 8):
        """
        Predict anchor positions one step forward for pose-based association.
        """
        predicted_anchors: Dict[str, Dict[str, Any]] = {}

        for anchor_name in POSE_ANCHOR_NAMES:
            anchor_filter = self.pose_anchor_filters.get(anchor_name)
            missing_count = int(self.pose_anchor_missing_counts.get(anchor_name, 0))
            reliability = float(self.pose_anchor_reliability.get(anchor_name, 0.0))

            if anchor_filter is not None:
                predicted_point = anchor_filter.predict()
                predicted_anchors[anchor_name] = {
                    "point": np.asarray(predicted_point, dtype=float),
                    "confidence": reliability,
                    "support": int(self.pose_anchors.get(anchor_name, {}).get("support", 0)),
                    "valid": missing_count <= stale_after,
                }
            else:
                previous_anchor = self.pose_anchors.get(anchor_name, {})
                predicted_anchors[anchor_name] = {
                    "point": previous_anchor.get("point", None),
                    "confidence": reliability,
                    "support": int(previous_anchor.get("support", 0)),
                    "valid": bool(previous_anchor.get("valid", False)) and missing_count <= stale_after,
                }

        self.predicted_pose_anchors = predicted_anchors

    def update(
        self,
        bbox: Optional[np.ndarray],
        cls: Optional[int],
        det_ind: Optional[int],
        keypoints: Optional[np.ndarray] = None,
    ):
        """
        Update tracker state with a matched detection.
        """
        self.det_ind = det_ind

        if bbox is not None:
            self.conf = float(bbox[-1])
            self.cls = int(cls)

            if self.last_observation.sum() >= 0:
                previous_box = None
                for step in range(self.delta_t):
                    delta = self.delta_t - step
                    if self.age - delta in self.observations:
                        previous_box = self.observations[self.age - delta]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)

            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            self.kf.update(xyxy2xysr(bbox))

            if keypoints is not None:
                self.keypoints = keypoints
                self.filtered_keypoints = self.filter_keypoints(
                    keypoints,
                    self.keypoint_confidence_threshold,
                )
            else:
                self.keypoints = None
                self.filtered_keypoints = None

            current_pose_anchors = extract_pose_anchors(
                self.filtered_keypoints,
                self.keypoint_confidence_threshold,
            )
            self._init_or_update_pose_anchor_filters(current_pose_anchors)
            self.predicted_pose_anchors = {
                anchor_name: dict(anchor_info)
                for anchor_name, anchor_info in self.pose_anchors.items()
            }
        else:
            # Preserve original unmatched-track behavior.
            self.kf.update(bbox)

    def predict(self):
        """
        Advance bbox KF and return predicted bbox.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self._predict_pose_anchors()
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Return current bbox estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def _continuous_velocity(
        self,
        name: str,
        k: int = 3,
        current_frame: Optional[int] = None,
        max_gap: int = 1,
    ):
        """
        Estimate a velocity vector from a continuous recent joint history.
        """
        history = self.joint_histories.get(name, None)
        if history is None or len(history) < k:
            return None

        recent_samples = list(history)[-k:]
        frames = [frame_idx for frame_idx, _ in recent_samples]

        if frames != list(range(frames[0], frames[0] + k)):
            return None
        if current_frame is not None and (current_frame - frames[-1] > max_gap):
            return None

        start_point = np.asarray(recent_samples[0][1], dtype=float)
        end_point = np.asarray(recent_samples[-1][1], dtype=float)
        dt = frames[-1] - frames[0]
        if dt == 0:
            return None

        return (end_point - start_point) / dt

    def update_map_position(
        self,
        pixel_mapper,
        xywh: np.ndarray,
        frame_idx: int,
        keypoints: Optional[np.ndarray] = None,
        keypoint_indices: Optional[Tuple[int, int]] = None,
        smoothing_factor: float = 0.5,
        reappearance_threshold: int = 5,
        dynamic_smoothing_min: float = 0.1,
        dynamic_smoothing_max: float = 0.3,
        dynamic_smoothing_thresh: float = 5,
    ):
        """
        Update map position using bbox-based or foot/ankle-based logic.
        """
        new_map_position = None

        if keypoint_indices is None:
            self.ankle_based_point = np.array(
                [xywh[0], (xywh[1] + xywh[3] / 2.0)],
                dtype=float,
            )
            new_map_position = pixel_mapper.detection_to_map(xywh)

        else:
            current_top_center = np.array(
                [xywh[0], xywh[1] - (xywh[3] / 2.0)],
                dtype=float,
            )
            self.joint_histories["bbox"].append((frame_idx, current_top_center))

            bbox_x1 = float(xywh[0]) - float(xywh[2]) / 2.0
            bbox_y1 = float(xywh[1]) - float(xywh[3]) / 2.0
            bbox_x2 = float(xywh[0]) + float(xywh[2]) / 2.0
            bbox_y2 = float(xywh[1]) + float(xywh[3]) / 2.0

            def _kp_inside_bbox(kp) -> bool:
                kx, ky = float(kp[0]), float(kp[1])
                if not (np.isfinite(kx) and np.isfinite(ky)):
                    return False
                return bbox_x1 <= kx <= bbox_x2 and bbox_y1 <= ky <= bbox_y2

            ankle_point = None

            if keypoints is not None:
                left_foot_indices = [20, 22, 24]
                right_foot_indices = [21, 23, 25]

                def collect_valid_points(indices: List[int]):
                    valid_points = []
                    for idx in indices:
                        if idx < 0 or idx >= len(keypoints):
                            continue
                        kp = keypoints[idx]
                        if kp[2] <= self.keypoint_confidence_threshold:
                            continue
                        if not _kp_inside_bbox(kp):
                            continue
                        valid_points.append(kp[:2])
                    return valid_points

                left_foot_points = collect_valid_points(left_foot_indices)
                right_foot_points = collect_valid_points(right_foot_indices)

                hip_and_knee_indices = {
                    "lhip": 11,
                    "rhip": 12,
                    "lknee": 13,
                    "rknee": 14,
                }
                for joint_name, joint_index in hip_and_knee_indices.items():
                    if joint_index < 0 or joint_index >= len(keypoints):
                        continue
                    kp = keypoints[joint_index]
                    if kp[2] <= self.keypoint_confidence_threshold:
                        continue
                    if not _kp_inside_bbox(kp):
                        continue
                    self.joint_histories[joint_name].append((frame_idx, kp[:2]))

                if left_foot_points or right_foot_points:
                    left_mean = np.mean(left_foot_points, axis=0) if left_foot_points else None
                    right_mean = np.mean(right_foot_points, axis=0) if right_foot_points else None

                    if left_mean is not None and right_mean is not None:
                        ankle_point = (left_mean + right_mean) / 2.0
                    else:
                        ankle_point = left_mean if left_mean is not None else right_mean
                elif keypoint_indices is not None:
                    keypoint_index_1, keypoint_index_2 = keypoint_indices
                    in_range_1 = 0 <= keypoint_index_1 < len(keypoints)
                    in_range_2 = 0 <= keypoint_index_2 < len(keypoints)
                    kp1 = keypoints[keypoint_index_1] if in_range_1 else None
                    kp2 = keypoints[keypoint_index_2] if in_range_2 else None

                    valid_ankle_candidates = []
                    for kp in (kp1, kp2):
                        if kp is None or kp[2] <= self.keypoint_confidence_threshold:
                            continue
                        if not _kp_inside_bbox(kp):
                            continue
                        valid_ankle_candidates.append(kp[:2])

                    if valid_ankle_candidates:
                        ankle_point = np.mean(valid_ankle_candidates, axis=0)

            if ankle_point is not None:
                ankle_point = np.asarray(ankle_point, dtype=float)
                if ankle_point.size < 2 or not np.isfinite(ankle_point[:2]).all():
                    ankle_point = None

            if ankle_point is not None:

                if self.missing_keypoints_frames >= reappearance_threshold:
                    if self.keypoint_kalman_filter is None:
                        self.keypoint_kalman_filter = KalmanFilterPoint()
                    self.keypoint_kalman_filter.initiate(ankle_point)

                if self.keypoint_kalman_filter is None:
                    self.keypoint_kalman_filter = KalmanFilterPoint()
                    self.keypoint_kalman_filter.initiate(ankle_point)

                self.keypoint_kalman_filter.predict()
                self.keypoint_kalman_filter.update(ankle_point)
                filtered_point = self.keypoint_kalman_filter.state[:2].flatten()

                new_map_position = pixel_mapper.pixel_to_map(filtered_point)
                self.ankle_based_point = filtered_point
                self.missing_keypoints_frames = 0
            else:
                self.missing_keypoints_frames += 1

                if self.keypoint_kalman_filter is not None:
                    velocity_priority = ["lhip", "rhip", "lknee", "rknee", "bbox"]
                    estimated_velocity = None

                    for joint_name in velocity_priority:
                        estimated_velocity = self._continuous_velocity(
                            joint_name,
                            k=5,
                            current_frame=frame_idx,
                            max_gap=3,
                        )
                        if estimated_velocity is not None:
                            break

                    if estimated_velocity is not None:
                        vx = float(estimated_velocity[0])
                        vy = float(estimated_velocity[1])
                        self.keypoint_kalman_filter.state[2:] = np.array([vx, vy], dtype=float).reshape(2, 1)

                    predicted_point = self.keypoint_kalman_filter.predict()
                    if predicted_point is not None:
                        new_map_position = pixel_mapper.pixel_to_map(predicted_point)
                        self.ankle_based_point = np.asarray(predicted_point, dtype=float)

            self.prev_top_center = current_top_center

        if (
            new_map_position is not None
            and self.current_map_pos is not None
            and dynamic_smoothing_thresh > 0
        ):
            current_map_position = np.asarray(self.current_map_pos, dtype=float)
            next_map_position = np.asarray(new_map_position, dtype=float)
            position_delta = np.linalg.norm(current_map_position - next_map_position)
            dynamic_smoothing = max(
                dynamic_smoothing_min,
                min(
                    dynamic_smoothing_max,
                    1.0 - (position_delta / float(dynamic_smoothing_thresh)),
                ),
            )
        else:
            dynamic_smoothing = smoothing_factor

        if self.current_map_pos is not None:
            self.last_map_pos = np.asarray(self.current_map_pos, dtype=float)
            if new_map_position is not None:
                next_map_position = np.asarray(new_map_position, dtype=float)
                self.current_map_pos = (
                    dynamic_smoothing * next_map_position
                    + (1.0 - dynamic_smoothing) * self.last_map_pos
                )
            else:
                self.current_map_pos = self.last_map_pos
        else:
            self.current_map_pos = (
                None if new_map_position is None else np.asarray(new_map_position, dtype=float)
            )

    def calculate_velocity(self):
        """
        Compute normalized velocity in map space.
        """
        velocity = np.array([None, None], dtype=object)

        if self.current_map_pos is not None and self.last_map_pos is not None:
            map_delta = np.asarray(self.current_map_pos, dtype=float) - np.asarray(self.last_map_pos, dtype=float)
            map_delta = normalize(map_delta.astype(float))
            velocity = map_delta

        return velocity


# ---------------------------------------------------------------------
# OC-SORT tracker
# ---------------------------------------------------------------------
class OCSort(BaseTracker):
    """
    OC-SORT tracker for video analysis.
    """

    INROOM_FINAL_ID_START = 99

    def __init__(
        self,
        per_class: bool = False,
        det_thresh: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
        asso_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
        pixel_mapper=None,
        limit_entry: bool = False,
        entry_polys=None,
        class_id_to_label: dict = None,
        entry_window_time=None,
        boundary=None,
        boundary_pad_pct: float = 0.0,
        track_enemy: bool = True,
        entry_conf_threshold: Optional[float] = None,
        detection_keypoint_mean_conf_threshold: Optional[float] = None,
        pose_hybrid_enabled: bool = True,
        pose_weight: float = 0.5,
        pose_min_affinity: float = 0.10,
        keypoint_mean_conf_threshold: Optional[float] = None,
        max_obs: int = 50,
        duplicate_birth_ioa: Optional[float] = None,
    ):
        super().__init__(max_age=max_age, class_id_to_label=class_id_to_label)

        self.per_class = per_class
        self.max_age = max_age
        self.min_hits = min_hits
        self.asso_threshold = asso_threshold
        self.frame_count = 0
        self.det_thresh = det_thresh
        # Duplicate-birth guard: refuse to birth a track from an unmatched
        # detection contained (IoA >= this) in a track matched this frame.
        # None disables (library default; the engine opts in explicitly).
        self.duplicate_birth_ioa = (
            float(duplicate_birth_ioa) if duplicate_birth_ioa is not None else None
        )
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.use_byte = use_byte
        self.max_obs = max_obs

        self.pose_association = self._build_pose_association_config(
            asso_threshold=asso_threshold,
            det_thresh=det_thresh,
            pose_hybrid_enabled=pose_hybrid_enabled,
            pose_weight=pose_weight,
            pose_min_affinity=pose_min_affinity,
            detection_keypoint_mean_conf_threshold=detection_keypoint_mean_conf_threshold,
            legacy_keypoint_mean_conf_threshold=keypoint_mean_conf_threshold,
        )

        self.entry_conf_threshold = (
            float(entry_conf_threshold)
            if entry_conf_threshold is not None
            else float(det_thresh)
        )

        KalmanBoxTracker.count = 0

        self.pixel_mapper = pixel_mapper
        self.limit_entry = limit_entry
        self.entry_polys = entry_polys
        self.next_final_id = 1
        self.next_inroom_final_id = self.INROOM_FINAL_ID_START

        self.entry_buffer_scale = 1.15
        self.inroom_confirmation_frames = max(3, int(self.min_hits))
        self.inroom_confirmation_hits = max(3, int(self.min_hits))
        self.entry_buffer_polys = []

        if self.limit_entry:
            if not self.entry_polys:
                raise ValueError("entry_polys cannot be None if limit_entry is True")
            if self.pixel_mapper is None:
                raise ValueError("pixel_mapper cannot be None if limit_entry is True")

        if self.entry_polys:
            self.entry_buffer_polys = [
                scale(poly, xfact=self.entry_buffer_scale, yfact=self.entry_buffer_scale, origin="center")
                for poly in self.entry_polys
            ]

        self.entry_window_time = entry_window_time
        self.entry_window_counter = 0
        self.entry_window_active = False

        self.boundary = boundary
        if self.boundary is not None:
            self.boundary_padded = (
                scale(self.boundary, xfact=1 + boundary_pad_pct, yfact=1 + boundary_pad_pct, origin="center")
                if boundary_pad_pct
                else self.boundary
            )
        else:
            self.boundary_padded = None

        self.track_enemy = track_enemy
        self.inroom_was_in_entry: Dict[int, bool] = {}

    def _point_in_entry_core(self, map_point: Optional[geo.Point]) -> bool:
        return bool(
            map_point is not None
            and self.entry_polys
            and any(poly.contains(map_point) for poly in self.entry_polys)
        )

    def _point_in_entry_buffer(self, map_point: Optional[geo.Point]) -> bool:
        return bool(
            map_point is not None
            and self.entry_buffer_polys
            and any(poly.contains(map_point) for poly in self.entry_buffer_polys)
        )

    def _point_in_interior_safe_zone(self, map_point: Optional[geo.Point]) -> bool:
        if map_point is None or self.boundary is None:
            return False
        if not self.boundary.contains(map_point):
            return False
        if self._point_in_entry_buffer(map_point):
            return False
        return True

    def _classify_birth_location(self, map_point: Optional[geo.Point]) -> str:
        if map_point is None:
            return "unknown"
        if self._point_in_entry_core(map_point):
            return "entry"
        if self._point_in_entry_buffer(map_point):
            return "entry_buffer"
        if self._point_in_interior_safe_zone(map_point):
            return "interior"
        if self.boundary_padded is not None and self.boundary_padded.contains(map_point):
            return "boundary_band"
        return "outside"

    def _update_track_role_evidence(self, track: KalmanBoxTracker):
        map_point = None
        if track.current_map_pos is not None:
            current_xy = np.asarray(track.current_map_pos, dtype=float).reshape(-1)
            if current_xy.size >= 2 and np.isfinite(current_xy[:2]).all():
                map_point = geo.Point(float(current_xy[0]), float(current_xy[1]))

        track.frames_since_birth += 1

        if self._point_in_entry_core(map_point):
            track.frames_in_entry += 1
        if self._point_in_entry_buffer(map_point):
            track.frames_in_entry_buffer += 1
        if self._point_in_interior_safe_zone(map_point):
            track.frames_in_interior += 1
            track.consecutive_interior_frames += 1
        else:
            track.consecutive_interior_frames = 0

    def _maybe_assign_track_role(self, track: KalmanBoxTracker):
        if track.role_locked:
            return

        if track.frames_in_entry > 0:
            track.identity_role = "entry"
            track.role_locked = True
            if track.final_id is None:
                track.final_id = self.next_final_id
                self.next_final_id += 1
            return

        if (
            self.track_enemy
            and track.frames_in_entry == 0
            and track.frames_in_entry_buffer == 0
            and track.frames_in_interior >= self.inroom_confirmation_frames
            and track.consecutive_interior_frames >= self.inroom_confirmation_frames
            and track.hit_streak >= self.inroom_confirmation_hits
        ):
            track.identity_role = "inroom"
            track.role_locked = True
            if track.final_id is None:
                track.final_id = self._allocate_inroom_final_id()
                self.inroom_was_in_entry[track.final_id] = False
            return

        if track.birth_location == "entry":
            track.identity_role = "entry_candidate"
        elif track.birth_location == "interior":
            track.identity_role = "inroom_candidate"
        else:
            track.identity_role = "unknown"

    def _is_inroom_final_id(self, final_id: Optional[int]) -> bool:
        return final_id is not None and int(final_id) >= self.INROOM_FINAL_ID_START

    def _allocate_inroom_final_id(self) -> int:
        final_id = self.next_inroom_final_id
        self.next_inroom_final_id += 1
        return final_id

    @staticmethod
    def _build_pose_association_config(
        asso_threshold: float,
        det_thresh: float,
        pose_hybrid_enabled: bool,
        pose_weight: float,
        pose_min_affinity: float,
        detection_keypoint_mean_conf_threshold: Optional[float],
        legacy_keypoint_mean_conf_threshold: Optional[float],
    ) -> PoseAssociationConfig:
        mean_conf_threshold = detection_keypoint_mean_conf_threshold
        if mean_conf_threshold is None:
            mean_conf_threshold = legacy_keypoint_mean_conf_threshold
        if mean_conf_threshold is None:
            mean_conf_threshold = det_thresh

        pose_weight = float(np.clip(pose_weight, 0.0, 1.0))

        return PoseAssociationConfig(
            enabled=bool(pose_hybrid_enabled),
            weight=pose_weight,
            min_affinity=float(pose_min_affinity),
            detection_mean_conf_threshold=float(mean_conf_threshold),
            primary=PoseAssociationGateConfig(
                hybrid_threshold=float(asso_threshold),
                box_evidence_floor=float(asso_threshold) * 0.5,
                pose_evidence_floor=float(asso_threshold) * 0.5,
            ),
            rematch=PoseAssociationGateConfig(
                hybrid_threshold=float(asso_threshold),
                box_evidence_floor=float(asso_threshold) * 0.4,
                pose_evidence_floor=float(asso_threshold) * 0.4,
            ),
        )

    @staticmethod
    def _split_detections_by_confidence(
        dets: np.ndarray,
        det_thresh: float,
        keypoints: Optional[np.ndarray],
        detection_keypoint_mean_conf_threshold: float,
    ):
        """
        Split detections into primary and secondary groups based on detection
        confidence and mean keypoint confidence.
        """
        detection_confidences = dets[:, 4]

        if keypoints is not None:
            keypoint_mean_confidences = np.asarray(
                [
                    float(np.mean(kp[:, 2])) if kp is not None and len(kp) > 0 else 0.0
                    for kp in keypoints
                ],
                dtype=float,
            )
        else:
            keypoint_mean_confidences = None

        low_confidence_mask = detection_confidences > 0.1
        below_primary_threshold_mask = detection_confidences < det_thresh
        secondary_detection_mask = np.logical_and(low_confidence_mask, below_primary_threshold_mask)

        if keypoint_mean_confidences is not None:
            secondary_detection_mask = np.logical_and(
                secondary_detection_mask,
                keypoint_mean_confidences > detection_keypoint_mean_conf_threshold,
            )

        primary_detection_mask = detection_confidences > det_thresh
        if keypoint_mean_confidences is not None:
            primary_detection_mask = np.logical_and(
                primary_detection_mask,
                keypoint_mean_confidences > detection_keypoint_mean_conf_threshold,
            )

        primary_detections = dets[primary_detection_mask]
        secondary_detections = dets[secondary_detection_mask]

        if keypoints is not None:
            primary_keypoints = keypoints[primary_detection_mask]
            secondary_keypoints = keypoints[secondary_detection_mask]
        else:
            primary_keypoints = None
            secondary_keypoints = None

        return primary_detections, secondary_detections, primary_keypoints, secondary_keypoints

    def _compute_pose_metrics(
        self,
        detections: np.ndarray,
        tracks: List[KalmanBoxTracker],
        keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute pose affinity and reliability matrices for a detection set.
        """
        if keypoints is None or len(tracks) == 0 or len(detections) == 0:
            return None, None

        pose_affinity = np.zeros((len(detections), len(tracks)), dtype=float)
        pose_reliability = np.zeros((len(detections), len(tracks)), dtype=float)

        for detection_index in range(len(detections)):
            similarities, reliabilities = compute_detection_track_pose_metrics(
                keypoints[detection_index],
                tracks,
                detections[detection_index, 0:4],
                keypoint_confidence_threshold,
            )
            pose_affinity[detection_index, :] = similarities
            pose_reliability[detection_index, :] = reliabilities

        pose_affinity[pose_affinity < self.pose_association.min_affinity] = 0.0
        pose_reliability[pose_affinity <= 0.0] = 0.0
        return pose_affinity, pose_reliability

    @staticmethod
    def _update_track_with_detection(
        track: KalmanBoxTracker,
        detection: np.ndarray,
        keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
    ):
        track.keypoint_confidence_threshold = float(keypoint_confidence_threshold)
        track.update(detection[:5], detection[5], detection[6], keypoints=keypoints)

    def _predict_active_tracks(self):
        """
        Predict all active tracks forward one step and remove invalid predictions.

        Returns:
          - predicted tracker boxes array of shape [N, 5]
        """
        predicted_track_boxes = np.zeros((len(self.active_tracks), 5), dtype=float)
        invalid_track_indices = []

        for track_index in range(len(predicted_track_boxes)):
            predicted_box = self.active_tracks[track_index].predict()[0]
            predicted_track_boxes[track_index, :] = [
                predicted_box[0],
                predicted_box[1],
                predicted_box[2],
                predicted_box[3],
                0.0,
            ]
            if np.any(np.isnan(predicted_box)):
                invalid_track_indices.append(track_index)

        predicted_track_boxes = np.ma.compress_rows(np.ma.masked_invalid(predicted_track_boxes))

        for track_index in reversed(invalid_track_indices):
            self.active_tracks.pop(track_index)

        return predicted_track_boxes

    def _run_primary_association(
        self,
        detections: np.ndarray,
        predicted_track_boxes: np.ndarray,
        primary_keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
        image_width: int,
        image_height: int,
    ):
        pose_affinity, pose_reliability = self._compute_pose_metrics(
            detections,
            self.active_tracks,
            primary_keypoints,
            keypoint_confidence_threshold,
        )

        track_velocities = np.array(
            [
                track.velocity if track.velocity is not None else np.array((0.0, 0.0))
                for track in self.active_tracks
            ],
            dtype=float,
        )
        previous_boxes = np.array([track.last_observation for track in self.active_tracks], dtype=float)
        k_step_observations = np.array(
            [
                k_previous_obs(track.observations, track.age, self.delta_t)
                for track in self.active_tracks
            ],
            dtype=float,
        )

        matches, unmatched_detections, unmatched_tracks = associate(
            detections[:, 0:5],
            predicted_track_boxes,
            self.asso_func,
            self.asso_threshold,
            track_velocities,
            k_step_observations,
            self.inertia,
            image_width,
            image_height,
            pose_affinity=pose_affinity,
            pose_reliability=pose_reliability,
            pose_weight=self.pose_association.weight if self.pose_association.enabled else 0.0,
            hybrid_threshold=self.pose_association.primary.hybrid_threshold,
            box_threshold_floor=self.pose_association.primary.box_evidence_floor,
            pose_threshold_floor=self.pose_association.primary.pose_evidence_floor,
        )

        return matches, unmatched_detections, unmatched_tracks, previous_boxes

    def _apply_primary_matches(
        self,
        matches: np.ndarray,
        detections: np.ndarray,
        primary_keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
    ):
        for match_pair in matches:
            detection_index, track_index = match_pair[0], match_pair[1]
            matched_keypoints = primary_keypoints[detection_index] if primary_keypoints is not None else None
            self._update_track_with_detection(
                self.active_tracks[track_index],
                detections[detection_index],
                matched_keypoints,
                keypoint_confidence_threshold,
            )

    def _run_byte_second_association(
        self,
        secondary_detections: np.ndarray,
        unmatched_track_indices: np.ndarray,
        predicted_track_boxes: np.ndarray,
        secondary_keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
    ):
        if not self.use_byte or len(secondary_detections) == 0 or unmatched_track_indices.shape[0] == 0:
            return unmatched_track_indices

        unmatched_predicted_boxes = predicted_track_boxes[unmatched_track_indices]
        iou_similarity = np.array(self.asso_func(secondary_detections, unmatched_predicted_boxes))

        if iou_similarity.max() <= self.asso_threshold:
            return unmatched_track_indices

        matched_indices = linear_assignment(-iou_similarity)
        matched_track_indices_to_remove = []

        for match_pair in matched_indices:
            secondary_detection_index = match_pair[0]
            unmatched_track_subindex = match_pair[1]
            track_index = unmatched_track_indices[unmatched_track_subindex]

            if iou_similarity[secondary_detection_index, unmatched_track_subindex] < self.asso_threshold:
                continue

            matched_keypoints = (
                secondary_keypoints[secondary_detection_index]
                if secondary_keypoints is not None
                else None
            )

            self._update_track_with_detection(
                self.active_tracks[track_index],
                secondary_detections[secondary_detection_index],
                matched_keypoints,
                keypoint_confidence_threshold,
            )
            matched_track_indices_to_remove.append(track_index)

        return np.setdiff1d(unmatched_track_indices, np.array(matched_track_indices_to_remove))

    def _run_rematch_association(
        self,
        unmatched_detection_indices: np.ndarray,
        unmatched_track_indices: np.ndarray,
        detections: np.ndarray,
        previous_boxes: np.ndarray,
        primary_keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
        image_width: int,
        image_height: int,
    ):
        if unmatched_detection_indices.shape[0] == 0 or unmatched_track_indices.shape[0] == 0:
            return unmatched_detection_indices, unmatched_track_indices

        unmatched_detections = detections[unmatched_detection_indices]
        unmatched_track_boxes = previous_boxes[unmatched_track_indices]
        rematch_tracks = [self.active_tracks[track_index] for track_index in unmatched_track_indices]

        rematch_pose_affinity, rematch_pose_reliability = self._compute_pose_metrics(
            unmatched_detections,
            rematch_tracks,
            primary_keypoints[unmatched_detection_indices] if primary_keypoints is not None else None,
            keypoint_confidence_threshold,
        )

        rematch_matches, _, _ = associate(
            unmatched_detections[:, 0:5],
            unmatched_track_boxes,
            self.asso_func,
            self.asso_threshold,
            np.zeros((len(unmatched_track_indices), 2), dtype=float),
            unmatched_track_boxes,
            0.0,
            image_width,
            image_height,
            pose_affinity=rematch_pose_affinity,
            pose_reliability=rematch_pose_reliability,
            pose_weight=self.pose_association.weight if self.pose_association.enabled else 0.0,
            hybrid_threshold=self.pose_association.rematch.hybrid_threshold,
            box_threshold_floor=self.pose_association.rematch.box_evidence_floor,
            pose_threshold_floor=self.pose_association.rematch.pose_evidence_floor,
        )

        if rematch_matches.size == 0:
            return unmatched_detection_indices, unmatched_track_indices

        matched_detection_indices_to_remove = []
        matched_track_indices_to_remove = []

        for match_pair in rematch_matches:
            detection_index = unmatched_detection_indices[match_pair[0]]
            track_index = unmatched_track_indices[match_pair[1]]
            matched_keypoints = primary_keypoints[detection_index] if primary_keypoints is not None else None

            self._update_track_with_detection(
                self.active_tracks[track_index],
                detections[detection_index],
                matched_keypoints,
                keypoint_confidence_threshold,
            )
            matched_detection_indices_to_remove.append(detection_index)
            matched_track_indices_to_remove.append(track_index)

        unmatched_detection_indices = np.setdiff1d(
            unmatched_detection_indices,
            np.array(matched_detection_indices_to_remove),
        )
        unmatched_track_indices = np.setdiff1d(
            unmatched_track_indices,
            np.array(matched_track_indices_to_remove),
        )

        return unmatched_detection_indices, unmatched_track_indices

    def _mark_unmatched_tracks(self, unmatched_track_indices: np.ndarray):
        for track_index in unmatched_track_indices:
            self.active_tracks[track_index].update(None, None, None)

    def _compute_birth_pixel_anchor(
        self,
        detection_xyxy: np.ndarray,
        keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
        keypoint_indices: Optional[Tuple[int, int]],
    ) -> Optional[np.ndarray]:
        """Pick the best pixel anchor for birth-time map projection.

        Priority: confident feet -> ankles -> knees, each filtered to keypoints
        that fall inside the detection bbox (defends against pose models that
        hallucinate the full skeleton outside small/partial bboxes). Falls back
        to bbox bottom-center when no usable keypoints are present.
        """
        x1 = float(detection_xyxy[0])
        y1 = float(detection_xyxy[1])
        x2 = float(detection_xyxy[2])
        y2 = float(detection_xyxy[3])
        bbox_bottom_center = np.array([(x1 + x2) / 2.0, y2], dtype=float)

        if keypoints is None:
            return bbox_bottom_center
        kps = np.asarray(keypoints)
        if kps.ndim != 2 or kps.shape[1] < 3 or kps.shape[0] == 0:
            return bbox_bottom_center

        def collect_inside_bbox(indices):
            valid = []
            for idx in indices:
                if idx < 0 or idx >= kps.shape[0]:
                    continue
                kp = kps[idx]
                if float(kp[2]) < keypoint_confidence_threshold:
                    continue
                kx, ky = float(kp[0]), float(kp[1])
                if not (np.isfinite(kx) and np.isfinite(ky)):
                    continue
                if x1 <= kx <= x2 and y1 <= ky <= y2:
                    valid.append(np.array([kx, ky], dtype=float))
            return valid

        # Halpe-26: feet (toes + heels)
        feet = collect_inside_bbox([20, 22, 24, 21, 23, 25])
        if feet:
            return np.mean(feet, axis=0)

        # Configured ankle pair (default Halpe-26 ankles 15/16)
        ankle_idx = tuple(keypoint_indices) if keypoint_indices is not None else (15, 16)
        ankles = collect_inside_bbox(list(ankle_idx))
        if ankles:
            return np.mean(ankles, axis=0)

        # Knees — fallback for partial-body / broken-leg manikins
        knees = collect_inside_bbox([13, 14])
        if knees:
            return np.mean(knees, axis=0)

        return bbox_bottom_center

    def _make_map_point_from_detection(
        self,
        detection_xyxy: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        keypoint_confidence_threshold: float = 0.5,
        keypoint_indices: Optional[Tuple[int, int]] = None,
    ) -> Optional[geo.Point]:
        if self.pixel_mapper is None:
            return None

        pixel_anchor = self._compute_birth_pixel_anchor(
            detection_xyxy,
            keypoints,
            keypoint_confidence_threshold,
            keypoint_indices,
        )
        if pixel_anchor is None:
            return None

        map_xy = self.pixel_mapper.pixel_to_map(pixel_anchor)
        map_xy = np.asarray(map_xy, dtype=float).reshape(-1)

        if map_xy.size >= 2 and np.isfinite(map_xy[:2]).all():
            return geo.Point(float(map_xy[0]), float(map_xy[1]))
        return None

    def _should_create_new_track(
        self,
        birth_location: str,
        detection_confidence: float,
    ) -> bool:
        create_track = False

        if birth_location == "interior" and self.track_enemy:
            create_track = True
        elif self.limit_entry:
            if self.entry_window_active:
                if (
                    self.entry_window_time is not None
                    and self.entry_window_counter >= self.entry_window_time
                ):
                    return False

            if birth_location in {"entry", "entry_buffer"} and detection_confidence >= self.entry_conf_threshold:
                create_track = True

            if create_track and birth_location == "entry" and not self.entry_window_active:
                self.entry_window_active = True
                self.entry_window_counter = 0
        else:
            create_track = True

        return create_track

    def _is_duplicate_of_matched_track(self, det_xyxy: np.ndarray) -> bool:
        """True when ``det_xyxy`` lies almost entirely on a person who already
        has a track matched this frame.

        Guards track birth against duplicate detections: a partially occluded
        person can yield two boxes of different extents (e.g. torso-only +
        full-body) whose mutual IoU passes detector NMS, leaving the surplus
        box unmatched — without this check it would be born as a rival track
        that then steals the identity. Containment (intersection over the
        smaller box's area) is used instead of IoU because the duplicate pair
        differs precisely in extent. Genuine new entrants are unaffected: a
        real person does not materialize ~fully inside an already-claimed box.
        """
        if self.duplicate_birth_ioa is None or self.duplicate_birth_ioa >= 1.0:
            return False
        dx1, dy1, dx2, dy2 = (float(v) for v in det_xyxy[:4])
        det_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
        if det_area <= 0:
            return False
        for track in self.active_tracks:
            if track.time_since_update != 0 or track.last_observation.sum() < 0:
                continue
            tx1, ty1, tx2, ty2 = (float(v) for v in track.last_observation[:4])
            trk_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
            if trk_area <= 0:
                continue
            iw = min(dx2, tx2) - max(dx1, tx1)
            ih = min(dy2, ty2) - max(dy1, ty1)
            if iw <= 0 or ih <= 0:
                continue
            ioa = (iw * ih) / min(det_area, trk_area)
            if ioa >= self.duplicate_birth_ioa:
                return True
        return False

    def _create_new_tracks(
        self,
        unmatched_detection_indices: np.ndarray,
        detections: np.ndarray,
        primary_keypoints: Optional[np.ndarray],
        keypoint_confidence_threshold: float,
        keypoint_indices: Optional[Tuple[int, int]] = None,
    ):
        if unmatched_detection_indices.shape[0] == 0:
            return

        for detection_index in unmatched_detection_indices:
            if self._is_duplicate_of_matched_track(detections[detection_index, 0:4]):
                continue
            detection_keypoints = (
                primary_keypoints[detection_index]
                if primary_keypoints is not None
                else None
            )
            detection_confidence = float(detections[detection_index, 4])

            map_point = self._make_map_point_from_detection(
                detections[detection_index, 0:4],
                keypoints=detection_keypoints,
                keypoint_confidence_threshold=keypoint_confidence_threshold,
                keypoint_indices=keypoint_indices,
            )
            birth_location = self._classify_birth_location(map_point)

            if not self._should_create_new_track(birth_location, detection_confidence):
                continue

            new_track = KalmanBoxTracker(
                detections[detection_index, :5],
                detections[detection_index, 5],
                detections[detection_index, 6],
                delta_t=self.delta_t,
                max_obs=self.max_obs,
                keypoints=detection_keypoints,
                keypoint_confidence_threshold=keypoint_confidence_threshold,
            )
            new_track.birth_frame = self.frame_count
            new_track.birth_location = birth_location

            if birth_location == "entry":
                new_track.identity_role = "entry_candidate"
            elif birth_location == "interior":
                new_track.identity_role = "inroom_candidate"
            else:
                new_track.identity_role = "unknown"

            self.active_tracks.append(new_track)

    def _build_track_output(
        self,
        track: KalmanBoxTracker,
        tlwh: np.ndarray,
        map_velocity,
    ) -> Dict[str, Any]:
        track_output = {
            "top_left_x": tlwh[0],
            "top_left_y": tlwh[1],
            "width": tlwh[2],
            "height": tlwh[3],
            "track_id": track.final_id,
            "track_id_raw": track.id + 1,
            "confidence": track.conf,
            "class": track.cls,
            "detection_index": track.det_ind,
            "keypoints": track.filtered_keypoints,
            "ankle_based_point": track.ankle_based_point,
            "identity_role": track.identity_role,
            "birth_location": track.birth_location,
            "is_inroom": track.identity_role == "inroom",
            "is_entry": track.identity_role == "entry",
        }

        if self.pixel_mapper is not None:
            track_output["current_map_pos"] = track.current_map_pos
            track_output["map_velocity"] = map_velocity
        else:
            track_output["current_map_pos"] = None
            track_output["map_velocity"] = [None, None]

        return track_output

    def _update_tracks_and_collect_outputs(
        self,
        keypoint_indices: Optional[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []

        track_index = len(self.active_tracks)
        for track in reversed(self.active_tracks):
            if track.last_observation.sum() < 0:
                tlwh = xyxy2tlwh(track.get_state()[0])
            else:
                tlwh = xyxy2tlwh(track.last_observation)

            if self.pixel_mapper is not None:
                track.update_map_position(
                    self.pixel_mapper,
                    xyxy2xywh(track.get_state()[0]),
                    self.frame_count,
                    keypoints=track.filtered_keypoints,
                    keypoint_indices=keypoint_indices,
                    smoothing_factor=0.7,
                )
                map_velocity = track.calculate_velocity()
            else:
                map_velocity = [None, None]

            if track.time_since_update < 1:
                self._update_track_role_evidence(track)
                self._maybe_assign_track_role(track)

            if (
                track.time_since_update < 1
                and track.hit_streak >= self.min_hits
                and track.final_id is not None
            ):
                outputs.append(self._build_track_output(track, tlwh, map_velocity))

            track_index -= 1
            if track.time_since_update > self.max_age:
                self.active_tracks.pop(track_index)

        return outputs

    def _run_inroom_exit_logic(self):
        if not self.track_enemy or self.boundary is None:
            return

        for track in list(self.active_tracks):
            final_id = getattr(track, "final_id", None)
            if not self._is_inroom_final_id(final_id):
                continue

            if self.pixel_mapper is not None and track.current_map_pos is None:
                fallback_point = self._make_map_point_from_detection(
                    track.get_state()[0],
                    keypoints=track.filtered_keypoints,
                    keypoint_confidence_threshold=track.keypoint_confidence_threshold,
                )
                if fallback_point is not None:
                    track.current_map_pos = np.array(
                        [fallback_point.x, fallback_point.y], dtype=float
                    )

            if track.current_map_pos is None:
                continue

            current_xy = np.asarray(track.current_map_pos, dtype=float).reshape(-1)
            if current_xy.size >= 2 and np.isfinite(current_xy[:2]).all():
                map_point = geo.Point(float(current_xy[0]), float(current_xy[1]))
            else:
                map_point = None

            if map_point is None:
                continue

            if self.entry_polys and not self.inroom_was_in_entry.get(final_id, False):
                self.inroom_was_in_entry[final_id] = any(
                    poly.contains(map_point) for poly in self.entry_polys
                )

            if self.inroom_was_in_entry.get(final_id, False) and (not self.boundary.contains(map_point)):
                self.active_tracks.remove(track)
                self.inroom_was_in_entry.pop(final_id, None)

        active_inroom_ids = {
            int(getattr(track, "final_id", -1))
            for track in self.active_tracks
            if self._is_inroom_final_id(getattr(track, "final_id", None))
        }
        for final_id in list(self.inroom_was_in_entry.keys()):
            if final_id not in active_inroom_ids:
                self.inroom_was_in_entry.pop(final_id, None)

    @PerClassDecorator
    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        keypoints: Optional[np.ndarray] = None,
        keypoint_confidence_threshold: float = 0.5,
        keypoint_indices: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        assert isinstance(dets, np.ndarray), (
            f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        )
        assert len(dets.shape) == 2, (
            "Unsupported 'dets' dimensions, valid number of dimensions is two"
        )
        assert dets.shape[1] == 6, (
            "Unsupported 'dets' 2nd dimension length, valid length is 6"
        )

        self.frame_count += 1

        if self.limit_entry and self.entry_window_active:
            self.entry_window_counter += 1

        image_height, image_width = img.shape[0:2]

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        primary_detections, secondary_detections, primary_keypoints, secondary_keypoints = (
            self._split_detections_by_confidence(
                dets,
                self.det_thresh,
                keypoints,
                self.pose_association.detection_mean_conf_threshold,
            )
        )

        predicted_track_boxes = self._predict_active_tracks()

        matches, unmatched_detections, unmatched_tracks, previous_boxes = self._run_primary_association(
            primary_detections,
            predicted_track_boxes,
            primary_keypoints,
            keypoint_confidence_threshold,
            image_width,
            image_height,
        )

        self._apply_primary_matches(
            matches,
            primary_detections,
            primary_keypoints,
            keypoint_confidence_threshold,
        )

        unmatched_tracks = self._run_byte_second_association(
            secondary_detections,
            unmatched_tracks,
            predicted_track_boxes,
            secondary_keypoints,
            keypoint_confidence_threshold,
        )

        unmatched_detections, unmatched_tracks = self._run_rematch_association(
            unmatched_detections,
            unmatched_tracks,
            primary_detections,
            previous_boxes,
            primary_keypoints,
            keypoint_confidence_threshold,
            image_width,
            image_height,
        )

        self._mark_unmatched_tracks(unmatched_tracks)

        self._create_new_tracks(
            unmatched_detections,
            primary_detections,
            primary_keypoints,
            keypoint_confidence_threshold,
            keypoint_indices,
        )

        outputs = self._update_tracks_and_collect_outputs(keypoint_indices)
        self._run_inroom_exit_logic()

        return np.array(outputs) if len(outputs) > 0 else np.array([])