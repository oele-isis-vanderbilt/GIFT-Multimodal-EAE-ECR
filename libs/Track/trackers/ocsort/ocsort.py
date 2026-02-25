# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))

"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
import numpy as np

from collections import deque
from typing import Optional, Dict, Any


from ...motion.kalman_filters.xysr_kf import KalmanFilterXYSR
from ...utils.association import associate, linear_assignment
from ...utils.iou import get_asso_func
from ...utils.iou import run_asso_func
from ..basetracker import BaseTracker
from ...utils import PerClassDecorator
from ...utils.ops import xyxy2xysr


#Change Begin
from ...utils.ops import xywh2xyxy, xyxy2xywh, xyxy2tlwh
import shapely.geometry as geo
from shapely.affinity import scale


class KalmanFilterPoint:
    def __init__(self):
        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1))  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # Covariance matrix with high initial uncertainty
        self.F = np.array([[1, 0, 1, 0],  # State transition matrix
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # Models constant velocity
        self.H = np.array([[1, 0, 0, 0],  # Measurement matrix
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 10  # Measurement noise
        self.Q = np.eye(4) * 0.01  # Process noise (small value)

    def initiate(self, measurement):
        """
        Initialize the Kalman filter with the first keypoint measurement.
        """
        self.state[:2] = measurement.reshape(2, 1)  # Set initial position
        self.state[2:] = 0  # Start with no velocity

    def predict(self):
        """
        Predict the next position based on the current state and velocity.
        """
        self.state = np.dot(self.F, self.state)  # Apply the state transition matrix
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q  # Update covariance
        return self.state[:2].flatten()  # Return predicted position [x, y]

    def update(self, measurement):
        """
        Update the state with the observed measurement (ankle keypoints).
        """
        y = measurement.reshape(2, 1) - np.dot(self.H, self.state)  # Residual (error)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Measurement uncertainty
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # Kalman gain
        self.state = self.state + np.dot(K, y)  # Update state with measurement
        I = np.eye(4)
        self.P = np.dot(I - np.dot(K, self.H), self.P)  # Update covariance
# Change End


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, cls, det_ind, delta_t=3, max_obs=50, keypoints=None, keypoint_confidence_threshold=0.2):
        """
        Initializes a tracker using the initial bounding box.

        Parameters:
        -----------
        - bbox: ndarray
            Initial bounding box in (x1, y1, x2, y2) format.
        - cls: int
            Class ID of the object.
        - det_ind: int
            Detection index.
        - delta_t: int, optional
            Time step for velocity calculation. Default is 3.
        - max_obs: int, optional
            Maximum number of observations to store. Default is 50.
        - keypoints: ndarray, optional
            The array of keypoints output from yolo in the data key it returns which contains a,y,conf
        - keypoint_confidence_threshold: float
            The minimum confidence of a keypoint to be associated with a track 

        Attributes:
        -----------
        - det_ind: int
            Detection index.
        - kf: KalmanFilterXYSR
            Kalman filter for motion prediction.
        - time_since_update: int
            Number of frames since the last update.
        - id: int
            Unique track ID.
        - max_obs: int
            Maximum number of observations to store.
        - history: deque
            History of bounding box predictions.
        - hits: int
            Number of times the object was detected.
        - hit_streak: int
            Number of consecutive frames the object was detected.
        - age: int
            Total number of frames since the object was first detected.
        - conf: float
            Confidence score of the detection.
        - cls: int
            Class ID of the object.
        - last_observation: ndarray
            Last observation of the bounding box.
        - observations: dict
            Dictionary of all observations with frame ID as keys.
        - history_observations: deque
            History of observations.
        - velocity: ndarray
            Velocity of the object.
        - delta_t: int
            Time step for velocity calculation.
        - current_map_pos: ndarray
            Current map position of the object.
        - last_map_pos: ndarray
            Last map position of the object.
        - final_id: int
            Final track ID for uniformity in output.
        """
        # define constant velocity model
        self.det_ind = det_ind
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
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = xyxy2xysr(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[-1]
        self.cls = cls
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity = None
        self.delta_t = delta_t

        #Change Begin
        self.current_map_pos = None
        self.last_map_pos = None
        self.final_id = None  # Final track ID for uniformity in output

        # Store keypoints and confidence threshold
        self.keypoints = keypoints  # List of keypoints
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self.filtered_keypoints = self.filter_keypoints(keypoints, keypoint_confidence_threshold)


        self.ankle_based_point = None
         # Initialize keypoint Kalman filter attribute
        self.keypoint_kalman_filter = None  # Add this line

        # Initialize the previous top_center as None (replace prev_top_middle)
        self.prev_top_center = None
        self.missing_keypoints_frames = 0

        self.previous_bbox = None
        # ------------------------------------------------------------------
        # Joint‑specific position history for velocity estimation
        # Stores (frame_idx, np.array([x, y])) for continuity checking
        # Priority: left‑hip → right‑hip → left‑knee → right‑knee → bbox
        self.joint_histories = {
            'lhip': deque([], maxlen=8),
            'rhip': deque([], maxlen=8),
            'lknee': deque([], maxlen=8),
            'rknee': deque([], maxlen=8),
            'bbox': deque([], maxlen=8)
        }
        # ------------------------------------------------------------------
        #Change End

    #Change Begin
    def filter_keypoints(self, keypoints, threshold):
        """
        Filter keypoints based on confidence threshold.

        Parameters:
        -----------
        - keypoints: ndarray
            Array of keypoints in (x, y, confidence) format.
        - threshold: float
            Confidence threshold.

        Returns:
        --------
        - filtered_keypoints: ndarray
            Array of keypoints where keypoints below the threshold are set to (0, 0, 0).
        """
        if keypoints is None:
            return None
        filtered_keypoints = keypoints.copy()
        for kp in filtered_keypoints:
            if kp[2] < threshold:
                kp[:] = [0, 0, 0]  # Set low-confidence keypoints to (0, 0, 0)
        return filtered_keypoints
    #Change End

    def update(self, bbox, cls, det_ind, keypoints=None):
        """
        Updates the state vector with the observed bounding box.

        Parameters:
        -----------
        - bbox: ndarray
            Bounding box in (x1, y1, x2, y2, score) format.
        - cls: int
            Class ID of the object.
        - det_ind: int
            Detection index.
        - keypoints: ndarray, optional
            The array of keypoints output from yolo in the data key it returns which contains a,y,conf
        """
        self.det_ind = det_ind
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(xyxy2xysr(bbox))
            #Change Begin
            # Update keypoints if provided
            if keypoints is not None:
                self.keypoints = keypoints
                self.filtered_keypoints = self.filter_keypoints(keypoints, self.keypoint_confidence_threshold)
            #Change End
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.

        Returns:
        --------
        - bbox: ndarray
            Bounding box in (x1, y1, x2, y2) format.
        """
        return convert_x_to_bbox(self.kf.x)
    
    #Change Begin
    def update_map_position(self, pixel_mapper, xywh, frame_idx,
                            keypoints=None, keypoint_indices=None, 
                            smoothing_factor=0.5, bbox_top_center_history_len=5,
                            reappearance_threshold=5, dynamic_smoothing_min=0.1, dynamic_smoothing_max=0.3,
                            dynamic_smoothing_thresh=5):
        """
        Use Traditional middle of bounding box if ankle indices are not specified by the use but if they are Proceed with ankle based approach 
        Update the map position of the object with Kalman filtering to predict ankle-based keypoint position 

        Parameters:
        -----------
        - pixel_mapper: object
            Mapper to convert pixel coordinates to map coordinates.
        - xywh: ndarray
            Bounding box in (xc, yc, w, h) format.
        - frame_idx: int, required. Index of the current video frame.
        - keypoints: list or ndarray, optional
            List of keypoints, where each keypoint has (x, y, confidence).
        - keypoint_indices: tuple of int, optional
            Indices of the keypoints to use (e.g., ankles). Should be a tuple of two indices.
        - smoothing_factor: float, optional
            Factor for smoothing the position (0 to 1). Higher values give more weight to the current measurement.
            Default is 0.5.
        - bbox_top_center_history_len: int, optional
            The length of the history deque to store bounding box top_centers for averaging velocity calculation.
            Default is 5 frames.
        - keypoint_confidence_threshold: float, optional
            Minimum confidence for keypoints to be considered valid. Default is 0.5.
        - reappearance_threshold: int, optional
            Number of consecutive frames where keypoints are missing before Kalman filter is re-initialized after reappearance.
            Default is 5 frames.
        - dynamic_smoothing_min: float, optional
            Minimum smoothing factor when calculating dynamic smoothing based on position delta. Default is 0.3.
        - dynamic_smoothing_max: float, optional
            Maximum smoothing factor when calculating dynamic smoothing based on position delta. Default is 0.8.
        - dynamic_smoothing_thresh: float, optional
            Position delta threshold used to calculate dynamic smoothing. Higher values reduce dynamic smoothing sensitivity.
            Default is 50.
        """
        if keypoint_indices is None:
            # Fallback to the bounding box-based position only when keypoints are missing
            self.ankle_based_point = np.array([xywh[0], (xywh[1] + xywh[3] / 2)])
            new_map_pos = pixel_mapper.detection_to_map(xywh)
        else:
            # If keypoints are provided, proceed to keypoint-based calculations
            # Assign current_top_center to top_center of bounding box as a fallback
            current_top_center = np.array([xywh[0], xywh[1] - (xywh[3] / 2)])  # Top_center of bounding box
            # Record bbox top‑centre in history
            self.joint_histories['bbox'].append((frame_idx, current_top_center))

            # Add current_top_center to bbox top_center history (legacy for velocity)
            if not hasattr(self, 'bbox_top_center_history'):
                self.bbox_top_center_history = deque(maxlen=bbox_top_center_history_len)
            self.bbox_top_center_history.append(current_top_center)

            # Check if keypoints are provided and valid
            if keypoints is not None:
                # Compute foot-based center using heel and toe keypoints
                foot_indices_left = [20, 22, 24]
                foot_indices_right = [21, 23, 25]
                left_pts = [
                    keypoints[i][:2]
                    for i in foot_indices_left
                    if i < len(keypoints) and keypoints[i][2] > self.keypoint_confidence_threshold
                ]
                right_pts = [
                    keypoints[i][:2]
                    for i in foot_indices_right
                    if i < len(keypoints) and keypoints[i][2] > self.keypoint_confidence_threshold
                ]
                # ---- Hip & knee joint history updates -------------------
                # Halpe‑26 indices: 12‑LHip, 13‑RHip, 14‑LKnee, 15‑RKnee
                hip_knee_map = {'lhip': 12, 'rhip': 13, 'lknee': 14, 'rknee': 15}
                for name, idx in hip_knee_map.items():
                    if idx < len(keypoints) and keypoints[idx][2] > self.keypoint_confidence_threshold:
                        self.joint_histories[name].append((frame_idx, keypoints[idx][:2]))
                # ---------------------------------------------------------
                if left_pts or right_pts:
                    left_mean = np.mean(left_pts, axis=0) if left_pts else None
                    right_mean = np.mean(right_pts, axis=0) if right_pts else None
                    if left_mean is not None and right_mean is not None:
                        ankle_point = (left_mean + right_mean) / 2
                    elif left_mean is not None:
                        ankle_point = left_mean
                    else:
                        ankle_point = right_mean
                else:
                    # Fallback to ankles
                    kp1 = keypoints[keypoint_indices[0]] if keypoint_indices[0] < len(keypoints) else None
                    kp2 = keypoints[keypoint_indices[1]] if keypoint_indices[1] < len(keypoints) else None
                    valid_ankles = [
                        kp[:2]
                        for kp in (kp1, kp2)
                        if kp is not None and kp[2] > self.keypoint_confidence_threshold
                    ]
                    if valid_ankles:
                        ankle_point = np.mean(valid_ankles, axis=0)
                    else:
                        ankle_point = None

                if ankle_point is not None:
                    # If keypoints were missing before and now reappear, reinitialize Kalman filter
                    if self.missing_keypoints_frames >= reappearance_threshold:
                        if self.keypoint_kalman_filter is None:
                            self.keypoint_kalman_filter = KalmanFilterPoint()  # Initialize Kalman filter if missing
                        self.keypoint_kalman_filter.initiate(ankle_point)  # Re-init Kalman filter
                    # Ensure the Kalman filter is initialized before updating
                    if self.keypoint_kalman_filter is None:
                        self.keypoint_kalman_filter = KalmanFilterPoint()
                        self.keypoint_kalman_filter.initiate(ankle_point)
                    # Update Kalman filter with the ankle keypoints
                    self.keypoint_kalman_filter.update(ankle_point)
                    # Use Kalman filter to predict the ankle-based point position
                    predicted_point = self.keypoint_kalman_filter.predict()
                    # Use the predicted ankle point for mapping
                    new_map_pos = pixel_mapper.pixel_to_map(predicted_point)
                    # Store the ankle-based point for future use
                    self.ankle_based_point = predicted_point
                    # Reset the missing keypoints counter
                    self.missing_keypoints_frames = 0
                else:
                    # If keypoints are missing or not confident, predict the position using keypoint Kalman filter
                    self.missing_keypoints_frames += 1
                    if self.keypoint_kalman_filter is not None:
                        # Select velocity from the highest‑priority continuous joint
                        priority = ['lhip', 'rhip', 'lknee', 'rknee', 'bbox']
                        vel_candidate = None
                        for name in priority:
                            vel_candidate = self._continuous_velocity(name, k=5,
                                                                      current_frame=frame_idx,
                                                                      max_gap=3)
                            if vel_candidate is not None:
                                break

                        if vel_candidate is not None:
                            velocity_x, velocity_y = vel_candidate
                            # Inject into Kalman state
                            self.keypoint_kalman_filter.state[2:] = np.array([velocity_x, velocity_y]).reshape(2, 1)
                        # Predict the ankle point based on the adjusted velocity
                        predicted_point = self.keypoint_kalman_filter.predict()
                        # Map the predicted ankle point
                        if predicted_point is not None:
                            new_map_pos = pixel_mapper.pixel_to_map(predicted_point)
                            self.ankle_based_point = predicted_point
                        else:
                            new_map_pos = None  # Kalman filter fails or isn't ready
                    else:
                        new_map_pos = None  # No Kalman filter and no valid keypoints
            self.prev_top_center = current_top_center  # Use top_center of the bounding box for future velocity
        self.previous_bbox = xywh

        # Dynamic smoothing adjustment based on position delta
        if 'new_map_pos' in locals() and new_map_pos is not None and self.current_map_pos is not None:
            # Calculate the change in position
            position_delta = np.linalg.norm(self.current_map_pos - new_map_pos)
            # Set dynamic smoothing factor based on how much movement is occurring
            dynamic_smoothing = max(dynamic_smoothing_min, min(dynamic_smoothing_max, 1.0 - (position_delta / dynamic_smoothing_thresh)))
        else:
            dynamic_smoothing = smoothing_factor

        # Apply smoothing to the new position
        if self.current_map_pos is not None:
            self.last_map_pos = self.current_map_pos
            if 'new_map_pos' in locals() and new_map_pos is not None:
                self.current_map_pos = (
                    dynamic_smoothing * new_map_pos +
                    (1 - dynamic_smoothing) * self.current_map_pos
                )
            else:
                # If no valid new_map_pos, keep using the last known position
                self.current_map_pos = self.current_map_pos
        else:
            if 'new_map_pos' in locals():
                self.current_map_pos = new_map_pos
    # ------------------------------------------------------------------
    def _continuous_velocity(self, name: str, k: int = 3, current_frame: Optional[int] = None, max_gap: int = 1):
        """
        Return a velocity vector (vx, vy) for the given joint name if we have
        at least `k` *consecutive* frame entries in `self.joint_histories[name]`,
        and the last stored frame is no more than `max_gap` away from the *current_frame* (if provided).
        Otherwise returns None.
        """
        hist = self.joint_histories.get(name, None)
        if hist is None or len(hist) < k:
            return None
        # Take the last k entries
        sub = list(hist)[-k:]
        frames = [f for f, _ in sub]
        # Require strictly consecutive frame indices
        if frames != list(range(frames[0], frames[0] + k)):
            return None
        # If recency is required make sure we are close to current frame
        if current_frame is not None and (current_frame - frames[-1] > max_gap):
            return None
        p0 = sub[0][1]
        p1 = sub[-1][1]
        dt = frames[-1] - frames[0]
        if dt == 0:
            return None
        return (p1 - p0) / dt
    # ------------------------------------------------------------------


    def calculate_velocity(self):
        vel = [None, None]
        if self.current_map_pos is not None and self.last_map_pos is not None:
            vel = self.current_map_pos - self.last_map_pos
            vel = normalize(vel)
        return vel
    

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm
#Change End 


class OCSort(BaseTracker):
    """
    OC-SORT Tracker for Video Analysis

    This tracker integrates advanced techniques to handle non-linear motion and occlusion, 
    enhancing the basic SORT tracker for robust and accurate tracking.

    Literature Summary:
    -------------------
    OC-SORT is an advanced multi-object tracking algorithm that enhances the basic SORT tracker 
    by addressing its limitations in handling non-linear motion and occlusion. OC-SORT remains 
    simple, online, and real-time, achieving state-of-the-art tracking performance.

    ### Technical README for OC-SORT Algorithm

    This document provides a concise technical overview of the OC-SORT (Observation-Centric SORT) algorithm, 
    detailing its components and their functionalities for multi-object tracking (MOT) in video streams.

    #### Overview
    OC-SORT is an advanced multi-object tracking algorithm that enhances the basic SORT tracker by addressing 
    its limitations in handling non-linear motion and occlusion. OC-SORT remains simple, online, and real-time, 
    achieving state-of-the-art tracking performance.

    ### 1. Introduction to OC-SORT

    - **Multi-Object Tracking (MOT):** The objective is to detect and track all objects in a scene while maintaining unique identifiers.
    - **Tracking-by-Detection Paradigm:** Utilizes object detection followed by a tracking step.
    - **Challenges Addressed:** Reduces error accumulation during occlusion and improves tracking robustness for non-linear motion.

    ### 2. Algorithm Components

    #### 2.1 Kalman Filter (KF)
    - **Motion Model:** Uses a Kalman filter with a constant-velocity model to predict the trajectory of objects.
    - **State Vector:** Includes position, velocity, scale, and aspect ratio to enhance trajectory predictions.

    #### 2.2 Observation-Centric Re-Update (ORU)
    - **Purpose:** Reduces error accumulation during occlusion by using virtual observations.
    - **Method:** Generates virtual trajectories between the last-seen observation before occlusion and the latest observation after re-association, correcting the accumulated error in KF parameters.

    #### 2.3 Observation-Centric Momentum (OCM)
    - **Purpose:** Incorporates motion direction consistency into the association stage.
    - **Method:** Adds a cost term based on the consistency of motion direction, calculated from observations instead of state estimations, to the association cost matrix.

    ### 3. Algorithm Workflow

    1. **Initialization:**
       - Initialize tracklets and set the detection score threshold.

    2. **Detection and Classification:**
       - Detect objects in each frame and classify detections based on the score threshold.

    3. **Kalman Filter Prediction:**
       - Predict new locations for tracklets using the Kalman filter.

    4. **Association with OCM:**
       - Compute similarity (IoU and direction consistency) between detections and tracklets.
       - Use the Hungarian algorithm for matching and update tracklets.

    5. **Re-Update with ORU:**
       - For tracklets re-associated after occlusion, apply ORU to correct accumulated errors.

    6. **Tracklet Update:**
       - Update matched tracklets with new positions and appearance features.
       - Remove unmatched tracklets if they remain unmatched for a certain number of frames.

    7. **Tracklet Initialization:**
       - Initialize new tracklets for unmatched high-confidence detections.

    ### 4. Performance and Benchmarks

    - **Datasets:** Evaluated on MOT17, MOT20, KITTI, and DanceTrack datasets.
    - **Metrics:** Uses HOTA, IDF1, MOTA, AssA, DetA, and FPS to evaluate tracking performance.
    - **Results:** Achieves state-of-the-art performance with significant improvements in HOTA, AssA, and IDF1 scores compared to previous methods.

    ### 5. Implementation Details

    - **Framework:** Implemented in PyTorch.
    - **Hardware:** Tested on systems with Intel i9 CPUs and NVIDIA GPUs.
    - **Source Code:** Available at [OC-SORT GitHub Repository](https://github.com/noahcao/OC_SORT).

    ### 6. Limitations and Future Work

    - **Non-linear Motion:** While improved, performance may still degrade in highly non-linear motion scenarios.
    - **Detection Quality:** Relies on high-quality detections; performance may degrade with noisy detections.
    - **Real-time Performance:** While real-time, further optimizations can be achieved with hardware accelerations and multi-threading.

    ### References

    - OC-SORT Paper: [arXiv:2203.14360v3](https://arxiv.org/abs/2203.14360v3)
    - MOTChallenge: [MOT17 Dataset](https://motchallenge.net/data/MOT17/), [MOT20 Dataset](https://motchallenge.net/data/MOT20/)


    Initialization:
    ---------------
    To initialize the BYTETracker with default values, you can use the following code:

    ```python
    tracker = OCSORT(
        per_class=False,         
        det_thresh=0.2,          
        max_age=30,               
        min_hits=5,               
        asso_threshold=0.3,       
        delta_t=3,               
        asso_func="iou",         
        inertia=0.2,             
        use_byte=True,           
        pixel_mapper=mapper,        
        limit_entry=False,        
        entry_polys=entry_polys          
    )         
    ```

    """
    # Special ID reserved for a single pre‑existing enemy already inside the room
    ENEMY_FINAL_ID = 99

    def __init__(
        self,
        per_class=False,
        det_thresh=0.2,
        max_age=30,
        min_hits=3,
        asso_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
        #Change Begin
        pixel_mapper=None,
        limit_entry=False,
        entry_polys=None,
        class_id_to_label: dict = None,
        entry_window_time=None,
        boundary=None,
        boundary_pad_pct=0.0,
        track_enemy=True,
        # Change End    
    ):
        """
        OC-SORT Tracker for Video Analysis

        This tracker integrates advanced techniques to handle non-linear motion and occlusion, 
        enhancing the basic SORT tracker for robust and accurate tracking.

        Parameters:
        -----------
        - per_class: bool, optional
            Whether to track objects on a per-class basis. Default is False.
            If True, the tracker will maintain separate trackers for each class of object.
        - det_thresh: float, optional
            Detection confidence threshold. Default is 0.2.
            Detections with confidence scores below this threshold will not be considered for tracking.
        - max_age: int, optional
            Maximum number of frames to keep a track alive without updates. Default is 30.
            Tracks that have not been updated for more than `max_age` frames will be deleted.
        - min_hits: int, optional
            Minimum number of hits before a track is considered confirmed. Default is 3.
            Tracks will be considered tentative until they have been updated at least `min_hits` times.
        - asso_threshold: float, optional
            Association threshold for matching. Default is 0.3.
            Detections and tracks will be associated if their IoU or other matching score exceeds this threshold.
        - delta_t: int, optional
            Time step for velocity calculation. Default is 3.
            Used to calculate the velocity of objects between detections.
        - asso_func: str, optional
            Association function for matching. Default is "iou".
            Specifies the function used to associate detections with existing tracks. Options include "iou", "giou", etc.
        - inertia: float, optional
            Inertia term for association. Default is 0.2.
            Controls the smoothness of motion predictions.
        - use_byte: bool, optional
            Whether to use BYTE association. Default is False.
            If True, a second round of association using BYTE will be performed for low-confidence detections.
        - pixel_mapper: object, optional
            Pixel mapper for converting pixel coordinates to map coordinates.
            If provided, the tracker will update and use map coordinates for tracking.
        - limit_entry: bool, optional
            Whether to limit track creation to specified entry zones. Default is False.
            If True, new tracks will only be created if detections fall within specified entry zones.
        - entry_polys: list of polygons, optional
            List of polygons defining entry zones.
            New tracks will only be created if detections fall within these polygons. Required if `limit_entry` is True.
        - class_id_to_label: dict, optional
            Dictionary mapping class IDs to class labels.
            Provides human-readable labels for class IDs.

        Attributes:
        -----------
        - per_class: bool
            Indicates whether the tracker is operating on a per-class basis.
        - max_age: int
            Maximum number of frames a track is kept alive without updates.
        - min_hits: int
            Minimum number of hits required for a track to be confirmed.
        - asso_threshold: float
            Threshold for associating detections with existing tracks.
        - frame_count: int
            Counter for the number of frames processed.
        - det_thresh: float
            Detection confidence threshold.
        - delta_t: int
            Time step used for velocity calculations.
        - asso_func: function
            Function used for associating detections with tracks.
        - inertia: float
            Inertia term used in the association process.
        - use_byte: bool
            Indicates whether BYTE association is used for second-round matching.
        - pixel_mapper: object, optional
            Mapper for converting pixel coordinates to map coordinates.
        - limit_entry: bool
            Indicates whether track creation is limited to specified entry zones.
        - entry_polys: list of polygons, optional
            Polygons defining entry zones for track creation.
        - next_final_id: int
            Counter for assigning unique final IDs to tracks.
        - active_tracks: list of KalmanBoxTracker
            List of currently active tracks.


        Methods:
        --------
        - __init__(self, per_class=False, det_thresh=0.2, max_age=30, min_hits=3, asso_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False, pixel_mapper=None, limit_entry=False, entry_polys=None, class_id_to_label=None)
            Initializes the tracker with specific parameters.
        - update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray=None) -> np.ndarray
            Updates the tracker with new detections and returns the current tracks.

        Output Structure from `update` Method:
        --------------------------------------
        The `update` method returns a list of dictionaries, each representing a track with the following keys:
        - 'top_left_x': Top-left x-coordinate of the bounding box.
        - 'top_left_y': Top-left y-coordinate of the bounding box.
        - 'width': Width of the bounding box.
        - 'height': Height of the bounding box.
        - 'track_id': Final track ID.
        - 'track_id_raw': Raw ID of the track.
        - 'confidence': Confidence score of the detection.
        - 'class': Class ID of the object.
        - 'current_map_pos': Current map position of the object (if `pixel_mapper` is provided).
        - 'map_velocity': Velocity of the object on the map (if `pixel_mapper` is provided).
        - 'detection_index': Index of the detection.
        """
        #Change Begin
        super().__init__(max_age=max_age, class_id_to_label=class_id_to_label)
        #Change End
        """
        Sets key parameters for SORT
        """
        self.per_class = per_class
        self.max_age = max_age
        self.min_hits = min_hits
        self.asso_threshold = asso_threshold
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0
        #Change Begin
        self.pixel_mapper = pixel_mapper
        self.limit_entry = limit_entry
        self.entry_polys = entry_polys
        self.next_final_id = 1  # To track the next final ID to assign
        if self.limit_entry:
            if not self.entry_polys:
                raise ValueError("entry_polys cannot be None if limit_entry is True")
            if self.pixel_mapper is None:
                raise ValueError("pixel_mapper cannot be None if limit_entry is True")
            
        self.entry_window_time = entry_window_time  # Duration of the entry window in seconds/frames
        self.entry_window_counter = 0  # Counter to track the entry window elapsed time
        self.entry_window_active = False  # Flag to indicate if the entry window is active

        # ---- Boundary padding & enemy‑track state ---------------------------------
        self.boundary = boundary                       # original room polygon
        if self.boundary is not None:
            # percentage padding → scaled polygon about centre
            self.boundary_padded = (
                scale(self.boundary, xfact=1 + boundary_pad_pct,
                      yfact=1 + boundary_pad_pct, origin='center')
                if boundary_pad_pct else self.boundary
            )
        else:
            self.boundary_padded = None

        # Enemy tracking flags
        self.enemy_active = False   # True while the enemy track exists
        self.enemy_done   = False   # Set True once enemy leaves; prevents re‑creation
        self.enemy_was_in_entry = False  # Becomes True once the enemy steps into any entry polygon
        # Global switch to enable / disable enemy‑tracking feature
        self.track_enemy = track_enemy

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, keypoints=None, keypoint_confidence_threshold=0.5, keypoint_indices=None) -> np.ndarray:
        """
        Update the tracker with new detections.

        Parameters:
        -----------
        - dets: np.ndarray
            Array of detections in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...].
        - img: np.ndarray
            Image corresponding to the detections.
        - embs: np.ndarray, optional
            Array of embeddings for appearance features.
        - keypoints: ndarray, optional
            The array of keypoints output from yolo in the data key it returns which contains a,y,conf
        - keypoint_confidence_threshold: float
            The minimum confidence of a keypoint to be associated with a track 

        Returns:
        --------
        - outputs: np.ndarray
            Array of tracking results, each with additional object ID information.
        """

        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1

        #Change Begin
        if self.limit_entry and self.entry_window_active:
            self.entry_window_counter += 1
        #Change End    

        h, w = img.shape[0:2]

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        inds_low = confs > 0.1
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(
            inds_low, inds_high
        )  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        
        remain_inds = confs > self.det_thresh
        dets = dets[remain_inds]
        # Assuming 'keypoints' is an array where each row corresponds to keypoints for a detection in 'dets'
        keypoints_first = keypoints[remain_inds]
        keypoints_second = keypoints[inds_second]

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.active_tracks), 5))
        to_del = []
        #change Begin
        outputs = []
        #change End
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.active_tracks
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.active_tracks
            ]
        )

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5], trks, self.asso_func, self.asso_threshold, velocities, k_observations, self.inertia, w, h
        )
        for m in matched:
            self.active_tracks[m[1]].update(dets[m[0], :5], dets[m[0], 5], dets[m[0], 6], keypoints=(keypoints_first[m[0]] if keypoints_first is not None else None))

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(
                dets_second, u_trks
            )  # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.asso_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(
                        dets_second[det_ind, :5], dets_second[det_ind, 5], dets_second[det_ind, 6], keypoints=(keypoints_second[m[0]] if keypoints_second is not None else None)
                    )
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = run_asso_func(self.asso_func, left_dets, left_trks, w, h)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.asso_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(dets[det_ind, :5], dets[det_ind, 5], dets[det_ind, 6], keypoints=(keypoints_first[det_ind] if keypoints_first is not None else None))
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None)

        # Change Begin
        # ------------------------------------------------------------------
        # create and initialise new trackers for unmatched detections
        # ------------------------------------------------------------------
        if unmatched_dets.shape[0] > 0:
            for i in unmatched_dets:
                # Convert current detection’s centre to map coordinates (if mapper available)
                map_point = None
                if self.pixel_mapper is not None:
                    map_point = geo.Point(*self.pixel_mapper.detection_to_map(xyxy2xywh(dets[i, 0:4])))

                # -------- Pre‑existing enemy handling ---------------------------------
                if self.track_enemy and (not self.enemy_active) and (not self.enemy_done) and \
                   (map_point is not None) and (self.boundary_padded is not None) and \
                   self.boundary_padded.contains(map_point):
                    # Create a dedicated enemy track with the reserved ID
                    trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], dets[i, 6],
                                           delta_t=self.delta_t, max_obs=self.max_obs)
                    trk.final_id = self.ENEMY_FINAL_ID
                    self.active_tracks.append(trk)
                    self.enemy_active = True
                    # IMPORTANT: do *not* touch entry‑window timers for the enemy
                    continue
                # ----------------------------------------------------------------------

                create = False
                if self.limit_entry:
                    # Check if the entry window is active and if it has expired
                    if self.entry_window_active:
                        if self.entry_window_time is not None and self.entry_window_counter >= self.entry_window_time:
                            continue  # Skip creating new tracks as the entry window has expired

                    # map_point already computed above; only recompute if absent
                    if map_point is None and self.pixel_mapper is not None:
                        map_point = geo.Point(*self.pixel_mapper.detection_to_map(xyxy2xywh(dets[i, 0:4])))
                    for poly in self.entry_polys:
                        if poly.contains(map_point):
                            create = True
                            break

                    # Start the entry window counter only on the first confirmed entry
                    if create and not self.entry_window_active:
                        self.entry_window_active = True
                        self.entry_window_counter = 0  # Start counting from the first entry
                else:
                    create = True

                if create:
                    trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], dets[i, 6], delta_t=self.delta_t, max_obs=self.max_obs)
                    self.active_tracks.append(trk)
           
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                tlwh = xyxy2tlwh(trk.get_state()[0])# Convert to tlwh format
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                tlwh = xyxy2tlwh(trk.last_observation)
            #if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits):  
                  
                if trk.final_id is None:  # Assign a final ID if not already assigned
                    trk.final_id = self.next_final_id
                    self.next_final_id += 1

                # Create dictionary for each track
                track_dict = {
                    'top_left_x': tlwh[0],
                    'top_left_y': tlwh[1],
                    'width': tlwh[2],
                    'height': tlwh[3],
                    'track_id':trk.final_id,
                    'track_id_raw': trk.id + 1,
                    'confidence': trk.conf,
                    'class': trk.cls,
                    'detection_index': trk.det_ind,
                    'keypoints': trk.filtered_keypoints,
                    'ankle_based_point': trk.ankle_based_point
                }


                if self.pixel_mapper is not None:
                    # Update map position and calculate velocity
                    trk.update_map_position(self.pixel_mapper,
                                           xyxy2xywh(trk.get_state()[0]),
                                           self.frame_count,
                                           keypoints=trk.filtered_keypoints,
                                           keypoint_indices=keypoint_indices,
                                           smoothing_factor=0.7)
                    map_vel = trk.calculate_velocity()
                    # Add map position and velocity to dictionary
                    track_dict['current_map_pos'] = trk.current_map_pos
                    track_dict['map_velocity'] = map_vel
                else:
                    track_dict['current_map_pos'] = None
                    track_dict['map_velocity'] = [None, None]

                outputs.append(track_dict)

            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)
        
        # ------------------------------------------------------------------
        # Enemy exit logic – only remove if it left via a doorway
        # ------------------------------------------------------------------
        if self.track_enemy and self.enemy_active:
            enemy_trk = next((t for t in self.active_tracks
                            if getattr(t, "final_id", None) == self.ENEMY_FINAL_ID), None)

            # Track disappeared unexpectedly (aged out etc.)
            if enemy_trk is None:
                self.enemy_active = False
                self.enemy_done   = True
                self.enemy_was_in_entry = False
            else:
                if enemy_trk.current_map_pos is not None and self.boundary is not None:
                    pt = geo.Point(*enemy_trk.current_map_pos)

                    # If the enemy touches any entry polygon, remember it
                    if not self.enemy_was_in_entry and self.entry_polys:
                        self.enemy_was_in_entry = any(poly.contains(pt) for poly in self.entry_polys)

                    # Only declare “enemy left” when it was in entry once *and* is now outside boundary
                    if self.enemy_was_in_entry and not self.boundary.contains(pt):
                        self.active_tracks.remove(enemy_trk)
                        self.enemy_active = False
                        self.enemy_done   = True
                        self.enemy_was_in_entry = False

        return np.array(outputs) if len(outputs) > 0 else np.array([])
        #Change End
