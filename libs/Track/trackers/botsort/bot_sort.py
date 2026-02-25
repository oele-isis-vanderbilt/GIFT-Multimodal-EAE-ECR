# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))

import numpy as np
from collections import deque

from ...appearance.reid_auto_backend import ReidAutoBackend
from ...motion.cmc.sof import SOF
from ...motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from .basetrack import BaseTrack, TrackState
from ...utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
from ...utils.ops import xywh2xyxy, xyxy2xywh, xyxy2tlwh
from ..basetracker import BaseTracker
from ...utils import PerClassDecorator
#Change Begin
import shapely.geometry as geo

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
#Change End

class STrack(BaseTrack):
    """
    STrack class for tracking individual objects.
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, det, feat=None, feat_history=50, max_obs=50, keypoints=None, keypoint_confidence_threshold=0.5):
        """
        Initialize an STrack instance.

        Parameters:
        -----------
        - det: ndarray
            Detection bounding box in (x1, y1, x2, y2, conf, class, det_ind) format.
        - feat: ndarray, optional
            Appearance feature of the detection.
        - feat_history: int, optional
            Number of past features to store.
        - max_obs: int, optional
            Maximum number of observations to store.
        - keypoints: ndarray, optional
            The array of keypoints output from yolo in the data key it returns which contains a,y,conf
        - keypoint_confidence_threshold: float
            The minimum confidence of a keypoint to be associated with a track 


         Attributes:
        -----------
        - xywh: ndarray
            Bounding box in (xc, yc, w, h) format.
        - conf: float
            Confidence score of the detection.
        - cls: int
            Class ID of the object.
        - det_ind: int
            Detection index.
        - max_obs: int
            Maximum number of observations to store.
        - kalman_filter: KalmanFilterXYWH
            Kalman filter for motion prediction.
        - mean: ndarray
            Mean state vector of the Kalman filter.
        - covariance: ndarray
            Covariance matrix of the Kalman filter.
        - is_activated: bool
            Whether the track is activated.
        - cls_hist: list
            History of class IDs.
        - history_observations: deque
            History of observations.
        - tracklet_len: int
            Length of the tracklet.
        - smooth_feat: ndarray
            Smoothed appearance feature.
        - curr_feat: ndarray
            Current appearance feature.
        - features: deque
            Queue of appearance features.
        - alpha: float
            Smoothing factor for features.
        - current_map_pos: ndarray
            Current map position.
        - last_map_pos: ndarray
            Last map position.
        - hit_streak: int
            Number of continuous frames a detection is present.
        - final_id: int
            Final track ID for uniformity in output.
        """
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs=max_obs
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.conf)
        self.history_observations = deque([], maxlen=self.max_obs)

        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        #Change Begin
        self.current_map_pos = None
        self.last_map_pos = None
        self.hit_streak = 0  # Track the number of continuous frames a detection is present
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

    def update_features(self, feat):
        """
        Update appearance features of the track.

        Parameters:
        -----------
        - feat: ndarray
            New appearance feature to update.
        """
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, conf):
        """
        Update class history with new detection.

        Parameters:
        -----------
        - cls: int
            Class ID of the detection.
        - conf: float
            Confidence score of the detection.
        """
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += conf
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, conf])
                self.cls = cls
        else:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    def predict(self):
        """
        Predict the next state of the track using Kalman filter.
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        """
        Predict the next state for multiple tracks using Kalman filter.

        Parameters:
        -----------
        - stracks: list
            List of STrack instances.
        """
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """
        Apply global motion compensation to multiple tracks.

        Parameters:
        -----------
        - stracks: list
            List of STrack instances.
        - H: ndarray, optional
            Homography matrix for motion compensation.
        """
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        Start a new tracklet.

        Parameters:
        -----------
        - kalman_filter: KalmanFilterXYWH
            Kalman filter for motion prediction.
        - frame_id: int
            ID of the current frame.
        """
        self.kalman_filter = kalman_filter
        self.id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        #Change Begin
        self.hit_streak = 0  # Reset hit streak on activation
        #Change End

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        Re-activate an existing track with new detection.

        Parameters:
        -----------
        - new_track: STrack
            New track to re-activate.
        - frame_id: int
            ID of the current frame.
        - new_id: bool, optional
            Whether to assign a new ID to the track.
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        #Change Begin
        self.hit_streak = 0  # Reset hit streak on reactivation
        # Update keypoints
        self.keypoints = new_track.keypoints
        self.filtered_keypoints = self.filter_keypoints(new_track.keypoints, self.keypoint_confidence_threshold)
        #Change End

        self.update_cls(new_track.cls, new_track.conf)

    def update(self, new_track, frame_id):
        """
        Update a matched track.

        Parameters:
        -----------
        - new_track: STrack
            New track to update.
        - frame_id: int
            ID of the current frame.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        #Change Begin
        self.hit_streak += 1  # Increment hit streak on update
        # Update keypoints
        self.keypoints = new_track.keypoints
        self.filtered_keypoints = self.filter_keypoints(new_track.keypoints, self.keypoint_confidence_threshold)
        #Change End
        self.update_cls(new_track.cls, new_track.conf)

    @property
    def xyxy(self):
        """
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.

        Returns:
        --------
        - ret: ndarray
            Bounding box in (min x, min y, max x, max y) format.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret
    
    #Change Begin
    def calculate_velocity(self):
        """
        Calculate velocity of the object.

        Returns:
        --------
        - vel: list
            Velocity in the form [vx, vy].
        """
        vel = [None, None]
        if self.current_map_pos is not None and self.last_map_pos is not None:
            vel = self.current_map_pos - self.last_map_pos
            vel = normalize(vel)
        return vel
    
    def calculate_top_center_velocity(self, current_top_center):
        """
        Calculate the velocity based on the top_center of the bounding box.
        
        Parameters:
        -----------
        - current_top_center: tuple
            The current top_center point (x, y) of the bounding box.
        
        Returns:
        --------
        - velocity: tuple
            Velocity (vx, vy) of the top_center.
        """
        if self.prev_top_center is None:
            velocity = (0, 0)  # No previous data, assume no movement
        else:
            prev_x, prev_y = self.prev_top_center
            curr_x, curr_y = current_top_center
            velocity = (curr_x - prev_x, curr_y - prev_y)  # Change in position
        return velocity


    def update_map_position(self, pixel_mapper, xywh, keypoints=None, keypoint_indices=None, 
                        smoothing_factor=0.5, bbox_top_center_history_len=5,
                        reappearance_threshold=5, dynamic_smoothing_min=0.3, dynamic_smoothing_max=0.6,
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
        
        Returns:
        --------
        - use_bounding_box: bool
            Indicator if bounding box fallback was used due to missing or low-confidence keypoints.
        
        Functionality Overview:
        -----------------------
        1. **Keypoint Detection**: 
        - Attempts to use keypoints for position updates. If valid keypoints are present, uses them to update the Kalman filter. If keypoints are missing or of low confidence, falls back to bounding box-based tracking.
        
        2. **Bounding Box Velocity Calculation**:
        - Stores bounding box top_centers over the last `bbox_top_center_history_len` frames to calculate average velocity. This velocity is used when keypoints are missing.
        
        3. **Kalman Filter Prediction**:
        - Uses the Kalman filter to predict future positions based on keypoints or bounding box data. If keypoints reappear after being missing for more than `reappearance_threshold` frames, the Kalman filter is reinitialized for a quick recovery.
        
        4. **Dynamic Smoothing**:
        - Adjusts the smoothing factor based on the distance between the current and new position (`position_delta`). The smoothing factor varies between `dynamic_smoothing_min` and `dynamic_smoothing_max`, with sensitivity controlled by `dynamic_smoothing_thresh`.
        
        5. **Error Handling**:
        - Ensures the Kalman filter is always initialized before attempting updates, preventing errors when keypoints are missing.

        Why Use These Functionalities:
        ------------------------------
        - **Keypoint Detection**: Ensures we use more reliable ankle points for precise tracking when available.
        - **Bounding Box Velocity**: Fallback when keypoints are unavailable, ensuring consistent tracking using bounding box-based movement.
        - **Kalman Filter Prediction**: Smoothly handles tracking over time, ensuring predictions account for velocity and direction.
        - **Dynamic Smoothing**: Provides flexibility in tracking stability by adjusting the smoothness based on movement. Helps avoid jittering in fast-moving or slow-moving objects.
        - **Error Handling**: Prevents crashes or unexpected behavior when keypoints are missing or reappear after occlusion.
        """
        
        
        # Only calculate bounding box-based movement when keypoint_indices is None
        if keypoint_indices is None:
            # Fallback to the bounding box-based position only when keypoints are missing
            self.ankle_based_point = np.array([xywh[0],(xywh[1]+xywh[3]/2)])
            new_map_pos = pixel_mapper.detection_to_map(xywh)
        else:
            # If keypoints are provided, proceed to keypoint-based calculations
            # Assign current_top_center to top_center of bounding box as a fallback
            current_top_center = np.array([xywh[0], xywh[1]-(xywh[3]/2)])  # Top_center of bounding box

            # Add current_top_center to bbox top_center history
            if not hasattr(self, 'bbox_top_center_history'):
                self.bbox_top_center_history = deque(maxlen=bbox_top_center_history_len)
            
            self.bbox_top_center_history.append(current_top_center)

            # Check if keypoints are provided and valid
            if keypoints is not None:
                # Extract the keypoints based on the provided indices
                kp1 = keypoints[keypoint_indices[0]] if len(keypoints) > keypoint_indices[0] else None
                kp2 = keypoints[keypoint_indices[1]] if len(keypoints) > keypoint_indices[1] else None

                # Check if both keypoints have sufficient confidence
                if kp1 is not None and kp2 is not None and kp1[2] > self.keypoint_confidence_threshold and kp2[2] > self.keypoint_confidence_threshold:
                    # Compute the average ankle-based point position
                    point_x = (kp1[0] + kp2[0]) / 2
                    point_y = (kp1[1] + kp2[1]) / 2
                    ankle_point = np.array([point_x, point_y])

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
                        predicted_point = self.keypoint_kalman_filter.predict()

                        # Calculate average velocity from bounding box history
                        if len(self.bbox_top_center_history) > 1:
                            avg_velocity_x = np.mean([self.bbox_top_center_history[i][0] - self.bbox_top_center_history[i - 1][0]
                                                    for i in range(1, len(self.bbox_top_center_history))])
                            avg_velocity_y = np.mean([self.bbox_top_center_history[i][1] - self.bbox_top_center_history[i - 1][1]
                                                    for i in range(1, len(self.bbox_top_center_history))])
                        else:
                            avg_velocity_x, avg_velocity_y = 0, 0

                        # Blend Kalman velocity with averaged bounding box velocity
                        velocity_x = 0.5 * self.keypoint_kalman_filter.state[2] + 0.5 * avg_velocity_x
                        velocity_y = 0.5 * self.keypoint_kalman_filter.state[3] + 0.5 * avg_velocity_y

                        # Update Kalman filter velocity using blended velocity
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
            # Store the current top_center point for future velocity calculation
            self.prev_top_center = current_top_center  # Use top_center of the bounding box for future velocity

        # Dynamic smoothing adjustment based on position delta
        if new_map_pos is not None and self.current_map_pos is not None:
            # Calculate the change in position
            position_delta = np.linalg.norm(self.current_map_pos - new_map_pos)
            # Set dynamic smoothing factor based on how much movement is occurring
            dynamic_smoothing = max(dynamic_smoothing_min, min(dynamic_smoothing_max, 1.0 - (position_delta / dynamic_smoothing_thresh)))
        else:
            dynamic_smoothing = smoothing_factor

        # Apply smoothing to the new position
        if self.current_map_pos is not None:
            self.last_map_pos = self.current_map_pos
            if new_map_pos is not None:
                self.current_map_pos = (
                    dynamic_smoothing * new_map_pos +
                    (1 - dynamic_smoothing) * self.current_map_pos
                )
            else:
                # If no valid new_map_pos, keep using the last known position
                self.current_map_pos = self.current_map_pos
        else:
            self.current_map_pos = new_map_pos

        

def normalize(v):
    """
    Normalize a vector.

    Parameters:
    -----------
    - v: ndarray
        Vector to normalize.

    Returns:
    --------
    - v: ndarray
        Normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm
#Change End 


class BoTSORT(BaseTracker):
    """
    BoTSORT Tracker for Video Analysis

    This tracker integrates appearance and motion cues for robust tracking.
    Extended from Mikel Broström's work on boxmot.

    Literature Summary:
    -------------------
    The BoTSORT (Bag-of-Tricks SORT) tracker leverages a combination of motion and appearance features to achieve robust and accurate tracking. By using a Kalman filter for motion prediction and a deep learning model for re-identification, the tracker can handle various challenges in multi-object tracking such as camera motion, detection accuracy, and identity consistency. BoTSORT incorporates camera-motion compensation, an improved Kalman filter, and a novel IoU-ReID fusion method.

    ### Technical README for BoT-SORT Algorithm

    This document provides an overview of the BoT-SORT algorithm, its components, and their functionalities, designed for multi-object tracking in video streams.

    #### Overview
    BoT-SORT (Bag-of-Tricks SORT) and BoT-SORT-ReID are state-of-the-art multi-object tracking algorithms that combine motion and appearance information to robustly track multiple pedestrians in various scenarios. These algorithms address limitations in existing SORT-like algorithms by incorporating camera-motion compensation, an improved Kalman filter, and a novel IoU-ReID fusion method.

    ### 1. Introduction to BoT-SORT

    - **Multi-Object Tracking (MOT):** Detects and tracks all objects in a scene while maintaining unique identifiers.
    - **Tracking-by-Detection Paradigm:** Uses object detection followed by a tracking step.
    - **Challenges Addressed:** Improves bounding box prediction accuracy, handles camera motion, and balances detection performance and identity consistency.

    ### 2. Algorithm Components

    #### 2.1 Kalman Filter (KF)
    - **Motion Model:** Uses a discrete Kalman filter with a constant-velocity model to predict object trajectories.
    - **State Vector:** Enhanced state vector \[x, y, width, height, x-velocity, y-velocity, width-velocity, height-velocity\] improves bounding box accuracy.
    - **Dynamic Noise Covariances:** Process noise covariance \( Q \) and measurement noise covariance \( R \) are updated dynamically based on frame content.

    #### 2.2 Camera Motion Compensation (CMC)
    - **Purpose:** Corrects the predicted bounding box locations affected by camera motion.
    - **Method:** Uses global image registration techniques to estimate camera motion and transform the bounding boxes accordingly.
    - **Affine Transformation:** Keypoints are extracted, and affine transformation matrices are computed to align frames.

    #### 2.3 IoU-ReID Fusion
    - **Appearance Features:** Extracted using a ResNeSt50 backbone model for robust Re-ID features.
    - **Fusion Method:** Combines IoU and cosine distance between embeddings. The minimum distance of the two is used to improve tracking accuracy.
    - **Thresholds:** Utilizes thresholds for IoU and appearance similarity to filter unlikely matches.

    ### 3. Algorithm Workflow

    1. **Initialization:**
       - Initialize tracklets and appearance features.

    2. **Detection and Feature Extraction:**
       - Detect objects in each frame using a pre-trained detector (e.g., YOLOX).
       - Extract high-confidence detections and compute their appearance features.

    3. **Kalman Filter Prediction:**
       - Predict new locations for tracklets using the Kalman filter.
       - Apply camera motion compensation to correct predictions.

    4. **First Association (High-confidence Detections):**
       - Compute IoU and appearance distances.
       - Use the Hungarian algorithm to match high-confidence detections with predicted tracklets.

    5. **Second Association (Low-confidence Detections):**
       - Match remaining low-confidence detections using IoU.

    6. **Tracklet Update:**
       - Update matched tracklets with new positions and appearance features.
       - Remove unmatched tracklets after a certain number of frames.

    7. **New Tracklet Initialization:**
       - Initialize new tracklets for unmatched high-confidence detections.

    8. **Post-processing (Optional):**
       - Apply linear interpolation to fill gaps in tracklets for smoother tracking.

    ### 4. Performance and Benchmarks

    - **Datasets:** Evaluated on MOT17 and MOT20 datasets.
    - **Metrics:** Measures include MOTA (Multiple-Object Tracking Accuracy), IDF1, HOTA, False Positives (FP), False Negatives (FN), and ID Switches (IDSW).
    - **Results:** BoT-SORT and BoT-SORT-ReID achieve top performance in terms of MOTA, IDF1, and HOTA, outperforming other state-of-the-art trackers.

    ### 5. Implementation Details

    - **Framework:** Implemented in PyTorch.
    - **Hardware:** Tested on systems with Intel i9 CPUs and NVIDIA RTX 3060 GPUs.
    - **Source Code:** Available at [GitHub Repository](https://github.com/NirAharon/BOT-SORT).

    ### 6. Limitations and Future Work

    - **High-Density Scenarios:** Camera motion estimation may fail in highly crowded scenes.
    - **Run-time Performance:** Camera motion compensation can be computationally intensive but can be optimized with multi-threading.
    - **Appearance Model:** Further improvements can be made by integrating the feature extraction network into the detection head.

    ### References

    - BoT-SORT Paper: [arXiv:2206.14651v2](https://arxiv.org/abs/2206.14651v2)
    - MOTChallenge: [MOT17 Dataset](https://motchallenge.net/data/MOT17/), [MOT20 Dataset](https://motchallenge.net/data/MOT20/)



    Initialization:
        ---------------
        To initialize the BoTSORT tracker with default values, you can use the following code:

        ```python
        BoTSORT(
            model_weights="path/to/model_weights",
            device="cuda",
            fp16=True,
            per_class=False,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method="sof",
            frame_rate=30,
            fuse_first_associate=False,
            with_reid=True,
            pixel_mapper=None,
            entry_polys=None,
            min_hits=3,
            limit_entry=False,
            class_id_to_label=None
        )
         ```
    """
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "sof",
        frame_rate=30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        #Change Begin
        pixel_mapper=None,
        entry_polys=None,  # Add this line
        min_hits=3,  # Add this line
        limit_entry=False,  # Add this line
        class_id_to_label: dict = None,
        #Change End
    ):
        """    
        Parameters:
        -----------
        - model_weights: str
            Path to the ReID model weights. Influences the appearance features extracted for tracking.
        - device: str
            Device to run the model on (e.g., 'cpu', 'cuda'). Determines the computational resource used.
        - fp16: bool
            Whether to use half-precision for computation. Affects the speed and memory usage.
        - per_class: bool
            Whether to track objects on a per-class basis. If True, tracks objects separately for each class.
        - track_high_thresh: float
            High confidence threshold for detections.
            - Lower values increase the number of detections considered high confidence.
            - Higher values are more selective.
        - track_low_thresh: float
            Low confidence threshold for detections.
            - Lower values increase the number of detections considered low confidence.
            - Higher values are more selective.
        - new_track_thresh: float
            Threshold for initializing new tracks.
            - Lower values increase the number of new tracks initialized.
            - Higher values are more selective.
        - track_buffer: int
            Buffer size for the tracker, determining how long a track can be kept without being updated.
            - Lower values reduce buffer size.
            - Higher values increase buffer size.
        - match_thresh: float
            Threshold for matching detections to tracks.
            - Lower values increase the likelihood of matches.
            - Higher values are more selective.
        - proximity_thresh: float
            Threshold for proximity-based matching.
            - Lower values increase the number of matches based on proximity.
            - Higher values are more selective.
        - appearance_thresh: float
            Threshold for appearance-based matching.
            - Lower values increase the number of matches based on appearance features.
            - Higher values are more selective.
        - cmc_method: str
            Method for camera motion compensation.
        - frame_rate: int
            Frame rate of the video.
        - fuse_first_associate: bool
            Whether to fuse scores during the first association.
        - with_reid: bool
            Whether to use re-identification features. Enhances the ability to track objects across frames based on appearance.
        - pixel_mapper: optional
            Pixel mapper for converting pixel coordinates to map coordinates. Used for spatial analysis.
        - entry_polys: optional
            List of polygons defining entry zones. Limits track creation to these zones if specified.
        - min_hits: int
            Minimum number of hits before a track is considered confirmed.
        - limit_entry: bool
            Whether to limit track creation to specified entry zones.
        - class_id_to_label: dict, optional
            Dictionary mapping class IDs to class labels. Provides a readable format for class identifiers.

        Attributes:
        -----------
        - lost_stracks: list
            List of lost tracks.
        - removed_stracks: list
            List of removed tracks.
        - buffer_size: int
            Adjusted buffer size based on the frame rate.
        - kalman_filter: KalmanFilterXYWH
            Instance of the Kalman filter used for motion prediction.
        - model: ReidAutoBackend
            Instance of the ReID model used for appearance feature extraction.
        - cmc: SOF
            Instance of the camera motion compensation method.
        - pixel_mapper: optional
            Instance of the pixel mapper for coordinate conversion.
        - entry_polys: list
            List of polygons defining entry zones.
        - next_final_id: int
            Counter for assigning final track IDs.
        - frame_count: int
            Counter for the number of frames processed.
        - active_tracks: list
            List of currently active tracks.

       Methods:
        --------
        - __init__(self, model_weights, device, fp16, per_class=False, track_high_thresh=0.5, ...)
            Initializes the tracker with specific parameters.
        - update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray
            Updates the tracker with new detections and returns the current tracks.


        Output Structure from `update` Method:
        --------------------------------------
        The `update` method returns a list of dictionaries, each representing a track with the following keys:
        - 'top_left_x': Top-left x-coordinate of the bounding box.
        - 'top_left_y': Top-left y-coordinate of the bounding box.
        - 'width': Width of the bounding box.
        - 'height': Height of the bounding box.
        - 'track_id_raw': Raw ID of the track.
        - 'confidence': Confidence score of the detection.
        - 'class': Class ID of the object.
        - 'detection_index': Index of the detection.
        - 'current_map_pos': Current map position of the object (if `pixel_mapper` is provided).
        - 'map_velocity': Velocity of the object on the map (if `pixel_mapper` is provided).
        - 'track_id': Final track ID (if the hit streak meets the `min_hits` threshold).
        """
        #Change Beign
        super().__init__(class_id_to_label=class_id_to_label)
        #Change End
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid
        if self.with_reid:
            rab = ReidAutoBackend(
                weights=model_weights, device=device, half=fp16
            )
            self.model = rab.get_backend()

        self.cmc = SOF()
        self.fuse_first_associate = fuse_first_associate
        #Change Begin
        self.pixel_mapper = pixel_mapper  # Store pixel_mapper
        self.entry_polys = entry_polys if entry_polys is not None else []  # Add this line
        self.min_hits = min_hits  # Add this line
        self.limit_entry = limit_entry  # Add this line
        self.next_final_id = 1  # To track the next final ID to assign
        if self.limit_entry:
            if not self.entry_polys:
                raise ValueError("entry_polys cannot be None if limit_entry is True")
            if self.pixel_mapper is None:
                raise ValueError("pixel_mapper cannot be None if limit_entry is True")
        #Change End

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, keypoints=None, keypoint_confidence_threshold=0.5, keypoint_indices=None) -> np.ndarray:
        """
        Update the tracker with new detections.

        Parameters:
        -----------
        - dets: np.ndarray
            Array of detections in the format (x1, y1, x2, y2, conf, class).
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
        - outputs: list
            List of dictionaries, each representing a track with the following keys:
            - 'top_left_x': Top-left x-coordinate of the bounding box.
            - 'top_left_y': Top-left y-coordinate of the bounding box.
            - 'width': Width of the bounding box.
            - 'height': Height of the bounding box.
            - 'track_id_raw': Raw ID of the track.
            - 'confidence': Confidence score of the detection.
            - 'class': Class ID of the object.
            - 'detection_index': Index of the detection.
            - 'current_map_pos': Current map position of the object (if `pixel_mapper` is provided).
            - 'map_velocity': Velocity of the object on the map (if `pixel_mapper` is provided).
            - 'track_id': Final track ID (if the hit streak meets the `min_hits` threshold).
        """
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        # Remove bad detections
        confs = dets[:, 4]

        # find second round association detections
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]

        


        # find first round association detections
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]

        # Assuming 'keypoints' is an array where each row corresponds to keypoints for a detection in 'dets'
        keypoints_first = keypoints[first_mask]
        keypoints_second = keypoints[second_mask]

        """Extract embeddings """
        # appearance descriptor extraction
        if self.with_reid:
            if embs is not None:
                features_high = embs
            else:
                # (Ndets x X) [512, 1024, 2048]
                features_high = self.model.get_features(dets_first[:, 0:4], img)

        if len(dets) > 0:
            """Detections"""
            #Change Begin
            if self.with_reid:
                detections = [STrack(det, f, max_obs=self.max_obs, keypoints=(keypoints_first[i] if keypoints_first is not None else None), keypoint_confidence_threshold=keypoint_confidence_threshold) 
                              for (i, (det, f)) in enumerate(zip(dets_first, features_high))]
            else:
                detections = [STrack(det, max_obs=self.max_obs, keypoints=(keypoints_first[i] if keypoints_first is not None else None), keypoint_confidence_threshold=keypoint_confidence_threshold) 
                              for (i, det) in enumerate(dets_first)]
            #Change End
        else:
            detections = []

        """ Add newly detected tracklets to active_tracks"""
        unconfirmed = []
        active_tracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.cmc.apply(img, dets_first)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high conf detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
          ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        if len(dets_second) > 0:
            """Detections"""
            #Change Begin
            detections_second = [STrack(det, max_obs=self.max_obs, 
                                keypoints=(keypoints_second[i] if keypoints_second is not None else None), 
                                keypoint_confidence_threshold=keypoint_confidence_threshold) 
                         for i, det in enumerate(dets_second)]
            #Change End
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        ious_dists = fuse_score(ious_dists, detections)
        
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        #Change Begin
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            # Add this block to limit tracker creation
            create = False
            if self.limit_entry:
                map_point = geo.Point(*self.pixel_mapper.detection_to_map(track.xywh))
                create = False
                for poly in self.entry_polys:
                    if poly.contains(map_point):
                        create = True
                        break
            else:
                create = True

            if create:
                track.activate(self.kalman_filter, self.frame_count)
                activated_starcks.append(track)
        
        #Change End

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        output_stracks = [track for track in self.active_tracks if track.is_activated]

        #Change Begin
        outputs = []
        for t in output_stracks:
            tlwh = xyxy2tlwh(t.xyxy)  # Convert to tlwh format
            
            # Create dictionary for each track
            track_dict = {
                'top_left_x': tlwh[0],
                'top_left_y': tlwh[1],
                'width': tlwh[2],
                'height': tlwh[3],
                'track_id_raw': t.id,
                'confidence': t.conf,
                'class': t.cls,
                'detection_index': int(t.det_ind),
                'keypoints': t.filtered_keypoints,
                'ankle_based_point': t.ankle_based_point
            }
            
            if self.pixel_mapper is not None:
                # Update map position and calculate velocity
                t.update_map_position(self.pixel_mapper, xyxy2xywh(t.xyxy), keypoints=t.filtered_keypoints, keypoint_indices=keypoint_indices, smoothing_factor=0.7)
                map_vel = t.calculate_velocity()
                # Add map position and velocity to dictionary
                track_dict['current_map_pos'] = t.current_map_pos
                track_dict['map_velocity'] = map_vel
            else:
                track_dict['current_map_pos'] = None
                track_dict['map_velocity'] = [None, None]
            
            if t.hit_streak >= self.min_hits:
                if t.final_id is None:  # Assign a final ID if not already assigned
                    t.final_id = self.next_final_id
                    self.next_final_id += 1
                track_dict['track_id'] = t.final_id
                outputs.append(track_dict)
                
        return outputs
    #Change End 


def joint_stracks(tlista, tlistb):
    """
    Join two lists of tracks.

    Parameters:
    -----------
    - tlista: list
        List of tracks.
    - tlistb: list
        List of tracks.

    Returns:
    --------
    - res: list
        Combined list of tracks.
    """
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """
    Subtract one list of tracks from another.

    Parameters:
    -----------
    - tlista: list
        List of tracks.
    - tlistb: list
        List of tracks to subtract.

    Returns:
    --------
    - res: list
        Resulting list of tracks.
    """
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """
    Remove duplicate tracks from two lists of tracks.

    Parameters:
    -----------
    - stracksa: list
        List of tracks.
    - stracksb: list
        List of tracks.

    Returns:
    --------
    - resa: list
        First list of tracks with duplicates removed.
    - resb: list
        Second list of tracks with duplicates removed.
    """
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
