# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))

import numpy as np
from collections import deque

from ...appearance.reid_auto_backend import ReidAutoBackend
from ...motion.cmc import get_cmc_method
from ...motion.kalman_filters.xysr_kf import KalmanFilterXYSR
from ...motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from ...utils.association import associate, linear_assignment
from ...utils.iou import get_asso_func
from ..basetracker import BaseTracker
from ...utils import PerClassDecorator
from ...utils.ops import xyxy2xysr
#Change Begin
from ...utils.ops import xywh2xyxy, xyxy2xywh, xyxy2tlwh
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
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    KalmanBoxTracker class for tracking individual objects.
    """

    count = 0

    def __init__(self, det, delta_t=3, emb=None, alpha=0, max_obs=50, keypoints=None, keypoint_confidence_threshold=0.5):
        """
        Initialises a tracker using initial bounding box.

        Parameters:
        -----------
        - det: ndarray
            Detection bounding box in (x1, y1, x2, y2, conf, class, det_ind) format.
        - delta_t: int, optional
            Time step for velocity calculation.
        - emb: ndarray, optional
            Appearance feature of the detection.
        - alpha: float, optional
            Smoothing factor for features.
        - max_obs: int, optional
            Maximum number of observations to store.
        - keypoints: ndarray, optional
            The array of keypoints output from yolo in the data key it returns which contains a,y,conf
        - keypoint_confidence_threshold: float
            The minimum confidence of a keypoint to be associated with a track 

        Attributes:
        -----------
        - max_obs: int
            Maximum number of observations to store.
        - conf: float
            Confidence score of the detection.
        - cls: int
            Class ID of the object.
        - det_ind: int
            Detection index.
        - kf: KalmanFilterXYSR
            Kalman filter for motion prediction.
        - time_since_update: int
            Number of frames since last update.
        - id: int
            Unique identifier for the tracker.
        - history: deque
            History of predicted bounding boxes.
        - hits: int
            Number of times the tracker was updated with a detection.
        - hit_streak: int
            Number of continuous frames a detection is present.
        - age: int
            Number of frames since the tracker was initialized.
        - last_observation: ndarray
            Last observed bounding box.
        - features: deque
            Queue of appearance features.
        - observations: dict
            Dictionary of past observations.
        - velocity: ndarray
            Velocity of the object.
        - delta_t: int
            Time step for velocity calculation.
        - history_observations: deque
            History of observations.
        - emb: ndarray
            Appearance embedding of the detection.
        - frozen: bool
            Whether the tracker is frozen (no updates).
        - current_map_pos: ndarray
            Current map position.
        - last_map_pos: ndarray
            Last map position.
        - final_id: int
            Final track ID for uniformity in output.
        
        """
        # define constant velocity model
        self.max_obs=max_obs
        bbox = det[0:5]
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                # x  y  s  r  x' y' s'
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
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.bbox_to_z_func = xyxy2xysr
        self.x_to_bbox_func = convert_x_to_bbox

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        # Used to output track after min_hits reached
        self.features = deque([], maxlen=self.max_obs)
        # Used for velocity
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t
        self.history_observations = deque([], maxlen=self.max_obs)

        self.emb = emb

        self.frozen = False

        #Change Begin
        self.current_map_pos = None
        self.last_map_pos = None
        self.final_id = None  # Final track ID for uniformity in output
        self.keypoints = keypoints  # Store keypoints
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

    def update(self, det, keypoints=None):
        """
        Updates the state vector with observed bbox.

        Parameters:
        -----------
        - det: ndarray
            Detection bounding box in (x1, y1, x2, y2, conf, class, det_ind) format.
        - keypoints: ndarray, optional
            The array of keypoints output from yolo in the data key it returns which contains a,y,conf
        """

        if det is not None:
            bbox = det[0:5]
            self.conf = det[4]
            self.cls = det[5]
            self.det_ind = det[6]
            self.frozen = False

            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
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

            self.kf.update(self.bbox_to_z_func(bbox))
            #Change Begin
            # Update keypoints if provided
            if keypoints is not None:
                self.keypoints = keypoints
                self.filtered_keypoints = self.filter_keypoints(keypoints, self.keypoint_confidence_threshold)
            #Change End
        else:
            self.kf.update(det)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        """
        Updates the appearance embedding of the tracker.

        Parameters:
        -----------
        - emb: ndarray
            New appearance embedding.
        - alpha: float, optional
            Smoothing factor for the embedding update.
        """
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        """
        Returns the current appearance embedding.

        Returns:
        --------
        - emb: ndarray
            Current appearance embedding.
        """
        return self.emb

    def apply_affine_correction(self, affine):
        """
        Applies affine transformation to correct for camera motion.

        Parameters:
        -----------
        - affine: ndarray
            Affine transformation matrix.
        """
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.

        Returns:
        --------
        - bbox: ndarray
            Predicted bounding box.
        """
        # Don't allow negative bounding boxes
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.

        Returns:
        --------
        - bbox: ndarray
            Current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """
        Calculates the Mahalanobis distance between the state vector and an observed bounding box.

        Parameters:
        -----------
        - bbox: ndarray
            Observed bounding box.

        Returns:
        --------
        - dist: float
            Mahalanobis distance.
        """
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
    
    #Change Begin
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


    def calculate_velocity(self):
        """
        Calculates the velocity of the object.

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
    

def normalize(v):
    """
    Normalizes a vector.

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



class DeepOCSort(BaseTracker):
    """
    DeepOCSort Tracker for Video Analysis

    This tracker integrates appearance and motion cues for robust tracking.
    Extended from Mikel Broström's work on boxmot.

    Literature Summary:
    -------------------
    Deep OC-SORT builds upon the motion-based OC-SORT by incorporating dynamic appearance cues and adaptive re-identification, resulting in improved robustness and accuracy in multi-object tracking. By using a Kalman filter for motion prediction and a deep learning model for re-identification, the tracker can handle various challenges in multi-object tracking such as camera motion, detection accuracy, and identity consistency. Deep OC-SORT incorporates camera-motion compensation, dynamic appearance integration, and adaptive weighting for appearance cues.

    ### Technical README for Deep OC-SORT Algorithm

    This document provides a concise technical overview of the Deep OC-SORT algorithm, detailing its components and their functionalities for multi-object tracking (MOT) in video streams.

    #### Overview
    Deep OC-SORT builds upon the motion-based OC-SORT by incorporating dynamic appearance cues and adaptive re-identification, resulting in improved robustness and accuracy in multi-object tracking.

    ### 1. Introduction to Deep OC-SORT

    - **Multi-Object Tracking (MOT):** The objective is to detect and track all objects in a scene while maintaining unique identifiers.
    - **Tracking-by-Detection Paradigm:** Utilizes object detection followed by a tracking step.
    - **Challenges Addressed:** Improves tracking robustness by integrating visual appearance with motion-based tracking methods.

    ### 2. Algorithm Components

    #### 2.1 Preliminary: OC-SORT
    - **OC-SORT:** Extends the SORT algorithm by addressing limitations in handling non-linear motion and occlusion through observation-centric modules.
      - **Observation-Centric Momentum (OCM):** Uses angular velocity from bounding box points to improve tracking.
      - **Observation-Centric Recovery (OCR):** Re-associates tracks using the last known position and bounding box points.
      - **Observation-Centric Online Smoothing (OOS):** Smooths tracks to reduce noise.

    #### 2.2 Camera Motion Compensation (CMC)
    - **Purpose:** Adjusts object positions in the presence of camera motion to improve localization accuracy.
    - **Method:** Applies scaled rotation matrix and translation to correct bounding box positions and velocities in the Kalman filter state.

    #### 2.3 Dynamic Appearance (DA)
    - **Purpose:** Integrates visual appearance cues adaptively based on detector confidence to improve object association.
    - **Method:** Uses a dynamic weighting factor \( \alpha_t \) in the Exponential Moving Average (EMA) of appearance embeddings, adjusting based on detection confidence to reject corrupted embeddings.

    #### 2.4 Adaptive Weighting (AW)
    - **Purpose:** Enhances the appearance cost matrix based on the discriminativeness of appearance embeddings to improve association accuracy.
    - **Method:** Boosts individual track-box scores based on the similarity of track and detection embeddings, using the highest and second-highest scores to measure discriminativeness.

    ### 3. Algorithm Workflow

    1. **Initialization:**
       - Initialize tracklets and set detection score threshold.

    2. **Detection and Classification:**
       - Detect objects in each frame and classify detections based on the score threshold.

    3. **Kalman Filter Prediction:**
       - Predict new locations for tracklets using the Kalman filter.

    4. **Camera Motion Compensation (CMC):**
       - Adjust predicted bounding box positions and velocities for camera motion.

    5. **Association with OCM and AW:**
       - Compute similarity (IoU and appearance) between detections and tracklets.
       - Use the Hungarian algorithm for matching, incorporating adaptive weighting for appearance cues.

    6. **Re-Update with OCR:**
       - For tracklets re-associated after occlusion, apply OCR to correct accumulated errors.

    7. **Tracklet Update:**
       - Update matched tracklets with new positions and appearance features.
       - Remove unmatched tracklets if they remain unmatched for a certain number of frames.

    8. **Tracklet Initialization:**
       - Initialize new tracklets for unmatched high-confidence detections.

    ### 4. Performance and Benchmarks

    - **Datasets:** Evaluated on MOT17, MOT20, and DanceTrack datasets.
    - **Metrics:** Uses HOTA, IDF1, MOTA, AssA, and DetA to evaluate tracking performance.
    - **Results:** Achieves state-of-the-art performance with significant improvements in HOTA, IDF1, and AssA scores compared to previous methods.

    ### 5. Implementation Details

    - **Framework:** Implemented in PyTorch.
    - **Hardware:** Tested on systems with Intel i9 CPUs and NVIDIA GPUs.
    - **Source Code:** Available at [Deep OC-SORT GitHub Repository](https://github.com/GerardMaggiolino/Deep-OC-SORT).

    ### 6. Limitations and Future Work

    - **Non-linear Motion:** While improved, performance may still degrade in highly non-linear motion scenarios.
    - **Detection Quality:** Relies on high-quality detections; performance may degrade with noisy detections.
    - **Real-time Performance:** While real-time, further optimizations can be achieved with hardware accelerations and multi-threading.

    ### References

    - Deep OC-SORT Paper: [arXiv:2302.11813v1](https://arxiv.org/abs/2302.11813v1)
    - MOTChallenge: [MOT17 Dataset](https://motchallenge.net/data/MOT17/), [MOT20 Dataset](https://motchallenge.net/data/MOT20/)

    Initialization:
    ---------------
    To initialize the DeepOCSort tracker with default values, you can use the following code:

    ```python
    DeepOCSort(
        model_weights="path/to/model_weights",
        device="cuda",
        fp16=True,
        per_class=False,
        det_thresh=0.3,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.5,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        pixel_mapper=None,
        limit_entry=False,
        entry_polys=None,
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
        det_thresh=0.3,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.5,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        #Change Begin
        pixel_mapper=None,
        limit_entry=False,
        entry_polys=None,
        class_id_to_label: dict = None,
        #Change End
        **kwargs
    ):  
        """
        Parameters:
        -----------
        - model_weights: str
            Path to the ReID model weights. Provides the path to the model weights for re-identification.
        - device: str
            Device to run the model on (e.g., 'cpu', 'cuda'). Specifies the hardware device for model computation.
        - fp16: bool
            Whether to use half-precision for computation. Enables faster computation with reduced precision.
        - per_class: bool
            Whether to track objects on a per-class basis. Allows separate tracking for different classes.
        - det_thresh: float
            Detection threshold.
            - Lower values will include more detections.
            - Higher values will be more selective.
        - max_age: int
            Maximum number of frames to keep a track alive without detections.
            - Lower values will remove tracks quicker.
            - Higher values will keep tracks longer.
        - min_hits: int
            Minimum number of associated detections before a track is confirmed.
            - Lower values will confirm tracks faster.
            - Higher values will require more detections.
        - iou_threshold: float
            Intersection over Union (IoU) threshold for matching.
            - Lower values will allow more matches.
            - Higher values will be more selective.
        - delta_t: int
            Time step for velocity calculation. Determines the interval for velocity updates.
        - asso_func: str ("iou" or "giou" or "ciou" or ""diou"" or "centroid")
            Association function for matching. Specifies the method for associating detections with tracks.
        - inertia: float
            Inertia for motion model. Controls the smoothness of motion predictions.
        - w_association_emb: float
            Weight for appearance association. Balances motion and appearance cues.
        - alpha_fixed_emb: float
            Fixed alpha for exponential moving average of embeddings. Smoothing factor for embedding updates.
        - aw_param: float
            Parameter for adaptive weighting. Adjusts the influence of appearance cues.
        - embedding_off: bool
            Whether to turn off embedding-based association. Disables appearance-based matching.
        - cmc_off: bool
            Whether to turn off camera motion compensation. Disables corrections for camera movements.
        - aw_off: bool
            Whether to turn off adaptive weighting. Disables adaptive weighting of appearance cues.
        - pixel_mapper: optional
            Optional pixel mapper for converting pixel coordinates to map coordinates. Converts pixel positions to real-world coordinates.
        - limit_entry: bool
            Whether to limit track creation to specified entry zones. Restricts tracking to defined areas.
        - entry_polys: optional
            List of polygons defining entry zones. Specifies zones for track initialization.
        - class_id_to_label: dict, optional
            Dictionary mapping class IDs to class labels. Maps class IDs to human-readable labels.

        Attributes:
        -----------
        - active_tracks: list
            List of currently active tracks.
        - frame_count: int
            Counter for the current frame.
        - height: int
            Height of the input image.
        - width: int
            Width of the input image.
        - model: object
            ReID model for extracting appearance features.
        - cmc: object
            Camera motion compensation method.
        - next_final_id: int
            Counter for assigning final track IDs.
        - max_age: int
            Maximum number of frames to keep a track alive without detections.
        - min_hits: int
            Minimum number of associated detections before a track is confirmed.
        - iou_threshold: float
            Intersection over Union (IoU) threshold for matching.
        - det_thresh: float
            Detection threshold.
        - delta_t: int
            Time step for velocity calculation.
        - asso_func: function
            Association function for matching.
        - inertia: float
            Inertia for motion model.
        - w_association_emb: float
            Weight for appearance association.
        - alpha_fixed_emb: float
            Fixed alpha for exponential moving average of embeddings.
        - aw_param: float
            Parameter for adaptive weighting.
        - embedding_off: bool
            Whether to turn off embedding-based association.
        - cmc_off: bool
            Whether to turn off camera motion compensation.
        - aw_off: bool
            Whether to turn off adaptive weighting.
        - pixel_mapper: optional
            Instance of the pixel mapper for coordinate conversion.
        - limit_entry: bool
            Whether to limit track creation to specified entry zones.
        - entry_polys: list
            List of polygons defining entry zones.

        Methods:
        --------
        - __init__(self, model_weights, device, fp16, per_class=False, det_thresh=0.3, ...)
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
        #Change Begin
        super().__init__(max_age=max_age, class_id_to_label=class_id_to_label)
        #Change End
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.per_class = per_class
        KalmanBoxTracker.count = 1

        rab = ReidAutoBackend(
            weights=model_weights, device=device, half=fp16
        )
        self.model = rab.get_backend()
        # "similarity transforms using feature point extraction, optical flow, and RANSAC"
        self.cmc = get_cmc_method('sof')()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
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
        #Change End

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, keypoints=None, keypoint_confidence_threshold=0.5, keypoint_indices=None) -> np.ndarray:
        """
        Updates the tracker with new detections.

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
        #dets, s, c = dets.data
        #print(dets, s, c)
        assert isinstance(dets, np.ndarray), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
        assert isinstance(img, np.ndarray), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
        assert len(dets.shape) == 2, "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert dets.shape[1] == 6, "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1
        self.height, self.width = img.shape[:2]

        scores = dets[:, 4]
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        assert dets.shape[1] == 7
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        keypoints = keypoints[remain_inds]

        # appearance descriptor extraction
        if self.embedding_off or dets.shape[0] == 0:
            dets_embs = np.ones((dets.shape[0], 1))
        elif embs is not None:
            dets_embs = embs
        else:
            # (Ndets x X) [512, 1024, 2048]
            dets_embs = self.model.get_features(dets[:, 0:4], img)

        # CMC
        if not self.cmc_off:
            transform = self.cmc.apply(img, dets[:, :4])
            for trk in self.active_tracks:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.active_tracks), 5))
        trk_embs = []
        to_del = []
        # Chnage Begin
        outputs = []
        # Change End
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.active_tracks[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        if len(trk_embs) > 0:
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)

        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.active_tracks])
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])

        """
            First round of association
        """
        # (M detections X N tracks, final score)
        if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = dets_embs @ trk_embs.T
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5],
            trks,
            self.asso_func,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            img.shape[1], # w
            img.shape[0], # h
            stage1_emb_cost,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
        )
        for m in matched:
            det_ind = m[0]
            self.active_tracks[m[1]].update(dets[m[0], :], keypoints=(keypoints[det_ind] if keypoints is not None else None))
            self.active_tracks[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            iou_left = self.asso_func(left_dets, left_trks)
            # TODO: is better without this
            emb_cost_left = left_dets_embs @ left_trks_embs.T
            if self.embedding_off:
                emb_cost_left = np.zeros_like(emb_cost_left)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.active_tracks[trk_ind].update(dets[det_ind, :], keypoints=(keypoints[det_ind] if keypoints is not None else None))
                    self.active_tracks[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.active_tracks[m].update(None)
        # Change Begin
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            create = False
            if self.limit_entry:
                map_point = geo.Point(*self.pixel_mapper.detection_to_map(dets[i, 0:4], [0, 1]))
                for poly in self.entry_polys:
                    if poly.contains(map_point):
                        create = True
                        break
            else:
                create = True
            
            if create:
                trk = KalmanBoxTracker(dets[i], delta_t=self.delta_t, emb=dets_embs[i], alpha=dets_alpha[i], max_obs=self.max_obs, keypoints=(keypoints[i] if keypoints is not None else None), keypoint_confidence_threshold=keypoint_confidence_threshold)
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
                    'track_id': trk.final_id,
                    'track_id_raw': trk.id,
                    'confidence': trk.conf,
                    'class': trk.cls,
                    'detection_index': trk.det_ind,
                    'keypoints': trk.filtered_keypoints,
                    'ankle_based_point': trk.ankle_based_point
                }

                if self.pixel_mapper is not None:
                    # Update map position and calculate velocity
                    trk.update_map_position(self.pixel_mapper, xyxy2xywh(trk.get_state()[0]), keypoints=trk.filtered_keypoints, keypoint_indices=keypoint_indices, smoothing_factor=0.7)
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

        return np.array(outputs) if len(outputs) > 0 else np.array([])
        #Change End
