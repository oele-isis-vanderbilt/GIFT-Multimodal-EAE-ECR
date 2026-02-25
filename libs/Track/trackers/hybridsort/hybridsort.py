# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))

"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""

from collections import deque  # [hgx0418] deque for reid feature

import numpy as np

from ...appearance.reid_auto_backend import ReidAutoBackend
from ...motion.cmc import get_cmc_method
from .association import (
    associate_4_points_with_score, associate_4_points_with_score_with_reid,
    cal_score_dif_batch_two_score, embedding_distance, linear_assignment)
from ...utils import PerClassDecorator
from ...utils.iou import get_asso_func
from ..basetracker import BaseTracker
from ...utils import PerClassDecorator
from ...utils.ops import xywh2xyxy, xyxy2xywh, xyxy2tlwh
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

# Change End

np.random.seed(0)


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    score = bbox[4]
    if score:
        return np.array([x, y, s, score, r]).reshape((5, 1))
    else:
        return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[4])
    h = x[2] / w
    score = x[3]
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    KalmanBoxTracker class for tracking individual objects.
    """
    count = 0

    def __init__(
        self,
        bbox,
        cls,
        det_ind,
        temp_feat,
        delta_t=3,
        orig=False,
        buffer_size=30,
        longterm_bank_length=30,
        alpha=0.8,
        max_obs=50,
        keypoints=None, 
        keypoint_confidence_threshold=0.5
    ):     # 'temp_feat' and 'buffer_size' for reid feature
        """
        Initialises a tracker using initial bounding box.

        Parameters:
        -----------
        - bbox: ndarray
            Detection bounding box in (x1, y1, x2, y2, score) format.
        - cls: int
            Class ID of the object.
        - det_ind: int
            Detection index.
        - temp_feat: ndarray
            Temporary feature for appearance embedding.
        - delta_t: int, optional
            Time step for velocity calculation.
        - orig: bool, optional
            Whether to use the original Kalman filter.
        - buffer_size: int, optional
            Buffer size for appearance features.
        - longterm_bank_length: int, optional
            Length of the long-term feature bank.
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
        - velocity_lt: ndarray
            Velocity of the object for the left-top corner.
        - velocity_rt: ndarray
            Velocity of the object for the right-top corner.
        - velocity_lb: ndarray
            Velocity of the object for the left-bottom corner.
        - velocity_rb: ndarray
            Velocity of the object for the right-bottom corner.
        - delta_t: int
            Time step for velocity calculation.
        - confidence_pre: float
            Previous confidence score.
        - confidence: float
            Current confidence score.
        - smooth_feat: ndarray
            Smoothed appearance feature.
        - current_map_pos: ndarray
            Current map position.
        - last_map_pos: ndarray
            Last map position.
        - final_id: int
            Final track ID for uniformity in output.
        """
        # define constant velocity model
        # if not orig and not args.kalman_GPR:
        from ...motion.kalman_filters.xysr_kf import KalmanFilterXYSR
        self.kf = KalmanFilterXYSR(dim_x=9, dim_z=5, max_obs=max_obs)

        # u, v, s, c, r, ~u, ~v, ~s, ~c
        self.kf.F = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01

        self.kf.x[:5] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[4]
        self.cls = cls
        self.det_ind = det_ind
        self.adapfs = False
        
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.last_observation_save = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None
        self.delta_t = delta_t
        self.confidence_pre = None
        self.confidence = bbox[4]

        # add the following values and functions
        self.smooth_feat = None
        buffer_size = longterm_bank_length
        self.features = deque([], maxlen=buffer_size)
        self.update_features(temp_feat)

        # momentum of embedding update
        self.alpha = alpha
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

    # ReID. for update embeddings during tracking
    def update_features(self, feat, score=-1):
        """
        Updates the appearance embedding of the tracker.

        Parameters:
        -----------
        - feat: ndarray
            New appearance embedding.
        - score: float, optional
            Detection confidence score.
        """
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if self.adapfs:
                assert score > 0
                pre_w = self.alpha * (self.confidence / (self.confidence + score))
                cur_w = (1 - self.alpha) * (score / (self.confidence + score))
                sum_w = pre_w + cur_w
                pre_w = pre_w / sum_w
                cur_w = cur_w / sum_w
                self.smooth_feat = pre_w * self.smooth_feat + cur_w * feat
            else:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def camera_update(self, warp_matrix):
        """
        Updates the mean of the current tracklet with ECC results.

        Parameters:
        -----------
        - warp_matrix: ndarray
            Warp matrix computed by ECC.
        """
        x1, y1, x2, y2, s = convert_x_to_bbox(self.kf.x)[0]
        x1_, y1_ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_ = warp_matrix @ np.array([x2, y2, 1]).T
        # w, h = x2_ - x1_, y2_ - y1_
        # cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:5] = convert_bbox_to_z([x1_, y1_, x2_, y2_, s])

    def update(self, bbox, cls, det_ind, id_feature, update_feature=True, keypoints=None):
        """
        Updates the state vector with observed bbox.

        Parameters:
        -----------
        - bbox: ndarray
            Detection bounding box in (x1, y1, x2, y2, score) format.
        - cls: int
            Class ID of the object.
        - det_ind: int
            Detection index.
        - id_feature: ndarray
            Appearance feature for the detection.
        - update_feature: bool, optional
            Whether to update the appearance feature.
        - keypoints: ndarray, optional
            The array of keypoints output from yolo in the data key it returns which contains a,y,conf
        """
        
        velocity_lt = None
        velocity_rt = None
        velocity_lb = None
        velocity_rb = None
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            self.det_ind = det_ind
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    # dt = self.delta_t - i
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if velocity_lt is not None:
                            velocity_lt += speed_direction_lt(previous_box, bbox)
                            velocity_rt += speed_direction_rt(previous_box, bbox)
                            velocity_lb += speed_direction_lb(previous_box, bbox)
                            velocity_rb += speed_direction_rb(previous_box, bbox)
                        else:
                            velocity_lt = speed_direction_lt(previous_box, bbox)
                            velocity_rt = speed_direction_rt(previous_box, bbox)
                            velocity_lb = speed_direction_lb(previous_box, bbox)
                            velocity_rb = speed_direction_rb(previous_box, bbox)
                        # break
                if previous_box is None:
                    previous_box = self.last_observation
                    self.velocity_lt = speed_direction_lt(previous_box, bbox)
                    self.velocity_rt = speed_direction_rt(previous_box, bbox)
                    self.velocity_lb = speed_direction_lb(previous_box, bbox)
                    self.velocity_rb = speed_direction_rb(previous_box, bbox)
                else:
                    self.velocity_lt = velocity_lt
                    self.velocity_rt = velocity_rt
                    self.velocity_lb = velocity_lb
                    self.velocity_rb = velocity_rb
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            
            self.last_observation = bbox
            self.last_observation_save = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            # Update keypoints if provided
            if keypoints is not None:
                self.keypoints = keypoints
                self.filtered_keypoints = self.filter_keypoints(keypoints, self.keypoint_confidence_threshold)
            #Change Begin
            
            # add interface for update feature or not
            if update_feature:
                if self.adapfs:
                    self.update_features(id_feature, score=bbox[4])
                else:
                    self.update_features(id_feature)
            self.confidence_pre = self.confidence
            self.confidence = bbox[4]
            
        else:
            
            self.kf.update(bbox)
            self.confidence_pre = None
        

    def predict(self, track_thresh=0.6):
        """
        Advances the state vector and returns the predicted bounding box estimate.

        Parameters:
        -----------
        - track_thresh: float, optional
            Tracking threshold.

        Returns:
        --------
        - bbox: ndarray
            Predicted bounding box.
        - kalman_score: float
            Kalman filter confidence score.
        - simple_score: float
            Simple confidence score.
        """
        if ((self.kf.x[7] + self.kf.x[2]) <= 0):
            self.kf.x[7] *= 0.0

        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        if not self.confidence_pre:
            return (
                self.history[-1],
                np.clip(self.kf.x[3], track_thresh, 1.0),
                np.clip(self.confidence, 0.1, track_thresh)
            )
        else:
            return (
                self.history[-1],
                np.clip(self.kf.x[3], track_thresh, 1.0),
                np.clip(self.confidence - (self.confidence_pre - self.confidence), 0.1, track_thresh)
            )

    def get_state(self):
        """
        Returns the current bounding box estimate.

        Returns:
        --------
        - bbox: ndarray
            Current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
    
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


class HybridSORT(BaseTracker):
    """
    HybridSORT Tracker for Video Analysis

    This tracker integrates appearance and motion cues for robust tracking.
    Extended from Mikel Broström's work on boxmot.

    Literature Summary:
    -------------------
    Hybrid-SORT is an advanced multi-object tracking algorithm that integrates both strong and weak cues to enhance the robustness and accuracy of object tracking. The algorithm maintains the Simple, Online, and Real-Time (SORT) characteristics while achieving superior performance by leveraging additional cues such as confidence state, height state, and velocity direction.

    ### Technical README for Hybrid-SORT Algorithm

    This document provides a concise technical overview of the Hybrid-SORT algorithm, detailing its components and their functionalities for multi-object tracking (MOT) in video streams.

    #### Overview
    Hybrid-SORT is an advanced multi-object tracking algorithm that integrates both strong and weak cues to enhance the robustness and accuracy of object tracking. The algorithm maintains the Simple, Online, and Real-Time (SORT) characteristics while achieving superior performance by leveraging additional cues such as confidence state, height state, and velocity direction.

    ### 1. Introduction to Hybrid-SORT

    - **Multi-Object Tracking (MOT):** The objective is to detect and track all objects in a scene while maintaining unique identifiers.
    - **Challenges Addressed:** Resolves issues of occlusion and clustering where strong cues (spatial and appearance information) become unreliable by incorporating weak cues (confidence state, height state, and velocity direction).

    ### 2. Algorithm Components

    #### 2.1 Weak Cues Modeling
    - **Tracklet Confidence Modeling (TCM):**
      - **Purpose:** Utilizes object confidence to indicate occlusion relationships, enhancing the association process.
      - **Method:** Extends the Kalman filter state vector to include confidence and its velocity component. For low-confidence detections, linear prediction is used to estimate tracklet confidence.

    - **Height Modulated IoU (HMIoU):**
      - **Purpose:** Incorporates object height as a cue for distinguishing objects, especially effective in occlusion and clustering scenarios.
      - **Method:** Computes IoU based on the height axis and combines it with the conventional IoU for enhanced discrimination.

    #### 2.2 Robust Observation-Centric Momentum (ROCM)
    - **Purpose:** Improves velocity direction modeling to handle complex motions more effectively.
    - **Method:** Uses multiple temporal intervals and four corners of the object box to calculate velocity direction, enhancing robustness against noise.

    #### 2.3 Appearance Modeling (Optional)
    - **Purpose:** Enhances tracking performance by incorporating appearance features.
    - **Method:** Uses an independent ReID model and Exponential Moving Average (EMA) for appearance feature representation, with cosine distance as the similarity metric.

    ### 3. Algorithm Workflow

    1. **Initialization:**
       - Initialize tracklets and set detection score threshold.

    2. **Detection and Classification:**
       - Detect objects in each frame and classify detections based on the score threshold.

    3. **Kalman Filter Prediction:**
       - Predict new locations for tracklets using the Kalman filter with extended states for confidence and height.

    4. **Association with TCM, HMIoU, and ROCM:**
       - Compute similarity (IoU, height-modulated IoU, confidence, and velocity direction) between detections and tracklets.
       - Use the Hungarian algorithm for matching and update tracklets.

    5. **Tracklet Update:**
       - Update matched tracklets with new positions and features.
       - Remove unmatched tracklets if they remain unmatched for a certain number of frames.

    6. **Tracklet Initialization:**
       - Initialize new tracklets for unmatched high-confidence detections.

    ### 4. Performance and Benchmarks

    - **Datasets:** Evaluated on DanceTrack, MOT17, and MOT20 datasets.
    - **Metrics:** Uses HOTA, IDF1, and MOTA to evaluate tracking performance.
    - **Results:** Achieves state-of-the-art performance with significant improvements in HOTA, IDF1, and MOTA scores compared to previous methods.

    ### 5. Implementation Details

    - **Framework:** Implemented in PyTorch.
    - **Hardware:** Tested on systems with Intel CPUs and NVIDIA GPUs.
    - **Source Code:** Available at [Hybrid-SORT GitHub Repository](https://github.com/ymzis69/HybridSORT).

    ### 6. Limitations and Future Work

    - **Non-linear Motion:** While improved, performance may still degrade in highly non-linear motion scenarios.
    - **Detection Quality:** Relies on high-quality detections; performance may degrade with noisy detections.
    - **Real-time Performance:** While real-time, further optimizations can be achieved with hardware accelerations and multi-threading.

    ### References

    - Hybrid-SORT Paper: [arXiv:2308.00783v2](https://arxiv.org/abs/2308.00783v2)
    - MOTChallenge: [MOT17 Dataset](https://motchallenge.net/data/MOT17/), [MOT20 Dataset](https://motchallenge.net/data/MOT20/)

    Initialization:
    ---------------
    To initialize the HybridSORT tracker with default values, you can use the following code:

    ```python
    HybridSORT(
        reid_weights="path/to/reid_weights",
        device="cuda",
        half=True,
        det_thresh=0.3,
        per_class=False,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        longterm_reid_weight=0,
        TCM_first_step_weight=0,
        use_byte=False,
        pixel_mapper=None,
        limit_entry=False,
        entry_polys=None,
        class_id_to_label=None
    )
    ```
    """
    def __init__(self, reid_weights, device, half, det_thresh, per_class=False, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, longterm_reid_weight=0, TCM_first_step_weight=0, use_byte=False, 
                 #Change Begin
        pixel_mapper=None,
        #Change End 
        # Change Begin
        limit_entry=False,
        entry_polys=None,
        class_id_to_label: dict = None,
        # Change End
        ):
        """
        Parameters:
        -----------
        - reid_weights: str
            Path to the ReID model weights. Provides the path to the model weights for re-identification.
        - device: str
            Device to run the model on (e.g., 'cpu', 'cuda'). Specifies the hardware device for model computation.
        - half: bool
            Whether to use half-precision for computation. Enables faster computation with reduced precision.
        - det_thresh: float
            Detection threshold.
            - Lower values will include more detections.
            - Higher values will be more selective.
        - per_class: bool
            Whether to track objects on a per-class basis. Allows separate tracking for different classes.
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
        - longterm_reid_weight: float
            Weight for long-term ReID matching. Adjusts the influence of long-term ReID cues.
        - TCM_first_step_weight: float
            Weight for the first step of Tracklet Confidence Modeling (TCM). Adjusts the influence of TCM in the first matching step.
        - use_byte: bool
            Whether to use BYTE association in the second matching step. Enables BYTE association.
        - pixel_mapper: object, optional
            Optional pixel mapper for converting pixel coordinates to map coordinates. Converts pixel positions to real-world coordinates.
        - limit_entry: bool
            Whether to limit track creation to specified entry zones. Restricts tracking to defined areas.
        - entry_polys: list, optional
            Optional list of polygons defining entry zones. Specifies zones for track initialization.
        - class_id_to_label: dict, optional
            Dictionary mapping class IDs to class labels. Maps class IDs to human-readable labels.

        Attributes:
        -----------
        - active_tracks: list
            List of currently active tracks.
        - frame_count: int
            Counter for the current frame.
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
        - use_byte: bool
            Whether to use BYTE association in the second matching step.
        - low_thresh: float
            Lower threshold for second-stage matching.
        - EG_weight_high_score: float
            Weight for high-score embedding matching.
        - EG_weight_low_score: float
            Weight for low-score embedding matching.
        - TCM_first_step: bool
            Whether to use Tracklet Confidence Modeling (TCM) in the first matching step.
        - with_longterm_reid: bool
            Whether to use long-term ReID matching.
        - with_longterm_reid_correction: bool
            Whether to use long-term ReID correction.
        - longterm_reid_weight: float
            Weight for long-term ReID matching.
        - TCM_first_step_weight: float
            Weight for the first step of Tracklet Confidence Modeling (TCM).
        - high_score_matching_thresh: float
            Threshold for high-score matching.
        - longterm_reid_correction_thresh: float
            Threshold for long-term ReID correction.
        - longterm_reid_correction_thresh_low: float
            Lower threshold for long-term ReID correction.
        - TCM_byte_step: bool
            Whether to use TCM in the BYTE step.
        - TCM_byte_step_weight: float
            Weight for TCM in the BYTE step.
        - dataset: str
            Dataset name.
        - ECC: bool
            Whether to use ECC for camera motion compensation.
        - pixel_mapper: object, optional
            Instance of the pixel mapper for coordinate conversion.
        - limit_entry: bool
            Whether to limit track creation to specified entry zones.
        - entry_polys: list
            List of polygons defining entry zones.

        Methods:
        --------
        - __init__(self, reid_weights, device, half, det_thresh, per_class=False, ...)
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
        self.per_class = per_class
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.use_byte = use_byte
        self.low_thresh = 0.1
        self.EG_weight_high_score = 1.3
        self.EG_weight_low_score = 1.2
        self.TCM_first_step = True
        self.with_longterm_reid = True
        self.with_longterm_reid_correction = True
        self.longterm_reid_weight = longterm_reid_weight
        self.TCM_first_step_weight = TCM_first_step_weight
        self.high_score_matching_thresh = 0.8
        self.longterm_reid_correction_thresh = 0.4
        self.longterm_reid_correction_thresh_low = 0.4
        self.TCM_byte_step = True
        self.TCM_byte_step_weight = 1.0
        self.dataset = 'dancetrack'
        self.ECC = False
        KalmanBoxTracker.count = 0

        rab = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        )
        self.model = rab.get_backend()
        self.cmc = get_cmc_method('ecc')()
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
        # Change End

    def camera_update(self, trackers, warp_matrix):
        """
        Updates the state of each tracker with ECC results.

        Parameters:
        -----------
        - trackers: list
            List of active trackers.
        - warp_matrix: ndarray
            Warp matrix computed by ECC.
        """
        for tracker in trackers:
            tracker.camera_update(warp_matrix)

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, keypoints=None, keypoint_confidence_threshold=0.5, keypoint_indices=None) -> np.ndarray:
        """
        Updates the tracker with new detections.

        Parameters:
        -----------
        - dets: np.ndarray
            Array of detections in the format (x1, y1, x2, y2, score, class, det_index).
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
        
        if dets is None:
            return np.empty((0, 7))

        if self.ECC:
            warp_matrix = self.cmc.apply(img, dets)
            if warp_matrix is not None:
                self.camera_update(self.active_tracks, warp_matrix)
        
        self.frame_count += 1
        scores = dets[:, 4]
        bboxes = dets[:, :4]
        #Change Begin
        dets_indices = np.arange(len(dets))  # Creating unique indices for each detection
        #Change End
        dets_embs = self.model.get_features(bboxes, img)
        #Change Begin
        dets0 = np.concatenate((dets, np.expand_dims(dets_indices, axis=-1)), axis=1)
        #change End
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > self.low_thresh
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        #Change Begin
        dets0_second = dets0[inds_second]
        keypoints_second = keypoints[inds_second]
        #change End
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        #Change Begin
        dets0 = dets0[remain_inds]
        keypoints_first = keypoints[remain_inds]
        #Change End
        id_feature_keep = dets_embs[remain_inds]  # ID feature of 1st stage matching
        id_feature_second = dets_embs[inds_second]  # ID feature of 2nd stage matching

        trks = np.zeros((len(self.active_tracks), 8))
        to_del = []
        #Change Begin
        outputs = []
        #change End
        
        for t, trk in enumerate(trks):
            pos, kalman_score, simple_score = self.active_tracks[t].predict()
            trk[:6] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score[0], simple_score]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities_lt = np.array(
            [trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.active_tracks])
        velocities_rt = np.array(
            [trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.active_tracks])
        velocities_lb = np.array(
            [trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.active_tracks])
        velocities_rb = np.array(
            [trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.active_tracks])
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])
        

        """
            First round of association
        """
        if self.EG_weight_high_score > 0 and self.TCM_first_step:
            track_features = np.asarray([track.smooth_feat for track in self.active_tracks],
                                        dtype=np.float64)
            emb_dists = embedding_distance(track_features, id_feature_keep).T
            if self.with_longterm_reid or self.with_longterm_reid_correction:
                long_track_features = np.asarray([np.vstack(list(track.features)).mean(0) for track in self.active_tracks],
                                                 dtype=np.float64)
                assert track_features.shape == long_track_features.shape
                long_emb_dists = embedding_distance(long_track_features, id_feature_keep).T
                assert emb_dists.shape == long_emb_dists.shape
                matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_with_reid(
                    dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                    k_observations, self.inertia, self.TCM_first_step_weight, self.asso_func, emb_cost=emb_dists,
                    weights=(1.0, self.EG_weight_high_score), thresh=self.high_score_matching_thresh,
                    long_emb_dists=long_emb_dists, with_longterm_reid=self.with_longterm_reid,
                    longterm_reid_weight=self.longterm_reid_weight,
                    with_longterm_reid_correction=self.with_longterm_reid_correction,
                    longterm_reid_correction_thresh=self.longterm_reid_correction_thresh,
                    dataset=self.dataset)
            else:
                matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_with_reid(
                    dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                    k_observations, self.inertia, self.TCM_first_step_weight, self.asso_func, emb_cost=emb_dists,
                    weights=(1.0, self.EG_weight_high_score), thresh=self.high_score_matching_thresh)
        elif self.TCM_first_step:
            matched, unmatched_dets, unmatched_trks = associate_4_points_with_score(
                dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                k_observations, self.inertia, self.TCM_first_step_weight, self.asso_func)
        # update with id feature
        
        for m in matched:
            #Change Begin
            self.active_tracks[m[1]].update(
            dets[m[0], :], dets0[m[0], 5], dets0[m[0], 6], id_feature_keep[m[0], :],
            keypoints=(keypoints_first[m[0]] if keypoints_first is not None else None)
        ) 
            #Change End
            
        
        
        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            u_tracklets = [self.active_tracks[index] for index in unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                if self.TCM_byte_step:
                    iou_left -= np.array(
                        cal_score_dif_batch_two_score(dets_second, u_trks) * self.TCM_byte_step_weight
                    )
                    iou_left_thre = iou_left
                if self.EG_weight_low_score > 0:
                    u_track_features = np.asarray([track.smooth_feat for track in u_tracklets], dtype=np.float64)
                    emb_dists_low_score = embedding_distance(u_track_features, id_feature_second).T
                    matched_indices = linear_assignment(-iou_left + self.EG_weight_low_score * emb_dists_low_score,
                                                        )
                else:
                    matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if self.with_longterm_reid_correction and self.EG_weight_low_score > 0:
                        if (iou_left_thre[m[0], m[1]] < self.iou_threshold) or \
                           (emb_dists_low_score[m[0], m[1]] > self.longterm_reid_correction_thresh_low):
                            print("correction 2nd:", emb_dists_low_score[m[0], m[1]])
                            continue
                    else:
                        if iou_left_thre[m[0], m[1]] < self.iou_threshold:
                            continue
                    self.active_tracks[trk_ind].update(
                        dets_second[det_ind, :],
                        #Change Begin
                        dets0_second[det_ind, 5],
                        dets0_second[det_ind, 6],
                        id_feature_second[det_ind, :],
                        keypoints=(keypoints_second[det_ind] if keypoints_second is not None else None),
                        #Change End
                        update_feature=False
                    )     # [hgx0523] do not update with id feature
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            
            left_dets = dets[unmatched_dets]
            #left_id_feature = id_feature_keep[unmatched_dets]       # update id feature, if needed
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
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
                    self.active_tracks[trk_ind].update(
                        #Change Begin
                        dets[det_ind, :],
                        dets0[det_ind, 5],
                        dets0[det_ind, 6],
                        id_feature_keep[det_ind, :],
                        keypoints=(keypoints_first[det_ind] if keypoints_first is not None else None),
                        #Change End
                        update_feature=False
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
            
        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None, None, None, None)

        # Change Begin
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            create = False
            if self.limit_entry:
                map_point = geo.Point(*self.pixel_mapper.detection_to_map(dets[i, 0:4]))
                for poly in self.entry_polys:
                    if poly.contains(map_point):
                        create = True
                        break
            else:
                create = True

            if create:
                trk = KalmanBoxTracker(
                        dets[i, :], dets0[i, 5], dets0[i, 6], id_feature_keep[i, :], delta_t=self.delta_t, max_obs=self.max_obs,
                        keypoints=(keypoints_first[i] if keypoints_first is not None else None),  # Initialize with keypoints
                        keypoint_confidence_threshold=keypoint_confidence_threshold  # Initialize with threshold
                    )
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
                    'track_id_raw': trk.id + 1,
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
            if (trk.time_since_update > self.max_age):
                self.active_tracks.pop(i)
        
        return np.array(outputs) if len(outputs) > 0 else np.array([])
        #Change End