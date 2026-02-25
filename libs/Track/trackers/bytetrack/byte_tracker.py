# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))

import numpy as np
from collections import deque

from ...motion.kalman_filters.xyah_kf import KalmanFilterXYAH
from .basetrack import BaseTrack, TrackState
from ...utils.matching import fuse_score, iou_distance, linear_assignment
from ...utils.ops import tlwh2xyah, xywh2tlwh, xywh2xyxy, xyxy2xywh, xyxy2tlwh
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

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, det, max_obs, keypoints=None, keypoint_confidence_threshold=0.5):
        """ Attributes:
            -----------
            - xywh: ndarray
                Bounding box in (xc, yc, w, h) format.
            - tlwh: ndarray
                Bounding box in (t, l, w, h) format.
            - xyah: ndarray
                Bounding box in (xc, yc, a, h) format.
            - conf: float
                Confidence score of the detection.
            - cls: int
                Class ID of the object.
            - det_ind: int
                Detection index.
            - max_obs: int
                Maximum number of observations to store.
            - kalman_filter: KalmanFilterXYAH
                Kalman filter for motion prediction.
            - mean: ndarray
                Mean state vector of the Kalman filter.
            - covariance: ndarray
                Covariance matrix of the Kalman filter.
            - is_activated: bool
                Whether the track is activated.
            - tracklet_len: int
                Length of the tracklet.
            - history_observations: deque
                History of observations.
            - current_map_pos: ndarray
                Current map position.
            - last_map_pos: ndarray
                Last map position.
            - hit_streak: int
                Number of continuous frames a detection is present.
            - final_id: int
                Final track ID for uniformity in output.
            - keypoints: ndarray, optional
            The array of keypoints output from yolo in the data key it returns which contains a,y,conf
            - keypoint_confidence_threshold: float
                The minimum confidence of a keypoint to be associated with a track 
                """
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.tlwh = xywh2tlwh(self.xywh)  # (xc, yc, w, h) --> (t, l, w, h)
        self.xyah = tlwh2xyah(self.tlwh)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs=max_obs
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
        self.history_observations = deque([], maxlen=self.max_obs)
        #Change begin
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
        #Change end

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

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        Start a new tracklet.

        Parameters:
        -----------
        - kalman_filter: KalmanFilterXYAH
            Kalman filter for motion prediction.
        - frame_id: int
            ID of the current frame.
        """
        self.kalman_filter = kalman_filter
        self.id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xyah)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        #Change begin
        self.hit_streak = 0  # Reset hit streak on activation
        #Change end

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        Re-activate an existing track with new detection.

        Parameters:
        -----------
        - new_track: STrack
            New track to re-activate.
        - frame_id: int
            ID of the current frame.
        - new_id: bool
            Whether to assign a new ID to the track.
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        #Change begin
        self.hit_streak = 0  # Reset hit streak on reactivation
        # Update keypoints
        self.keypoints = new_track.keypoints
        self.filtered_keypoints = self.filter_keypoints(new_track.keypoints, self.keypoint_confidence_threshold)
        #Change end

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
            self.mean, self.covariance, new_track.xyah
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        #Change begin
        self.hit_streak += 1  # Increment hit streak on update
        # Update keypoints
        self.keypoints = new_track.keypoints
        self.filtered_keypoints = self.filter_keypoints(new_track.keypoints, self.keypoint_confidence_threshold)
        #Change end

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
            ret = self.mean[:4].copy()  # kf (xc, yc, a, h)
            ret[2] *= ret[3]  # (xc, yc, a, h)  -->  (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret
    
    #Change begin
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
#Change end 


class BYTETracker(BaseTracker):
    """
    BYTETracker class for tracking multiple objects.

    This tracker integrates both high-confidence and low-confidence detections to enhance tracking accuracy and identity preservation.

    Literature Summary:
    -------------------
    ByteTrack is an advanced multi-object tracking algorithm that improves the association of detection boxes by incorporating almost every detection box, including those with low detection scores, to enhance tracking accuracy and identity preservation.

    ### Overview
    ByteTrack is an advanced multi-object tracking algorithm that improves the association of detection boxes by incorporating almost every detection box, including those with low detection scores, to enhance tracking accuracy and identity preservation.

    ### 1. Introduction to ByteTrack

    - **Multi-Object Tracking (MOT):** The objective is to estimate bounding boxes and identities of objects in videos.
    - **Tracking-by-Detection Paradigm:** Utilizes object detection followed by a tracking step.
    - **Challenges Addressed:** Reduces missing detections and fragmented trajectories by associating both high and low detection score boxes.

    ### 2. Algorithm Components

    #### 2.1 High-Confidence and Low-Confidence Detections
    - **Detection Thresholding:** Separates detection boxes into high-confidence and low-confidence based on a score threshold.
    - **High-Confidence Detections:** Directly associated with existing tracklets.
    - **Low-Confidence Detections:** Utilized to recover true objects and filter out background detections.

    #### 2.2 Kalman Filter (KF)
    - **Motion Model:** Uses a Kalman filter with a constant-velocity model to predict the trajectory of objects.
    - **State Vector:** Includes position and velocity to enhance trajectory predictions.

    #### 2.3 Association Strategy
    - **First Association:** Matches high-confidence detections with tracklets using motion similarity or appearance similarity.
    - **Second Association:** Matches remaining low-confidence detections with unmatched tracklets using motion similarity to recover occluded or low-confidence objects.

    ### 3. Algorithm Workflow

    1. **Initialization:**
       - Initialize tracklets and set the detection score threshold \( \tau \).

    2. **Detection and Classification:**
       - Detect objects in each frame and classify detections as high-confidence or low-confidence based on the score threshold \( \tau \).

    3. **Kalman Filter Prediction:**
       - Predict new locations for tracklets using the Kalman filter.

    4. **First Association:**
       - Compute similarity (IoU or appearance) between high-confidence detections and tracklets.
       - Use the Hungarian algorithm for matching and update tracklets.

    5. **Second Association:**
       - Compute IoU similarity between remaining low-confidence detections and unmatched tracklets.
       - Match low-confidence detections to tracklets to reduce missing detections and false negatives.

    6. **Tracklet Update:**
       - Update matched tracklets with new positions and appearance features.
       - Remove unmatched tracklets if they remain unmatched for a certain number of frames.

    7. **Tracklet Initialization:**
       - Initialize new tracklets for unmatched high-confidence detections.

    ### 4. Performance and Benchmarks

    - **Datasets:** Evaluated on MOT17, MOT20, HiEve, and BDD100K datasets.
    - **Metrics:** Uses MOTA (Multiple Object Tracking Accuracy), IDF1, HOTA, False Positives (FP), False Negatives (FN), and ID Switches (IDSW).
    - **Results:** Achieves state-of-the-art performance with significant improvements in MOTA, IDF1, and HOTA scores compared to previous methods.

    ### 5. Implementation Details

    - **Framework:** Implemented in PyTorch.
    - **Hardware:** Tested on systems with NVIDIA V100 GPUs.
    - **Source Code:** Available at [ByteTrack GitHub Repository](https://github.com/ifzhang/ByteTrack).

    ### 6. Limitations and Future Work

    - **High-Density Scenarios:** Performance may degrade in extremely crowded scenes due to occlusions.
    - **Run-time Performance:** While efficient, further optimization can be achieved with hardware accelerations and multi-threading.
    - **Appearance Model:** Potential improvements by integrating more advanced feature extraction networks.

    ### References

    - ByteTrack Paper: [arXiv:2110.06864v3](https://arxiv.org/abs/2110.06864v3)
    - MOTChallenge: [MOT17 Dataset](https://motchallenge.net/data/MOT17/), [MOT20 Dataset](https://motchallenge.net/data/MOT20/)

    Initialization:
    ---------------
    To initialize the BYTETracker with default values, you can use the following code:

    ```python
    BYTETracker(
        track_thresh=0.45,
        match_thresh=0.8,
        track_buffer=25,
        frame_rate=30,
        per_class=False,
        pixel_mapper=None,
        entry_polys=None,
        limit_entry=False,
        min_hits=3,  
        class_id_to_label: dict = None,
    )
    ```

    """
    def __init__(
        self,
        track_thresh=0.45,
        match_thresh=0.8,
        track_buffer=25,
        frame_rate=30,
        per_class=False,
        #Change Beign
        pixel_mapper= None,
        entry_polys=None,  # Add this line
        limit_entry=False,  # Add this line
        min_hits=3,  # Add this line for min_hits
        class_id_to_label: dict = None,
        #Change End
    ):
        """
        Parameters:
        -----------
        - track_thresh: float
            Threshold for tracking detections.
            - Lower values will increase the number of detections considered for tracking.
            - Higher values will be more selective.
        - match_thresh: float
            Threshold for matching detections to tracks.
            - Lower values increase the likelihood of matches.
            - Higher values are more selective.
        - track_buffer: int
            Number of frames to keep tracks alive without matching detections.
            - Lower values reduce buffer size, leading to quicker track updates.
            - Higher values increase buffer size, allowing for more stable tracking.
        - frame_rate: int
            Frame rate of the video.
        - per_class: bool
            Whether to track objects on a per-class basis.
        - pixel_mapper: optional
            Optional pixel mapper for converting pixel coordinates to map coordinates.
        - entry_polys: optional
            List of polygons defining entry zones. Limits track creation to these zones if specified.
        - limit_entry: bool
            Whether to limit track creation to specified entry zones.
        - min_hits: int
            Minimum number of hits before a track is considered confirmed.
        - class_id_to_label: dict, optional
            Dictionary mapping class IDs to class labels. Provides a readable format for class identifiers.

        Attributes:
        -----------
        - active_tracks: list
            List of currently active tracks.
        - lost_stracks: list
            List of lost tracks.
        - removed_stracks: list
            List of removed tracks.
        - frame_id: int
            Counter for the current frame ID.
        - track_buffer: int
            Number of frames to keep tracks alive without matching detections.
        - per_class: bool
            Whether to track objects on a per-class basis.
        - track_thresh: float
            Threshold for tracking detections.
        - match_thresh: float
            Threshold for matching detections to tracks.
        - det_thresh: float
            Detection threshold for filtering detections.
        - buffer_size: int
            Adjusted buffer size based on the frame rate.
        - max_time_lost: int
            Maximum time a track can be lost before being removed.
        - kalman_filter: KalmanFilterXYAH
            Instance of the Kalman filter used for motion prediction.
        - pixel_mapper: optional
            Instance of the pixel mapper for coordinate conversion.
        - entry_polys: list
            List of polygons defining entry zones.
        - limit_entry: bool
            Whether to limit track creation to specified entry zones.
        - min_hits: int
            Minimum number of hits before a track is considered confirmed.
        - next_final_id: int
            Counter for assigning final track IDs.

        Methods:
        --------
        - __init__(self, track_thresh=0.45, match_thresh=0.8, track_buffer=25, frame_rate=30, ...)
            Initializes the tracker with specific parameters.
        - update(self, dets: np.ndarray, img: np.ndarray = None, embs: np.ndarray = None) -> np.ndarray
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
        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_buffer = track_buffer

        self.per_class = per_class
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYAH()
        #Change Begin
        self.pixel_mapper = pixel_mapper  # Store pixel_mapper
        self.entry_polys = entry_polys if entry_polys is not None else []  # Add this line
        self.limit_entry = limit_entry  # Add this line
        self.min_hits = min_hits  # Initialize min_hits
        self.next_final_id = 1  # To track the next final ID to assign
        if self.limit_entry:
            if not self.entry_polys:
                raise ValueError("entry_polys cannot be None if limit_entry is True")
            if self.pixel_mapper is None:
                raise ValueError("pixel_mapper cannot be None if limit_entry is True")
        #Change End

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray = None, embs: np.ndarray = None, keypoints=None, keypoint_confidence_threshold=0.5, keypoint_indices=None) -> np.ndarray:
        """
        Update the tracker with new detections.

        Parameters:
        -----------
        - dets: np.ndarray
            Array of detections in the format (x1, y1, x2, y2, conf, class).
        - img: np.ndarray, optional
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
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        confs = dets[:, 4]

        remain_inds = confs > self.track_thresh

        inds_low = confs > 0.1
        inds_high = confs < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = dets[inds_second]
        dets = dets[remain_inds]

        #Change Begin
        keypoints_first = keypoints[remain_inds]
        keypoints_second = keypoints[inds_second]
        #Change End
        

        if len(dets) > 0:
            """Detections"""
            #Change Begin
            detections = [
                STrack(det, max_obs=self.max_obs, keypoints=(keypoints_first[i] if keypoints_first is not None else None), keypoint_confidence_threshold=keypoint_confidence_threshold) 
                              for i, det in enumerate(dets)
            ]
            #Change End
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
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
        # association the untrack to the low conf detections
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
        dists = iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
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
            if track.conf < self.det_thresh:
                continue
            
            # Only create if the map point is within an entry region
            create = False
            if self.limit_entry:
                map_point = geo.Point(*self.pixel_mapper.detection_to_map(track.xywh))
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
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

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
        # get confs of lost tracks
        output_stracks = [track for track in self.active_tracks if track.is_activated]
        #Change begin
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
    #Change end 


# id, class_id, conf


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
    - list: list
        Resulting list of tracks after subtraction.
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
    Remove duplicate tracks from two lists.

    Parameters:
    -----------
    - stracksa: list
        List of tracks.
    - stracksb: list
        List of tracks.

    Returns:
    --------
    - resa: list
        List of unique tracks from stracksa.
    - resb: list
        List of unique tracks from stracksb.
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
