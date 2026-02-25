# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot)

import numpy as np

from ....motion.kalman_filters.xyah_kf import KalmanFilterXYAH

#Change Begin
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


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(
        self,
        detection,
        id,
        n_init,
        max_age,
        ema_alpha,
    ):
        self.id = id
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        #Change Begin
        self.time_since_update = 0
        self.ema_alpha = ema_alpha
        self.current_map_pos = None
        self.last_map_pos = None
        self.state = TrackState.Tentative
        self.final_id = None  # Final track ID for uniformity in output
        # Keypoints handling
        self.keypoints = detection.keypoints
        self.keypoint_confidence_threshold = detection.keypoint_confidence_threshold
        self.filtered_keypoints = self.filter_keypoints(self.keypoints, self.keypoint_confidence_threshold)


        self.ankle_based_point = None
         # Initialize keypoint Kalman filter attribute
        self.keypoint_kalman_filter = None  # Add this line

        # Initialize the previous top_center as None (replace prev_top_middle)
        self.prev_top_center = None
        self.missing_keypoints_frames = 0
        #Change End
        self.features = []
        if detection.feat is not None:
            detection.feat /= np.linalg.norm(detection.feat)
            self.features.append(detection.feat)

        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilterXYAH()
        self.mean, self.covariance = self.kf.initiate(self.bbox)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def camera_update(self, warp_matrix):
        [a, b] = warp_matrix
        warp_matrix = np.array([a, b, [0, 0, 1]])
        warp_matrix = warp_matrix.tolist()
        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = warp_matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        """
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.bbox, self.conf
        )

        feature = detection.feat / np.linalg.norm(detection.feat)

        smooth_feat = (
            self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
        )
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.features = [smooth_feat]

        self.hits += 1
        self.time_since_update = 0
        #Change Begin
        # Update keypoints
        self.keypoints = detection.keypoints
        self.filtered_keypoints = self.filter_keypoints(self.keypoints, self.keypoint_confidence_threshold)
        #Change End
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    #Change Begin
    def filter_keypoints(self, keypoints, threshold):
        if keypoints is None:
            return None
        filtered_keypoints = keypoints.copy()
        for kp in filtered_keypoints:
            if kp[2] < threshold:
                kp[:] = [0, 0, 0]
        return filtered_keypoints
    #Change End
        
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
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
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm
#Change End
