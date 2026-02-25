# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot)

import numpy as np

from ...appearance.reid_auto_backend import ReidAutoBackend
from ...motion.cmc import get_cmc_method
from .sort.detection import Detection
from .sort.tracker import Tracker
from ...utils.matching import NearestNeighborDistanceMetric
from ...utils.ops import xywh2xyxy, xyxy2xywh, xyxy2tlwh
from ...utils import PerClassDecorator
from ..basetracker import BaseTracker


class StrongSORT(BaseTracker):
    """
    StrongSORT Tracker for Video Analysis

    This tracker integrates advanced detection, embedding models, and inference tricks 
    to improve upon the classic DeepSORT tracker for robust and accurate tracking.

    Literature Summary:
    -------------------
    StrongSORT is an advanced multi-object tracking algorithm that significantly improves upon 
    the classic DeepSORT tracker by integrating advanced detection, embedding models, and inference 
    tricks. It also introduces two lightweight, plug-and-play algorithms, AFLink and GSI, to address 
    the common problems of missing association and missing detection.

    ### Technical README for StrongSORT Algorithm

    This document provides a concise technical overview of the StrongSORT algorithm, detailing its components 
    and their functionalities for multi-object tracking (MOT) in video streams.

    #### Overview
    StrongSORT is an advanced multi-object tracking algorithm that significantly improves upon the classic DeepSORT tracker 
    by integrating advanced detection, embedding models, and inference tricks. It also introduces two lightweight, plug-and-play 
    algorithms, AFLink and GSI, to address the common problems of missing association and missing detection.

    ### 1. Introduction to StrongSORT

    - **Multi-Object Tracking (MOT):** The objective is to detect and track all objects in a scene while maintaining their unique identifiers.
    - **Tracking-by-Detection Paradigm:** Utilizes object detection followed by a tracking step.
    - **Challenges Addressed:** Enhances detection accuracy, reduces fragmented trajectories, and improves overall tracking robustness.

    ### 2. Algorithm Components

    #### 2.1 Advanced Modules
    - **Detector:** Uses a strong object detector (e.g., YOLOX-X) for improved detection accuracy.
    - **Embedding Model:** Employs a stronger appearance feature extractor (e.g., BoT) for more discriminative features.
    - **EMA Feature Updating:** Replaces the feature bank mechanism with an Exponential Moving Average (EMA) updating strategy to reduce detection noise.

    #### 2.2 Camera Motion Compensation (ECC)
    - **Purpose:** Corrects the predicted bounding box locations affected by camera motion.
    - **Method:** Uses enhanced correlation coefficient maximization to estimate global rotation and translation between frames.

    #### 2.3 NSA Kalman Filter
    - **Motion Model:** Uses a noise-scale adaptive (NSA) Kalman filter to predict object trajectories with adaptive measurement noise covariance based on detection confidence.

    #### 2.4 Matching Strategies
    - **Motion Cost Integration:** Combines appearance and motion costs for a more robust matching process.
    - **Vanilla Matching:** Uses a simple global linear assignment algorithm for matching, improving accuracy by removing additional prior constraints.

    ### 3. Plug-and-Play Algorithms

    #### 3.1 AFLink (Appearance-Free Link Model)
    - **Purpose:** Solves the problem of missing association by linking short tracklets into complete trajectories without using appearance information.
    - **Method:** Uses a two-branch framework to extract spatiotemporal features and predict the connectivity between tracklets based on spatiotemporal information alone.

    #### 3.2 GSI (Gaussian-Smoothed Interpolation)
    - **Purpose:** Addresses the problem of missing detection by smoothing trajectories and filling gaps using Gaussian process regression.
    - **Method:** Models nonlinear motion and applies Gaussian process regression to produce more accurate and stable localizations.

    ### 4. Algorithm Workflow

    1. **Initialization:**
       - Initialize tracklets and set the detection score threshold.

    2. **Detection and Classification:**
       - Detect objects in each frame and classify detections based on the score threshold.

    3. **Kalman Filter Prediction:**
       - Predict new locations for tracklets using the NSA Kalman filter.

    4. **Camera Motion Compensation:**
       - Apply ECC to correct predictions affected by camera motion.

    5. **First Association:**
       - Compute similarity (appearance and motion) between high-confidence detections and tracklets.
       - Use global linear assignment for matching and update tracklets.

    6. **Second Association:**
       - Use AFLink to link short tracklets into complete trajectories.

    7. **Tracklet Update:**
       - Update matched tracklets with new positions and appearance features.
       - Remove unmatched tracklets if they remain unmatched for a certain number of frames.

    8. **Tracklet Initialization:**
       - Initialize new tracklets for unmatched high-confidence detections.

    9. **Interpolation:**
       - Apply GSI to smooth trajectories and fill gaps due to missing detections.

    ### 5. Performance and Benchmarks

    - **Datasets:** Evaluated on MOT17, MOT20, DanceTrack, and KITTI datasets.
    - **Metrics:** Uses HOTA, IDF1, MOTA, AssA, DetA, and FPS to evaluate tracking performance.
    - **Results:** Achieves state-of-the-art performance with significant improvements in IDF1, HOTA, and AssA scores compared to previous methods.

    ### 6. Implementation Details

    - **Framework:** Implemented in PyTorch.
    - **Hardware:** Tested on systems with NVIDIA V100 GPUs.
    - **Source Code:** Available at [StrongSORT GitHub Repository](https://github.com/dyhBUPT/StrongSORT) and [MMTracking GitHub Repository](https://github.com/open-mmlab/mmtracking).

    ### 7. Limitations and Future Work

    - **Running Speed:** Relatively low compared to joint trackers and appearance-free separate trackers due to reliance on a strong detector and appearance model.
    - **Detection Threshold:** High threshold may lead to missing detections, which can be addressed by a more elaborate threshold strategy.
    - **False Associations:** AFLink does not handle false associations well, indicating a need for stronger global link strategies.

    ### References

    - StrongSORT Paper: [arXiv:2202.13514v2](https://arxiv.org/abs/2202.13514v2)
    - MOTChallenge: [MOT17 Dataset](https://motchallenge.net/data/MOT17/), [MOT20 Dataset](https://motchallenge.net/data/MOT20/)

    Initialization:
    ---------------
    To initialize the BYTETracker with default values, you can use the following code:

    ```python
    tracker = StrongSORT(
        model_weights=Path('ReID_models/osnet_x0_25_msmt17.pt'),  
        device='mps',  
        fp16=False,  
        per_class=False, 
        max_dist=0.2,  
        max_iou_dist=0.7,  
        max_age=30,  
        n_init=20,  
        nn_budget=100,  
        mc_lambda=0.995, 
        ema_alpha=0.9,  
        pixel_mapper=mapper, 
        limit_entry=False, 
        entry_polys=entry_polys, 
    )    
    ```
    """
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=1,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
        #Change Begin
        pixel_mapper=None,
        limit_entry=False,
        entry_polys=None,
        class_id_to_label: dict = None,
        #Change End
    ):
        """
        Parameters:
        -----------
        - model_weights: str
            Path to the ReID model weights.
        - device: str
            Device to run the model on (e.g., 'cpu', 'cuda').
        - fp16: bool
            Whether to use half-precision for computation.
        - per_class: bool, optional
            Whether to track objects on a per-class basis. Default is False.
            If True, the tracker will maintain separate trackers for each class of object.
        - max_dist: float, optional
            Maximum distance for appearance matching. Default is 0.2.
            Detections and tracks will be associated if their appearance distance is below this threshold.
        - max_iou_dist: float, optional
            Maximum IOU distance for matching. Default is 0.7.
            Detections and tracks will be associated if their IoU is below this threshold.
        - max_age: int, optional
            Maximum number of frames to keep a track alive without updates. Default is 30.
            Tracks that have not been updated for more than `max_age` frames will be deleted.
        - n_init: int, optional
            Number of frames that a track remains in the initialization phase. Default is 1.
            Tracks will be considered tentative until they have been updated at least `n_init` times.
        - nn_budget: int, optional
            Maximum size of the appearance descriptor buffer. Default is 100.
            Limits the number of appearance descriptors stored for each track.
        - mc_lambda: float, optional
            Weighting factor for motion cost. Default is 0.995.
            Controls the influence of the motion model in the tracking process.
        - ema_alpha: float, optional
            Smoothing factor for EMA updating. Default is 0.9.
            Determines the smoothing applied to the appearance descriptors.
        - pixel_mapper: object, optional
            Pixel mapper for converting pixel coordinates to map coordinates.
            If provided, the tracker will update and use map coordinates for tracking.
        - limit_entry: bool, optional
            Whether to limit track creation to specified entry zones. Default is False.
            If True, new tracks will only be created if detections fall within specified entry zones.
        - entry_polys: list of polygons, optional
            List of polygons defining entry zones. Required if `limit_entry` is True.
            New tracks will only be created if detections fall within these polygons.
        - class_id_to_label: dict, optional
            Dictionary mapping class IDs to class labels.
            Provides human-readable labels for class IDs.

        Attributes:
        -----------
        - per_class: bool
            Indicates whether the tracker is operating on a per-class basis.
        - pixel_mapper: object
            Mapper for converting pixel coordinates to map coordinates.
        - next_final_id: int
            Counter for assigning unique final IDs to tracks.
        - tracker: Tracker
            The main tracking object that handles the logic for tracking and associating detections.
        - model: Model
            The appearance feature extractor model.
        - cmc: object
            Camera Motion Compensation (CMC) object for handling camera motion corrections.

        Methods:
        --------
        - __init__(self, model_weights, device, fp16, per_class=False, max_dist=0.2, max_iou_dist=0.7, max_age=30, n_init=1, nn_budget=100, mc_lambda=0.995, ema_alpha=0.9, pixel_mapper=None, limit_entry=False, entry_polys=None, class_id_to_label=None)
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
        - 'track_id_raw': Raw ID of the track.
        - 'track_id': Final track ID.
        - 'confidence': Confidence score of the detection.
        - 'class': Class ID of the object.
        - 'detection_index': Index of the detection.
        - 'current_map_pos': Current map position of the object (if `pixel_mapper` is provided).
        - 'map_velocity': Velocity of the object on the map (if `pixel_mapper` is provided).
        """

        #Change Beign
        super().__init__(class_id_to_label=class_id_to_label)
        #Change End
        self.per_class = per_class
        rab = ReidAutoBackend(
            weights=model_weights, device=device, half=fp16
        )
        self.model = rab.get_backend()
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
            #Change Begin
            limit_entry=limit_entry,
            entry_polys=entry_polys,
            pixel_mapper=pixel_mapper,
            #Change End 
        )
        self.cmc = get_cmc_method('ecc')()
        #Change Begin
        self.pixel_mapper = pixel_mapper  # Store pixel_mapper
        self.next_final_id = 1  # To track the next final ID to assign
        if limit_entry:
            if not entry_polys:
                raise ValueError("entry_polys cannot be None if limit_entry is True")
            if self.pixel_mapper is None:
                raise ValueError("pixel_mapper cannot be None if limit_entry is True")
        #Change End 

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, keypoints: np.ndarray = None, keypoint_confidence_threshold=0.5, keypoint_indices=None) -> np.ndarray:
        """
        Update the tracker with new detections.

        Parameters:
        -----------
        - dets: np.ndarray
            Array of detections in the format (x1, y1, x2, y2, conf, cls).
        - img: np.ndarray
            Image corresponding to the detections.
        - embs: np.ndarray, optional
            Array of embeddings for appearance features.
        - keypoints: np.ndarray, optional
            Array of keypoints in the format (num_detections, num_keypoints, 3), where each keypoint has (x, y, confidence).
        - keypoint_confidence_threshold: float
            The confidence threshold for filtering keypoints.

        Returns:
        --------
        - outputs: list
            List of dictionaries, each representing a track.
        """
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # extract appearance information for each detection
        if embs is not None:
            features = embs
        else:
            features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        # Include keypoints in Detection creation
        #Change Begin
        detections = [
            Detection(
                box, 
                conf, 
                cls, 
                det_ind, 
                feat, 
                keypoints=keypoints[i] if keypoints is not None else None, 
                keypoint_confidence_threshold=keypoint_confidence_threshold
            )
            for i, (box, conf, cls, det_ind, feat) in enumerate(zip(tlwh, confs, clss, det_ind, features))
        ]
        #Change End

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        #Change Begin
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue

            if track.final_id is None:  # Assign a final ID if not already assigned
                    track.final_id = self.next_final_id
                    self.next_final_id += 1

            
            
            if self.pixel_mapper is not None:
                # Update map position and calculate velocity
                track.update_map_position(self.pixel_mapper, xyxy2xywh(track.to_tlbr()), keypoints=track.filtered_keypoints, keypoint_indices=keypoint_indices, smoothing_factor=0.7)
                map_vel = track.calculate_velocity()
                # Add map position and velocity to dictionary
                map_pos = track.current_map_pos
            else:
                map_pos = None
                map_vel = [None, None]

            t,l,w,h = track.to_tlwh()

            outputs.append({
                    'top_left_x': t,
                    'top_left_y': l,
                    'width': w,
                    'height': h,
                    'track_id_raw': track.id,
                    'track_id':track.final_id,
                    'confidence': track.conf,
                    'class': track.cls,
                    'detection_index': track.det_ind,
                    'current_map_pos': map_pos,
                    'map_velocity': map_vel,
                    'keypoints': track.filtered_keypoints,
                    'ankle_based_point': track.ankle_based_point
                })

        return outputs if len(outputs) > 0 else np.array([])
        #Change End 
