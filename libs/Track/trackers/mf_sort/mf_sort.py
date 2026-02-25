from typing import List, Tuple, Dict
import numpy as np
import shapely.geometry as geo
from .tracklet import Tracklet
from .distance import iou_cost_function, iou
from . import matching
from .detection import Detection
from ..basetracker import BaseTracker


class MF_SORT(BaseTracker):
    """
    MF-SORT Tracker for Video Analysis

    This tracker integrates motion features for robust and efficient tracking,
    particularly suited for static camera scenarios.

    Literature Summary:
    -------------------
    MF-SORT (Simple Online and Realtime Tracking with Motion Features) is a multi-object tracking 
    algorithm designed to balance performance and efficiency by focusing on motion features 
    instead of appearance features during data association. This approach is particularly suited 
    for static camera video applications such as pedestrian surveillance and sports video analysis.

    ### Technical README for MF-SORT Algorithm

    This document provides a concise technical overview of the MF-SORT algorithm, detailing its 
    components and their functionalities for multi-object tracking (MOT) in video streams, especially 
    for static camera scenarios.

    #### Overview
    MF-SORT (Simple Online and Realtime Tracking with Motion Features) is a multi-object tracking 
    algorithm designed to balance performance and efficiency by focusing on motion features instead 
    of appearance features during data association. This approach is particularly suited for static 
    camera video applications such as pedestrian surveillance and sports video analysis.

    ### 1. Introduction to MF-SORT

    - **Multi-Object Tracking (MOT):** The goal is to estimate bounding boxes and identities of objects in videos.
    - **Tracking-by-Detection Paradigm:** Utilizes object detection followed by a tracking step.
    - **Challenges Addressed:** Reduces computational complexity while maintaining accuracy by using motion features for data association.

    ### 2. Algorithm Components

    #### 2.1 Kalman Filter (KF)
    - **Motion Model:** Uses a Kalman filter to predict the trajectory of objects.
    - **State Vector:** Includes position, velocity, aspect ratio, and height of the bounding box to enhance trajectory predictions.

    #### 2.2 Motion Features for Data Association
    - **Mahalanobis Distance:** Computes the similarity between detection and tracking boxes using squared Mahalanobis distance, which is faster and more reliable than appearance-based metrics.
    - **Thresholding:** Uses specific thresholds (thca for cascade matching and thgo for global matching) to ensure robust association.

    ### 3. Algorithm Workflow

    1. **Initialization:**
       - Initialize tracklets and set the necessary thresholds for Mahalanobis distance.

    2. **Detection and Classification:**
       - Detect objects in each frame.

    3. **Kalman Filter Prediction:**
       - Predict new locations for tracklets using the Kalman filter.

    4. **Matching Module:**
       - Perform cascade matching based on the priority of tracking boxes.
       - Use global matching for unmatched tracking boxes.

    5. **Tracklet Update:**
       - Update matched tracklets with new positions.
       - Remove unmatched tracklets if they remain unmatched for a set number of frames (max_age).

    6. **Tracklet Initialization:**
       - Initialize new tracklets for unmatched detection boxes, considering IoU gating to avoid false positives.

    ### 4. Performance and Benchmarks

    - **Datasets:** Evaluated on MOTChallenge and MOT-SOCCER datasets.
    - **Metrics:** Uses MOTA (Multiple Object Tracking Accuracy), False Positives (FP), False Negatives (FN), and ID Switches (IDSW).
    - **Results:** Achieves competitive accuracy with significantly reduced computational complexity compared to DeepSORT.

    ### 5. Implementation Details

    - **Framework:** Implemented by modifying DeepSORT, primarily in the initialization and matching stages.
    - **Hardware:** Tested on systems with i7 7700HQ CPUs and Nvidia GTX 1060 GPUs.
    - **Source Code:** Detailed implementation steps and parameters are provided in the paper.

    ### 6. Limitations and Future Work

    - **Static Cameras:** Specifically optimized for static camera scenarios; may not perform as well with moving cameras.
    - **Appearance Model:** Does not utilize appearance features, which can be beneficial in certain scenarios.
    - **Run-time Performance:** While efficient, further optimization can be achieved with advanced hardware.

    ### References

    - MF-SORT Paper: [MF-SORT: Simple Online and Realtime Tracking with Motion Features](https://doi.org/10.1007/978-3-030-34120-6_13)
    - MOTChallenge: [MOTChallenge Dataset](https://motchallenge.net/data/MOT17/)
    - MOT-SOCCER: [MOT-SOCCER Dataset](https://github.com/jozeeandfish/motsoccer)    

    Initialization:
    ---------------
    To initialize the BYTETracker with default values, you can use the following code:

    ```python
    tracker = MF_SORT(
        pixel_mapper=mapper,         
        entry_polys=entry_polys,            
        max_age=30,                        
        min_hits=5,           
        iou_threshold=0.5,              
        limit_entry=False,                   
    ```
    """
    def __init__(self, pixel_mapper: None, entry_polys: List[geo.Polygon], max_age=30, min_hits=3, iou_threshold=0.7, limit_entry=False, class_id_to_label: dict = None,):
        """
        Initialize the MF_SORT tracker.

        Parameters:
        -----------
        - pixel_mapper: Optional pixel mapper for converting pixel coordinates to map coordinates.
        - entry_polys: List of polygons defining entry zones.
        - max_age: int
            Maximum number of frames to keep a track alive without updates.
        - min_hits: int
            Minimum number of hits before a track is considered confirmed.
        - iou_threshold: float
            Threshold for Intersection over Union (IoU) matching.
        - limit_entry: bool
            Whether to limit track creation to specified entry zones.
        - class_id_to_label: dict, optional
            Dictionary mapping class IDs to class labels.

        Methods:
        --------
        - __init__(self, pixel_mapper, entry_polys, max_age=30, min_hits=3, iou_threshold=0.7, limit_entry=False, class_id_to_label=None)
            Initializes the tracker with specific parameters.
        - get_trackers(self)
            Returns the current and lost trackers.
        - _init_tracklets(self, dets, indices)
            Initializes new tracklets from detections.
        - predict(self)
            Predicts the next state for all tracklets using the Kalman filter.
        - _match(self, dets)
            Matches detections to existing tracklets.
        - step(self, dets)
            Processes a step of tracking by predicting, matching, and updating tracklets.
        - remove_trackers(self)
            Removes tracklets that have not been updated for a specified number of frames.
        - get_output(self)
            Returns the current state of tracklets in a structured format.
        - update(self, detections, frame=None)
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
        - 'current_map_pos': Current map position of the object (if `pixel_mapper` is provided).
        - 'map_velocity': Velocity of the object on the map (if `pixel_mapper` is provided).
        - 'detection_index': Index of the detection.
        """
        #Change Beign
        super().__init__(class_id_to_label=class_id_to_label)
        #Change End
        self.pixel_mapper = pixel_mapper
        self.entry_polys = entry_polys
        self.limit_entry = limit_entry
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracklet_counter = 1
        self.iou_threshold = iou_threshold
        self.frames = 0

        self.trackers: List[Tracklet] = []
        self.lost_trackers: List[Tracklet] = []
        # Change Begin
        self.next_final_id = 1  # To track the next final ID to assign
        if self.limit_entry:
            if not self.entry_polys:
                raise ValueError("entry_polys cannot be None if limit_entry is True")
            if self.pixel_mapper is None:
                raise ValueError("pixel_mapper cannot be None if limit_entry is True")
        # Change End

    def get_trackers(self):
        """
        Returns the current and lost trackers.

        Returns:
        --------
        - trackers: list
            List of current trackers.
        - lost_trackers: list
            List of lost trackers.
        """
        return self.trackers, self.lost_trackers

    def _init_tracklets(self, dets: List[Detection], indices: List[int]):
        """
        Initialize new tracklets from detections.

        Parameters:
        -----------
        - dets: List[Detection]
            List of detections.
        - indices: List[int]
            List of detection indices.
        """
        new_trackers = []
        for det, ind in zip(dets, indices):
            create = False
            # Only create if the map point is within an entry region
            if self.limit_entry:
                map_point = geo.Point(*self.pixel_mapper.detection_to_map(det.to_xywh(), [0, 1]))
                for poly in self.entry_polys:
                    if poly.contains(map_point):
                        create = True
            else:
                create = True

            # Gate detection by IOU with current trackers
            if create:
                for trk in self.trackers:
                    if iou(det, trk.get_bbox()) > self.iou_threshold:
                        create = False
                        break

            if create:
                
                trk = Tracklet(self.frames, self.tracklet_counter, det, self.pixel_mapper, self.min_hits, ind)
                new_trackers.append(trk)
                self.tracklet_counter += 1
                # print(f"New tracklet created: trk_id={trk.trk_id}, det={det}")

        self.trackers += new_trackers

    def predict(self):
        """
        Predict the next state for all tracklets using the Kalman filter.
        """
        self.frames += 1
        for trk in self.trackers:
            trk.predict()
    
    def _match(self, dets: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracklets.

        Parameters:
        -----------
        - dets: List[Detection]
            List of detections.

        Returns:
        --------
        - matches: list of tuple
            List of matched tracklet and detection indices.
        - unmatched_tracks: list
            List of unmatched tracklet indices.
        - unmatched_detections: list
            List of unmatched detection indices.
        """
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = []
        unconfirmed_tracks = []
        for i, t in enumerate(self.trackers):
            if t.in_probation:
                unconfirmed_tracks.append(i)
            else:
                confirmed_tracks.append(i)

        # Associate confirmed tracks using matching cascade
        matches_a, unmatched_tracks_a, unmatched_detections = \
            matching.matching_cascade(
                self.max_age, self.trackers, dets, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + \
            [t for t in unmatched_tracks_a if self.trackers[t].time_since_update == 1]
        unmatched_tracks_a = [
            t for t in unmatched_tracks_a if self.trackers[t].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = matching.min_cost_matching(
            iou_cost_function, 1-self.iou_threshold, self.trackers, dets, iou_track_candidates, unmatched_detections)

        # Construct final matching results
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        # print(f"Matches: {matches}")
        # print(f"Unmatched tracks: {unmatched_tracks}")
        # print(f"Unmatched detections: {unmatched_detections}")


        return matches, unmatched_tracks, unmatched_detections

    def step(self, dets: List[Detection]):
        """
        Process a step of tracking by predicting, matching, and updating tracklets.

        Parameters:
        -----------
        - dets: List[Detection]
            List of detections.
        """
        # Associate predictions to observations
        matched, unmatched_trks, unmatched_dets = self._match(dets)
        for trk_id, det_id in matched:
            trk = self.trackers[trk_id]
            trk.update(self.frames, dets[det_id], det_id)

        # Create new trackers for unmatched detections
        self._init_tracklets([dets[d] for d in unmatched_dets], unmatched_dets)

    def remove_trackers(self):
        """
        Remove tracklets that have not been updated for a specified number of frames.
        """
        new_trackers = []
        # Delete trackers with too high age
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                self.lost_trackers.append(trk)
            else:
                new_trackers.append(trk)

        # Update the active trackers
        self.trackers = new_trackers

    def get_output(self) -> List[Dict]:
        """
        Returns the current state of tracklets in a structured format.

        Returns:
        --------
        - outputs: list of dict
            List of dictionaries, each representing a track with the following keys:
            - 'top_left_x': Top-left x-coordinate of the bounding box.
            - 'top_left_y': Top-left y-coordinate of the bounding box.
            - 'width': Width of the bounding box.
            - 'height': Height of the bounding box.
            - 'track_id_raw': Raw ID of the track.
            - 'track_id': Final track ID.
            - 'confidence': Confidence score of the detection.
            - 'class': Class ID of the object.
            - 'current_map_pos': Current map position of the object (if `pixel_mapper` is provided).
            - 'map_velocity': Velocity of the object on the map (if `pixel_mapper` is provided).
            - 'detection_index': Index of the detection.
        """
        outputs = []
        for trk in self.trackers:
            # If alive and not in probation, add it to the return
            if (trk.time_since_update == 0 and not trk.in_probation):

                if trk.final_id is None:  # Assign a final ID if not already assigned
                    trk.final_id = self.next_final_id
                    self.next_final_id += 1

                box = trk.get_bbox()
                tlwh = box.tlwh
                map_point, map_vel = trk.get_map_point()

                # Create dictionary for each track
                track_dict = {
                    'top_left_x': tlwh[0,0],
                    'top_left_y': tlwh[1,0],
                    'width': tlwh[2,0],
                    'height': tlwh[3,0],
                    'track_id_raw': trk.trk_id,
                    'track_id':trk.final_id,
                    'confidence': trk.detection.confidence,
                    'class': trk.detection.cls,
                    'current_map_pos': map_point,
                    'map_velocity': map_vel,
                    'detection_index' : trk.det_ind,
                    'keypoints': trk.keypoints  # Include filtered keypoints in the output
                }

                outputs.append(track_dict)
        return outputs
    
    def update(self, detections: np.ndarray, frame=None, keypoints=None, keypoint_confidence_threshold=0.5) -> List[Dict]:
        """
        Updates the tracker with new detections and returns the current tracks.

        Parameters:
        -----------
        - detections: np.ndarray
            Array of detections in the format (x1, y1, x2, y2, conf, cls).
        - frame: np.ndarray, optional
            Frame image corresponding to the detections.

        Returns:
        --------
        - outputs: list
            List of dictionaries, each representing a track.
        """
        # Convert detections from format N X (x1, y1, x2, y2, conf, cls) to list of Detection objects
        det_list = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            tlwh = [x1, y1, x2 - x1, y2 - y1]
            kp = keypoints[i] if keypoints is not None else None
            detection = Detection(tlwh, conf, cls, keypoints=kp, keypoint_confidence_threshold=keypoint_confidence_threshold)
            det_list.append(detection)
        
        self.predict()
        self.step(det_list)
        self.remove_trackers()
        return self.get_output()