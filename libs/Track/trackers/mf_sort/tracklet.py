from .kalman_utils import construct_position_filter, project
import numpy as np
from .detection import Detection

class Tracklet():
    def __init__(self, frame_num, trk_id, det: Detection, mapper: None, min_hits=3, det_ind=-1):
        self.trk_id = trk_id
        self.mapper = mapper
        self.min_hits = min_hits
        self.hit_streak = 1  # Number of consecutive detections for the tracker
        self.time_since_update = 0
        self.last_update_frame = frame_num
        self.age = 1
        self.in_probation = True
        self.history = []

        self.current_map_pos = None
        self.last_map_pos = None

        self.kf = construct_position_filter(det)
    
        self.body_R = self.kf.R
        self.detection = det
        self.det_ind = det_ind
        self.update_current_map_pos()
        #Change Begin
        self.final_id = None  # Final track ID for uniformity in output
        self.keypoints = det.keypoints  # Add keypoints to the tracklet
        #Change End
        

        

    def predict(self):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        
        # Keep aspect ratio positive
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        # Keep height positive
        if((self.kf.x[7]+self.kf.x[3])<=0):
            self.kf.x[7] *= 0.0

        self.kf.predict()
        if self.mapper is not None:
            self.last_map_pos = self.current_map_pos
            self.update_current_map_pos()

    def update_current_map_pos(self):
        
        d = self.get_bbox()
        if self.mapper is not None:
        # Update map position 
            self.current_map_pos = self.mapper.detection_to_map(d.to_xywh())
        else:
            self.current_map_pos = None   
        

    def update(self, frame_num: int, detection: Detection, det_ind: int):
        self.kf.update(detection.to_xyah().reshape((4,1)))
        self.detection = detection
        self.det_ind = det_ind
        self.update_current_map_pos()
        self.hit(frame_num)
        self.keypoints = detection.keypoints  # Update keypoints

    def hit(self, frame_num: int):
        # Update hit streak if it has not already this frame
        if frame_num != self.last_update_frame:
            self.last_update_frame = frame_num
            self.hit_streak += 1
            self.time_since_update = 0
            if self.hit_streak >= self.min_hits:
                self.in_probation = False

    def get_bbox(self):
        ret = self.kf.x[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return Detection(ret, self.detection.confidence, self.detection.cls)

    def get_map_point(self):
        vel = [None, None]
        if self.current_map_pos is not None and self.last_map_pos is not None:
            vel = self.current_map_pos - self.last_map_pos
            vel = normalize(vel)
        return self.current_map_pos, vel

    def get_state(self):
        return project(self.kf)

    def record_state(self):
        state = self.get_state()
        self.history.append(state)

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm