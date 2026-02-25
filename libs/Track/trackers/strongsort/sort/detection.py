# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot)

class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray or None
        A feature vector that describes the object contained in this image.
    keypoints : array_like, optional
        Keypoints for the detected object in the format `(x, y, confidence)`.
    keypoint_confidence_threshold : float
        Confidence threshold for filtering keypoints.

    """

    def __init__(self, tlwh, conf, cls, det_ind, feat, keypoints=None, keypoint_confidence_threshold=0.5):
        self.tlwh = tlwh
        self.conf = conf
        self.cls = cls
        self.det_ind = det_ind
        self.feat = feat
        #Change Begin
        self.keypoints = keypoints
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        #Change End
        
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    #Change Begin
    def to_xywh(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret
    #Change End

