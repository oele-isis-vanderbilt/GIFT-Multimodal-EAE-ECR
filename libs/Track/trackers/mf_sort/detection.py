import numpy as np

class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    cls : int
        Class ID of the detected object.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : float
        Detector confidence score.
    cls : int
        Class ID of the detected object.
    """

    def __init__(self, tlwh, confidence, cls, keypoints=None, keypoint_confidence_threshold=0.5):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.cls = int(cls)
        self.keypoints = self.filter_keypoints(keypoints, keypoint_confidence_threshold)

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

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xywh(self):
        """Convert bounding box to format `(center x, center y, width, height)`
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret