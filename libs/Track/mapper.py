import numpy as np
import cv2
from typing import TYPE_CHECKING, Iterable


class PixelMapper(object):
    def __init__(self, pixel_array, target_array):
        pixel_array = np.float32(np.asarray(pixel_array).reshape(-1,1,2))
        target_array = np.float32(np.asarray(target_array).reshape(-1,1,2))

        self.M, self.mask = cv2.findHomography(pixel_array, target_array)

    def detection_to_map(self, coordinates, weights=None):
        ## using the average of center and bottom
        # x, y, w, h = detection.to_xywh()
        # center_approx = self.pixel_to_map((x, y))
        # bottom_center_x = x
        # bottom_center_y = y + h / 2
        # bottom_approx = self.pixel_to_map((bottom_center_x, bottom_center_y))
        # points = np.stack((center_approx, bottom_approx))
        # return np.average(points, axis=0, weights=weights)

        # using just the bottom of the bounding box
            
        x, y, w, h = coordinates
        bottom_center_x = x
        bottom_center_y = y + h / 2
        bottom_approx = self.pixel_to_map((bottom_center_x, bottom_center_y))
        return bottom_approx

    def pixel_to_map(self, points: Iterable[float]) -> np.ndarray:
        pixel = np.float32(np.asarray(points).reshape(-1,1,2))
        trans = cv2.perspectiveTransform(pixel, self.M)
        return np.squeeze(trans)
