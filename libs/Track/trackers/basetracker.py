import numpy as np
import cv2 as cv
import hashlib
import colorsys
from abc import ABC, abstractmethod
from ..utils import logger as LOGGER


class BaseTracker(ABC):
    #Change Begin
    COLOR_PRIME = 137.50776405003785  # Large prime number for better distribution while obtaining the colour
    #Change End
    def __init__(
        self, 
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_obs: int = 50,
        #Change Begin
        class_id_to_label: dict = None
        #Change End
    ):
        """
        Initialize the BaseTracker object with detection threshold, maximum age, minimum hits, 
        and Intersection Over Union (IOU) threshold for tracking objects in video frames.

        Parameters:
        - det_thresh (float): Detection threshold for considering detections.
        - max_age (int): Maximum age of a track before it is considered lost.
        - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
        - iou_threshold (float): IOU threshold for determining match between detection and tracks.
        - max_obs (int): Maximum number of observations to store.
        - class_id_to_label (dict, optional): Dictionary mapping class IDs to class labels.

        Attributes:
        - frame_count (int): Counter for the frames processed.
        - active_tracks (list): List to hold active tracks, may be used differently in subclasses.
        """
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.per_class_active_tracks = {}

        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes
        #Change Begin
        self.class_id_to_label = class_id_to_label if class_id_to_label is not None else {}
        #Change End
        
        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5
            #Change Begin
            LOGGER.info(f"Updated max observations to {self.max_obs}")
            #Change End

    @abstractmethod
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method 
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The update method needs to be implemented by the subclass.")
    
    #Change Begin
    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using a systematic approach in the HSV space.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Generate a consistent hue based on the ID
        hue = (id * self.COLOR_PRIME) % 360  # Use a large prime number for better distribution

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
        
        # Convert RGB from 0-1 range to 0-255 range and ensure the values are within range
        rgb_255 = tuple(int(min(max(component * 255, 0), 255)) for component in rgb)
        
        # Convert RGB to BGR for OpenCV
        bgr = rgb_255[::-1]
        
        return bgr
    #Change End
        
    #Change Begin
    def plot_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, conf_class: bool) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - conf_class (bool): Whether to display confidence and class information.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        thickness = 2
        color = self.id_to_color(id)

        # Get the class label from the dictionary, default to class ID if not found
        class_label = self.class_id_to_label.get(cls, cls)

        img = cv.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            thickness
        )

        if conf_class:
            # Display ID with larger font size
            img = cv.putText(
                img,
                f'id: {int(id)}',
                (int(box[0]), int(box[1]) - 20),  # Adjust y-position to fit all text
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,  # Larger font size for the ID
                color,
                thickness
            )

            # Calculate the width of the ID text to position the rest of the text correctly
            id_text_size = cv.getTextSize(f'id: {int(id)}', cv.FONT_HERSHEY_SIMPLEX, 1.0, thickness)[0]

            # Display the rest of the information with a smaller font size
            img = cv.putText(
                img,
                f' conf: {conf:.2f}, class: {class_label}',
                (int(box[0]) + id_text_size[0] + 10, int(box[1]) - 20),  # Adjust x-position based on ID text width
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,  # Smaller font size for confidence and class
                color,
                thickness
            )
        else:
            # Display only ID with larger font size
            img = cv.putText(
                img,
                f'id: {int(id)}',
                (int(box[0]), int(box[1]) - 20),  # Adjust y-position to fit all text
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,  # Larger font size for the ID
                color,
                thickness
            )

        return img, color
    #Change End


    def plot_trackers_trajectories(self, img: np.ndarray, observations: list, id: int) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations. Each point
        in the trajectory is represented by a circle, with the thickness increasing for more
        recent observations to visualize the path of movement.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories.
        - observations (list): A list of bounding box coordinates representing the historical
        observations of a tracked object. Each observation is in the format (x1, y1, x2, y2).
        - id (int): The unique identifier of the tracked object for color consistency in visualization.

        Returns:
        - np.ndarray: The image array with the trajectories drawn on it.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float (i + 1)) * 1.2)
            img = cv.circle(
                img,
                (int((box[0] + box[2]) / 2),
                int((box[1] + box[3]) / 2)), 
                2,
                color=self.id_to_color(int(id)),
                thickness=trajectory_thickness
            )
        return img


    def plot_results(self, img: np.ndarray, show_trajectories: bool) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
                
        return img

