# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))
"""
This section provides functionalities for processing video frames using YOLO for object detection and a tracker for object tracking. It includes modules for video processing and utility functions to facilitate the handling of video and image data, including frame extraction and point mapping.

Modules:
--------
- `video_processing`: Contains the main function for processing video frames with object detection and tracking.
- `utils`: Provides utility functions for tasks such as loading pixel mappers, extracting frames, and creating point selection UIs.

Key Functions:
--------------
- `process_video_frames`: Processes video frames using YOLO and a tracker, optionally mapping tracked positions to a provided map.
- `load_pixel_mapper`: Loads point mappings and creates a PixelMapper instance.
- `load_entry_polygons`: Loads entry polygons and creates a list of Polygon instances.
- `extract_and_save_frame`: Extracts a specific frame from a video and saves it as an image.
- `create_point_selection_ui`: Creates a UI for robust point selection from frame and map images.
- `extract_entry_polygons_from_image`: Extract entry polygons from a map image through user interaction.

```
"""
from .SUPRA_Processing import run_detection
from .SUPRA_Processing import run_box_pose_detection
from .SUPRA_Processing import process_video

#from .SUPRA_Processing import process_video1