# Motion Trackers for Video Analysis -Surya ⎜(Extended from Mikel Broström's work on boxmot(10.0.81))


"""
Track Package
=============

The `Track` package is designed for video analysis, focusing on object detection and tracking using YOLO models and various trackers. This package includes modules for processing video frames, utility functions for handling video and image data, and a variety of trackers for different tracking needs.

Package Structure:
------------------
- `appearance`: 
  Contains modules for handling appearance-based tracking and object re-identification.

- `configs`: 
  Contains configuration files and settings for different models and trackers.

- `mapper`: 
  Contains classes and functions for mapping pixel coordinates to world coordinates.
  - `PixelMapper`: Class for handling coordinate transformations.

  Importing:
  ```
  from Track.mapper import PixelMapper
  ```

- `motion`: 
  Contains modules for motion-based tracking and analysis.

- `postprocessing`: 
  Contains functions for post-processing tracking results, such as trajectory smoothing and interpolation.

- `processing`: 
  This folder contains functions and utilities for video frame processing, mapping, and point selection. It is crucial for application purposes.
  - `process_video_frames`: Processes video frames using YOLO for object detection and a tracker for object tracking, optionally mapping tracked positions to a provided map.
  - `combine_frames`: Combines two frames side-by-side.
  - `create_directory_if_not_exists`: Creates a directory if it does not exist.
  - `utils`: Utility functions for processing.
    - `load_pixel_mapper`: Loads point mappings from a file and creates a `PixelMapper` instance for spatial mapping.
    - `load_entry_polygons`: Loads entry polygons from a file and creates a list of Polygon instances for defining regions of interest.
    - `extract_and_save_frame`: Extracts a specific frame from a video and saves it as an image file.
    - `create_point_selection_ui`: Creates a user interface for selecting and mapping points between frame and map images.
    - `extract_entry_polygons_from_image`: Extract entry polygons from a map image through user interaction.

  Importing:
  ```
  from Track.processing import process_video_frames, combine_frames, create_directory_if_not_exists
  from Track.processing.utils import load_pixel_mapper, load_entry_polygons, extract_and_save_frame, create_point_selection_ui, extract_entry_polygons_from_image
  ```

- `tracker_zoo`(Continous development): 
  A place to define a collection of pre-trained trackers optimized for various tracking scenarios.

- `trackers`: 
  This folder contains different tracking algorithms and their implementations. It is another key module for application purposes.
  - Various tracker classes implement methods for updating and managing tracked objects.
  - Example Trackers:
    - `BoTSORT`
    - `BYTETracker`
    - `DeepOCSort`
    - `HybridSORT`
    - `MF_SORT`
    - `OCSort`
    - `StrongSORT`

  Importing:
  ```
  from Track.trackers import BoTSORT, BYTETracker, DeepOCSort, HybridSORT, MF_SORT, OCSort, StrongSORT
  ```

- `utils`: 
  This folder contains general utility functions for the `Track` package.

  Importing:
  ```
  from Track.utils import general_utility_function
  ```

Usage:
------
For detailed information on each tracker and utility function, use the `help()` function in Python. This will provide a full overview of the class or function, including its parameters, return values, and usage examples.

Example:
--------
To get help on a specific tracker class or utility function:
```
help(TrackerClassName)
help(load_pixel_mapper)
```

Summary:
--------
The `Track` package is a powerful tool for video analysis, offering a range of functionalities for object detection, tracking, and spatial mapping. By using the provided modules and functions, users can efficiently process videos, extract meaningful data, and analyze object movements within defined regions of interest.
"""


__version__ = '2.0'

from .postprocessing.gsi import gsi
from .tracker_zoo import create_tracker, get_tracker_config

from .trackers.bytetrack.byte_tracker import BYTETracker

from .trackers.botsort.bot_sort import BoTSORT

from .trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT

from .trackers.hybridsort.hybridsort import HybridSORT

from .trackers.ocsort.ocsort import OCSort as OCSORT

from .trackers.strongsort.strong_sort import StrongSORT

from .trackers.mf_sort.mf_sort import MF_SORT

TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort']

__all__ = ("__version__",
           "StrongSORT", "OCSORT", "BYTETracker", "BoTSORT", "DeepOCSORT", "HybridSORT", "MF_SORT",
           "create_tracker", "get_tracker_config", "gsi", "PixelMapper")
