"""
utils.py

This module provides utility functions for computer vision tasks, which are essential for preprocessing and interacting with video and image data. These utilities play a crucial role in various stages of video analytics and object tracking projects.

Functions:
- load_pixel_mapper(point_map_path): Loads point mappings from a file and creates a PixelMapper instance.
    - Helpful for converting detected object coordinates in video frames to corresponding coordinates on a map. This is crucial for applications like object tracking and movement analysis in a spatial context. In the project, it is used to map the positions of tracked objects from the video frame to a spatial map for visualization and further analysis.

- load_entry_polygons(entry_polys_path): Loads entry polygons from a file and creates a list of Polygon instances.
    - Useful for defining regions of interest (ROIs) in the map, such as entry and exit zones. This helps in analyzing object movements and detecting events like entering or leaving specific areas. In the project, it is used to define areas on the map where specific actions or conditions need to be monitored, such as detecting when an object enters a restricted zone.

- extract_and_save_frame(video_path, frame_number, output_image_path): Extracts a specific frame from a video and saves it as an image.
    - Essential for isolating particular frames for detailed analysis, debugging, or use as reference images in further processing steps. In the project, it is used to capture specific frames that may require closer inspection or manual annotation, such as frames where an object is first detected or tracked.

- capture_frame_points(event, x, y, flags, param): Captures points from the frame image based on mouse events.
    - Used in the interactive point selection process to map coordinates between the video frame and the map. This function captures the selected points in the frame image. In the project, it is used in the point selection UI to select corresponding points in the video frame for accurate mapping to the spatial map.

- capture_map_points(event, x, y, flags, param): Captures points from the map image based on mouse events.
    - Complements capture_frame_points by capturing the corresponding points in the map image. Together, these functions facilitate accurate point mapping between frames and maps. In the project, it is used in the point selection UI to select corresponding points on the map image to complete the mapping from video frames to spatial coordinates.

- create_point_selection_ui(extracted_frame_path, map_image_path, output_file_path): Creates a UI for robust point selection from frame and map images.
    - Provides a user-friendly interface for selecting and mapping points between video frames and map images. This is vital for tasks that require precise spatial mapping and calibration. In the project, it is used to create an interactive interface where users can manually select and map points between extracted video frames and the corresponding map images, ensuring accurate spatial alignment and calibration.
"""

import numpy as np
from ..mapper import PixelMapper
import shapely.geometry as geo
import cv2


def load_pixel_mapper(point_map_path):
    """
    Load the point mapping file and create a PixelMapper instance.

    Parameters:
    - point_map_path (str): The path to the point mapping file.
        - The file should contain comma-separated values representing pixel coordinates and corresponding map coordinates.

    Returns:
    - PixelMapper: An instance of the PixelMapper class, initialized with the loaded pixel and map coordinates.

    Example:
    point_map_path = "path/to/point_map.csv"
    pixel_mapper = load_pixel_mapper(point_map_path)
    """
    # Load the point mapping file
    point_mapping = np.loadtxt(point_map_path, delimiter=",", dtype="int")
    pixel_arr = point_mapping[:, :2]
    map_arr = point_mapping[:, 2:]

    # Create and return the PixelMapper instance
    mapper = PixelMapper(pixel_arr, map_arr)
    return mapper

def load_entry_polygons(entry_polys_path):
    """
    Load the entry polygons file and create a list of Polygon instances.

    Parameters:
    - entry_polys_path (str): The path to the entry polygons file.
        - The file should contain comma-separated values representing the vertices of polygons. Lines starting with '#' are treated as comments.

    Returns:
    - list: A list of shapely.geometry.Polygon instances representing the entry polygons.

    Example:
    entry_polys_path = "path/to/entry_polygons.txt"
    entry_polygons = load_entry_polygons(entry_polys_path)
    """
    lines = []
    with open(entry_polys_path, "r") as entry_file:
        lines = entry_file.readlines()
    
    entry_polys = []
    for line in lines:
        if line[0] == "#":
            continue
        values = [int(v) for v in line.split(",")]
        points = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
        entry_polys.append(geo.Polygon(points))
    
    return entry_polys

def extract_and_save_frame(video_path, output_image_path, frame_number):
    """
    Extract a specified frame from a video and save it as an image file.

    Parameters:
    - video_path (str): The path to the video file.
        - The video file should be in a format supported by OpenCV.
    - frame_number (int): The frame number to extract.
        - The frame number should be within the range of total frames in the video.
    - output_image_path (str): The path to save the extracted frame image.
        - The output path should include the desired file name and extension (e.g., .jpg, .png).

    Example:
    video_path = "path/to/video.mp4"
    frame_number = 150
    output_image_path = "path/to/output/frame150.jpg"
    extract_and_save_frame(video_path, frame_number, output_image_path)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} saved as {output_image_path}")
    else:
        print(f"Failed to extract frame {frame_number}")
    cap.release()


# Function to capture points from the frame image
def capture_frame_points(event, x, y, flags, param):
    """
    Capture points from the frame image based on mouse events.

    Parameters:
    - event: The mouse event.
        - Should be cv2.EVENT_LBUTTONDOWN for capturing the point.
    - x (int): The x-coordinate of the mouse event.
    - y (int): The y-coordinate of the mouse event.
    - flags: Any relevant flags passed by OpenCV.
    - param: Additional parameters.

    Example:
    cv2.setMouseCallback('Frame Image', capture_frame_points)
    """
    global frame_coords, points, is_frame, frame_point_selected
    if event == cv2.EVENT_LBUTTONDOWN and is_frame:
        frame_coords.append((x, y))
        points.append((x, y, 0, 0))  # Placeholder for map coordinates
        print(f"Frame point selected: {x}, {y}")
        is_frame = False  # Now switch to map selection
        frame_point_selected = True  # Set the flag to indicate a frame point has been selected

# Function to capture points from the map image
def capture_map_points(event, x, y, flags, param):
    """
    Capture points from the map image based on mouse events.

    Parameters:
    - event: The mouse event.
        - Should be cv2.EVENT_LBUTTONDOWN for capturing the point.
    - x (int): The x-coordinate of the mouse event.
    - y (int): The y-coordinate of the mouse event.
    - flags: Any relevant flags passed by OpenCV.
    - param: Additional parameters.

    Example:
    cv2.setMouseCallback('Map Image', capture_map_points)
    """
    global map_coords, points, is_frame, frame_point_selected
    if event == cv2.EVENT_LBUTTONDOWN and not is_frame:
        map_coords.append((x, y))
        if len(points) > 0:
            points[-1] = (points[-1][0], points[-1][1], x, y)
        print(f"Map point selected: {x}, {y}")
        is_frame = True  # Now switch to frame selection
        frame_point_selected = False  # Reset the flag

# Create a UI for robust point selection
def create_point_selection_ui(extracted_frame_path, map_image_path, output_file_path):
    """
    Create a UI for robust point selection from frame and map images.

    Parameters:
    - extracted_frame_path (str): Path to the extracted frame image file.
        - The image should be in a format supported by OpenCV (e.g., .jpg, .png).
    - map_image_path (str): Path to the map image file.
        - The image should be in a format supported by OpenCV (e.g., .jpg, .png).
    - output_file_path (str): Path to save the selected points.
        - The output path should include the desired file name and extension (e.g., .txt, .csv).

    Instructions:
    1. Click on a point in the frame image window.
    2. Click on the corresponding point in the map image window.
    3. Repeat as many times as needed.
    4. Press ESC to finish and save the points.

    The function displays the frame and map images in separate windows and allows the user to select corresponding points by clicking. The selected points are saved to the specified output file.

    Example:
    extracted_frame_path = "path/to/extracted_frame.jpg"
    map_image_path = "path/to/map_image.jpg"
    output_file_path = "path/to/output/points.csv"
    create_point_selection_ui(extracted_frame_path, map_image_path, output_file_path)
    """
    global frame_coords, map_coords, points, is_frame, frame_point_selected
    frame_coords = []
    map_coords = []
    points = []
    is_frame = True  # Start with frame selection
    frame_point_selected = False  # Flag to indicate if a frame point has been selected

    # Load images
    frame_img = cv2.imread(extracted_frame_path)
    map_img = cv2.imread(map_image_path)

    cv2.namedWindow('Frame Image')
    cv2.namedWindow('Map Image')
    cv2.setMouseCallback('Frame Image', capture_frame_points)
    cv2.setMouseCallback('Map Image', capture_map_points)

    print("Instructions:")
    print("1. Click on a point in the frame image window.")
    print("2. Click on the corresponding point in the map image window.")
    print("3. Repeat as many times as needed.")
    print("4. Press ESC to finish and save the points.")

    while True:
        display_frame_img = frame_img.copy()
        display_map_img = map_img.copy()

        for p in points:
            cv2.circle(display_frame_img, (p[0], p[1]), 5, (0, 255, 0), -1)  # Frame points
            if p[2] != 0 and p[3] != 0:
                cv2.circle(display_map_img, (p[2], p[3]), 5, (0, 0, 255), -1)  # Map points

        if is_frame:
            cv2.putText(display_frame_img, "Select a point in this image", (10, display_frame_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_map_img, "Waiting for selection in the frame image...", (10, display_map_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_map_img, "Select a corresponding point in this image", (10, display_map_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame_img, "Waiting for selection in the map image...", (10, display_frame_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if frame_point_selected and not is_frame:
            cv2.putText(display_frame_img, "Please select the corresponding point in the map image before exiting.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Frame Image', display_frame_img)
        cv2.imshow('Map Image', display_map_img)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            if frame_point_selected:
                # Display error message if a frame point has been selected but not its corresponding map point
                print("Please select the corresponding point in the map image before exiting.")
                continue
            else:
                break

    cv2.destroyAllWindows()

    # Save points to file
    with open(output_file_path, 'w') as f:
        for p in points:
            f.write(f"{p[0]}, {p[1]}, {p[2]}, {p[3]}\n")

    print(f"Points saved to {output_file_path}")




def extract_entry_polygons_from_image(map_image_path, output_file_path, min_points=3):
    """
    Extract entry polygons from a map image through user interaction.

    Parameters:
    - map_image_path (str): The path to the map image file.
        - The image should be in a format supported by OpenCV (e.g., .jpg, .png).
    - output_file_path (str): The path to save the extracted polygons.
        - The output path should include the desired file name and extension (e.g., .txt).
    - min_points (int): The minimum number of points required to form a polygon.
        - Default is 3, which is the minimum number of points for a valid polygon.

    Instructions:
    1. Left-click to create polygon vertices in the map image window.
    2. Right-click to remove the last point.
    3. Press 'c' to confirm the current polygon.
    4. Press 'n' to start a new polygon only after the current one is confirmed.
    5. Press 'q' to save the polygons to the file and quit.

    Example:
    map_image_path = "path/to/map_image.jpg"
    output_file_path = "path/to/output/polygons.txt"
    extract_entry_polygons_from_image(map_image_path, output_file_path)
    """
    global entry_polygons, current_polygon, polygon_confirmed, message

    entry_polygons = []
    current_polygon = []
    polygon_confirmed = False
    message = "Left-click to add points, Right-click to remove last point, 'c' to confirm, 'n' for new, 'q' to save and quit"

    def draw_polygons(event, x, y, flags, param):
        global entry_polygons, current_polygon, polygon_confirmed, message

        if event == cv2.EVENT_LBUTTONDOWN and not polygon_confirmed:
            current_polygon.append((x, y))
            message = "Left-click to add points, Right-click to remove last point, 'c' to confirm, 'n' for new, 'q' to save and quit"
        
        elif event == cv2.EVENT_RBUTTONDOWN and len(current_polygon) > 0:
            current_polygon.pop()
            message = "Point removed. Left-click to add points, Right-click to remove last point, 'c' to confirm, 'n' for new, 'q' to save and quit"

    map_img = cv2.imread(map_image_path)

    cv2.namedWindow('Map Image')
    cv2.namedWindow('Instructions', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Map Image', draw_polygons)

    while True:
        display_map_img = map_img.copy()

        # Draw existing polygons
        for poly in entry_polygons:
            pts = np.array(poly, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_map_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            for point in poly:
                cv2.circle(display_map_img, point, 5, (0, 255, 0), -1)

        # Draw current polygon
        if len(current_polygon) > 0:
            pts = np.array(current_polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = (0, 255, 0) if polygon_confirmed else (0, 0, 255)
            cv2.polylines(display_map_img, [pts], isClosed=polygon_confirmed, color=color, thickness=2)
            for point in current_polygon:
                cv2.circle(display_map_img, point, 5, color, -1)

        # Check minimum points for current polygon
        if len(current_polygon) > 0 and len(current_polygon) < min_points:
            cv2.putText(display_map_img, f"Polygon needs at least {min_points} points", (10, display_map_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the map image
        cv2.imshow('Map Image', display_map_img)

        # Create a small overlay for instructions and messages
        overlay = np.zeros((150, 500, 3), dtype=np.uint8)
        y0, dy = 20, 30
        for i, line in enumerate(message.split(',')):
            y = y0 + i * dy
            cv2.putText(overlay, line.strip(), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Instructions', overlay)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            if len(current_polygon) >= min_points:
                polygon_confirmed = True
                message = "Polygon confirmed. Press 'n' to start a new polygon."
            else:
                message = f"Polygon needs at least {min_points} points."
        elif key == ord('n'):
            if polygon_confirmed:
                entry_polygons.append(current_polygon)
                current_polygon = []
                polygon_confirmed = False
                message = "Starting a new polygon. Left-click to add points, Right-click to remove last point, 'c' to confirm, 'n' for new, 'q' to save and quit"
            else:
                message = "Confirm the current polygon before starting a new one."
        elif key == ord('q'):
            if len(current_polygon) == 0 or (len(current_polygon) >= min_points and polygon_confirmed):
                if len(current_polygon) >= min_points:
                    entry_polygons.append(current_polygon)
                with open(output_file_path, 'w') as f:
                    for poly in entry_polygons:
                        f.write(", ".join([f"{x},{y}" for x, y in poly]) + "\n")
                message = f"Polygons saved to {output_file_path}"
                break
            else:
                message = f"Polygon needs at least {min_points} confirmed points."
    
    if len(current_polygon) < min_points and len(current_polygon) > 0:
        message = f"Polygon needs at least {min_points} points."

    cv2.destroyAllWindows()