import numpy as np
import cv2
import os
import csv
from shapely.geometry import Polygon, Point, LineString
from typing import List, Tuple, Dict, Optional, Any, Iterable, Union
import math
from shapely.ops import nearest_points


# ----------------------------------------------------------------------
# Configurable Halpe26 face keypoint indices (defaults)
# ----------------------------------------------------------------------
_DEFAULT_FACE_KP = {
    "NOSE": 0,
    "LEYE": 1,
    "REYE": 2,
    "LEAR": 3,
    "REAR": 4,
}

# Active indices used throughout this module (may be overridden by config)
FACE_KP = dict(_DEFAULT_FACE_KP)

def initialize_keypoint_indices(config: Optional[dict] = None) -> None:
    """Initialize (or reset) face keypoint indices.

    Reads optional `gaze_keypoint_map` from config, e.g.:
        {
          "gaze_keypoint_map": {"NOSE": 0, "LEYE": 1, "REYE": 2, "LEAR": 3, "REAR": 4}
        }

    Any missing keys fall back to defaults.
    """
    global FACE_KP, NOSE, LEYE, REYE, LEAR, REAR

    FACE_KP = dict(_DEFAULT_FACE_KP)

    if isinstance(config, dict):
        kp_map = config.get("gaze_keypoint_map", {})
        if isinstance(kp_map, dict):
            for k, v in kp_map.items():
                if not isinstance(k, str):
                    continue
                kk = k.strip().upper()
                if kk in FACE_KP:
                    try:
                        FACE_KP[kk] = int(v)
                    except Exception:
                        pass

    # Refresh aliases
    NOSE = FACE_KP["NOSE"]
    LEYE = FACE_KP["LEYE"]
    REYE = FACE_KP["REYE"]
    LEAR = FACE_KP["LEAR"]
    REAR = FACE_KP["REAR"]
    
def compute_gaze_vector(keypoints: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute a 2D gaze vector from Halpe26-format keypoints.
    Args:
        keypoints (np.ndarray): Array of shape (26, 3), where each row is (x, y, confidence).
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (origin, direction) as 2D vectors, both numpy arrays of shape (2,).
                                               Returns None if there is insufficient data.
    Algorithm:
        1. Determine gaze origin:
           - If both eyes have confidence > 0, origin = midpoint of LEYE and REYE.
           - Else if exactly one eye is present, origin = that eye.
           - Else if no eyes but NOSE, LEAR, and REAR are all present, origin = NOSE.
           - Otherwise return None.
        2. Determine gaze direction:
           - If both ears present, use midpoint(LEAR, REAR) - origin.
           - Else if exactly one ear present, use that ear - origin.
           - Otherwise return None.
        3. Normalize the direction vector to unit length.
    """
    # Validate input shape
    if keypoints.ndim != 2 or keypoints.shape[0] != 26 or keypoints.shape[1] < 2:
        return None

    def is_valid(idx: int) -> bool:
        return keypoints[idx, 2] > 0.3

    # 1. Origin calculation (priority: both eyes, then nose, then single eye)
    # Collect valid eyes
    eyes = []
    if is_valid(LEYE):
        eyes.append(keypoints[LEYE, :2])
    if is_valid(REYE):
        eyes.append(keypoints[REYE, :2])

    if len(eyes) == 2:
        origin = np.mean(eyes, axis=0)
    elif is_valid(NOSE):
        origin = keypoints[NOSE, :2]
    elif len(eyes) == 1:
        origin = eyes[0]
    else:
        return None

    # 2. Direction calculation
    ears = []
    if is_valid(LEAR):
        ears.append(keypoints[LEAR, :2])
    if is_valid(REAR):
        ears.append(keypoints[REAR, :2])

    if len(ears) == 2:
        ear_mid = np.mean(ears, axis=0)
        direction = origin - ear_mid
    elif len(ears) == 1:
        direction = origin - ears[0]
    else:
        return None

    # 3. Normalize direction
    norm = np.linalg.norm(direction)
    if norm <= 0:
        return None
    direction = direction / norm

    return origin, direction


# --- detect_enemy_fall ---
def detect_enemy_fall(
    tracker_output,
    *,
    enemy_ids: Optional[List[int]] = None,
    threshold: float = 0.5,
    px_tol: float = 10.0,
    avg_window: int = 5,
    conf_threshold: float = 0.3,
    only_torso_and_below: bool = True
) -> Dict[int, Optional[int]]:
    """
    Detect the first frame in which each enemy in `enemy_ids` is considered fallen.

    Args:
        tracker_output (list): List of frame dicts with 'frame' and 'objects'.
        enemy_ids (List[int] or None): list of enemy track IDs. If None, defaults to [99].
        threshold (float): Fraction of keypoints that must match reference pose.
        px_tol (float): Pixel tolerance for keypoint matching.
        avg_window (int): Number of frames to average for reference pose.
        conf_threshold (float): Confidence threshold to consider a keypoint valid.
        only_torso_and_below (bool): If True, restrict comparison to lower‑body keypoints.

    Returns:
        Dict[int, Optional[int]]: { enemy_id: fall_frame or None }
    """
    if enemy_ids is None:
        enemy_ids = [99]

    # Halpe26 indices ...
    lower_indices = {
        11, 12, 19,       # hips
        13, 14,           # knees
        15, 16,           # ankles
        20, 21,           # big toes
        22, 23,           # small toes
        24, 25            # heels
    }

    # Sort frames chronologically
    frames = sorted(tracker_output, key=lambda x: x['frame'])

    # Locate solitary starts per enemy
    solitary_starts: Dict[int, Optional[int]] = {}
    for eid in enemy_ids:
        last_multi_idx = None
        for i, frame_data in enumerate(frames):
            if len(frame_data.get('objects', [])) > 1:
                last_multi_idx = i
        if last_multi_idx is not None and last_multi_idx + 1 < len(frames):
            solitary_starts[eid] = frames[last_multi_idx + 1]['frame']
        else:
            solitary_starts[eid] = None

    fall_frames: Dict[int, Optional[int]] = {eid: None for eid in enemy_ids}

    # Build reference poses and detect fall per enemy
    for eid in enemy_ids:
        start = solitary_starts[eid]
        if start is None:
            continue

        # Build reference pose average
        ref_kps = []
        ref_scores = []
        for fr in range(start, start + avg_window):
            fd = next((f for f in frames if f['frame'] == fr), None)
            if not fd:
                continue
            eo = next((o for o in fd['objects'] if o.get('id') == eid), None)
            if eo and eo.get('keypoints'):
                ref_kps.append(np.array(eo['keypoints']))
                ref_scores.append(np.array(eo.get('keypoint_scores', [])))

        if len(ref_kps) < avg_window:
            continue  # insufficient data

        ref_pose  = np.mean(np.stack(ref_kps),  axis=0)  # (26,2)
        ref_score = np.mean(np.stack(ref_scores), axis=0)  # (26,)

        conf_mask = ref_score >= conf_threshold
        if only_torso_and_below:
            torso_mask = np.array([i in lower_indices for i in range(ref_score.shape[0])])
            conf_mask = conf_mask & torso_mask

        num_conf = int(conf_mask.sum())
        if num_conf == 0:
            continue

        # Scan all frames for first match
        for fd in frames:
            eo = next((o for o in fd.get('objects', []) if o.get('id') == eid), None)
            if not eo or not eo.get('keypoints'):
                continue
            kps = np.array(eo['keypoints'])
            if kps.shape != ref_pose.shape:
                continue
            dists = np.linalg.norm(kps - ref_pose, axis=1)
            match_frac = (dists[conf_mask] <= px_tol).sum() / num_conf
            if match_frac >= threshold:
                fall_frames[eid] = fd['frame']
                break

    return fall_frames


def annotate_camera_video(raw_frames: List[np.ndarray],
                          tracker_output: List[Dict],
                          frame_rate: int,
                          output_directory: str,
                          video_basename: str,
                          enemy_ids: List[int] = None,
                          gaze_conf_threshold: float = 0.3):
    # Define skeleton connections (pairs of keypoint indices)
    skeleton = [
        (15, 13), (13, 11), (11, 19),
        (16, 14), (14, 12), (12, 19),
        (17, 18), (18, 19),
        (18, 5),  (5, 7),  (7, 9),
        (18, 6),  (6, 8),  (8, 10),
        (1, 2),   (0, 1),  (0, 2),
        (1, 3),   (2, 4),  (3, 5),
        (4, 6),   (15, 20),(15, 22),
        (15, 24),(16, 21),(16, 23),
        (16, 25)
    ]
    """
    Generate the annotated camera video.
    - raw_frames: list of original video frames (np.ndarray), indexed by frame number starting at 1
    - tracker_output: list of dicts {'frame': int, 'objects': [{'id', 'bbox', 'keypoints', 'keypoint_scores'}]}
    - fall_frame: frame index at which the enemy is considered fallen
    - frame_rate: frames per second for output video
    - output_directory: directory to save the video
    - video_basename: base name of the video files
    - enemy_id: track ID of the enemy (default 99)
    - gaze_conf_threshold: minimum confidence to draw gaze vectors
    """
    # --- color palette & id→color cache (matches ProcessingEngine) ---
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    # Prepare output writer
    video_path = os.path.join(output_directory, f"{video_basename}_Tracking.mp4")
    height, width = raw_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (width, height)
    )

    if enemy_ids is None:
        enemy_ids = [99]
    # Iterate frames and overlay annotations
    for frame_data in tracker_output:
        idx = frame_data["frame"] - 1
        frame = raw_frames[idx].copy()
        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            # Draw bbox and ID
            if trk_id in enemy_ids:
                color = (255, 255, 255)
            else:
                color = track_colors.setdefault(trk_id, predefined_colors[trk_id % len(predefined_colors)])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Compute and draw ID text (use "Enemy" label for the enemy track)
            if trk_id in enemy_ids:
                id_text = "ID: Enemy"
            else:
                id_text = f"ID: {trk_id}"
            (id_w, id_h), id_base = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
            cv2.putText(frame, id_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

            # ---- Keypoints and skeleton ----
            kps = obj.get("keypoints", [])
            scores = obj.get("keypoint_scores", [])

            # Only proceed if we have a full 26‑keypoint set **and** matching scores
            if len(kps) == 26 and len(scores) == 26:
                # Draw keypoints
                for (x, y), s in zip(kps, scores):
                    if s >= gaze_conf_threshold:
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

                # Draw skeleton links
                for i1, i2 in skeleton:
                    s1 = scores[i1]
                    s2 = scores[i2]
                    if s1 >= gaze_conf_threshold and s2 >= gaze_conf_threshold:
                        p1 = kps[i1]
                        p2 = kps[i2]
                        pt1 = (int(p1[0]), int(p1[1]))
                        pt2 = (int(p2[0]), int(p2[1]))
                        cv2.line(frame, pt1, pt2, color, 1)

        writer.write(frame)

    writer.release()


def annotate_camera_with_gaze_triangle(raw_frames: List[np.ndarray],
                                       tracker_output: List[Dict],
                                       gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
                                       frame_rate: int,
                                       output_directory: str,
                                       video_basename: str,
                                       enemy_ids: List[int] = None,
                                       half_angle_deg: float = 30.0,
                                       alpha: float = 0.2,
                                       show_enemy_gaze: bool = True):
    """
    Generate the annotated camera video with bounding boxes, skeletons, and semi-transparent gaze triangles.
    - raw_frames: list of original video frames (np.ndarray), indexed by frame number starting at 1
    - tracker_output: list of dicts {'frame': int, 'objects': [{'id', 'bbox', 'keypoints', 'keypoint_scores'}]}
    - gaze_info: dictionary mapping (frame, track_id) to (ox, oy, dx, dy)
    - fall_frame: frame index at which the enemy is considered fallen
    - frame_rate: frames per second for output video
    - output_directory: directory to save the video
    - video_basename: base name of the video files
    - enemy_id: track ID of the enemy (default 99)
    - gaze_conf_threshold: minimum confidence to consider keypoints for gaze
    - half_angle_deg: half of the visual angle (in degrees) for the gaze triangle cone (default 10.0)
    - alpha: transparency factor for gaze triangles (0.0 fully transparent, 1.0 opaque)
    - show_enemy_gaze: if False, suppress drawing gaze cones for enemy IDs.
    """

    # Prepare color palette & id→color cache (matches ProcessingEngine)
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    if enemy_ids is None:
        enemy_ids = [99]

    # Prepare output writer
    video_path = os.path.join(output_directory, f"{video_basename}_GazeTracking.mp4")
    height, width = raw_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (width, height)
    )

    # Pre-compute triangle length (diagonal of frame)
    tri_len = math.hypot(width, height) * 1.5
    half_angle = half_angle_deg  # use as-is

    # Iterate frames and overlay annotations
    for frame_data in tracker_output:
        idx = frame_data["frame"] - 1
        frame = raw_frames[idx].copy()

        # --- 1. Draw bounding boxes & IDs ---
        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            # Color selection
            if trk_id in enemy_ids:
                color = (255, 255, 255)
                id_text = "ID: Enemy"
            else:
                color = track_colors.setdefault(trk_id, predefined_colors[trk_id % len(predefined_colors)])
                id_text = f"ID: {trk_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, id_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

        # --- 2. Draw each gaze cone on‑the‑fly so overlapping cones mix colours ---
        overlay = np.zeros_like(frame, dtype=np.uint8)

        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            gaze = gaze_info.get((frame_data["frame"], trk_id))
            if gaze is None:
                continue
            # Optionally skip enemy gaze overlay
            if trk_id in enemy_ids and not show_enemy_gaze:
                continue

            if trk_id in enemy_ids:
                color = (255, 255, 255)
            else:
                color = track_colors.setdefault(trk_id, predefined_colors[trk_id % len(predefined_colors)])

            ox, oy, dx, dy = gaze
            tri = _gaze_triangle((ox, oy), (dx, dy), half_angle, length=tri_len).astype(int)
            cv2.fillConvexPoly(overlay, tri, color)

            # Create a transparent overlay for this single triangle
            tri_overlay = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillConvexPoly(tri_overlay, tri, color)
            
            # Blend the triangle overlay onto the frame
            cv2.addWeighted(tri_overlay, alpha, frame, 1.0, 0, dst=frame)


        writer.write(frame)

    writer.release()


def annotate_fall_video(raw_frames: List[np.ndarray],
                        tracker_output: List[Dict],
                        fall_frames: Dict[int, Optional[int]] = None,
                        frame_rate: int = 30,
                        output_directory: str = "",
                        video_basename: str = "",
                        enemy_ids: List[int] = None):
    """
    Generate a video showing bounding boxes and IDs, and label 'FALLEN!' when the enemy falls.
    Only saved if fall_frames is not None.
    - raw_frames: list of original video frames (np.ndarray), indexed by frame number starting at 1
    - tracker_output: list of dicts {'frame': int, 'objects': [{'id', 'bbox'}]}
    - fall_frames: dict mapping enemy_id to fall_frame (frame index at which the enemy is considered fallen)
    - frame_rate: frames per second for output video
    - output_directory: directory to save the video
    - video_basename: base name of the video files
    - enemy_ids: list of enemy track IDs (default [99])
    """
    if fall_frames is None:
        return
    if enemy_ids is None:
        enemy_ids = [99]

    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    video_path = os.path.join(output_directory, f"{video_basename}_FallTracking.mp4")
    height, width = raw_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (width, height)
    )

    for frame_data in tracker_output:
        idx = frame_data["frame"] - 1
        frame = raw_frames[idx].copy()

        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            color = (255, 255, 255) if trk_id in enemy_ids else track_colors.setdefault(trk_id, predefined_colors[trk_id % len(predefined_colors)])
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # Draw ID label
            id_text = "ID: Enemy" if trk_id in enemy_ids else f"ID: {trk_id}"
            (id_w, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
            cv2.putText(frame, id_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)
            # Overlay "FALLEN!" if past fall_frame and is enemy
            if (trk_id in enemy_ids and
                fall_frames.get(trk_id) is not None and
                frame_data["frame"] >= fall_frames[trk_id]):
                fallen_text = "FALLEN!"
                fallen_x = x1 + id_w + 10
                fallen_y = y1 - 10
                cv2.putText(frame, fallen_text, (fallen_x, fallen_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

        writer.write(frame)

    writer.release()
    
def annotate_clearance_video(raw_frames: List[np.ndarray],
                             tracker_output: List[Dict],
                             clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
                             frame_rate: int,
                             output_directory: str,
                             video_basename: str,
                             enemy_ids: List[int] = None):
    """
    Generate a video showing bounding boxes and IDs, and overlay 'CLEARED!'
    when each enemy’s clearance start frame is reached.

    - raw_frames: list of original video frames (np.ndarray), indexed by frame number starting at 1
    - tracker_output: list of dicts {'frame': int, 'objects': [{'id', 'bbox'}]}
    - clearance_map: dict mapping track_id → (start_frame, end_frame, clearing_friend_id)
                     start_frame is the frame at which track_id is considered 'cleared'
    - frame_rate: frames per second for output video
    - output_directory: directory to save the video
    - video_basename: base name of the video files
    - enemy_ids: list of enemy track IDs (default [99])
    """
    if enemy_ids is None:
        enemy_ids = [99]
    # --- color palette & id→color cache (matches ProcessingEngine) ---
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    # Prepare output writer
    video_path = os.path.join(output_directory, f"{video_basename}_ClearanceTracking.mp4")
    height, width = raw_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (width, height)
    )

    # Iterate frames and overlay annotations
    for frame_data in tracker_output:
        idx = frame_data["frame"] - 1
        frame = raw_frames[idx].copy()

        # First, draw bounding boxes and IDs for all tracks
        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            # Determine if this track is the enemy (by enemy_ids)
            if trk_id in enemy_ids:
                color = (255, 255, 255)
                id_text = "ID: Enemy"
            else:
                color = track_colors.setdefault(trk_id, predefined_colors[trk_id % len(predefined_colors)])
                id_text = f"ID: {trk_id}"
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # Compute text size and draw ID label
            (id_w, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
            cv2.putText(frame, id_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

        # Overlay "CLEARED!" for each enemy when its end_frame has passed
        for eid in enemy_ids:
            start, end, _ = clearance_map.get(eid, (None, None, None))
            if end is not None and frame_data["frame"] >= end:
                obj = next((o for o in frame_data["objects"] if o.get("id") == eid), None)
                if obj:
                    x1, y1, x2, y2 = obj["bbox"]
                    color = (255, 255, 255)
                    # Recompute id_text and id_w for positioning "CLEARED!"
                    id_text = "ID: Enemy"
                    (id_w, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
                    # Compute position for "CLEARED!" text, just to the right of the box
                    cleared_x = x1 + id_w + 10
                    cleared_y = y1 - 10
                    cv2.putText(frame, "CLEARED!", (cleared_x, cleared_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

        writer.write(frame)

    writer.release()


def annotate_map_video(map_image: np.ndarray,
                       all_map_points: List[Tuple[int, int, float, float]],
                       frame_rate: int,
                       output_directory: str,
                       video_basename: str,
                       enemy_ids: List[int] = None,
                       total_frames: Optional[int] = None):
    """
    Generate the map-view tracking video.
    - map_image: the empty map as a numpy array
    - all_map_points: list of tuples (frame, id, mapX, mapY)
    - frame_rate: frames per second for output video
    - output_directory: directory to save the map video
    - video_basename: base name of the video files
    - enemy_ids: list of enemy track IDs (default [99])
    """
    if enemy_ids is None:
        enemy_ids = [99]
    map_out = os.path.join(output_directory, f"{video_basename}_MapTracking.mp4")
    h, w = map_image.shape[:2]
    writer = cv2.VideoWriter(
        map_out,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (w, h)
    )

    # Build a dict of points per frame
    points_per_frame = {}
    for frm, tid, mx, my in all_map_points:
        points_per_frame.setdefault(frm, []).append((tid, mx, my))

    # Determine total number of frames to write
    if total_frames is not None:
        max_frame = total_frames
    else:
        max_frame = max(points_per_frame.keys()) if points_per_frame else 0

    # --- color palette & id→color cache (matches ProcessingEngine) ---
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    permanent_vis = map_image.copy()

    # Iterate through all frames (including those without tracks) for full-length video
    for frame_num in range(1, max_frame + 1):
        temp_vis = permanent_vis.copy()
        for tid, mx, my in points_per_frame.get(frame_num, []):
            if tid in enemy_ids:
                color = (255, 255, 255)
            else:
                color = track_colors.setdefault(tid, predefined_colors[tid % len(predefined_colors)])
            # draw current position
            cv2.circle(temp_vis, (int(mx), int(my)), 8, color, -1)
            # draw trajectory
            # find previous positions
            traj = [(px, py) for f, t, px, py in all_map_points if t == tid and f <= frame_num]
            if tid not in enemy_ids:
                for p1, p2 in zip(traj, traj[1:]):
                    cv2.line(permanent_vis,
                             (int(p1[0]), int(p1[1])),
                             (int(p2[0]), int(p2[1])),
                             color, 2)
        writer.write(temp_vis)

    writer.release()


# ----------------------------------------------------------------------
# Map‑view POD visualisation (points + active POD regions)
# ----------------------------------------------------------------------
def annotate_map_pod_video(
    map_image: np.ndarray,
    *,
    all_map_points: List[Tuple[int, int, float, float]],
    assignment: Dict[int, Optional[int]],
    dynamic_work_areas: Dict[int, Dict[int, Polygon]],
    pod_capture_data: Dict[int, Dict[str, Optional[float]]],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    total_frames: Optional[int] = None,
    enemy_ids: Optional[List[int]] = None
):
    """
    Create a video similar to `annotate_map_video` but:
        • Draws ONLY the current position of each track (no trajectory lines).
        • Draws each POD working area circle per frame:
              – outline (non‑filled) while *not yet captured*,
              – filled circle once captured.
        • Colours match the track colour palette.

    Args
    ----
    map_image, all_map_points, frame_rate, output_directory, video_basename
        – Same semantics as annotate_map_video.
    assignment          : {track_id -> pod_idx or None}
    dynamic_work_areas  : {frame_idx -> {pod_idx -> shapely Polygon}}
    pod_capture_data    : {pod_idx -> {"assigned_id": id, "capture_time_sec": float|None}}
    total_frames        : optional max frame count to render.
    enemy_ids           : list of enemy IDs (drawn white).
    """
    if enemy_ids is None:
        enemy_ids = [99]

    # Build per‑frame point dict
    pts_per_frame: Dict[int, List[Tuple[int, float, float]]] = {}
    for frm, tid, mx, my in all_map_points:
        pts_per_frame.setdefault(frm, []).append((tid, mx, my))

    # Determine max_frame
    if total_frames is None:
        max_frame = max(pts_per_frame.keys()) if pts_per_frame else 0
    else:
        max_frame = total_frames

    # Pre‑compute capture frame for each pod (in map video frame indices)
    pod_capture_frame: Dict[int, Optional[int]] = {}
    fps = frame_rate
    # Build first_frame per track for conversion
    first_frame_track: Dict[int, int] = {}
    for frm in sorted(pts_per_frame.keys()):
        for tid, _, _ in pts_per_frame[frm]:
            if tid not in first_frame_track:
                first_frame_track[tid] = frm
    for pod_idx, info in pod_capture_data.items():
        tid = info.get("assigned_id")
        cap_sec = info.get("capture_time_sec")
        if tid is not None and cap_sec is not None:
            f0 = first_frame_track.get(tid)
            if f0 is not None:
                pod_capture_frame[pod_idx] = int(round(f0 + cap_sec * fps))
            else:
                pod_capture_frame[pod_idx] = None
        else:
            pod_capture_frame[pod_idx] = None

    # Colour palette
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    # Map pod_idx -> track colour (based on assigned_id)
    pod_colors: Dict[int, Tuple[int, int, int]] = {}
    for tid, pod_idx in assignment.items():
        if pod_idx is None:
            continue
        if tid in enemy_ids:
            pod_colors[pod_idx] = (255, 255, 255)
        else:
            pod_colors[pod_idx] = track_colors.setdefault(
                tid, predefined_colors[tid % len(predefined_colors)]
            )

    # Prepare writer
    os.makedirs(output_directory, exist_ok=True)
    out_path = os.path.join(output_directory, f"{video_basename}_PodAreas.mp4")
    h, w = map_image.shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (w, h)
    )

    for frame_idx in range(1, max_frame + 1):
        vis = map_image.copy()

        # --- Draw POD working areas ---
        frame_polys = dynamic_work_areas.get(frame_idx, {})
        for pod_idx, poly in frame_polys.items():
            color = pod_colors.get(pod_idx, (200, 200, 200))
            cap_frame = pod_capture_frame.get(pod_idx)

            # Draw every component polygon (handles Polygon or MultiPolygon)
            if isinstance(poly, Polygon):
                polys_iter = [poly]
            else:  # MultiPolygon
                polys_iter = list(poly.geoms)

            for sub_poly in polys_iter:
                pts = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                if cap_frame is not None and frame_idx >= cap_frame:
                    cv2.fillConvexPoly(vis, pts, color)
                else:
                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        # --- Draw current positions ---
        for tid, mx, my in pts_per_frame.get(frame_idx, []):
            if tid in enemy_ids:
                clr = (255, 255, 255)
            else:
                clr = track_colors.setdefault(tid, predefined_colors[tid % len(predefined_colors)])
            cv2.circle(vis, (int(mx), int(my)), 6, clr, -1)

        writer.write(vis)

    writer.release()


# ----------------------------------------------------------------------
# Combined map-view: POD working areas + persistent trails + positions
# ----------------------------------------------------------------------
def annotate_map_pod_with_paths_video(
    map_image: np.ndarray,
    *,
    all_map_points: List[Tuple[int, int, float, float]],
    assignment: Dict[int, Optional[int]],
    dynamic_work_areas: Dict[int, Dict[int, Polygon]],
    pod_capture_data: Dict[int, Dict[str, Optional[float]]],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    total_frames: Optional[int] = None,
    enemy_ids: Optional[List[int]] = None,
    fill_alpha: float = 0.35
):
    """
    Combined map-view annotator:
      • Draws POD working areas per frame (outline until captured, then filled).
      • Draws persistent trajectory paths for non-enemy tracks.
      • Draws current positions for all tracks.
      • Captured POD fills are blended translucently (controlled by fill_alpha) so trails remain visible.

    Args match the union of `annotate_map_video` and `annotate_map_pod_video`.
    The output file is named `<video_basename>_PodAreasWithTrails.mp4`.
    """
    if enemy_ids is None:
        enemy_ids = [99]

    # --- Build per‑frame points dict (same as other map functions) ---
    pts_per_frame: Dict[int, List[Tuple[int, float, float]]] = {}
    for frm, tid, mx, my in all_map_points:
        pts_per_frame.setdefault(frm, []).append((tid, mx, my))

    # Determine max_frame
    if total_frames is None:
        max_frame = max(pts_per_frame.keys()) if pts_per_frame else 0
    else:
        max_frame = total_frames

    # --- Pre‑compute capture frame for each POD (map video frame indices) ---
    pod_capture_frame: Dict[int, Optional[int]] = {}
    fps = frame_rate
    # First appearance per track (to convert capture seconds → frame index)
    first_frame_track: Dict[int, int] = {}
    for frm in sorted(pts_per_frame.keys()):
        for tid, _, _ in pts_per_frame[frm]:
            if tid not in first_frame_track:
                first_frame_track[tid] = frm
    for pod_idx, info in pod_capture_data.items():
        tid = info.get("assigned_id")
        cap_sec = info.get("capture_time_sec")
        if tid is not None and cap_sec is not None:
            f0 = first_frame_track.get(tid)
            pod_capture_frame[pod_idx] = int(round(f0 + cap_sec * fps)) if f0 is not None else None
        else:
            pod_capture_frame[pod_idx] = None

    # --- Colour palette (matches the rest of the module) ---
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    # Map pod_idx -> track colour (based on assigned_id)
    pod_colors: Dict[int, Tuple[int, int, int]] = {}
    for tid, pod_idx in assignment.items():
        if pod_idx is None:
            continue
        if tid in enemy_ids:
            pod_colors[pod_idx] = (255, 255, 255)
        else:
            pod_colors[pod_idx] = track_colors.setdefault(
                tid, predefined_colors[tid % len(predefined_colors)]
            )

    # --- Prepare writer ---
    os.makedirs(output_directory, exist_ok=True)
    out_path = os.path.join(output_directory, f"{video_basename}_PodAreasWithTrails.mp4")
    h, w = map_image.shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (w, h)
    )

    # This canvas will accumulate trajectory lines over time.
    permanent_trails = map_image.copy()

    # Iterate frames
    for frame_idx in range(1, max_frame + 1):
        # Start from the trail canvas accumulated so far
        vis = permanent_trails.copy()

        # --- Draw POD working areas for this frame ---
        frame_polys = dynamic_work_areas.get(frame_idx, {})
        for pod_idx, poly in frame_polys.items():
            color = pod_colors.get(pod_idx, (200, 200, 200))
            cap_frame = pod_capture_frame.get(pod_idx)

            # Handle Polygon or MultiPolygon
            if isinstance(poly, Polygon):
                polys_iter = [poly]
            else:
                polys_iter = list(getattr(poly, "geoms", []))

            for sub_poly in polys_iter:
                pts = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                if cap_frame is not None and frame_idx >= cap_frame:
                    # Translucent fill so underlying trails remain visible
                    overlay = vis.copy()
                    cv2.fillConvexPoly(overlay, pts, color)
                    cv2.addWeighted(overlay, fill_alpha, vis, 1.0 - fill_alpha, 0, dst=vis)
                    # Keep a crisp boundary on top
                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
                else:
                    # Outline if not yet captured
                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        # --- Draw current positions for this frame ---
        for tid, mx, my in pts_per_frame.get(frame_idx, []):
            clr = (255, 255, 255) if tid in enemy_ids else track_colors.setdefault(
                tid, predefined_colors[tid % len(predefined_colors)]
            )
            cv2.circle(vis, (int(mx), int(my)), 6, clr, -1)

        # --- Update the permanent trail canvas with trajectory lines up to this frame ---
        for tid in {t for _, t, _, _ in all_map_points}:
            if tid in enemy_ids:
                continue
            # Collect the trajectory for this ID up to the current frame
            traj = [(px, py) for f, t, px, py in all_map_points if t == tid and f <= frame_idx]
            if len(traj) < 2:
                continue
            clr = track_colors.setdefault(tid, predefined_colors[tid % len(predefined_colors)])
            for p1, p2 in zip(traj, traj[1:]):
                cv2.line(permanent_trails,
                         (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])),
                         clr, 2)

        writer.write(vis)

    writer.release()

# --- Helper function: save_position_cache ---
def save_position_cache(all_map_points: List[Tuple[int, int, float, float]],
                        output_directory: str,
                        video_basename: str):
    """
    Save position cache: frame, track ID, mapX, mapY.
    """
    cache_path = os.path.join(output_directory, f"{video_basename}_PositionCache.txt")
    with open(cache_path, "w") as f:
        f.write("frame,id,mapX,mapY\n")
        for frm, tid, mx, my in all_map_points:
            f.write(f"{frm},{tid},{mx},{my}\n")


# --- Helper function: save_gaze_cache ---
def save_gaze_cache(gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
                    output_directory: str,
                    video_basename: str):
    """
    Save gaze cache: frame, track ID, origin x/y, direction dx/dy.
    """
    cache_path = os.path.join(output_directory, f"{video_basename}_GazeCache.txt")
    with open(cache_path, "w") as f:
        f.write("frame,id,ox,oy,dx,dy\n")
        for (frm, tid), (ox, oy, dx, dy) in sorted(gaze_info.items()):
            f.write(f"{frm},{tid},{ox},{oy},{dx},{dy}\n")


# --- Helper function: save_enemy_fall_cache ---
def save_enemy_fall_cache(fall_frames: Dict[int, Optional[int]],
                          output_directory: str,
                          video_basename: str):
    """
    Save enemy fall frame cache.
    """
    cache_path = os.path.join(output_directory, f"{video_basename}_EnemyFallCache.txt")
    with open(cache_path, "w") as f:
        f.write("enemy_id,fall_frame\n")
        for eid, ff in fall_frames.items():
            f.write(f"{eid},{ff if ff is not None else ''}\n")


# --- Helper: compute_threat_clearance ---------------------------------------

def _boxes_intersect(boxA: Tuple[float, float, float, float],
                     boxB: Tuple[float, float, float, float]) -> bool:
    """Axis‑aligned bbox intersection."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


# --- Gaze triangle and triangle-box intersection helpers ---
def _gaze_triangle(origin: Tuple[float, float],
                   direction: Tuple[float, float],
                   half_angle_deg: float,
                   length: float = 10000.0) -> np.ndarray:
    """
    Create an isosceles triangle that represents the visual cone.

    Args:
        origin: (x, y) coordinates of the gaze origin.
        direction: (dx, dy) gaze direction (need not be normalised).
        half_angle_deg: Half of the visual angle in degrees.
        length: Length in pixels to extend the cone.

    Returns:
        3×2 float32 array with vertices [origin, left_ray_end, right_ray_end].
    """
    # Normalise direction
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    if n == 0:
        return np.zeros((3, 2), dtype=np.float32)
    d /= n

    # Rotation matrices for ±half_angle
    ang = math.radians(half_angle_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    rot_left  = np.array([[cos_a, -sin_a],
                          [sin_a,  cos_a]], dtype=np.float32)
    rot_right = np.array([[cos_a,  sin_a],
                          [-sin_a, cos_a]], dtype=np.float32)

    left_vec  = rot_left  @ d * length
    right_vec = rot_right @ d * length

    o = np.asarray(origin, dtype=np.float32)
    p_left  = o + left_vec
    p_right = o + right_vec
    return np.stack([o, p_left, p_right], axis=0)


def _triangle_box_intersect(triangle: np.ndarray,
                            box: Tuple[float, float, float, float]) -> bool:
    """
    Check whether a triangle intersects an axis‑aligned bounding box.

    Uses cv2.intersectConvexConvex to compute intersection area.
    """
    tri = triangle.reshape(-1, 1, 2).astype(np.float32)
    x1, y1, x2, y2 = map(float, box)
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32).reshape(-1, 1, 2)
    inter_area, _ = cv2.intersectConvexConvex(tri, rect)
    return inter_area > 0.0

# --- Helper: draw a dotted polygon outline ---
def _draw_dotted_polygon(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], thickness: int = 1, dash_length: int = 10, gap_length: int = 5):
    """
    Draw a dotted (dashed) outline of a polygon.
    pts: Nx2 array of vertex coordinates.
    """
    import numpy as _np
    import cv2 as _cv
    n = len(pts)
    for i in range(n):
        start = _np.array(pts[i], dtype=_np.float32)
        end = _np.array(pts[(i+1) % n], dtype=_np.float32)
        edge_vec = end - start
        length = _np.linalg.norm(edge_vec)
        if length == 0:
            continue
        direction = edge_vec / length
        step = dash_length + gap_length
        num_dashes = int(length // step) + 1
        for d in range(num_dashes):
            seg_start = start + direction * (d * step)
            seg_end = start + direction * (d * step + dash_length)
            seg_start = tuple(seg_start.astype(int))
            seg_end = tuple(seg_end.astype(int))
            _cv.line(img, seg_start, seg_end, color, thickness)


def _has_valid_run(
    info_list: List[Tuple[int, int, bool, bool]],
    intersection_thr: int,
    wrist_thr: int,
    gaze_thr: int
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Look for the first consecutive run of `intersection_thr` frames
    where at least `wrist_thr` frames have wrist_flag=True and
    at least `gaze_thr` frames have gaze_flag=True.

    Returns (first_frame_in_run, last_frame_in_run, contributing_friend_id)  
    or (None, None, None) if no valid run is found.
    """
    if not info_list:
        return None, None, None

    # Build a per‑frame map
    frame_map: Dict[int, List[Tuple[int, bool, bool]]] = {}
    for f_idx, f_id, w_flag, g_flag in info_list:
        frame_map.setdefault(f_idx, []).append((f_id, w_flag, g_flag))

    frames_sorted = sorted(frame_map.keys())
    i = 0
    while i < len(frames_sorted):
        run = [frames_sorted[i]]
        j = i + 1
        while j < len(frames_sorted) and frames_sorted[j] == frames_sorted[j - 1] + 1:
            run.append(frames_sorted[j])
            j += 1

        if len(run) >= intersection_thr:
            wrist_cnt = 0
            gaze_cnt = 0
            first_fid: Optional[int] = None
            early_frame: Optional[int] = None
            for fr in run:
                for fid, w_flag, g_flag in frame_map[fr]:
                    if w_flag:
                        wrist_cnt += 1
                    if g_flag:
                        gaze_cnt += 1
                    if (w_flag or g_flag) and first_fid is None:
                        first_fid = fid
                    # Record the first frame where all three thresholds are met
                    if (wrist_cnt >= wrist_thr and gaze_cnt >= gaze_thr and
                        len(run) >= intersection_thr and early_frame is None):
                        early_frame = fr
            if wrist_cnt >= wrist_thr and gaze_cnt >= gaze_thr:
                return early_frame if early_frame is not None else run[0], run[-1], first_fid
        i = j

    return None, None, None


def compute_threat_clearance(
    tracker_output: List[Dict],
    keypoint_details: Dict[Tuple[int, int], Tuple[List, List]],
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    *,
    enemy_ids: Optional[List[int]] = None,
    visual_angle_deg: float = 20.0,
    intersection_frames: int = 30,
    wrist_frames: int = 7,
    gaze_frames: int = 15,
) -> Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]]:
    """
    Detects the first valid clearance per enemy using temporal thresholds.

    Returns:
        { enemy_id: (first_clear_frame, last_clear_frame, clearing_friend_id) }
    """
    if enemy_ids is None:
        enemy_ids = [99]

    half_angle = visual_angle_deg / 2.0
    clearance: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]] = {eid: (None, None, None) for eid in enemy_ids}
    per_enemy: Dict[int, List[Tuple[int, int, bool, bool]]] = {eid: [] for eid in enemy_ids}

    # Pass 1: collect per‑frame info when boxes intersect
    for frame_entry in tracker_output:
        fidx = frame_entry["frame"]
        bboxes = {obj["id"]: tuple(obj["bbox"]) for obj in frame_entry["objects"]}
        enemies = [tid for tid in bboxes if tid in enemy_ids]
        friends = [tid for tid in bboxes if tid not in enemy_ids]

        for eid in enemies:
            ebox = bboxes[eid]
            ex1, ey1, ex2, ey2 = ebox

            for fid in friends:
                if clearance[eid][0] is not None:
                    break  # already cleared
                fbox = bboxes[fid]
                if not _boxes_intersect(ebox, fbox):
                    continue

                # wrist check
                wrist_flag = False
                kp_tuple = keypoint_details.get((fidx, fid))
                if kp_tuple and len(kp_tuple[0]) > 10:
                    kp_list, _ = kp_tuple
                    for wi in (9, 10):
                        wx, wy = kp_list[wi]
                        if ex1 <= wx <= ex2 and ey1 <= wy <= ey2:
                            wrist_flag = True
                            break

                # gaze check  (triangle / cone intersection)
                gaze_flag = False
                g = gaze_info.get((fidx, fid))
                if g:
                    ox, oy, dx, dy = g
                    tri = _gaze_triangle((ox, oy), (dx, dy), half_angle)
                    if _triangle_box_intersect(tri, ebox):
                        gaze_flag = True

                per_enemy[eid].append((fidx, fid, wrist_flag, gaze_flag))

    # Pass 2: evaluate thresholds per enemy
    for eid, info in per_enemy.items():
        frm_start, frm_end, fid = _has_valid_run(info, intersection_frames, wrist_frames, gaze_frames)
        clearance[eid] = (frm_start, frm_end, fid)

    return clearance



# --- Helper: save_threat_clearance_cache ------------------------------------
def save_threat_clearance_cache(clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
                                output_directory: str,
                                video_basename: str):
    """
    Write threat‑clearance cache:
        enemy_id,start_frame,end_frame,clearing_friend
    """
    cache_path = os.path.join(output_directory, f"{video_basename}_ThreatClearanceCache.txt")
    with open(cache_path, "w") as f:
        f.write("enemy_id,immediate_frame,contact_end_frame,clearing_friend\n")
        for eid, (start, end, fid) in clearance_map.items():
            f.write(f"{eid},{start if start is not None else -1},{end if end is not None else -1},{fid if fid is not None else -1}\n")


# --- Map Gaze Annotation: Only Gaze Regions as Mask Overlays ---
def annotate_map_with_gaze(
    map_image: np.ndarray,
    pixel_mapper,
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    room_boundary_coords: List[Tuple[float, float]],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    enemy_ids: List[int] = None,
    show_enemy_gaze: bool = True,
    half_angle_deg: float = 20.0,
    alpha: float = 0.3,
    enable_alpha: bool = True,
    enable_fill: bool = True,
    enable_boundary: bool = True,
    total_frames: Optional[int] = None,
    accumulated_clear: bool = False
):
    """
    Generate a map‐view video showing gaze cones, with two modes:
      • Normal mode (accumulated_clear=False): each frame overlays only that frame's clipped gaze triangles.
      • Accumulated-clear mode (accumulated_clear=True): the room starts completely black (blurred)
        and gradually “clears” regions as they are covered by gaze over time. Each frame also draws
        that frame’s gaze triangles on top.

    Args:
        map_image:               A BGR numpy array of the empty floor‐plan (map‐space pixels).
        pixel_mapper:            Object with .pixel_to_map((x_px, y_px)) → (x_map, y_map).
        gaze_info:               Dict[(frame_idx, track_id) → (ox, oy, dx, dy)] in pixel‐space.
        room_boundary_coords:    List of (mapX, mapY) tuples defining the room boundary polygon.
        frame_rate:              Frames per second for the output video.
        output_directory:        Directory where output video will be saved.
        video_basename:          Base name for the output file (no extension).
        enemy_ids:               List of track IDs considered “enemies” (drawn in white). Defaults to [99].
        show_enemy_gaze:         If False, skip drawing gaze for any ID in enemy_ids.
        half_angle_deg:          Half of the visual angle (in degrees) for the gaze cone.
        alpha:                   Transparency for the gaze overlay (only used in normal mode).
        - enable_alpha: if False, draw triangles opaquely (skip cv2.addWeighted for triangles); room‑blur overlay is always blended.
        total_frames:            Total number of frames to render. If None, derived from gaze_info.
        accumulated_clear:       If True, produce a progressively‐clearing blurred‐black mask + draw each
                                 frame’s gaze triangles. If False, run the original “only gaze overlays” mode.
    """
    if enemy_ids is None:
        enemy_ids = [99]

    # --- ACCUMULATED‐CLEAR MODE (translucent blurred black + per‐frame triangles) ---
    if accumulated_clear:
        # Build room polygon for clipping & determine max_frame
        room_polygon = Polygon(room_boundary_coords)
        if total_frames is None:
            if gaze_info:
                max_frame = max(frame_idx for (frame_idx, _) in gaze_info.keys())
            else:
                raise ValueError("`total_frames` must be provided if `gaze_info` is empty.")
        else:
            max_frame = total_frames

        # Create output folder and VideoWriter
        os.makedirs(output_directory, exist_ok=True)
        video_path = os.path.join(output_directory, f"{video_basename}_GazeMapCleared.mp4")
        map_h, map_w = map_image.shape[:2]
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"avc1"),
            frame_rate,
            (map_w, map_h)
        )

        # Precompute a binary mask of the room area (uint8 0/1)
        room_mask = np.zeros((map_h, map_w), dtype=np.uint8)
        room_poly_xy = np.array(room_boundary_coords, dtype=np.int32)
        cv2.fillPoly(room_mask, [room_poly_xy], 1)

        # This mask will accumulate all covered pixels over time
        covered_mask = np.zeros((map_h, map_w), dtype=np.uint8)

        # Compute diagonal length for gaze cones
        xs = [pt[0] for pt in room_boundary_coords]
        ys = [pt[1] for pt in room_boundary_coords]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        diag_px = math.hypot(width, height)
        length_map = diag_px * 1.5

        # Prepare a color palette & cache for per-frame triangle drawing
        predefined_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128)
        ]
        track_colors: Dict[int, Tuple[int, int, int]] = {}

        # Iterate through each frame index
        for frame_idx in range(1, max_frame + 1):
            # Accumulate covered pixels for this frame
            for (fidx, trk_id), (ox_px, oy_px, dx, dy) in gaze_info.items():
                if fidx != frame_idx:
                    continue
                if trk_id in enemy_ids:
                    continue

                # Convert origin + direction into map-space
                o_map_x, o_map_y = pixel_mapper.pixel_to_map((ox_px, oy_px))
                ref_px = (ox_px + dx, oy_px + dy)
                ref_map_x, ref_map_y = pixel_mapper.pixel_to_map(ref_px)

                dir_map = np.array([ref_map_x - o_map_x, ref_map_y - o_map_y], dtype=np.float32)
                norm_dir = np.linalg.norm(dir_map)
                if norm_dir == 0:
                    continue
                dir_map /= norm_dir

                origin_map = (float(o_map_x), float(o_map_y))
                tri_map = _gaze_triangle(origin_map, tuple(dir_map.tolist()), half_angle_deg, length_map)
                tri_map_pts = [(float(pt[0]), float(pt[1])) for pt in tri_map]

                tri_polygon = Polygon(tri_map_pts)
                clipped = tri_polygon.intersection(room_polygon)
                if clipped.is_empty:
                    continue

                if clipped.is_empty:
                    continue
                # Handle Polygon and collections of polygons
                if isinstance(clipped, Polygon):
                    polys = [clipped]
                elif hasattr(clipped, 'geoms'):
                    polys = [g for g in clipped.geoms if isinstance(g, Polygon)]
                else:
                    polys = []
                for sub_poly in polys:
                    coords = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                    temp_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                    cv2.fillConvexPoly(temp_mask, coords, 1)
                    covered_mask |= (temp_mask & room_mask)

            # Build visible_frame from base map
            visible_frame = map_image.copy()

            # Create a blurred-black overlay and blend over uncovered regions
            black_overlay = np.zeros_like(map_image, dtype=np.uint8)
            cv2.fillPoly(black_overlay, [room_poly_xy], (0, 0, 0))
            blurred_overlay = cv2.GaussianBlur(black_overlay, (51, 51), sigmaX=0, sigmaY=0)

            mask_uncovered = ((room_mask == 1) & (covered_mask == 0)).astype(np.uint8)
            mask_uncovered_3ch = cv2.merge([mask_uncovered, mask_uncovered, mask_uncovered])

            alpha_blur = 0.6
            blend = cv2.addWeighted(visible_frame, 1.0 - alpha_blur, blurred_overlay, alpha_blur, 0)
            visible_frame[mask_uncovered_3ch.astype(bool)] = blend[mask_uncovered_3ch.astype(bool)]

            # Draw this frame's gaze triangles on top (blended/transparent)
            for (fidx, trk_id), (ox_px, oy_px, dx, dy) in gaze_info.items():
                if fidx != frame_idx:
                    continue
                if trk_id in enemy_ids and not show_enemy_gaze:
                    continue

                if trk_id in enemy_ids:
                    color = (255, 255, 255)
                else:
                    color = track_colors.setdefault(
                        trk_id,
                        predefined_colors[trk_id % len(predefined_colors)]
                    )

                o_map_x, o_map_y = pixel_mapper.pixel_to_map((ox_px, oy_px))
                ref_px = (ox_px + dx, oy_px + dy)
                ref_map_x, ref_map_y = pixel_mapper.pixel_to_map(ref_px)

                dir_map = np.array([ref_map_x - o_map_x, ref_map_y - o_map_y], dtype=np.float32)
                norm_dir_map = np.linalg.norm(dir_map)
                if norm_dir_map == 0:
                    continue
                dir_map /= norm_dir_map

                origin_map = (float(o_map_x), float(o_map_y))
                tri_map = _gaze_triangle(origin_map, tuple(dir_map.tolist()), half_angle_deg, length_map)
                tri_map_pts_int = [(int(pt[0]), int(pt[1])) for pt in tri_map]

                tri_polygon = Polygon(tri_map_pts_int)
                clipped = tri_polygon.intersection(room_polygon)
                if clipped.is_empty:
                    continue

                if clipped.is_empty:
                    continue
                if isinstance(clipped, Polygon):
                    polys = [clipped]
                elif hasattr(clipped, 'geoms'):
                    polys = [g for g in clipped.geoms if isinstance(g, Polygon)]
                else:
                    polys = []
                for sub_poly in polys:
                    coords = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                    tri_overlay = np.zeros_like(visible_frame, dtype=np.uint8)
                    if enable_fill:
                        cv2.fillConvexPoly(tri_overlay, coords, color)
                        if enable_alpha:
                            cv2.addWeighted(tri_overlay, alpha, visible_frame, 1.0, 0, dst=visible_frame)
                        else:
                            tri_mask = np.any(tri_overlay != 0, axis=2)
                            visible_frame[tri_mask] = tri_overlay[tri_mask]
                        if enable_boundary:
                            _draw_dotted_polygon(visible_frame, coords, color, thickness=2)
                    else:
                        _draw_dotted_polygon(tri_overlay, coords, color, thickness=2)
                        # Overlay the dotted lines (always opaque)
                        tri_mask = np.any(tri_overlay != 0, axis=2)
                        visible_frame[tri_mask] = tri_overlay[tri_mask]

            writer.write(visible_frame)

        writer.release()
        return

    # --- NORMAL (NON-ACCUMULATED) MODE: just draw per-frame translucent gaze cones ---

    # Build room polygon for clipping
    room_polygon = Polygon(room_boundary_coords)

    # Determine max_frame
    if total_frames is None:
        if gaze_info:
            max_frame = max(frame_idx for (frame_idx, _) in gaze_info.keys())
        else:
            raise ValueError("`total_frames` must be provided if `gaze_info` is empty.")
    else:
        max_frame = total_frames

    # Prepare color palette & cache
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    # Create VideoWriter for normal mode
    os.makedirs(output_directory, exist_ok=True)
    video_path = os.path.join(output_directory, f"{video_basename}_GazeMap.mp4")
    map_h, map_w = map_image.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (map_w, map_h)
    )

    # Compute diagonal length for gaze cones
    xs = [pt[0] for pt in room_boundary_coords]
    ys = [pt[1] for pt in room_boundary_coords]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    diag_px = math.hypot(width, height)
    length_px = diag_px * 1.5

    # Iterate frames and draw translucent cones
    for frame_idx in range(1, max_frame + 1):
        base_map = map_image.copy()
        for (fidx, trk_id), (ox_px, oy_px, dx, dy) in gaze_info.items():
            if fidx != frame_idx:
                continue
            if trk_id in enemy_ids and not show_enemy_gaze:
                continue

            if trk_id in enemy_ids:
                color = (255, 255, 255)
            else:
                color = track_colors.setdefault(
                    trk_id,
                    predefined_colors[trk_id % len(predefined_colors)]
                )

            o_map_x, o_map_y = pixel_mapper.pixel_to_map((ox_px, oy_px))
            ref_px = (ox_px + dx, oy_px + dy)
            ref_map_x, ref_map_y = pixel_mapper.pixel_to_map(ref_px)
            dir_map = np.array([ref_map_x - o_map_x, ref_map_y - o_map_y], dtype=np.float32)
            norm_dir_map = np.linalg.norm(dir_map)
            if norm_dir_map == 0:
                continue
            dir_map /= norm_dir_map

            origin_map = (float(o_map_x), float(o_map_y))
            tri_map = _gaze_triangle(origin_map, tuple(dir_map.tolist()), half_angle_deg, length_px)
            tri_map_pts = [(float(pt[0]), float(pt[1])) for pt in tri_map]

            tri_polygon = Polygon(tri_map_pts)
            clipped = tri_polygon.intersection(room_polygon)
            if clipped.is_empty:
                continue

            if clipped.is_empty:
                continue
            if isinstance(clipped, Polygon):
                polys = [clipped]
            elif hasattr(clipped, 'geoms'):
                polys = [g for g in clipped.geoms if isinstance(g, Polygon)]
            else:
                polys = []
            for sub_poly in polys:
                poly_xy = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                tri_overlay = np.zeros_like(base_map, dtype=np.uint8)
                if enable_fill:
                    cv2.fillConvexPoly(tri_overlay, poly_xy, color)
                    if enable_alpha:
                        cv2.addWeighted(tri_overlay, alpha, base_map, 1.0, 0, dst=base_map)
                    else:
                        tri_mask = np.any(tri_overlay != 0, axis=2)
                        base_map[tri_mask] = tri_overlay[tri_mask]
                    if enable_boundary:
                        _draw_dotted_polygon(base_map, poly_xy, color, thickness=2)
                else:
                    _draw_dotted_polygon(tri_overlay, poly_xy, color, thickness=2)
                    tri_mask = np.any(tri_overlay != 0, axis=2)
                    base_map[tri_mask] = tri_overlay[tri_mask]

        writer.write(base_map)

    writer.release()


# ----------------------------------------------------------------------
# Combined camera-view: tracking overlays + clearance callouts
# ----------------------------------------------------------------------
def annotate_camera_tracking_with_clearance(raw_frames: List[np.ndarray],
                                            tracker_output: List[Dict],
                                            clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
                                            frame_rate: int,
                                            output_directory: str,
                                            video_basename: str,
                                            enemy_ids: List[int] = None,
                                            gaze_conf_threshold: float = 0.3,
                                            show_clearing_id: bool = True):
    """
    Combined camera-view annotator:
      • Draws tracking overlays (bounding boxes, IDs, and 26‑keypoint skeletons).
      • Adds a clearance callout when an enemy's clearance *end* frame is reached.

    Output file: `<video_basename>_TrackingWithClearance.mp4`
    """
    # Define skeleton connections (pairs of keypoint indices) — same as annotate_camera_video
    skeleton = [
        (15, 13), (13, 11), (11, 19),
        (16, 14), (14, 12), (12, 19),
        (17, 18), (18, 19),
        (18, 5),  (5, 7),  (7, 9),
        (18, 6),  (6, 8),  (8, 10),
        (1, 2),   (0, 1),  (0, 2),
        (1, 3),   (2, 4),  (3, 5),
        (4, 6),   (15, 20),(15, 22),
        (15, 24),(16, 21),(16, 23),
        (16, 25)
    ]

    if enemy_ids is None:
        enemy_ids = [99]

    # Colour palette
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    track_colors: Dict[int, Tuple[int, int, int]] = {}

    # Prepare writer
    os.makedirs(output_directory, exist_ok=True)
    video_path = os.path.join(output_directory, f"{video_basename}_TrackingWithClearance.mp4")
    height, width = raw_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        frame_rate,
        (width, height)
    )

    # Iterate frames
    for frame_data in tracker_output:
        idx = frame_data["frame"] - 1
        frame = raw_frames[idx].copy()

        # First pass: draw boxes, IDs, and skeletons
        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]

            # Color rule
            if trk_id in enemy_ids:
                color = (255, 255, 255)
                id_text = "ID: Enemy"
            else:
                color = track_colors.setdefault(trk_id, predefined_colors[trk_id % len(predefined_colors)])
                id_text = f"ID: {trk_id}"

            # Box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (id_w, id_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
            cv2.putText(frame, id_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

            # Skeleton (requires full 26-keypoint set and scores)
            kps = obj.get("keypoints", [])
            scores = obj.get("keypoint_scores", [])
            if len(kps) == 26 and len(scores) == 26:
                # keypoints
                for (kx, ky), s in zip(kps, scores):
                    if s >= gaze_conf_threshold:
                        cv2.circle(frame, (int(kx), int(ky)), 3, color, -1)
                # skeleton links
                for i1, i2 in skeleton:
                    s1 = scores[i1]
                    s2 = scores[i2]
                    if s1 >= gaze_conf_threshold and s2 >= gaze_conf_threshold:
                        p1 = kps[i1]; p2 = kps[i2]
                        pt1 = (int(p1[0]), int(p1[1])); pt2 = (int(p2[0]), int(p2[1]))
                        cv2.line(frame, pt1, pt2, color, 1)

        # Second pass: overlay clearance labels for enemies whose end frame has passed
        for eid in enemy_ids:
            start, end, fid = clearance_map.get(eid, (None, None, None))
            if end is None:
                continue
            if frame_data["frame"] >= end:
                # find the enemy's bbox this frame
                obj = next((o for o in frame_data["objects"] if o.get("id") == eid), None)
                if obj:
                    x1, y1, x2, y2 = obj["bbox"]
                    color = (255, 255, 255)
                    id_text = "ID: Enemy"
                    (id_w, _), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
                    label = "CLEARED!"
                    if show_clearing_id and fid is not None and fid != -1:
                        label = f"CLEARED by {fid}!"
                    label_pos = (x1 + id_w + 10, y1 - 10)
                    cv2.putText(frame, label, label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        writer.write(frame)

    writer.release()


# --- Room Coverage Analysis and Cache ---
def compute_room_coverage(
    map_image: np.ndarray,
    pixel_mapper,
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    room_boundary_coords: List[Tuple[float, float]],
    frame_rate: int,
    total_frames: Optional[int] = None,
    enemy_ids: List[int] = None,
    half_angle_deg: float = 30.0
) -> Dict[str, object]:
    """
    Walk through every frame from 1..max_frame, build a binary "covered_mask" exactly as in
    the 'accumulated_clear' mode of annotate_map_with_gaze, and compute:
      1. coverage_fraction_per_frame: a list of (frame_idx, fraction_covered_in_room)
      2. time_to_full_coverage: if coverage ever reaches 1.0, (frame_idx_of_full - first_non_enemy_frame)/frame_rate
         otherwise None.
      3. final_fraction: coverage fraction at the last frame processed.

    Args:
        map_image:            BGR numpy array of the empty floor‐plan (map‐space pixels).
        pixel_mapper:         Object with .pixel_to_map((x_px, y_px)) → (x_map, y_map).
        gaze_info:            Dict[(frame_idx, track_id) → (ox_px, oy_px, dx, dy)] in pixel‐space.
        room_boundary_coords: List of (mapX, mapY) tuples defining the room boundary polygon.
        frame_rate:           Frames per second of the source video (used to convert frames → seconds).
        total_frames:         If provided, the maximum frame index to iterate. Otherwise deduced from gaze_info.
        enemy_ids:            List of track IDs considered “enemies” (their gaze is ignored). Defaults to [99].
        half_angle_deg:       Half of the visual angle (in degrees) for the gaze cone (same as annotate_map_with_gaze).

    Returns:
        A dict with keys:
          • "coverage_per_frame": List of (frame_idx (int), fraction_covered (float in [0..1]))
          • "time_to_full": float in seconds (time from first non-enemy frame to full coverage), or None
          • "final_fraction": float in [0..1], coverage fraction at the last frame
          • "first_non_enemy_frame": the first frame index where any non-enemy gaze appeared, or None if never

    Usage:
        result = compute_room_coverage(
            map_image,
            pixel_mapper,
            gaze_info,
            room_boundary_coords,
            frame_rate=30,
            total_frames=None,
            enemy_ids=[99],
            half_angle_deg=10.0
        )
        coverage_list = result["coverage_per_frame"]
        time_full     = result["time_to_full"]
        final_frac    = result["final_fraction"]
    """
    if enemy_ids is None:
        enemy_ids = [99]

    # 1) Build the room polygon and binary room_mask
    map_h, map_w = map_image.shape[:2]
    room_polygon = Polygon(room_boundary_coords)

    # Create a binary mask of the room area (1 inside room, 0 outside)
    room_mask = np.zeros((map_h, map_w), dtype=np.uint8)
    room_poly_xy = np.array(room_boundary_coords, dtype=np.int32)
    cv2.fillPoly(room_mask, [room_poly_xy], 1)

    total_room_pixels = int(room_mask.sum())
    if total_room_pixels == 0:
        raise ValueError("Room boundary polygon has zero area or is outside image bounds.")

    # 2) Determine max_frame
    if total_frames is None:
        if gaze_info:
            max_frame = max(frame_idx for (frame_idx, _) in gaze_info.keys())
        else:
            raise ValueError("`total_frames` must be provided if `gaze_info` is empty.")
    else:
        max_frame = total_frames

    # 3) Precompute length_map for cones (diagonal of room × 1.5)
    xs = [pt[0] for pt in room_boundary_coords]
    ys = [pt[1] for pt in room_boundary_coords]
    width  = max(xs) - min(xs)
    height = max(ys) - min(ys)
    diag_px = math.hypot(width, height)
    length_map = diag_px * 1.5

    # 4) Prepare the accumulating covered_mask
    covered_mask = np.zeros((map_h, map_w), dtype=np.uint8)

    # 5) Identify the first frame where any non-enemy gaze appears:
    non_enemy_frames = [fidx for (fidx, tid) in gaze_info.keys() if tid not in enemy_ids]
    first_non_enemy_frame = min(non_enemy_frames) if non_enemy_frames else None

    coverage_per_frame: List[Tuple[int, float]] = []
    time_to_full: Optional[float] = None
    full_frame_idx: Optional[int] = None

    # 6) Iterate frame by frame
    for frame_idx in range(1, max_frame + 1):
        # a) For this frame, accumulate all gaze cones from non-enemy tracks
        for (fidx, trk_id), (ox_px, oy_px, dx, dy) in gaze_info.items():
            if fidx != frame_idx:
                continue
            if trk_id in enemy_ids:
                continue

            # Convert pixel origin + direction → map-space
            o_map_x, o_map_y = pixel_mapper.pixel_to_map((ox_px, oy_px))
            ref_px = (ox_px + dx, oy_px + dy)
            ref_map_x, ref_map_y = pixel_mapper.pixel_to_map(ref_px)

            dir_map = np.array([ref_map_x - o_map_x, ref_map_y - o_map_y], dtype=np.float32)
            norm_dir = np.linalg.norm(dir_map)
            if norm_dir == 0:
                continue
            dir_map /= norm_dir

            origin_map = (float(o_map_x), float(o_map_y))
            # Build the full triangle in map-space
            tri_map = _gaze_triangle(origin_map, tuple(dir_map.tolist()), half_angle_deg, length_map)
            tri_pts = [(float(pt[0]), float(pt[1])) for pt in tri_map]

            tri_polygon = Polygon(tri_pts)
            clipped = tri_polygon.intersection(room_polygon)
            if clipped.is_empty:
                continue

            # Rasterize clipped polygon into a temporary mask, then OR it into covered_mask
            if isinstance(clipped, Polygon):
                coords = np.array(clipped.exterior.coords, dtype=np.int32)
                temp_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                cv2.fillConvexPoly(temp_mask, coords, 1)
                covered_mask |= (temp_mask & room_mask)
            else:
                # MultiPolygon case: iterate sub-polygons
                for geom in clipped.geoms:
                    if isinstance(geom, Polygon):
                        coords = np.array(geom.exterior.coords, dtype=np.int32)
                        temp_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                        cv2.fillConvexPoly(temp_mask, coords, 1)
                        covered_mask |= (temp_mask & room_mask)

        # b) Compute fraction covered so far
        covered_pixels = int(covered_mask.sum())
        fraction = covered_pixels / total_room_pixels
        coverage_per_frame.append((frame_idx, fraction))

        # c) If coverage just reached 1.0 and we haven't recorded it yet, record time_to_full
        if fraction >= 1.0 and time_to_full is None and first_non_enemy_frame is not None:
            full_frame_idx = frame_idx
            # Time elapsed = (full_frame_idx - first_non_enemy_frame) / frame_rate
            time_to_full = (full_frame_idx - first_non_enemy_frame) / frame_rate

    # 7) After all frames, record final fraction
    final_fraction = coverage_per_frame[-1][1] if coverage_per_frame else 0.0

    return {
        "coverage_per_frame": coverage_per_frame,
        "time_to_full": time_to_full,
        "final_fraction": final_fraction,
        "first_non_enemy_frame": first_non_enemy_frame
    }


def save_room_coverage_cache(
    coverage_data: Dict[str, object],
    output_directory: str,
    video_basename: str
) -> None:
    """
    Write out a "RoomCoverageCache.txt" file containing:
      1. A header "frame,coverage_fraction"
      2. One line per frame: "frame_idx,fraction_covered"
      3. A blank line, then a small summary block:
           first_non_enemy_frame,<value or ''>
           time_to_full_seconds,<value or ''>
           final_fraction,<value>

    Args:
        coverage_data: Dict as returned by compute_room_coverage, with keys:
            • "coverage_per_frame": list of (frame_idx, fraction)
            • "time_to_full": float or None
            • "final_fraction": float
            • "first_non_enemy_frame": int or None
        output_directory: Directory where we write the cache file.
        video_basename:    Base name of the video (no extension).

    Produces:
        <output_directory>/<video_basename>_RoomCoverageCache.txt
    """
    os.makedirs(output_directory, exist_ok=True)
    cache_path = os.path.join(output_directory, f"{video_basename}_RoomCoverageCache.txt")

    cov_list = coverage_data.get("coverage_per_frame", [])
    time_to_full       = coverage_data.get("time_to_full", None)
    final_fraction     = coverage_data.get("final_fraction", 0.0)
    first_non_enemy    = coverage_data.get("first_non_enemy_frame", None)

    with open(cache_path, "w") as f:
        # 1) Write per-frame coverage
        f.write("frame,coverage_fraction\n")
        for frame_idx, frac in cov_list:
            f.write(f"{frame_idx},{frac:.6f}\n")

        # 2) Blank line, then summary
        f.write("\n")
        f.write(f"first_non_enemy_frame,{first_non_enemy if first_non_enemy is not None else ''}\n")
        f.write(f"time_to_full_seconds,{time_to_full if time_to_full is not None else ''}\n")
        f.write(f"final_fraction,{final_fraction:.6f}\n")


# ----------------------------------------------------------------------
# POD‑related helpers (step 1: assign PODs to track IDs by entry logic)
# ----------------------------------------------------------------------

def _first_valid_index(traj: Iterable[tuple]) -> int:
    """Return index of the first non‑None point in a trajectory list."""
    for i, pt in enumerate(traj):
        if pt is not None:
            return i
    raise ValueError("Trajectory contains no valid points")


def assign_pods_by_entry(
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]],
    pods: List[Tuple[float, float]],
    *,
    enemy_ids: Optional[List[int]] = None,
    boundary: Optional[Polygon] = None,
) -> Dict[int, Optional[int]]:
    """
    Geometry‑aware POD assignment.

    1. Project first friend’s entry point to the boundary (“door”).
    2. Infer global CW (+1) / CCW (‑1) flow from the next ≈30 points.
    3. Split the room by a divider (door → centroid → opposite wall).
    4. Within each side (+1 / ‑1) order PODs by perimeter distance
       along the flow, furthest → nearest.
    5. Alternate soldier assignment between the two sides, starting on
       the entrance side.  Lists are printed for debugging.
    """

    if boundary is None:
        raise ValueError("`boundary` Polygon is required.")

    if isinstance(pods, np.ndarray):
        pods = pods.tolist()
    if enemy_ids is None:
        enemy_ids = [99]

    # --- 1. Filter non‑enemy tracks and choose candidates ---
    friend_tracks = {tid: traj for tid, traj in tracks_by_id.items()
                     if tid not in enemy_ids}
    if not friend_tracks or not pods:
        return {}

    def first_frame(item):
        trk = item[1]
        try:
            return _first_valid_index(trk)
        except ValueError:
            return float("inf")

    items = sorted(friend_tracks.items(),
                   key=lambda kv: sum(p is not None for p in kv[1]),
                   reverse=True)[: len(pods)]
    items.sort(key=first_frame)          # earliest appearance first

    # --- 2. Door reference (project entry onto boundary) ---
    first_tid, first_trk = items[0]
    idx0 = _first_valid_index(first_trk)
    entry_pt = Point(first_trk[idx0])

    boundary_line = LineString(boundary.exterior.coords)
    door_pt = boundary_line.interpolate(boundary_line.project(entry_pt))
    centre_pt = Point(boundary.centroid.coords[0])

    # --- 3. Global flow direction (+1 CW / ‑1 CCW) ---
    pts_future = [pt for pt in first_trk[idx0 + 1: idx0 + 91] if pt is not None]
    if pts_future:
        mean_vec = np.mean(np.array(pts_future), axis=0) - np.array(entry_pt.coords[0])
        v_centre = np.array([centre_pt.x, centre_pt.y]) - np.array(entry_pt.coords[0])
        z_cross = np.cross(v_centre, mean_vec)
        movement_sign = -1 if z_cross < 0 else +1
    else:
        movement_sign = -1  # default CCW if no extra points

    # --- 4. Divider and side helper ---
    divider_v = np.array([centre_pt.x, centre_pt.y]) - np.array([door_pt.x, door_pt.y])

    def _side(pt: Point) -> int:
        v = np.array([pt.x, pt.y]) - np.array([door_pt.x, door_pt.y])
        return 1 if np.cross(divider_v, v) >= 0 else -1

    # --- 5. POD metadata: side + perimeter distance (furthest first) ---
    door_s = boundary_line.project(door_pt)
    pod_meta = []
    for idx, pod in enumerate(pods):
        proj_pt = boundary_line.interpolate(boundary_line.project(Point(pod)))
        side = _side(proj_pt)
        pod_s = boundary_line.project(proj_pt)
        perim = ((pod_s - door_s) if side == -1 else (door_s - pod_s)) % boundary_line.length
        pod_meta.append({"idx": idx, "side": side, "perim": perim})

    pods_pos = sorted((d for d in pod_meta if d["side"] == +1),
                      key=lambda d: d["perim"], reverse=True)
    pods_neg = sorted((d for d in pod_meta if d["side"] == -1),
                      key=lambda d: d["perim"], reverse=True)

    # print(f"[assign_pods_by_entry] first entrant {first_tid}, dir={movement_sign}")
    # print(f"  +1 PODs: {[d['idx'] for d in pods_pos]}")
    # print(f"  -1 PODs: {[d['idx'] for d in pods_neg]}")

    # --- 6. Alternate assignment starting on entrance side ---
    assignment: Dict[int, Optional[int]] = {}
    idx_pos = idx_neg = 0
    current_side = movement_sign   # entrance side

    for tid, _ in items:
        if current_side == +1:
            if idx_pos < len(pods_pos):
                assignment[tid] = pods_pos[idx_pos]["idx"]
                idx_pos += 1
            elif idx_neg < len(pods_neg):
                assignment[tid] = pods_neg[idx_neg]["idx"]
                idx_neg += 1
            else:
                assignment[tid] = None
        else:
            if idx_neg < len(pods_neg):
                assignment[tid] = pods_neg[idx_neg]["idx"]
                idx_neg += 1
            elif idx_pos < len(pods_pos):
                assignment[tid] = pods_pos[idx_pos]["idx"]
                idx_pos += 1
            else:
                assignment[tid] = None
        current_side *= -1  # flip for next soldier

    return assignment





# ----------------------------------------------------------------------
# POD‑related helpers (step 2: compute/adjust POD working areas)
# ----------------------------------------------------------------------

def compute_pod_working_areas(
    pods: List[Tuple[float, float]],
    *,
    boundary: Polygon,
    working_radius: float
) -> Dict[int, Dict[str, Union[Tuple[float, float], float]]]:
    """
    For each POD centre, build a circular working area with `working_radius`
    and shift its centre *only along the X and/or Y axes* if the circle would
    fall outside the room `boundary`.

    The algorithm is intentionally simple and fast:

        1. Clamp the centre so that its bounding box [cx-r .. cx+r] lies
           entirely within the boundary's bounding rectangle.  This already
           guarantees the circle is fully inside for rectangular rooms
           (the common case).
        2. If the boundary is non‑rectangular and the resulting circle still
           overlaps outside, perform a small corrective search:
              – Try ±dx steps (1 pixel) along X then Y until the circle is
                entirely within, or a max of 50 iterations each direction.

    Args
    ----
    pods : list[(x, y)]
        POD centres in *map* coordinates.
    boundary : shapely.geometry.Polygon
        Room boundary polygon in the same coordinate space.
    working_radius : float
        Desired radius of the working area.

    Returns
    -------
    dict {pod_index -> {"center": (cx, cy), "radius": r, "shift": (dx, dy)} }
    """
    results: Dict[int, Dict[str, Tuple[float, float] | float]] = {}
    min_x, min_y, max_x, max_y = boundary.bounds

    for i, (cx, cy) in enumerate(pods):
        # --- coarse clamping inside bounding rectangle ---
        cx_clamp = min(max(cx, min_x + working_radius), max_x - working_radius)
        cy_clamp = min(max(cy, min_y + working_radius), max_y - working_radius)
        center = Point(cx_clamp, cy_clamp)

        circle = center.buffer(working_radius)
        if not circle.within(boundary):
            # simple local search: step inward until fits
            step = working_radius * 0.05  # 5 % of radius per step
            moved = False
            for _ in range(100):  # 100 total attempts is plenty
                if circle.within(boundary):
                    break
                # Move towards the polygon centroid (approx inward)
                centroid = boundary.centroid
                vec_x = centroid.x - center.x
                vec_y = centroid.y - center.y
                norm = math.hypot(vec_x, vec_y)
                if norm == 0:
                    break
                vec_x /= norm; vec_y /= norm
                center = Point(center.x + vec_x * step, center.y + vec_y * step)
                circle = center.buffer(working_radius)
                moved = True
            if moved:
                cx_clamp, cy_clamp = center.x, center.y

        dx_shift = cx_clamp - cx
        dy_shift = cy_clamp - cy
        results[i] = {
            "center": (cx_clamp, cy_clamp),
            "radius": working_radius,
            "shift": (dx_shift, dy_shift)
        }

    return results


# ----------------------------------------------------------------------
# POD‑related helpers (step 3: per‑frame dynamic adjustment w/ enemy overlap)
# ----------------------------------------------------------------------

def _bbox_to_map_polygon(bbox_px: Tuple[int, int, int, int],
                         pixel_mapper) -> Polygon:
    """
    Convert a bbox in pixel space (x1,y1,x2,y2) to a shapely Polygon
    in **map coordinates** using `pixel_mapper.pixel_to_map`.
    """
    x1, y1, x2, y2 = bbox_px
    p1 = pixel_mapper.pixel_to_map((x1, y1))
    p2 = pixel_mapper.pixel_to_map((x2, y1))
    p3 = pixel_mapper.pixel_to_map((x2, y2))
    p4 = pixel_mapper.pixel_to_map((x1, y2))
    return Polygon([p1, p2, p3, p4])


def dynamic_pod_working_areas(
    tracker_output: List[Dict],
    *,
    assignment: Dict[int, int],
    initial_working_areas: Dict[int, Dict[str, Union[Tuple[float, float], float]]],
    pixel_mapper,
    boundary: Polygon,
    enemy_ids: List[int],
    working_radius: float,
    overlap_threshold_frac: float = 0.1
) -> Dict[int, Dict[int, Polygon]]:
    """
    Build a per‑frame map of POD working‑area polygons (map space), accounting for enemy overlap.

    NEW LOGIC
    ---------
    • Each *enemy* bbox (converted to a map‑polygon) is assigned to **one** POD — specifically,
      the POD whose current circle shares the largest intersection **area** with that bbox.
    • Only that POD’s circle is shifted / unioned with the enemy polygon.
      This prevents an enemy that straddles two circles from expanding both.
    • Fallback behaviour (no intersection): circles remain unchanged.
    • If an enemy’s overlap with multiple PODs exceeds this threshold (overlap fraction (intersection area / pod circle area)), it will be applied to all such PODs.

    Returns
    -------
    { frame_idx : { pod_idx : shapely Polygon } }
    """
    # Pre‑compute the static circles for every POD
    pod_circles: Dict[int, Polygon] = {
        idx: Point(info["center"]).buffer(working_radius)
        for idx, info in initial_working_areas.items()
    }
    # Pre-compute each POD circle's total area for fraction calculation
    pod_circle_areas: Dict[int, float] = {
        idx: circle.area for idx, circle in pod_circles.items()
    }

    dynamic_map: Dict[int, Dict[int, Polygon]] = {}

    for frame_entry in tracker_output:
        fidx = frame_entry["frame"]

        # ---- 1. Build enemy polygons in map space --------------------
        enemy_polys: List[Polygon] = [
            _bbox_to_map_polygon(tuple(obj["bbox"]), pixel_mapper)
            for obj in frame_entry["objects"]
            if obj["id"] in enemy_ids
        ]

        # ---- 2. Assign each enemy polygon to PODs based on overlap fraction ---
        blocked_by: Dict[int, List[Polygon]] = {idx: [] for idx in pod_circles}
        for epoly in enemy_polys:
            # compute fraction overlap per POD
            fracs = []
            for idx, circle in pod_circles.items():
                if circle.intersects(epoly):
                    raw_area = circle.intersection(epoly).area
                    frac = raw_area / pod_circle_areas[idx]
                    fracs.append((idx, frac))
            if not fracs:
                continue
            max_frac = max(frac for (_, frac) in fracs)
            for idx, frac in fracs:
                # apply if fraction meets threshold and is either max or above threshold
                if frac >= overlap_threshold_frac and (frac == max_frac or frac >= overlap_threshold_frac):
                    blocked_by[idx].append(epoly)

        # ---- 3. Build working area for each POD ----------------------
        pod_polys_this_frame: Dict[int, Polygon] = {}
        for pod_idx, base_circle in pod_circles.items():
            adjusted = base_circle

            for epoly in blocked_by.get(pod_idx, []):
                # Attempt a +Y shift; if that fails, try −Y; otherwise keep original position
                cx, cy = initial_working_areas[pod_idx]["center"]
                shift_up  = Point(cx, cy + working_radius).buffer(working_radius)
                shift_down = Point(cx, cy - working_radius).buffer(working_radius)

                if shift_up.within(boundary) and not shift_up.intersects(epoly):
                    adjusted = shift_up
                elif shift_down.within(boundary) and not shift_down.intersects(epoly):
                    adjusted = shift_down

                # Union with enemy polygon, then clip to room boundary
                adjusted = adjusted.union(epoly).intersection(boundary)

            # Ensure final area respects the boundary even if there was no enemy
            adjusted = adjusted.intersection(boundary)
            pod_polys_this_frame[pod_idx] = adjusted

        dynamic_map[fidx] = pod_polys_this_frame

    return dynamic_map


# ----------------------------------------------------------------------
# POD‑related helpers (step 4: capture‑time evaluation)
# ----------------------------------------------------------------------

def compute_pod_capture_times(
    *,
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]],
    assignment: Dict[int, Optional[int]],
    dynamic_work_areas: Dict[int, Dict[int, Polygon]],
    frame_rate: float,
    capture_threshold_sec: float
) -> Dict[int, Dict[str, Optional[float]]]:
    """
    For every POD that has an assigned soldier, compute how long that soldier
    takes to 'capture' their working area.

    A POD is considered captured when the assigned soldier's map‑position
    remains INSIDE the POD's working area for
        `capture_threshold_sec` *consecutive seconds*.

    Returns
    -------
    { pod_idx :
        {
            "assigned_id"      : track_id or None,
            "capture_time_sec" : float or None
        }
    }
    """
    threshold_frames = int(capture_threshold_sec * frame_rate)
    results: Dict[int, Dict[str, Optional[float]]] = {}

    # Build quick access to first‑appearance frame per track
    first_frame_of_id: Dict[int, int] = {}
    for tid, traj in tracks_by_id.items():
        try:
            idx = _first_valid_index(traj)
            first_frame_of_id[tid] = idx + 1  # frames are 1‑based
        except ValueError:
            first_frame_of_id[tid] = None

    # Determine total frames we have working‑area data for
    max_frame = max(dynamic_work_areas.keys()) if dynamic_work_areas else 0

    for tid, pod_idx in assignment.items():
        if pod_idx is None:
            continue
        traj = tracks_by_id.get(tid, [])
        first_f = first_frame_of_id.get(tid)
        capture_frame: Optional[int] = None
        consec = 0

        for fidx in range(1, max_frame + 1):
            if fidx > len(traj):
                break
            pos = traj[fidx - 1]
            if pos is None:
                consec = 0
                continue

            area_poly = dynamic_work_areas.get(fidx, {}).get(pod_idx)
            if area_poly is None:
                consec = 0
                continue

            if area_poly.contains(Point(pos)):
                consec += 1
                if consec >= threshold_frames and capture_frame is None:
                    capture_frame = fidx - threshold_frames + 1
            else:
                consec = 0

        capture_time_sec: Optional[float]
        if capture_frame is not None and first_f is not None:
            capture_time_sec = (capture_frame - first_f) / frame_rate
        else:
            capture_time_sec = None

        results[pod_idx] = {
            "assigned_id": tid,
            "capture_time_sec": capture_time_sec,
            "capture_frame": capture_frame
        }

    return results


# ----------------------------------------------------------------------
# POD‑related helpers (step 5: cache writer)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# POD‑related helpers (step 5: cache writer)
# ----------------------------------------------------------------------

def save_pod_cache(
    pod_capture_data: Dict[int, Dict[str, Optional[float]]],
    output_directory: str,
    video_basename: str
) -> None:
    """
    Write out "<video_basename>_PodCache.txt" with columns:
        pod_idx,assigned_id,capture_time_sec
    """
    os.makedirs(output_directory, exist_ok=True)
    cache_path = os.path.join(output_directory, f"{video_basename}_PodCache.txt")
    with open(cache_path, "w") as f:
        f.write("pod_idx,assigned_id,capture_time_sec,capture_frame\n")
        for pod_idx in sorted(pod_capture_data.keys()):
            data = pod_capture_data[pod_idx]
            aid = data.get("assigned_id")
            ctime = data.get("capture_time_sec")
            cframe = data.get("capture_frame")
            ctime_str = f"{ctime:.2f}" if ctime is not None else ""
            cframe_str = str(cframe) if cframe is not None else ""
            f.write(f"{pod_idx},{aid if aid is not None else ''},{ctime_str},{cframe_str}\n")


# --- Helper function: save_metrics_cache ---
def save_metrics_cache(metrics: Iterable[Dict[str, Any]],
                       output_directory: str,
                       video_basename: str) -> None:
    """
    Save computed metrics to CSV: metric name, score, assessment.
    Args:
        metrics: iterable of dicts, each containing 'metric_name', 'score', and 'assessment'.
        output_directory: path to save the CSV.
        video_basename: base name for the file.
    """
    os.makedirs(output_directory, exist_ok=True)
    cache_path = os.path.join(output_directory, f"{video_basename}_Metrics.csv")
    with open(cache_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric_name", "score", "assessment"])
        for entry in metrics:
            writer.writerow([entry["metric_name"], entry["score"], entry["assessment"]])


# ----------------------------------------------------------------------
# POD‑related convenience wrapper (one‑call pipeline)
# ----------------------------------------------------------------------
def run_pod_analysis(
    *,
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]],
    tracker_output: List[Dict],
    pods_cfg: List[Tuple[float, float]],
    pixel_mapper,
    boundary: Polygon,
    enemy_ids: List[int],
    working_radius: float,
    frame_rate: float,
    capture_threshold_sec: float,
    save_cache: bool = False,
    output_directory: str = "",
    video_basename: str = ""
) -> Tuple[
        Dict[int, Optional[int]],            # assignment
        Dict[int, Dict[int, Polygon]],       # dynamic_work_areas
        Dict[int, Dict[str, Optional[float]]]# pod_capture_data
]:
    """
    Convenience wrapper that runs the full POD pipeline:

        1. assign_pods_by_entry
        2. compute_pod_working_areas
        3. dynamic_pod_working_areas
        4. compute_pod_capture_times
        5. (optional) save_pod_cache

    Returns
    -------
    (assignment, dynamic_work_areas, pod_capture_data)
    """
    assignment = assign_pods_by_entry(
        tracks_by_id,
        pods_cfg,
        enemy_ids=enemy_ids,
        boundary=boundary
    )

    initial_work_areas = compute_pod_working_areas(
        pods_cfg,
        boundary=boundary,
        working_radius=working_radius
    )

    dynamic_work_areas = dynamic_pod_working_areas(
        tracker_output,
        assignment=assignment,
        initial_working_areas=initial_work_areas,
        pixel_mapper=pixel_mapper,
        boundary=boundary,
        enemy_ids=enemy_ids,
        working_radius=working_radius
    )

    pod_capture_data = compute_pod_capture_times(
        tracks_by_id=tracks_by_id,
        assignment=assignment,
        dynamic_work_areas=dynamic_work_areas,
        frame_rate=frame_rate,
        capture_threshold_sec=capture_threshold_sec
    )

    if save_cache and output_directory and video_basename:
        save_pod_cache(pod_capture_data, output_directory, video_basename)

    return assignment, dynamic_work_areas, pod_capture_data