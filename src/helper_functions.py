import csv
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import cv2
import numpy as np
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon

# ``src.metrics._shared`` is imported lazily inside ``assign_pods_by_entry`` to
# avoid an import cycle: ``src.metrics.__init__`` pulls in ``threat_clearance``,
# which itself imports from this module. Type-only references are gated behind
# ``TYPE_CHECKING`` so static analyzers still resolve them.
if TYPE_CHECKING:
    from src.metrics._shared import DoorAxes  # noqa: F401


# ----------------------------------------------------------------------
# Drill-window helpers
# ----------------------------------------------------------------------


def _in_window(frame_idx: int, start_frame: Optional[int], end_frame: Optional[int]) -> bool:
    """Return True when ``frame_idx`` is within the inclusive drill window.

    Either bound may be ``None`` to disable that side of the clamp. When both
    are ``None`` the call is a no-op and every frame is in-window.
    """
    if start_frame is not None and frame_idx < start_frame:
        return False
    if end_frame is not None and frame_idx > end_frame:
        return False
    return True


def _windowed_frame_range(
    max_frame: int, start_frame: Optional[int], end_frame: Optional[int]
) -> range:
    """Return a clamped 1-indexed inclusive range over ``[start, end] ∩ [1, max_frame]``."""
    s = max(1, int(start_frame)) if start_frame is not None else 1
    e = min(int(max_frame), int(end_frame)) if end_frame is not None else int(max_frame)
    return range(s, e + 1) if e >= s else range(0)


# ----------------------------------------------------------------------
# Frame source helpers: stream frames from a video file.
# ----------------------------------------------------------------------


def _open_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video_path: {video_path}")
    return cap


def _get_frame_stream(video_path: str):
    """Return (cap, get_frame_fn) where get_frame_fn(frame_idx_1based) -> frame|None."""
    if not video_path:
        raise ValueError("video_path must be set for streaming frame access.")

    cap = _open_video_capture(video_path)
    state = {"cur": 0, "last": None}

    def _get(frame_idx_1based: int):
        if frame_idx_1based <= 0:
            return None

        if frame_idx_1based <= state["cur"]:
            return state["last"]

        while state["cur"] < frame_idx_1based:
            ok, fr = cap.read()
            if not ok or fr is None:
                return None
            state["cur"] += 1
            state["last"] = fr

        return state["last"]

    return cap, _get


# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------


def _normalize_inroom_ids(inroom_ids: Optional[List[int]] = None) -> set:
    if inroom_ids is None:
        return set()
    return set(inroom_ids)


def _build_track_color_cache() -> Tuple[List[Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]]:
    predefined_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
    ]
    return predefined_colors, {}


def _get_track_color(
    track_id: int,
    inroom_ids: set,
    track_colors: Dict[int, Tuple[int, int, int]],
    predefined_colors: List[Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    if track_id in inroom_ids:
        return (255, 255, 255)
    return track_colors.setdefault(track_id, predefined_colors[track_id % len(predefined_colors)])


# ----------------------------------------------------------------------
# PixelMapper safety helpers
# ----------------------------------------------------------------------


def _pm_xy(pixel_mapper, pt) -> Optional[Tuple[float, float]]:
    """Robustly get a finite (x, y) from pixel_mapper.pixel_to_map for a single point."""
    if pixel_mapper is None:
        return None
    try:
        xy = pixel_mapper.pixel_to_map(pt)
    except Exception:
        return None
    xy = np.asarray(xy, dtype=float).reshape(-1)
    if xy.size < 2 or (not np.isfinite(xy[:2]).all()):
        return None
    return float(xy[0]), float(xy[1])


def safe_intersection(a, b):
    """Return a.intersection(b) but robust to invalid geometries."""
    if a is None or b is None:
        return None

    try:
        return a.intersection(b)
    except GEOSException:
        try:
            a2 = a.buffer(0)
            b2 = b.buffer(0)
            return a2.intersection(b2)
        except Exception:
            return None


def safe_union(a, b):
    """Return a.union(b) but robust to invalid geometries."""
    if a is None or b is None:
        return None
    try:
        return a.union(b)
    except GEOSException:
        try:
            a2 = a.buffer(0)
            b2 = b.buffer(0)
            if a2.is_empty and b2.is_empty:
                return a2
            if a2.is_empty:
                return b2
            if b2.is_empty:
                return a2
            return a2.union(b2)
        except Exception:
            return None



def _extract_polygons(geom):
    """Return only polygonal pieces from any shapely geometry."""
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]

    if isinstance(geom, GeometryCollection):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]

    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]

    return []

# ----------------------------------------------------------------------
# Configurable Halpe26 face keypoint indices
# ----------------------------------------------------------------------

_DEFAULT_FACE_KP = {
    "NOSE": 0,
    "LEYE": 1,
    "REYE": 2,
    "LEAR": 3,
    "REAR": 4,
}

FACE_KP = dict(_DEFAULT_FACE_KP)
NOSE = FACE_KP["NOSE"]
LEYE = FACE_KP["LEYE"]
REYE = FACE_KP["REYE"]
LEAR = FACE_KP["LEAR"]
REAR = FACE_KP["REAR"]


def initialize_keypoint_indices(config: Optional[dict] = None) -> None:
    """Initialize or reset face keypoint indices."""
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

    NOSE = FACE_KP["NOSE"]
    LEYE = FACE_KP["LEYE"]
    REYE = FACE_KP["REYE"]
    LEAR = FACE_KP["LEAR"]
    REAR = FACE_KP["REAR"]


def compute_gaze_vector(keypoints: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute a 2D gaze vector from Halpe26-format keypoints."""
    if keypoints.ndim != 2 or keypoints.shape[0] != 26 or keypoints.shape[1] < 3:
        return None

    def is_valid(idx: int) -> bool:
        return keypoints[idx, 2] > 0.3

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

    norm = np.linalg.norm(direction)
    if norm <= 0:
        return None

    return origin, direction / norm


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------


def _draw_dotted_polygon(
    img: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
    gap_length: int = 5,
):
    n = len(pts)
    for i in range(n):
        start = np.array(pts[i], dtype=np.float32)
        end = np.array(pts[(i + 1) % n], dtype=np.float32)
        edge_vec = end - start
        length = np.linalg.norm(edge_vec)
        if length == 0:
            continue
        direction = edge_vec / length
        step = dash_length + gap_length
        num_dashes = int(length // step) + 1
        for d in range(num_dashes):
            seg_start = start + direction * (d * step)
            seg_end = start + direction * (d * step + dash_length)
            cv2.line(img, tuple(seg_start.astype(int)), tuple(seg_end.astype(int)), color, thickness)


def _draw_point_with_border(
    img: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    fill_color: Tuple[int, int, int],
    border_color: Tuple[int, int, int] = (0, 0, 0),
    border_thickness: int = 2,
) -> None:
    if border_thickness > 0:
        cv2.circle(img, center, radius + border_thickness, border_color, -1)
    cv2.circle(img, center, radius, fill_color, -1)


def _gaze_triangle(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    half_angle_deg: float,
    length: float = 10000.0,
) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    if n == 0:
        return np.zeros((3, 2), dtype=np.float32)
    d /= n

    ang = math.radians(half_angle_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    rot_left = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rot_right = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)

    left_vec = rot_left @ d * length
    right_vec = rot_right @ d * length

    o = np.asarray(origin, dtype=np.float32)
    p_left = o + left_vec
    p_right = o + right_vec
    return np.stack([o, p_left, p_right], axis=0)


def _triangle_box_intersect(triangle: np.ndarray, box: Tuple[float, float, float, float]) -> bool:
    tri = triangle.reshape(-1, 1, 2).astype(np.float32)
    x1, y1, x2, y2 = map(float, box)
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
    inter_area, _ = cv2.intersectConvexConvex(tri, rect)
    return inter_area > 0.0


def _boxes_intersect(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


# ----------------------------------------------------------------------
# Camera-view annotations
#
# Every camera-view overlay follows the same recipe: stream the source
# video, copy each frame, draw, write. Decoding the source dominates that
# cost, so ``annotate_camera_videos_combined`` writes N overlays in a single
# decode pass. Each overlay contributes ``(out_path, draw_fn)`` built by a
# ``*_overlay_spec`` factory; the public ``annotate_*`` functions remain as
# signature-compatible single-spec wrappers.
# ----------------------------------------------------------------------

_CAMERA_SKELETON = [
    (15, 13), (13, 11), (11, 19),
    (16, 14), (14, 12), (12, 19),
    (17, 18), (18, 19),
    (18, 5), (5, 7), (7, 9),
    (18, 6), (6, 8), (8, 10),
    (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5),
    (4, 6), (15, 20), (15, 22),
    (15, 24), (16, 21), (16, 23),
    (16, 25),
]


def annotate_camera_videos_combined(
    tracker_output: List[Dict],
    frame_rate: int,
    specs: Sequence[Tuple[str, Any]],
    *,
    video_path: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> None:
    """Write N camera-view overlay videos in one pass over the source video.

    ``specs`` is a sequence of ``(out_path, draw_fn)`` where
    ``draw_fn(frame, frame_data)`` mutates its private copy of the decoded
    frame for that overlay.
    """
    if not specs:
        return
    if start_frame is not None or end_frame is not None:
        tracker_output = [
            f for f in tracker_output if _in_window(int(f.get("frame", 0)), start_frame, end_frame)
        ]

    cap, get_frame = _get_frame_stream(video_path)
    writers: List[Tuple[cv2.VideoWriter, Any]] = []
    try:
        first_idx = tracker_output[0]["frame"] if tracker_output else 1
        first_frame = get_frame(first_idx)
        if first_frame is None:
            raise ValueError(
                "Unable to read first frame for camera annotations: "
                + ", ".join(os.path.basename(p) for p, _ in specs)
            )

        height, width = first_frame.shape[:2]
        for out_path, draw_fn in specs:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            writers.append(
                (
                    cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"avc1"), frame_rate, (width, height)
                    ),
                    draw_fn,
                )
            )

        for frame_data in tracker_output:
            frame = get_frame(frame_data["frame"])
            if frame is None:
                break
            for writer, draw_fn in writers:
                overlay = frame.copy()
                draw_fn(overlay, frame_data)
                writer.write(overlay)
    finally:
        for writer, _ in writers:
            writer.release()
        if cap is not None:
            cap.release()


def _draw_track_boxes(
    frame: np.ndarray,
    frame_data: Dict,
    inroom_ids: set,
    track_colors: Dict[int, Tuple[int, int, int]],
    predefined_colors: List[Tuple[int, int, int]],
) -> None:
    """Box + ID label for every tracked object (shared overlay first pass)."""
    for obj in frame_data["objects"]:
        trk_id = obj["id"]
        x1, y1, x2, y2 = obj["bbox"]
        color = _get_track_color(trk_id, inroom_ids, track_colors, predefined_colors)
        id_text = f"ID: InRoom {trk_id}" if trk_id in inroom_ids else f"ID: {trk_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)


from libs.giftpose.meta.wholebody133 import SKELETON as _WB_SKELETON


def camera_tracking_overlay_spec(
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    gaze_conf_threshold: float = 0.3,
) -> Tuple[str, Any]:
    inroom = _normalize_inroom_ids(inroom_ids)
    predefined_colors, track_colors = _build_track_color_cache()
    out_path = os.path.join(output_directory, f"{video_basename}_Tracking_Overlays.mp4")

    def draw(frame: np.ndarray, frame_data: Dict) -> None:
        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            color = _get_track_color(trk_id, inroom, track_colors, predefined_colors)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            id_text = f"ID: InRoom {trk_id}" if trk_id in inroom else f"ID: {trk_id}"
            cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

            wb = obj.get("keypoints_wb")
            wbs = obj.get("keypoint_scores_wb")
            if wb and wbs and len(wb) == 133:
                # Whole-body backends: draw the FULL 133-kp skeleton — body +
                # feet + hand finger links (edges from the COCO-WholeBody
                # metainfo) and the 68 face landmarks as fine dots. The
                # canonical-26 subset is NOT drawn on top (it is derived from
                # these same points and would double the limb lines).
                for i1, i2 in _WB_SKELETON:
                    if wbs[i1] >= gaze_conf_threshold and wbs[i2] >= gaze_conf_threshold:
                        cv2.line(frame, (int(wb[i1][0]), int(wb[i1][1])),
                                 (int(wb[i2][0]), int(wb[i2][1])), color, 1)
                # attach hands to the arms (wrist -> hand root)
                for wr, hr in ((9, 91), (10, 112)):
                    if wbs[wr] >= gaze_conf_threshold and wbs[hr] >= gaze_conf_threshold:
                        cv2.line(frame, (int(wb[wr][0]), int(wb[wr][1])),
                                 (int(wb[hr][0]), int(wb[hr][1])), color, 1)
                for i in range(23):              # body + feet points
                    if wbs[i] >= gaze_conf_threshold:
                        cv2.circle(frame, (int(wb[i][0]), int(wb[i][1])), 3, color, -1)
                for i in range(23, 91):          # face landmarks (dots only)
                    if wbs[i] >= gaze_conf_threshold:
                        cv2.circle(frame, (int(wb[i][0]), int(wb[i][1])), 1, color, -1)
                for i in range(91, 133):         # hand points
                    if wbs[i] >= gaze_conf_threshold:
                        cv2.circle(frame, (int(wb[i][0]), int(wb[i][1])), 2, color, -1)
                continue

            kps = obj.get("keypoints", [])
            scores = obj.get("keypoint_scores", [])
            if len(kps) == 26 and len(scores) == 26:
                for (x, y), s in zip(kps, scores):
                    if s >= gaze_conf_threshold:
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

                for i1, i2 in _CAMERA_SKELETON:
                    if scores[i1] >= gaze_conf_threshold and scores[i2] >= gaze_conf_threshold:
                        p1 = (int(kps[i1][0]), int(kps[i1][1]))
                        p2 = (int(kps[i2][0]), int(kps[i2][1]))
                        cv2.line(frame, p1, p2, color, 1)

    return out_path, draw


def gaze_triangle_overlay_spec(
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    half_angle_deg: float = 30.0,
    alpha: float = 0.2,
    show_inroom_gaze: bool = True,
) -> Tuple[str, Any]:
    inroom = _normalize_inroom_ids(inroom_ids)
    predefined_colors, track_colors = _build_track_color_cache()
    out_path = os.path.join(output_directory, f"{video_basename}_Gaze_Triangles.mp4")

    def draw(frame: np.ndarray, frame_data: Dict) -> None:
        _draw_track_boxes(frame, frame_data, inroom, track_colors, predefined_colors)

        tri_len = math.hypot(frame.shape[1], frame.shape[0]) * 1.5
        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            if trk_id in inroom and not show_inroom_gaze:
                continue

            gaze = gaze_info.get((frame_data["frame"], trk_id))
            if gaze is None:
                continue

            color = _get_track_color(trk_id, inroom, track_colors, predefined_colors)
            ox, oy, dx, dy = gaze
            tri = _gaze_triangle((ox, oy), (dx, dy), half_angle_deg, length=tri_len).astype(int)

            tri_overlay = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillConvexPoly(tri_overlay, tri, color)
            cv2.addWeighted(tri_overlay, alpha, frame, 1.0, 0, dst=frame)

    return out_path, draw


def clearance_overlay_spec(
    clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
) -> Tuple[str, Any]:
    inroom = _normalize_inroom_ids(inroom_ids)
    predefined_colors, track_colors = _build_track_color_cache()
    out_path = os.path.join(output_directory, f"{video_basename}_Clearance_Callouts.mp4")

    def draw(frame: np.ndarray, frame_data: Dict) -> None:
        _draw_track_boxes(frame, frame_data, inroom, track_colors, predefined_colors)

        for inroom_id in inroom:
            start, end, _ = clearance_map.get(inroom_id, (None, None, None))
            if end is None or frame_data["frame"] < end:
                continue

            obj = next((o for o in frame_data["objects"] if o.get("id") == inroom_id), None)
            if obj is None:
                continue

            x1, y1, x2, y2 = obj["bbox"]
            color = (255, 255, 255)
            id_text = f"ID: InRoom {inroom_id}"
            (id_w, _), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
            cv2.putText(frame, "CLEARED!", (x1 + id_w + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

    return out_path, draw


def tracking_with_clearance_overlay_spec(
    clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    gaze_conf_threshold: float = 0.3,
    show_clearing_id: bool = True,
) -> Tuple[str, Any]:
    inroom = _normalize_inroom_ids(inroom_ids)
    predefined_colors, track_colors = _build_track_color_cache()
    out_path = os.path.join(output_directory, f"{video_basename}_Tracking_WithClearance.mp4")

    def draw(frame: np.ndarray, frame_data: Dict) -> None:
        for obj in frame_data["objects"]:
            trk_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            color = _get_track_color(trk_id, inroom, track_colors, predefined_colors)
            id_text = f"ID: InRoom {trk_id}" if trk_id in inroom else f"ID: {trk_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

            kps = obj.get("keypoints", [])
            scores = obj.get("keypoint_scores", [])
            if len(kps) == 26 and len(scores) == 26:
                for (kx, ky), s in zip(kps, scores):
                    if s >= gaze_conf_threshold:
                        cv2.circle(frame, (int(kx), int(ky)), 3, color, -1)

                for i1, i2 in _CAMERA_SKELETON:
                    if scores[i1] >= gaze_conf_threshold and scores[i2] >= gaze_conf_threshold:
                        pt1 = (int(kps[i1][0]), int(kps[i1][1]))
                        pt2 = (int(kps[i2][0]), int(kps[i2][1]))
                        cv2.line(frame, pt1, pt2, color, 1)

        for inroom_id in inroom:
            start, end, fid = clearance_map.get(inroom_id, (None, None, None))
            if end is None or frame_data["frame"] < end:
                continue

            obj = next((o for o in frame_data["objects"] if o.get("id") == inroom_id), None)
            if obj is None:
                continue

            x1, y1, x2, y2 = obj["bbox"]
            color = (255, 255, 255)
            id_text = f"ID: InRoom {inroom_id}"
            (id_w, _), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
            label = f"CLEARED by {fid}!" if show_clearing_id and fid is not None and fid != -1 else "CLEARED!"
            cv2.putText(frame, label, (x1 + id_w + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    return out_path, draw


def annotate_camera_video(
    tracker_output: List[Dict],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    gaze_conf_threshold: float = 0.3,
    *,
    video_path: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    annotate_camera_videos_combined(
        tracker_output,
        frame_rate,
        [camera_tracking_overlay_spec(output_directory, video_basename, inroom_ids, gaze_conf_threshold)],
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def annotate_camera_with_gaze_triangle(
    tracker_output: List[Dict],
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    half_angle_deg: float = 30.0,
    alpha: float = 0.2,
    show_inroom_gaze: bool = True,
    *,
    video_path: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    annotate_camera_videos_combined(
        tracker_output,
        frame_rate,
        [
            gaze_triangle_overlay_spec(
                gaze_info,
                output_directory,
                video_basename,
                inroom_ids,
                half_angle_deg,
                alpha,
                show_inroom_gaze,
            )
        ],
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def annotate_clearance_video(
    tracker_output: List[Dict],
    clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    *,
    video_path: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    annotate_camera_videos_combined(
        tracker_output,
        frame_rate,
        [clearance_overlay_spec(clearance_map, output_directory, video_basename, inroom_ids)],
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )


# ----------------------------------------------------------------------
# Map-view annotations
# ----------------------------------------------------------------------


def annotate_map_video(
    map_image: np.ndarray,
    all_map_points: List[Tuple[int, int, float, float]],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    total_frames: Optional[int] = None,
    *,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    inroom_ids = _normalize_inroom_ids(inroom_ids)
    predefined_colors, track_colors = _build_track_color_cache()

    map_out = os.path.join(output_directory, f"{video_basename}_Tracking_Map.mp4")
    h, w = map_image.shape[:2]
    writer = cv2.VideoWriter(map_out, cv2.VideoWriter_fourcc(*"avc1"), frame_rate, (w, h))

    points_per_frame: Dict[int, List[Tuple[int, float, float]]] = {}
    for frm, tid, mx, my in all_map_points:
        points_per_frame.setdefault(frm, []).append((tid, mx, my))

    max_frame = total_frames if total_frames is not None else (max(points_per_frame.keys()) if points_per_frame else 0)
    permanent_vis = map_image.copy()
    last_pos: Dict[int, Tuple[float, float]] = {}

    for frame_num in _windowed_frame_range(max_frame, start_frame, end_frame):
        temp_vis = permanent_vis.copy()
        for tid, mx, my in points_per_frame.get(frame_num, []):
            color = _get_track_color(tid, inroom_ids, track_colors, predefined_colors)
            _draw_point_with_border(temp_vis, (int(mx), int(my)), 8, color)

            if tid not in inroom_ids:
                prev = last_pos.get(tid)
                if prev is not None:
                    cv2.line(permanent_vis, (int(prev[0]), int(prev[1])), (int(mx), int(my)), color, 2)
                last_pos[tid] = (mx, my)

        writer.write(temp_vis)

    writer.release()


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
    inroom_ids: Optional[List[int]] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    inroom_ids = _normalize_inroom_ids(inroom_ids)
    predefined_colors, track_colors = _build_track_color_cache()

    pts_per_frame: Dict[int, List[Tuple[int, float, float]]] = {}
    for frm, tid, mx, my in all_map_points:
        pts_per_frame.setdefault(frm, []).append((tid, mx, my))

    max_frame = total_frames if total_frames is not None else (max(pts_per_frame.keys()) if pts_per_frame else 0)

    pod_capture_frame: Dict[int, Optional[int]] = {}
    fps = frame_rate
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

    pod_colors: Dict[int, Tuple[int, int, int]] = {}
    for tid, pod_idx in assignment.items():
        if pod_idx is None:
            continue
        pod_colors[pod_idx] = _get_track_color(tid, inroom_ids, track_colors, predefined_colors)

    os.makedirs(output_directory, exist_ok=True)
    out_path = os.path.join(output_directory, f"{video_basename}_Tracking_PodAreas.mp4")
    h, w = map_image.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), frame_rate, (w, h))

    for frame_idx in _windowed_frame_range(max_frame, start_frame, end_frame):
        vis = map_image.copy()
        frame_polys = dynamic_work_areas.get(frame_idx, {})

        for pod_idx, poly in frame_polys.items():
            color = pod_colors.get(pod_idx, (200, 200, 200))
            cap_frame = pod_capture_frame.get(pod_idx)

            polys_iter = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
            for sub_poly in polys_iter:
                pts = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                if cap_frame is not None and frame_idx >= cap_frame:
                    cv2.fillConvexPoly(vis, pts, color)
                else:
                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        for tid, mx, my in pts_per_frame.get(frame_idx, []):
            clr = _get_track_color(tid, inroom_ids, track_colors, predefined_colors)
            _draw_point_with_border(vis, (int(mx), int(my)), 6, clr)

        writer.write(vis)

    writer.release()


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
    inroom_ids: Optional[List[int]] = None,
    fill_alpha: float = 0.35,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    inroom_ids = _normalize_inroom_ids(inroom_ids)
    predefined_colors, track_colors = _build_track_color_cache()

    pts_per_frame: Dict[int, List[Tuple[int, float, float]]] = {}
    for frm, tid, mx, my in all_map_points:
        pts_per_frame.setdefault(frm, []).append((tid, mx, my))

    max_frame = total_frames if total_frames is not None else (max(pts_per_frame.keys()) if pts_per_frame else 0)

    pod_capture_frame: Dict[int, Optional[int]] = {}
    fps = frame_rate
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

    pod_colors: Dict[int, Tuple[int, int, int]] = {}
    for tid, pod_idx in assignment.items():
        if pod_idx is None:
            continue
        pod_colors[pod_idx] = _get_track_color(tid, inroom_ids, track_colors, predefined_colors)

    os.makedirs(output_directory, exist_ok=True)
    out_path = os.path.join(output_directory, f"{video_basename}_Tracking_PodAreasWithTrails.mp4")
    h, w = map_image.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), frame_rate, (w, h))

    permanent_trails = map_image.copy()
    last_pos: Dict[int, Tuple[float, float]] = {}

    for frame_idx in _windowed_frame_range(max_frame, start_frame, end_frame):
        vis = permanent_trails.copy()
        frame_polys = dynamic_work_areas.get(frame_idx, {})

        for pod_idx, poly in frame_polys.items():
            color = pod_colors.get(pod_idx, (200, 200, 200))
            cap_frame = pod_capture_frame.get(pod_idx)
            polys_iter = [poly] if isinstance(poly, Polygon) else list(getattr(poly, "geoms", []))

            for sub_poly in polys_iter:
                pts = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                if cap_frame is not None and frame_idx >= cap_frame:
                    overlay = vis.copy()
                    cv2.fillConvexPoly(overlay, pts, color)
                    cv2.addWeighted(overlay, fill_alpha, vis, 1.0 - fill_alpha, 0, dst=vis)
                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
                else:
                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        for tid, mx, my in pts_per_frame.get(frame_idx, []):
            clr = _get_track_color(tid, inroom_ids, track_colors, predefined_colors)
            _draw_point_with_border(vis, (int(mx), int(my)), 6, clr)

        for tid, mx, my in pts_per_frame.get(frame_idx, []):
            if tid in inroom_ids:
                continue
            clr = _get_track_color(tid, inroom_ids, track_colors, predefined_colors)
            prev = last_pos.get(tid)
            if prev is not None:
                cv2.line(permanent_trails, (int(prev[0]), int(prev[1])), (int(mx), int(my)), clr, 2)
            last_pos[tid] = (mx, my)

        writer.write(vis)

    writer.release()


# ----------------------------------------------------------------------
# Cache writers
# ----------------------------------------------------------------------


def save_position_cache(
    all_map_points: List[Tuple[int, int, float, float]],
    output_directory: str,
    video_basename: str,
):
    cache_path = os.path.join(output_directory, f"{video_basename}_PositionCache.txt")
    with open(cache_path, "w") as f:
        f.write("frame,id,mapX,mapY\n")
        for frm, tid, mx, my in all_map_points:
            f.write(f"{frm},{tid},{mx},{my}\n")


def save_gaze_cache(
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    output_directory: str,
    video_basename: str,
):
    cache_path = os.path.join(output_directory, f"{video_basename}_GazeCache.txt")
    with open(cache_path, "w") as f:
        f.write("frame,id,ox,oy,dx,dy\n")
        for (frm, tid), (ox, oy, dx, dy) in sorted(gaze_info.items()):
            f.write(f"{frm},{tid},{ox},{oy},{dx},{dy}\n")


def save_threat_clearance_cache(
    clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
    output_directory: str,
    video_basename: str,
):
    cache_path = os.path.join(output_directory, f"{video_basename}_ThreatClearanceCache.txt")
    with open(cache_path, "w") as f:
        f.write("inroom_id,immediate_frame,contact_end_frame,clearing_friend\n")
        for inroom_id, (start, end, fid) in clearance_map.items():
            f.write(
                f"{inroom_id},"
                f"{start if start is not None else -1},"
                f"{end if end is not None else -1},"
                f"{fid if fid is not None else -1}\n"
            )


def save_room_coverage_cache(
    coverage_data: Dict[str, object],
    output_directory: str,
    video_basename: str,
) -> None:
    os.makedirs(output_directory, exist_ok=True)
    cache_path = os.path.join(output_directory, f"{video_basename}_RoomCoverageCache.txt")

    cov_list = coverage_data.get("coverage_per_frame", [])
    time_to_full = coverage_data.get("time_to_full", None)
    final_fraction = coverage_data.get("final_fraction", 0.0)
    first_non_enemy = coverage_data.get("first_non_enemy_frame", None)

    with open(cache_path, "w") as f:
        f.write("frame,coverage_fraction\n")
        for frame_idx, frac in cov_list:
            f.write(f"{frame_idx},{frac:.2f}\n")

        f.write("\n")
        f.write(f"first_non_enemy_frame,{first_non_enemy if first_non_enemy is not None else ''}\n")
        time_to_full_str = f"{time_to_full:.2f}" if time_to_full is not None else ""
        f.write(f"time_to_full_seconds,{time_to_full_str}\n")
        f.write(f"final_fraction,{final_fraction:.2f}\n")


def save_pod_cache(
    pod_capture_data: Dict[int, Dict[str, Optional[float]]],
    output_directory: str,
    video_basename: str,
) -> None:
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


def save_metrics_cache(
    metrics: Iterable[Dict[str, Any]],
    output_directory: str,
    video_basename: str,
) -> None:
    os.makedirs(output_directory, exist_ok=True)
    cache_path = os.path.join(output_directory, f"{video_basename}_Metrics.csv")
    with open(cache_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_name", "score"])
        for entry in metrics:
            writer.writerow([entry["metric_name"], entry["score"]])


# ----------------------------------------------------------------------
# Threat clearance
# ----------------------------------------------------------------------


def _has_valid_run(
    info_list: List[Tuple[int, bool, bool]],
    intersection_thr: int,
    wrist_thr: int,
    gaze_thr: int,
) -> Tuple[Optional[int], Optional[int]]:
    if not info_list:
        return None, None

    frame_map: Dict[int, Tuple[bool, bool]] = {}
    for f_idx, w_flag, g_flag in info_list:
        prev_w, prev_g = frame_map.get(f_idx, (False, False))
        frame_map[f_idx] = (prev_w or w_flag, prev_g or g_flag)

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
            early_frame: Optional[int] = None

            for fr in run:
                w_flag, g_flag = frame_map[fr]
                if w_flag:
                    wrist_cnt += 1
                if g_flag:
                    gaze_cnt += 1
                if wrist_cnt >= wrist_thr and gaze_cnt >= gaze_thr and early_frame is None:
                    early_frame = fr

            if wrist_cnt >= wrist_thr and gaze_cnt >= gaze_thr:
                return early_frame if early_frame is not None else run[0], run[-1]

        i = j

    return None, None


def compute_threat_clearance(
    tracker_output: List[Dict],
    keypoint_details: Dict[Tuple[int, int], Tuple[List, List]],
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    *,
    inroom_ids: Optional[List[int]] = None,
    visual_angle_deg: float = 20.0,
    intersection_frames: int = 30,
    wrist_frames: int = 7,
    gaze_frames: int = 15,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]]:
    if start_frame is not None or end_frame is not None:
        tracker_output = [
            f for f in tracker_output if _in_window(int(f.get("frame", 0)), start_frame, end_frame)
        ]

    inroom_ids = _normalize_inroom_ids(inroom_ids)
    if not inroom_ids:
        return {}
    half_angle = visual_angle_deg / 2.0

    clearance: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]] = {
        inroom_id: (None, None, None) for inroom_id in inroom_ids
    }

    per_pair: Dict[Tuple[int, int], List[Tuple[int, bool, bool]]] = {}

    for frame_entry in tracker_output:
        fidx = frame_entry["frame"]
        bboxes = {obj["id"]: tuple(obj["bbox"]) for obj in frame_entry["objects"]}
        inrooms = [tid for tid in bboxes if tid in inroom_ids]
        friends = [tid for tid in bboxes if tid not in inroom_ids]

        for inroom_id in inrooms:
            ibox = bboxes[inroom_id]
            ix1, iy1, ix2, iy2 = ibox

            for fid in friends:
                fbox = bboxes[fid]
                if not _boxes_intersect(ibox, fbox):
                    continue

                wrist_flag = False
                kp_tuple = keypoint_details.get((fidx, fid))
                if kp_tuple and len(kp_tuple[0]) > 10:
                    kp_list, kp_scores = kp_tuple
                    for wi in (9, 10):
                        try:
                            wx, wy = kp_list[wi]
                        except Exception:
                            continue

                        score_ok = True
                        if kp_scores is not None and len(kp_scores) > wi:
                            try:
                                score_ok = float(kp_scores[wi]) > 0.0
                            except Exception:
                                score_ok = True

                        if score_ok and ix1 <= wx <= ix2 and iy1 <= wy <= iy2:
                            wrist_flag = True
                            break

                gaze_flag = False
                g = gaze_info.get((fidx, fid))
                if g:
                    ox, oy, dx, dy = g
                    tri = _gaze_triangle((ox, oy), (dx, dy), half_angle)
                    if _triangle_box_intersect(tri, ibox):
                        gaze_flag = True

                per_pair.setdefault((inroom_id, fid), []).append((fidx, wrist_flag, gaze_flag))

    for inroom_id in inroom_ids:
        best_result: Optional[Tuple[int, int, int]] = None
        pair_keys = [pair_key for pair_key in per_pair.keys() if pair_key[0] == inroom_id]

        for _, fid in pair_keys:
            frm_start, frm_end = _has_valid_run(
                per_pair[(inroom_id, fid)],
                intersection_frames,
                wrist_frames,
                gaze_frames,
            )
            if frm_start is None or frm_end is None:
                continue

            candidate = (frm_start, frm_end, fid)
            if best_result is None:
                best_result = candidate
                continue

            best_start, best_end, best_fid = best_result
            if (
                frm_start < best_start
                or (frm_start == best_start and frm_end < best_end)
                or (frm_start == best_start and frm_end == best_end and fid < best_fid)
            ):
                best_result = candidate

        if best_result is not None:
            clearance[inroom_id] = best_result

    return clearance


# ----------------------------------------------------------------------
# Map gaze / coverage
# ----------------------------------------------------------------------


def annotate_map_with_gaze(
    map_image: np.ndarray,
    pixel_mapper,
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    room_boundary_coords: List[Tuple[float, float]],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    show_inroom_gaze: bool = True,
    half_angle_deg: float = 20.0,
    alpha: float = 0.3,
    enable_alpha: bool = True,
    enable_fill: bool = True,
    enable_boundary: bool = True,
    total_frames: Optional[int] = None,
    accumulated_clear: bool = False,
    *,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    inroom_ids = _normalize_inroom_ids(inroom_ids)

    gaze_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for (fidx, tid), g in gaze_info.items():
        gaze_by_frame.setdefault(fidx, []).append((tid, g))

    room_polygon = Polygon(room_boundary_coords)
    max_frame = total_frames if total_frames is not None else (
        max(frame_idx for (frame_idx, _) in gaze_info.keys()) if gaze_info else None
    )
    if max_frame is None:
        raise ValueError("`total_frames` must be provided if `gaze_info` is empty.")

    predefined_colors, track_colors = _build_track_color_cache()

    xs = [pt[0] for pt in room_boundary_coords]
    ys = [pt[1] for pt in room_boundary_coords]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    diag_px = math.hypot(width, height)
    length_map = diag_px * 1.5

    os.makedirs(output_directory, exist_ok=True)
    output_name = f"{video_basename}_Gaze_MapCleared.mp4" if accumulated_clear else f"{video_basename}_Gaze_Map.mp4"
    video_path = os.path.join(output_directory, output_name)
    map_h, map_w = map_image.shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"avc1"), frame_rate, (map_w, map_h))

    if accumulated_clear:
        room_mask = np.zeros((map_h, map_w), dtype=np.uint8)
        room_poly_xy = np.array(room_boundary_coords, dtype=np.int32)
        cv2.fillPoly(room_mask, [room_poly_xy], 1)
        covered_mask = np.zeros((map_h, map_w), dtype=np.uint8)

        for frame_idx in _windowed_frame_range(max_frame, start_frame, end_frame):
            for trk_id, (ox_px, oy_px, dx, dy) in gaze_by_frame.get(frame_idx, []):
                if trk_id in inroom_ids:
                    continue

                o_xy = _pm_xy(pixel_mapper, (ox_px, oy_px))
                ref_xy = _pm_xy(pixel_mapper, (ox_px + dx, oy_px + dy))
                if o_xy is None or ref_xy is None:
                    continue

                o_map_x, o_map_y = o_xy
                ref_map_x, ref_map_y = ref_xy
                dir_map = np.array([ref_map_x - o_map_x, ref_map_y - o_map_y], dtype=np.float32)
                norm_dir = np.linalg.norm(dir_map)
                if norm_dir == 0:
                    continue
                dir_map /= norm_dir

                tri_map = _gaze_triangle((o_map_x, o_map_y), tuple(dir_map.tolist()), half_angle_deg, length_map)
                tri_polygon = Polygon([(float(pt[0]), float(pt[1])) for pt in tri_map])
                clipped = safe_intersection(tri_polygon, room_polygon)
                if clipped is None or clipped.is_empty:
                    continue

                polys = _extract_polygons(clipped)
                if not polys:
                    continue

                for sub_poly in polys:
                    coords = np.array(list(sub_poly.exterior.coords), dtype=np.int32)
                    temp_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                    cv2.fillConvexPoly(temp_mask, coords, 1)
                    covered_mask |= (temp_mask & room_mask)

            visible_frame = map_image.copy()
            black_overlay = np.zeros_like(map_image, dtype=np.uint8)
            cv2.fillPoly(black_overlay, [room_poly_xy], (0, 0, 0))
            blurred_overlay = cv2.GaussianBlur(black_overlay, (51, 51), sigmaX=0, sigmaY=0)

            mask_uncovered = ((room_mask == 1) & (covered_mask == 0)).astype(np.uint8)
            mask_uncovered_3ch = cv2.merge([mask_uncovered, mask_uncovered, mask_uncovered])
            alpha_blur = 0.6
            blend = cv2.addWeighted(visible_frame, 1.0 - alpha_blur, blurred_overlay, alpha_blur, 0)
            visible_frame[mask_uncovered_3ch.astype(bool)] = blend[mask_uncovered_3ch.astype(bool)]

            for trk_id, (ox_px, oy_px, dx, dy) in gaze_by_frame.get(frame_idx, []):
                if trk_id in inroom_ids and not show_inroom_gaze:
                    continue

                color = _get_track_color(trk_id, inroom_ids, track_colors, predefined_colors)

                o_xy = _pm_xy(pixel_mapper, (ox_px, oy_px))
                ref_xy = _pm_xy(pixel_mapper, (ox_px + dx, oy_px + dy))
                if o_xy is None or ref_xy is None:
                    continue

                o_map_x, o_map_y = o_xy
                ref_map_x, ref_map_y = ref_xy
                dir_map = np.array([ref_map_x - o_map_x, ref_map_y - o_map_y], dtype=np.float32)
                norm_dir = np.linalg.norm(dir_map)
                if norm_dir == 0:
                    continue
                dir_map /= norm_dir

                tri_map = _gaze_triangle((o_map_x, o_map_y), tuple(dir_map.tolist()), half_angle_deg, length_map)
                tri_polygon = Polygon([(int(pt[0]), int(pt[1])) for pt in tri_map])
                clipped = safe_intersection(tri_polygon, room_polygon)
                if clipped is None or clipped.is_empty:
                    continue

                polys = _extract_polygons(clipped)
                if not polys:
                    continue

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
                        tri_mask = np.any(tri_overlay != 0, axis=2)
                        visible_frame[tri_mask] = tri_overlay[tri_mask]

            writer.write(visible_frame)

        writer.release()
        return

    for frame_idx in _windowed_frame_range(max_frame, start_frame, end_frame):
        base_map = map_image.copy()
        for trk_id, (ox_px, oy_px, dx, dy) in gaze_by_frame.get(frame_idx, []):
            if trk_id in inroom_ids and not show_inroom_gaze:
                continue

            color = _get_track_color(trk_id, inroom_ids, track_colors, predefined_colors)

            o_xy = _pm_xy(pixel_mapper, (ox_px, oy_px))
            ref_xy = _pm_xy(pixel_mapper, (ox_px + dx, oy_px + dy))
            if o_xy is None or ref_xy is None:
                continue

            o_map_x, o_map_y = o_xy
            ref_map_x, ref_map_y = ref_xy
            dir_map = np.array([ref_map_x - o_map_x, ref_map_y - o_map_y], dtype=np.float32)
            norm_dir = np.linalg.norm(dir_map)
            if norm_dir == 0:
                continue
            dir_map /= norm_dir

            tri_map = _gaze_triangle((o_map_x, o_map_y), tuple(dir_map.tolist()), half_angle_deg, length_map)
            tri_polygon = Polygon([(float(pt[0]), float(pt[1])) for pt in tri_map])
            clipped = safe_intersection(tri_polygon, room_polygon)
            if clipped is None or clipped.is_empty:
                continue

            polys = _extract_polygons(clipped)
            if not polys:
                continue

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


def annotate_camera_tracking_with_clearance(
    tracker_output: List[Dict],
    clearance_map: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]],
    frame_rate: int,
    output_directory: str,
    video_basename: str,
    inroom_ids: List[int] = None,
    gaze_conf_threshold: float = 0.3,
    show_clearing_id: bool = True,
    *,
    video_path: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    annotate_camera_videos_combined(
        tracker_output,
        frame_rate,
        [
            tracking_with_clearance_overlay_spec(
                clearance_map,
                output_directory,
                video_basename,
                inroom_ids,
                gaze_conf_threshold,
                show_clearing_id,
            )
        ],
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def compute_room_coverage(
    map_image: np.ndarray,
    pixel_mapper,
    gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]],
    room_boundary_coords: List[Tuple[float, float]],
    frame_rate: int,
    total_frames: Optional[int] = None,
    inroom_ids: List[int] = None,
    half_angle_deg: float = 30.0,
    *,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Dict[str, object]:
    inroom_ids = _normalize_inroom_ids(inroom_ids)

    gaze_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for (fidx, tid), g in gaze_info.items():
        gaze_by_frame.setdefault(fidx, []).append((tid, g))

    map_h, map_w = map_image.shape[:2]
    room_polygon = Polygon(room_boundary_coords)

    room_mask = np.zeros((map_h, map_w), dtype=np.uint8)
    room_poly_xy = np.array(room_boundary_coords, dtype=np.int32)
    cv2.fillPoly(room_mask, [room_poly_xy], 1)

    total_room_pixels = int(room_mask.sum())
    if total_room_pixels == 0:
        raise ValueError("Room boundary polygon has zero area or is outside image bounds.")

    max_frame = total_frames if total_frames is not None else (
        max(frame_idx for (frame_idx, _) in gaze_info.keys()) if gaze_info else None
    )
    if max_frame is None:
        raise ValueError("`total_frames` must be provided if `gaze_info` is empty.")

    xs = [pt[0] for pt in room_boundary_coords]
    ys = [pt[1] for pt in room_boundary_coords]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    diag_px = math.hypot(width, height)
    length_map = diag_px * 1.5

    covered_mask = np.zeros((map_h, map_w), dtype=np.uint8)

    non_enemy_frames = [fidx for (fidx, tid) in gaze_info.keys() if tid not in inroom_ids]
    first_non_enemy_frame = min(non_enemy_frames) if non_enemy_frames else None

    coverage_per_frame: List[Tuple[int, float]] = []
    time_to_full: Optional[float] = None

    for frame_idx in _windowed_frame_range(max_frame, start_frame, end_frame):
        for trk_id, (ox_px, oy_px, dx, dy) in gaze_by_frame.get(frame_idx, []):
            if trk_id in inroom_ids:
                continue

            o_xy = _pm_xy(pixel_mapper, (ox_px, oy_px))
            ref_xy = _pm_xy(pixel_mapper, (ox_px + dx, oy_px + dy))
            if o_xy is None or ref_xy is None:
                continue

            o_map_x, o_map_y = o_xy
            ref_map_x, ref_map_y = ref_xy
            dir_map = np.array([ref_map_x - o_map_x, ref_map_y - o_map_y], dtype=np.float32)
            norm_dir = np.linalg.norm(dir_map)
            if norm_dir == 0:
                continue
            dir_map /= norm_dir

            tri_map = _gaze_triangle((o_map_x, o_map_y), tuple(dir_map.tolist()), half_angle_deg, length_map)
            tri_polygon = Polygon([(float(pt[0]), float(pt[1])) for pt in tri_map])
            clipped = safe_intersection(tri_polygon, room_polygon)
            if clipped is None or clipped.is_empty:
                continue

            polys = _extract_polygons(clipped)
            if not polys:
                continue

            for geom in polys:
                coords = np.array(geom.exterior.coords, dtype=np.int32)
                temp_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                cv2.fillConvexPoly(temp_mask, coords, 1)
                covered_mask |= (temp_mask & room_mask)

        covered_pixels = int(covered_mask.sum())
        fraction = round(covered_pixels / total_room_pixels, 2)
        coverage_per_frame.append((frame_idx, fraction))

        if fraction >= 1.0 and time_to_full is None and first_non_enemy_frame is not None:
            time_to_full = round((frame_idx - first_non_enemy_frame) / frame_rate, 2)

    final_fraction = coverage_per_frame[-1][1] if coverage_per_frame else 0.0

    return {
        "coverage_per_frame": coverage_per_frame,
        "time_to_full": time_to_full,
        "final_fraction": final_fraction,
        "first_non_enemy_frame": first_non_enemy_frame,
    }


# ----------------------------------------------------------------------
# POD helpers
# ----------------------------------------------------------------------


def _first_valid_index(traj: Iterable[tuple]) -> int:
    for i, pt in enumerate(traj):
        if pt is not None:
            return i
    raise ValueError("Trajectory contains no valid points")


def assign_pods_by_entry(
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]],
    pods: List[Tuple[float, float]],
    *,
    inroom_ids: Optional[List[int]] = None,
    boundary: Optional[Polygon] = None,
    pod_groups: Optional[List[str]] = None,
    frame_rate: float = 30.0,
    door_axes: "Optional[Sequence[DoorAxes]]" = None,
) -> Dict[int, Optional[int]]:
    """Assign each entrant to a POD using door-axis-derived entry direction.

    Direction estimator: least-squares 2D velocity over the entry window
    (shared with ``EntranceVectors_Metric`` — single source of truth).

    Side classification: projection of the entrant's displacement onto the
    door's two reference path axes ``(p̂_A, p̂_B)``; whichever projection is
    larger wins. Works uniformly for straight and corner doors.

    The function falls back gracefully when door axes are unavailable
    (e.g. legacy callers without entry polygons): in that case the previous
    boundary-projection behavior is used purely as a last-resort heuristic.
    """
    # Lazy imports to dodge the helper_functions ↔ src.metrics package cycle:
    # importing ``src.metrics.*`` at module-load time would re-enter this
    # module before its top-level definitions are bound.
    from src.metrics._shared import (
        classify_entry_side,
        door_for_entry,
        first_entry_frame,
        fit_entry_velocity,
        select_entry_tracks,
    )
    from src.metrics.entrance_vectors import ENTRY_VECTOR_WINDOW_SEC

    if boundary is None:
        raise ValueError("`boundary` Polygon is required.")

    if isinstance(pods, np.ndarray):
        pods = pods.tolist()

    inroom_ids = _normalize_inroom_ids(inroom_ids)
    if not tracks_by_id or not pods:
        return {}

    # The assigner caps by available POD count rather than team_size — fewer
    # PODs than entrants → only the first N entrants get assignments.
    items = select_entry_tracks(
        tracks_by_id,
        inroom_ids=inroom_ids,
        num_tracks=len(pods),
    )
    if not items:
        return {}

    doors_seq: list = list(door_axes) if door_axes else []

    first_tid, first_trk = items[0]
    start_idx = first_entry_frame(first_trk, doors_seq)
    if start_idx is None:
        start_idx = _first_valid_index(first_trk)
    entry_xy = first_trk[start_idx] if start_idx is not None else None
    if entry_xy is None:
        # Degenerate input — return empty assignment rather than asserting.
        return {tid: None for tid, _ in items}

    door = door_for_entry(entry_xy, doors_seq) if doors_seq else None

    # Reference point for POD-side geometry: door polygon centroid when we
    # have door axes; otherwise the boundary-projection point (legacy fallback).
    if door is not None:
        door_pt = Point(door.centroid)
    else:
        boundary_line = LineString(boundary.exterior.coords)
        door_pt = boundary_line.interpolate(boundary_line.project(Point(entry_xy)))

    # LSQ velocity → window-equivalent displacement vector.
    v = fit_entry_velocity(first_trk, start_idx, frame_rate, ENTRY_VECTOR_WINDOW_SEC)
    if v is not None:
        m_vec = v * float(ENTRY_VECTOR_WINDOW_SEC)
    else:
        m_vec = None
    sign, _, _, _ = classify_entry_side(m_vec, door)

    def _valid_groups(pg: Optional[List[str]]) -> Optional[List[str]]:
        if pg is None or not isinstance(pg, list) or len(pg) != len(pods):
            return None
        out: List[str] = []
        for g in pg:
            gs = str(g).strip().upper()
            if gs not in ("A", "B"):
                return None
            out.append(gs)
        return out

    pg = _valid_groups(pod_groups)

    # ------------------------------------------------------------------
    # Path A — explicit pod_groups (A/B) supplied
    # ------------------------------------------------------------------
    if pg is not None:
        pods_A = [i for i, g in enumerate(pg) if g == "A"]
        pods_B = [i for i, g in enumerate(pg) if g == "B"]

        if not pods_A or not pods_B:
            ordered = pods_A + pods_B

            def _score(pod_idx: int) -> float:
                x, y = pods[pod_idx]
                return float(Point(x, y).distance(door_pt))

            ordered.sort(key=_score, reverse=True)

            assignment: Dict[int, Optional[int]] = {}
            for (tid, _), pod_idx in zip(items, ordered):
                assignment[tid] = pod_idx
            for tid, _ in items[len(ordered):]:
                assignment[tid] = None
            return assignment

        # Group alignment: which group's centroid-from-door direction better
        # matches the first entrant's movement direction.
        def _group_alignment(group_indices: List[int]) -> float:
            if m_vec is None:
                return 0.0
            mn = float(np.linalg.norm(m_vec))
            if mn <= 1e-9:
                return 0.0
            m_hat = np.asarray(m_vec, dtype=float) / mn
            best = -1e9
            ox, oy = float(door_pt.x), float(door_pt.y)
            for pi in group_indices:
                px, py = pods[pi]
                v = np.array([float(px) - ox, float(py) - oy], dtype=float)
                n = float(np.linalg.norm(v))
                if n <= 1e-6:
                    continue
                v /= n
                best = max(best, float(np.dot(v, m_hat)))
            return best

        score_A = _group_alignment(pods_A)
        score_B = _group_alignment(pods_B)
        start_group = "A" if score_A >= score_B else "B"

        def _pod_dist(pod_idx: int) -> float:
            x, y = pods[pod_idx]
            return float(Point(x, y).distance(door_pt))

        pods_A_sorted = sorted(pods_A, key=_pod_dist, reverse=True)
        pods_B_sorted = sorted(pods_B, key=_pod_dist, reverse=True)

        assignment: Dict[int, Optional[int]] = {}
        ia = ib = 0
        current = start_group

        for tid, _ in items:
            if current == "A":
                if ia < len(pods_A_sorted):
                    assignment[tid] = pods_A_sorted[ia]
                    ia += 1
                elif ib < len(pods_B_sorted):
                    assignment[tid] = pods_B_sorted[ib]
                    ib += 1
                else:
                    assignment[tid] = None
            else:
                if ib < len(pods_B_sorted):
                    assignment[tid] = pods_B_sorted[ib]
                    ib += 1
                elif ia < len(pods_A_sorted):
                    assignment[tid] = pods_A_sorted[ia]
                    ia += 1
                else:
                    assignment[tid] = None
            current = "B" if current == "A" else "A"

        return assignment

    # ------------------------------------------------------------------
    # Path B — no pod_groups: split PODs by which path axis they fall on,
    # then alternate starting from the first entrant's classified side.
    # ------------------------------------------------------------------
    if door is not None:
        # Project each POD onto (p̂_A, p̂_B) from the door centroid; whichever
        # axis it aligns with more becomes its side.
        p_a = np.asarray(door.p_a, dtype=float)
        p_b = np.asarray(door.p_b, dtype=float)
        ox, oy = float(door.centroid[0]), float(door.centroid[1])

        def _pod_side(idx: int) -> int:
            px, py = pods[idx]
            v = np.array([float(px) - ox, float(py) - oy], dtype=float)
            n = float(np.linalg.norm(v))
            if n <= 1e-9:
                return +1
            v_hat = v / n
            pa = float(np.dot(v_hat, p_a))
            pb = float(np.dot(v_hat, p_b))
            return +1 if pa >= pb else -1

        def _pod_dist(idx: int) -> float:
            px, py = pods[idx]
            return float(Point(px, py).distance(door_pt))

        pod_meta = []
        for idx in range(len(pods)):
            pod_meta.append({"idx": idx, "side": _pod_side(idx), "dist": _pod_dist(idx)})

        # Within each side, prefer the deeper PODs first (mirrors prior
        # "perim sort" intent in a coordinate-free way).
        pods_pos = sorted(
            (d for d in pod_meta if d["side"] == +1),
            key=lambda d: d["dist"],
            reverse=True,
        )
        pods_neg = sorted(
            (d for d in pod_meta if d["side"] == -1),
            key=lambda d: d["dist"],
            reverse=True,
        )

        # First entrant's classified side. UNKNOWN → default to +1 so we
        # still produce a deterministic assignment.
        movement_sign = sign if sign != 0 else +1
    else:
        # Legacy fallback — keep the old boundary-cross-product split when no
        # door axes are available. This branch is never hit in normal runs
        # because `processing_engine.py` now passes `door_axes` through.
        boundary_line = LineString(boundary.exterior.coords)
        centre_pt = Point(boundary.centroid.coords[0])
        if m_vec is not None:
            v_centre = (
                np.array([centre_pt.x, centre_pt.y], dtype=float)
                - np.array(entry_xy, dtype=float)
            )
            z_cross = float(np.cross(v_centre, np.asarray(m_vec, dtype=float)))
            movement_sign = -1 if z_cross < 0 else +1
        else:
            movement_sign = -1

        divider_v = (
            np.array([centre_pt.x, centre_pt.y], dtype=float)
            - np.array([door_pt.x, door_pt.y], dtype=float)
        )

        def _side(pt: Point) -> int:
            v = np.array([pt.x, pt.y], dtype=float) - np.array([door_pt.x, door_pt.y], dtype=float)
            return 1 if float(np.cross(divider_v, v)) >= 0 else -1

        door_s = boundary_line.project(door_pt)
        pod_meta = []
        for idx, pod in enumerate(pods):
            proj_pt = boundary_line.interpolate(boundary_line.project(Point(pod)))
            side = _side(proj_pt)
            pod_s = boundary_line.project(proj_pt)
            perim = ((pod_s - door_s) if side == -1 else (door_s - pod_s)) % boundary_line.length
            pod_meta.append({"idx": idx, "side": side, "perim": perim})

        pods_pos = sorted(
            (d for d in pod_meta if d["side"] == +1),
            key=lambda d: d["perim"],
            reverse=True,
        )
        pods_neg = sorted(
            (d for d in pod_meta if d["side"] == -1),
            key=lambda d: d["perim"],
            reverse=True,
        )

    assignment: Dict[int, Optional[int]] = {}
    idx_pos = idx_neg = 0
    current_side = movement_sign

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
        current_side *= -1

    return assignment


def compute_pod_working_areas(
    pods: List[Tuple[float, float]],
    *,
    boundary: Polygon,
    working_radius: float,
) -> Dict[int, Dict[str, Union[Tuple[float, float], float]]]:
    results: Dict[int, Dict[str, Union[Tuple[float, float], float]]] = {}
    min_x, min_y, max_x, max_y = boundary.bounds

    for i, (cx, cy) in enumerate(pods):
        cx_clamp = min(max(cx, min_x + working_radius), max_x - working_radius)
        cy_clamp = min(max(cy, min_y + working_radius), max_y - working_radius)
        center = Point(cx_clamp, cy_clamp)

        circle = center.buffer(working_radius)
        if not circle.within(boundary):
            step = working_radius * 0.05
            moved = False
            for _ in range(100):
                if circle.within(boundary):
                    break
                centroid = boundary.centroid
                vec_x = centroid.x - center.x
                vec_y = centroid.y - center.y
                norm = math.hypot(vec_x, vec_y)
                if norm == 0:
                    break
                vec_x /= norm
                vec_y /= norm
                center = Point(center.x + vec_x * step, center.y + vec_y * step)
                circle = center.buffer(working_radius)
                moved = True
            if moved:
                cx_clamp, cy_clamp = center.x, center.y

        results[i] = {
            "center": (cx_clamp, cy_clamp),
            "radius": working_radius,
            "shift": (cx_clamp - cx, cy_clamp - cy),
        }

    return results


def _bbox_to_map_polygon(bbox_px: Tuple[int, int, int, int], pixel_mapper) -> Polygon:
    x1, y1, x2, y2 = bbox_px
    p1 = _pm_xy(pixel_mapper, (x1, y1))
    p2 = _pm_xy(pixel_mapper, (x2, y1))
    p3 = _pm_xy(pixel_mapper, (x2, y2))
    p4 = _pm_xy(pixel_mapper, (x1, y2))
    if any(p is None for p in (p1, p2, p3, p4)):
        return Polygon()
    return Polygon([p1, p2, p3, p4])


def dynamic_pod_working_areas(
    tracker_output: List[Dict],
    *,
    assignment: Dict[int, int],
    initial_working_areas: Dict[int, Dict[str, Union[Tuple[float, float], float]]],
    pixel_mapper,
    boundary: Polygon,
    inroom_ids: List[int],
    working_radius: float,
    overlap_threshold_frac: float = 0.1,
) -> Dict[int, Dict[int, Polygon]]:
    inroom_ids = _normalize_inroom_ids(inroom_ids)

    pod_circles: Dict[int, Polygon] = {
        idx: Point(info["center"]).buffer(working_radius)
        for idx, info in initial_working_areas.items()
    }
    pod_circle_areas: Dict[int, float] = {idx: circle.area for idx, circle in pod_circles.items()}
    dynamic_map: Dict[int, Dict[int, Polygon]] = {}

    for frame_entry in tracker_output:
        fidx = frame_entry["frame"]

        inroom_polys: List[Polygon] = []
        for obj in frame_entry["objects"]:
            if obj["id"] not in inroom_ids:
                continue
            poly = _bbox_to_map_polygon(tuple(obj["bbox"]), pixel_mapper)
            if poly.is_empty:
                continue
            try:
                poly = poly.buffer(0)
            except Exception:
                pass
            inroom_polys.append(poly)

        blocked_by: Dict[int, List[Polygon]] = {idx: [] for idx in pod_circles}
        for ipoly in inroom_polys:
            fracs = []
            for idx, circle in pod_circles.items():
                if circle.intersects(ipoly):
                    clipped = safe_intersection(circle, ipoly)
                    raw_area = 0.0 if (clipped is None or clipped.is_empty) else float(clipped.area)
                    frac = raw_area / pod_circle_areas[idx]
                    fracs.append((idx, frac))
            if not fracs:
                continue

            max_frac = max(frac for _, frac in fracs)
            for idx, frac in fracs:
                if frac >= overlap_threshold_frac and (frac == max_frac or frac >= overlap_threshold_frac):
                    blocked_by[idx].append(ipoly)

        pod_polys_this_frame: Dict[int, Polygon] = {}
        for pod_idx, base_circle in pod_circles.items():
            adjusted = base_circle

            for ipoly in blocked_by.get(pod_idx, []):
                cx, cy = initial_working_areas[pod_idx]["center"]
                shift_up = Point(cx, cy + working_radius).buffer(working_radius)
                shift_down = Point(cx, cy - working_radius).buffer(working_radius)

                if shift_up.within(boundary) and not shift_up.intersects(ipoly):
                    adjusted = shift_up
                elif shift_down.within(boundary) and not shift_down.intersects(ipoly):
                    adjusted = shift_down

                tmp = safe_union(adjusted, ipoly)
                if tmp is None or tmp.is_empty:
                    tmp = adjusted

                tmp2 = safe_intersection(tmp, boundary)
                adjusted = tmp2 if tmp2 is not None and not tmp2.is_empty else tmp

            tmp2 = safe_intersection(adjusted, boundary)
            if tmp2 is not None and not tmp2.is_empty:
                adjusted = tmp2
            pod_polys_this_frame[pod_idx] = adjusted

        dynamic_map[fidx] = pod_polys_this_frame

    return dynamic_map


def compute_pod_capture_times(
    *,
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]],
    assignment: Dict[int, Optional[int]],
    dynamic_work_areas: Dict[int, Dict[int, Polygon]],
    frame_rate: float,
    capture_threshold_sec: float,
) -> Dict[int, Dict[str, Optional[float]]]:
    threshold_frames = max(1, int(round(capture_threshold_sec * frame_rate)))
    results: Dict[int, Dict[str, Optional[float]]] = {}

    first_frame_of_id: Dict[int, Optional[int]] = {}
    for tid, traj in tracks_by_id.items():
        try:
            idx = _first_valid_index(traj)
            first_frame_of_id[tid] = idx + 1
        except ValueError:
            first_frame_of_id[tid] = None

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

        capture_time_sec = (capture_frame - first_f) / frame_rate if capture_frame is not None and first_f is not None else None
        results[pod_idx] = {
            "assigned_id": tid,
            "capture_time_sec": capture_time_sec,
            "capture_frame": capture_frame,
        }

    return results


def run_pod_analysis(
    *,
    tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]],
    tracker_output: List[Dict],
    pods_cfg: List[Tuple[float, float]],
    pod_groups: Optional[List[str]] = None,
    pixel_mapper,
    boundary: Polygon,
    inroom_ids: List[int],
    working_radius: float,
    frame_rate: float,
    capture_threshold_sec: float,
    save_cache: bool = False,
    output_directory: str = "",
    video_basename: str = "",
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    door_axes: "Optional[Sequence[DoorAxes]]" = None,
) -> Tuple[
    Dict[int, Optional[int]],
    Dict[int, Dict[int, Polygon]],
    Dict[int, Dict[str, Optional[float]]],
]:
    # ``start_frame`` / ``end_frame`` are accepted for forward-compatibility
    # with the drill-window pipeline. Today the inner POD analysis still
    # consumes the full sequence — the consuming POD metrics window their
    # own outputs via ``MetricContext.drill_*`` so trim semantics are
    # preserved at the metric layer.
    del start_frame, end_frame  # currently unused; see comment above.

    inroom_ids = _normalize_inroom_ids(inroom_ids)

    assignment = assign_pods_by_entry(
        tracks_by_id,
        pods_cfg,
        inroom_ids=inroom_ids,
        boundary=boundary,
        pod_groups=pod_groups,
        frame_rate=frame_rate,
        door_axes=door_axes,
    )

    initial_work_areas = compute_pod_working_areas(
        pods_cfg,
        boundary=boundary,
        working_radius=working_radius,
    )

    dynamic_work_areas = dynamic_pod_working_areas(
        tracker_output,
        assignment=assignment,
        initial_working_areas=initial_work_areas,
        pixel_mapper=pixel_mapper,
        boundary=boundary,
        inroom_ids=inroom_ids,
        working_radius=working_radius,
    )

    pod_capture_data = compute_pod_capture_times(
        tracks_by_id=tracks_by_id,
        assignment=assignment,
        dynamic_work_areas=dynamic_work_areas,
        frame_rate=frame_rate,
        capture_threshold_sec=capture_threshold_sec,
    )

    if save_cache and output_directory and video_basename:
        save_pod_cache(pod_capture_data, output_directory, video_basename)

    return assignment, dynamic_work_areas, pod_capture_data