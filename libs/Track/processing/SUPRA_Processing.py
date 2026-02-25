"""
Refactored for efficiency/cleanliness WITHOUT changing intended functionality.

Key improvements (no behavioral change intended):
- Remove duplicate imports.
- Avoid shared-reference bug in matched_keypoints initialization.
- Avoid map_point unbound usage.
- Convert defaultdicts to normal dicts before JSON dump.
- Reduce unnecessary frame copies (only when needed for outputs/show).
- Avoid redrawing full permanent trajectory each frame: draw only the newest segment.
- Consolidate repeated VideoWriter init logic.
- Keep your centroid/inside-box matching + uniqueness semantics.
"""

import os
import json
import cv2
import numpy as np
from collections import defaultdict
from tqdm import trange


# ----------------------------
# Small utilities
# ----------------------------
def create_results_directory(base_name="results"):
    """Create a new results directory with an incremented name if one already exists."""
    dir_index = 0
    new_dir_name = base_name
    while os.path.exists(new_dir_name):
        dir_index += 1
        new_dir_name = f"{base_name}{dir_index}"
    os.makedirs(new_dir_name)
    return new_dir_name


def combine_frames(img1, img2, img3=None):
    """
    Combines two or three frames side-by-side and below.

    img1 | img2
         | img3 (optional, below img2)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if img3 is not None:
        h3, w3 = img3.shape[:2]
        vis = np.zeros((max(h1, h2 + h3), w1 + w2, 3), np.uint8)
        vis[:h1, :w1, :3] = img1
        vis[:h2, w1:w1 + w2, :3] = img2
        vis[h2:h2 + h3, w1:w1 + w3, :3] = img3
    else:
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1, :3] = img1
        vis[:h2, w1:w1 + w2, :3] = img2

    return vis


def _ensure_writer(writer, out_path, fps, frame_shape, fourcc="DIVX"):
    """Create a cv2.VideoWriter if it doesn't exist yet; otherwise return existing."""
    if writer is not None:
        return writer
    h, w = frame_shape[:2]
    return cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))


def _to_jsonable(obj):
    """
    Convert nested defaultdicts to normal dicts recursively so json.dump works.
    """
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


# ----------------------------
# Ultralytics extraction helpers
# ----------------------------
def _extract_boxes_xyxy_conf_cls(box_result):
    """
    Extract detections into Nx6 numpy array: [x1,y1,x2,y2,conf,cls].
    Keeps your existing behavior: filter by conf threshold in the caller.
    """
    # Ultralytics: box_result.boxes.data is (N,6) tensor: xyxy, conf, cls
    boxes = box_result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    data = boxes.data
    # Move once to CPU + numpy
    arr = data.detach().cpu().numpy()
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _extract_pose_keypoints(pose_result):
    """
    Extract pose keypoints into list of (K,3) numpy arrays.
    """
    kps = []
    if pose_result.keypoints is None:
        return kps
    data = pose_result.keypoints.data
    if data is None or len(data) == 0:
        return kps
    arr = data.detach().cpu().numpy()
    # arr shape: (M, K, 3)
    for kp_set in arr:
        kps.append(kp_set.astype(np.float32, copy=False))
    return kps


def _filter_bad_keypoints(kp_set):
    """
    Keep your original semantics:
    "Filter out keypoints that have (0,0) and low confidence (<0.5)" by zeroing them out.
    """
    # kp_set: (K,3)
    kp = kp_set.copy()
    bad = (kp[:, 0] == 0) & (kp[:, 1] == 0) & (kp[:, 2] < 0.5)
    kp[bad] = 0
    return kp


def _match_keypoints_to_boxes_centroid_inside_unique(dets, all_keypoints):
    """
    Matches keypoints to boxes using your logic:
    - centroid distance
    - accept only if ALL valid keypoints (conf>0) are inside the box
    - enforce uniqueness across boxes by kp tuple

    Returns:
      matched_keypoints: (N, K, 3) float32; zeros when no match
    """
    N = dets.shape[0]
    if N == 0:
        return np.zeros((0, 17, 3), dtype=np.float32)

    # Default: zeros for each detection (IMPORTANT: no shared reference bug)
    matched = np.zeros((N, 17, 3), dtype=np.float32)

    if all_keypoints is None or len(all_keypoints) == 0:
        return matched

    # Build centroids for each kp set using only valid keypoints conf>0
    kp_sets = []
    centroids = []
    for kp_set in all_keypoints:
        if kp_set is None or kp_set.size == 0:
            continue
        valid = kp_set[kp_set[:, 2] > 0]
        if valid.size == 0:
            continue
        cx = float(valid[:, 0].mean())
        cy = float(valid[:, 1].mean())
        kp_sets.append(kp_set)
        centroids.append((cx, cy))

    if not kp_sets:
        return matched

    centroids = np.array(centroids, dtype=np.float32)  # (M,2)

    # Box centroids (N,2)
    box_centroids = np.column_stack(
        ((dets[:, 0] + dets[:, 2]) * 0.5, (dets[:, 1] + dets[:, 3]) * 0.5)
    ).astype(np.float32)

    # Distances (N,M)
    dxy = box_centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt((dxy ** 2).sum(axis=2)).astype(np.float32)

    assigned_keypoints = set()

    for i in range(N):
        x1, y1, x2, y2 = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3]
        order = np.argsort(dist[i])

        best_match = None
        for j in order:
            kp_set = kp_sets[int(j)]
            kp_tuple = tuple(map(tuple, kp_set))
            if kp_tuple in assigned_keypoints:
                continue

            valid = kp_set[kp_set[:, 2] > 0]
            if valid.size == 0:
                continue

            inside = (
                (valid[:, 0] > x1) & (valid[:, 0] < x2) &
                (valid[:, 1] > y1) & (valid[:, 1] < y2)
            )
            if inside.all():
                best_match = kp_set
                assigned_keypoints.add(kp_tuple)
                break

        if best_match is not None:
            matched[i] = best_match

    return matched


# ----------------------------
# Your original functions (refactored)
# ----------------------------
def run_detection(
    input_video_path,
    model,
    output_video_path=None,
    output_json_path=None,
    show_video=False,
    conf_threshold=0.4,
    device="cpu",
):
    """
    Run a YOLO detection model on a video, render, and optionally save the results.
    (Kept same behavior: uses results[0].plot() for annotation.)
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    json_data = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # In Ultralytics you can reduce overhead with verbose=False
        results = model(frame, device=device, conf=conf_threshold, verbose=False)

        # Build JSON (same structure)
        frame_detections = []
        # Use tensor batch extraction when possible (but still same output)
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            for bb, c, s in zip(xyxy, cls, conf):
                frame_detections.append(
                    {
                        "frame": frame_index,
                        "class": int(c),
                        "confidence": float(s),
                        "bbox": [float(x) for x in bb],
                    }
                )

        json_data.append({"frame_index": frame_index, "detections": frame_detections})

        annotated_frame = results[0].plot()

        if out:
            out.write(annotated_frame)

        if show_video:
            cv2.imshow("YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    if out:
        out.release()
    if show_video:
        cv2.destroyAllWindows()

    if output_json_path:
        with open(output_json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)


def run_box_pose_detection(
    video_path,
    output_video_path=None,
    output_json_path=None,
    box_model=None,
    pose_model=None,
    box_conf_threshold=0.4,
    pose_conf_threshold=0.4,
    show_video=False,
    device="cuda",
):
    """
    Run bounding box and pose detection on a video, rendering results and saving them to a video and JSON file.
    (Preserves your original behavior: pose is run on full frame; match by centroid; remove duplicates.)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    output_data = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Box detection
        box_results = box_model(frame, device=device, conf=box_conf_threshold, verbose=False)
        dets_list = []

        for result in box_results:
            arr = _extract_boxes_xyxy_conf_cls(result)  # (N,6)
            if arr.shape[0] == 0:
                continue
            # Filter conf (even though model conf already applied, keep your existing check)
            keep = arr[:, 4] >= box_conf_threshold
            arr = arr[keep]
            if arr.shape[0] == 0:
                continue
            # Ensure cls is int-like in last column
            arr[:, 5] = arr[:, 5].astype(np.int32)
            dets_list.append(arr)

        dets = np.concatenate(dets_list, axis=0) if dets_list else np.empty((0, 6), dtype=np.float32)

        # Pose detection
        pose_results = pose_model(frame, device=device, conf=pose_conf_threshold, verbose=False)
        all_keypoints = []
        for result in pose_results:
            all_keypoints.extend(_extract_pose_keypoints(result))
        all_keypoints = np.array(all_keypoints, dtype=np.float32) if all_keypoints else None

        # Match by centroid distance
        matched_keypoints = [None] * len(dets)
        if all_keypoints is not None and dets.size > 0:
            keypoint_centroids = []
            for keypoint_set in all_keypoints:
                centroid_x = float(np.mean(keypoint_set[:, 0]))
                centroid_y = float(np.mean(keypoint_set[:, 1]))
                keypoint_centroids.append((centroid_x, centroid_y, keypoint_set))

            for i, det in enumerate(dets):
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                box_centroid_x = (x1 + x2) / 2.0
                box_centroid_y = (y1 + y2) / 2.0

                best_distance = float("inf")
                best_match = None
                for cx, cy, kp_set in keypoint_centroids:
                    distance = float(np.sqrt((box_centroid_x - cx) ** 2 + (box_centroid_y - cy) ** 2))
                    if distance < best_distance:
                        best_distance = distance
                        best_match = kp_set

                if best_match is not None:
                    matched_keypoints[i] = best_match

            # Remove duplicate matches (keep first; your original kept closest but didn't store distances per kp_set.
            # Your code effectively removed duplicates by nulling later entries.)
            seen = {}
            for i, kp_set in enumerate(matched_keypoints):
                if kp_set is None:
                    continue
                kp_tuple = tuple(map(tuple, kp_set))
                if kp_tuple not in seen:
                    seen[kp_tuple] = i
                else:
                    matched_keypoints[i] = None

        # Draw + save JSON
        for i, box in enumerate(dets):
            x1, y1, x2, y2, confidence, cls = box
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

            if matched_keypoints[i] is not None:
                kp_set = matched_keypoints[i]
                adjusted_keypoints = []
                for kp in kp_set:
                    kp_x, kp_y, kp_conf = float(kp[0]), float(kp[1]), float(kp[2])
                    adjusted_keypoints.append({"x": kp_x, "y": kp_y, "confidence": kp_conf})
                    if kp_conf > pose_conf_threshold:
                        cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (0, 0, 255), -1)

                output_data.append(
                    {
                        "frame": frame_index,
                        "box": {"x1": x1i, "y1": y1i, "x2": x2i, "y2": y2i},
                        "keypoints": adjusted_keypoints,
                    }
                )

        if out:
            out.write(frame)

        if show_video:
            cv2.imshow("Processed Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    cap.release()
    if out:
        out.release()
    if show_video:
        cv2.destroyAllWindows()

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=4)


def process_video(
    video_path,
    box_model,
    pose_model,
    tracker,
    map_path=None,
    save_videos=True,
    save_keypoints=True,
    save_trajectories=True,
    show_video=False,
    box_conf_threshold=0.4,
    pose_conf_threshold=0.4,
    pose_sample_rate=1,
    keypoint_indices=None,
    device="cpu",
):
    """
    Process a video using YOLO models for bounding box and pose detection, and track objects with a tracker.

    Preserves behavior:
    - Box detection every frame.
    - Pose detection every pose_sample_rate frames (same as your original).
    - Keypoint filtering (0,0,<0.5 -> zero).
    - Matching: centroid distance + all valid keypoints inside box + uniqueness.
    - Tracking update call signature is unchanged.
    - Same output filenames and JSON structure intent.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    video_output_path = os.path.join(results_dir, "All-Options.mp4") if save_videos else None
    map_video_output_path = os.path.join(results_dir, "Mapped_Positions.mp4") if save_videos and map_path else None
    mapped_trajectories_output_path = os.path.join(results_dir, "Mapped_Trajectories.mp4") if save_videos and map_path else None
    bbox_video_output_path = os.path.join(results_dir, "Bounding-Boxes.mp4") if save_videos else None
    tracking_info_output_path = os.path.join(results_dir, "Tracking_and_Keypoints_and_Map.json") if save_keypoints else None
    mapped_trajectories_data_path = os.path.join(results_dir, "Mapped_Trajectories.json") if save_trajectories else None
    bbox_detections_video_output_path = os.path.join(results_dir, "Bounding-Box-Detections.mp4") if save_videos else None

    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise ValueError("Could not open input video.")

    frame_total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = int(vs.get(cv2.CAP_PROP_FPS)) or 30

    map_img = cv2.imread(map_path) if map_path else None
    permanent_trajectory_vis = map_img.copy() if map_img is not None else None

    # Writers (lazy init)
    all_options_vwriter = None
    map_positions_vwriter = None
    mapped_trajectories_vwriter = None
    bbox_vwriter = None
    detection_vwriter = None

    # Data
    complete_tracking_info = defaultdict(lambda: defaultdict(dict))
    complete_map_track_points = defaultdict(list)

    # Pose cache: if pose_sample_rate > 1, frames in between have no new pose;
    # your original sets matched_keypoints to zeros when no pose run.
    # We'll keep that behavior (no reuse).
    for frame_num in trange(1, frame_total + 1, unit="frame"):
        ret, frame = vs.read()
        if not ret or frame is None:
            break

        # Only allocate copies we actually need
        need_vis = True  # we always draw on vis (for show and/or all-options video)
        vis = frame.copy() if need_vis else frame

        bbox_vis = frame.copy() if save_videos else None
        detection_vis = frame.copy() if save_videos else None

        # Map visuals only if map exists and (show/save)
        need_map_outputs = map_img is not None and (save_videos or show_video)
        map_vis = map_img.copy() if need_map_outputs else None
        trajectory_vis = permanent_trajectory_vis.copy() if need_map_outputs and permanent_trajectory_vis is not None else None

        # 1) Box detection
        box_results = box_model(frame, device=device, conf=box_conf_threshold, verbose=False)

        dets_list = []
        for result in box_results:
            arr = _extract_boxes_xyxy_conf_cls(result)  # (N,6)
            if arr.shape[0] == 0:
                continue
            keep = arr[:, 4] >= box_conf_threshold
            arr = arr[keep]
            if arr.shape[0] == 0:
                continue
            arr[:, 5] = arr[:, 5].astype(np.int32)
            dets_list.append(arr)

        dets = np.concatenate(dets_list, axis=0) if dets_list else np.empty((0, 6), dtype=np.float32)

        # 2) Pose detection on sampled frames
        all_keypoints = None
        if pose_sample_rate > 0 and (frame_num % pose_sample_rate == 0):
            pose_results = pose_model(frame, device=device, conf=pose_conf_threshold, verbose=False)
            kps = []
            for result in pose_results:
                for kp_set in _extract_pose_keypoints(result):
                    kps.append(_filter_bad_keypoints(kp_set))
            all_keypoints = kps if kps else None

        # 3) Match keypoints to boxes
        matched_keypoints = _match_keypoints_to_boxes_centroid_inside_unique(dets, all_keypoints)

        # 4) Update tracker
        trackers = tracker.update(
            dets,
            frame,
            keypoints=matched_keypoints,
            keypoint_confidence_threshold=pose_conf_threshold,
            keypoint_indices=keypoint_indices,
        )

        # 5) Draw + collect
        for trk in trackers:
            trk_id = trk.get("track_id", None)
            if trk_id is None:
                continue

            # Always reset per-track to avoid unbound usage
            map_point = None

            box = (
                trk["top_left_x"],
                trk["top_left_y"],
                trk["top_left_x"] + trk["width"],
                trk["top_left_y"] + trk["height"],
            )

            vis, color = tracker.plot_box_on_img(vis, box, trk["confidence"], trk["class"], trk_id, conf_class=False)

            if bbox_vis is not None:
                bbox_vis, _ = tracker.plot_box_on_img(bbox_vis, box, trk["confidence"], trk["class"], trk_id, conf_class=False)

            if detection_vis is not None:
                detection_vis = cv2.rectangle(
                    detection_vis,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0),
                    thickness=2,
                )

            # Map tracking
            if tracker.pixel_mapper is not None and trk.get("current_map_pos", None) is not None:
                map_point = tuple(int(i) for i in np.squeeze(trk["current_map_pos"]))

                # append only if different from last point
                pts = complete_map_track_points[trk_id]
                if not pts or map_point != pts[-1]:
                    pts.append(map_point)

                    # draw only the newest segment on permanent trajectory (huge speed win, same result)
                    if permanent_trajectory_vis is not None and len(pts) >= 2:
                        cv2.line(
                            permanent_trajectory_vis,
                            pts[-2],
                            pts[-1],
                            tracker.id_to_color(trk_id),
                            thickness=2,
                        )

                if map_vis is not None:
                    cv2.circle(map_vis, map_point, 10, tracker.id_to_color(trk_id), -1)
                if trajectory_vis is not None:
                    cv2.circle(trajectory_vis, map_point, 10, tracker.id_to_color(trk_id), -1)

            # Keypoints drawing + JSON
            trk_kps = trk.get("keypoints", None)
            if trk_kps is not None and np.any(trk_kps[:, 2] > 0):
                keypoints_info = {}
                for i, (x, y, conf) in enumerate(trk_kps):
                    if conf >= pose_conf_threshold:
                        cv2.circle(vis, (int(x), int(y)), 4, color, thickness=-1)
                        keypoints_info[f"kp_{i}"] = {"x": int(x), "y": int(y), "conf": float(conf)}

                complete_tracking_info[frame_num][trk_id] = {
                    "box": {"x1": trk["top_left_x"], "y1": trk["top_left_y"], "x2": box[2], "y2": box[3]},
                    "keypoints": keypoints_info,
                    "map_position": map_point,
                }
            else:
                complete_tracking_info[frame_num][trk_id] = {
                    "box": {"x1": trk["top_left_x"], "y1": trk["top_left_y"], "x2": box[2], "y2": box[3]},
                    "keypoints": None,
                    "map_position": map_point,
                }

        # Show
        if show_video:
            if map_vis is not None and trajectory_vis is not None:
                combined_vis = combine_frames(vis, map_vis, trajectory_vis)
                cv2.imshow("Processed Frame, Map, and Trajectories", combined_vis)
            elif map_vis is not None:
                combined_vis = combine_frames(vis, map_vis)
                cv2.imshow("Processed Frame and Map", combined_vis)
            else:
                cv2.imshow("Processed Frame", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Save videos
        if save_videos:
            all_options_vwriter = _ensure_writer(all_options_vwriter, video_output_path, fps, vis.shape, fourcc="DIVX")
            all_options_vwriter.write(vis)

            if bbox_vis is not None:
                bbox_vwriter = _ensure_writer(bbox_vwriter, bbox_video_output_path, fps, bbox_vis.shape, fourcc="DIVX")
                bbox_vwriter.write(bbox_vis)

            if detection_vis is not None:
                detection_vwriter = _ensure_writer(
                    detection_vwriter, bbox_detections_video_output_path, fps, detection_vis.shape, fourcc="DIVX"
                )
                detection_vwriter.write(detection_vis)

            if map_vis is not None and map_video_output_path:
                map_positions_vwriter = _ensure_writer(map_positions_vwriter, map_video_output_path, fps, map_vis.shape, fourcc="DIVX")
                map_positions_vwriter.write(map_vis)

            if trajectory_vis is not None and mapped_trajectories_output_path:
                mapped_trajectories_vwriter = _ensure_writer(
                    mapped_trajectories_vwriter, mapped_trajectories_output_path, fps, trajectory_vis.shape, fourcc="DIVX"
                )
                mapped_trajectories_vwriter.write(trajectory_vis)

    # Cleanup
    vs.release()
    if all_options_vwriter:
        all_options_vwriter.release()
    if map_positions_vwriter:
        map_positions_vwriter.release()
    if mapped_trajectories_vwriter:
        mapped_trajectories_vwriter.release()
    if bbox_vwriter:
        bbox_vwriter.release()
    if detection_vwriter:
        detection_vwriter.release()
    if show_video:
        cv2.destroyAllWindows()

    # JSON outputs
    if save_keypoints and tracking_info_output_path:
        with open(tracking_info_output_path, "w") as f:
            json.dump(_to_jsonable(complete_tracking_info), f, indent=4)

    if save_trajectories and mapped_trajectories_data_path:
        with open(mapped_trajectories_data_path, "w") as f:
            json.dump({str(k): v for k, v in complete_map_track_points.items()}, f, indent=4)