######################################################Change Begin#####################################################
#Removed some unrequired older modules traceback, from mf_sort import MF_SORT, from .mapping import TrackMapper, from .detection.detector import Detector
from functools import cmp_to_key
import json
import cv2
import os
from tqdm import trange, tqdm
import logging
import numpy as np

from src.metrics.utils import len_comparator

# Unified metrics data context
from .metrics.context import MetricContext

from .utils.video import get_video_framerate, count_video_frames
from .utils.config import load_vmeta, load_config
from .utils.transcode import transcode
from .metrics import *
from .utils.vmeta import generate_vmeta
from shapely.geometry import Point
from shapely.ops import nearest_points
from src.helper_functions import (
    initialize_keypoint_indices,
    compute_gaze_vector,
    detect_enemy_fall,
    annotate_camera_video,
    annotate_camera_with_gaze_triangle,
    annotate_fall_video,
    annotate_clearance_video,
    annotate_camera_tracking_with_clearance,
    annotate_map_video,
    annotate_map_pod_video,
    annotate_map_pod_with_paths_video,
    annotate_map_with_gaze,
    save_position_cache,
    save_gaze_cache,
    save_enemy_fall_cache,
    compute_threat_clearance,
    save_threat_clearance_cache,
    compute_room_coverage,
    save_room_coverage_cache,
    # --- POD helpers ---
    run_pod_analysis,
    save_pod_cache,  # keep for backward compatibility if needed elsewhere
    save_metrics_cache
)

# Import new modules
from pathlib import Path
from mmpose.apis import MMPoseInferencer
from libs.Track.processing.utils import load_pixel_mapper
from libs.Track.processing.utils import load_entry_polygons
from libs.Track import OCSORT

# Suppress torchvision and model loading warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress verbose logging from mmengine, mmdet, and the tracker
logging.getLogger("mmengine").setLevel(logging.ERROR)
logging.getLogger("mmdet").setLevel(logging.ERROR)
logging.getLogger("libs.Track.trackers.basetracker").setLevel(logging.ERROR)
######################################################Change End#####################################################

class ProcessingEngine:
    def __init__(self, force_transcode=False, config=None):
        self.force_transcode = force_transcode
        self.metrics = []
        self.processed_vmetas = set()
        self.playback_time = 0
        self.metric_names = set()
        self.output_directory = ''
        self.video_basename = ''
        self.writeXML = True
        logging.debug("Ready to receive requests.")

        ######################################################Change Begin#####################################################
        # Initialize configuration if provided while calling Processing Engine
        if config:
            self.config = config
            self._initialize_components()
        else:
            self.config = None  # Will be set later in __load_config
        ######################################################Change End#####################################################

    ######################################################Change Begin#####################################################
    # Function to initialize components necessary for new processing.
    def _initialize_components(self, vmeta_path):
        # Load the PixelMapper instance
        point_mapping_path = os.path.join(os.path.dirname(vmeta_path), self.config["point_mapping_path"])
        self.mapper = load_pixel_mapper(point_mapping_path)

        # Load the entry polygons
        entry_polys_path = os.path.join(os.path.dirname(vmeta_path), self.config["entry_polys_path"])
        self.entry_polys = load_entry_polygons(entry_polys_path)

        #Load the Boundary of the room
        self.boundary = self.config.get("Boundary", None)
        
        # Initialize configured keypoint indices (NOSE/eyes/ears)
        initialize_keypoint_indices(self.config)

        # Initialize the MMPose inferencer
        self.inferencer = MMPoseInferencer(
            pose2d=self.config["pose2d_config"],
            pose2d_weights=self.config["pose2d_weights"],
            device=self.config.get("device", "cpu"),
            det_model=self.config["det_model"],
            det_weights=self.config["det_weights"],
            det_cat_ids=self.config.get("det_cat_ids", (0,)),
        )

        # Initialize the tracker
        self.tracker = OCSORT(
            per_class=False,
            det_thresh=0.3,
            max_age=1000,
            min_hits=3,
            asso_threshold=0.6,
            delta_t=5,
            asso_func="diou", 
            inertia=0.2,
            use_byte=True,
            pixel_mapper=self.mapper,
            limit_entry=True,
            entry_polys=self.entry_polys,
            entry_window_time=float('inf'),
            boundary=self.boundary,
            boundary_pad_pct=self.config.get("boundary_pad_pct", 0.05),
            track_enemy=self.config.get("track_enemy", True)
        )

        # Prepare additional parameters
        self.box_conf_threshold = self.config.get("box_conf_threshold", 0.5)
        self.pose_conf_threshold = self.config.get("pose_conf_threshold", 0.5)
        self.device = self.config.get("device", 'cpu')  # Device to run models on
        self.map_img = self.config.get("Map Image", None)
        self.keypoint_indices = self.config.get("keypoint_indices", None)  # Indices of keypoints to use
    ######################################################Change End#####################################################

    def mt_initialize(self, messages, vmeta_paths, output_path):
        logging.debug(f"Received initialization message with {len(vmeta_paths)} vmeta files.")
        #self.num_messages = len(messages)

        try:
            for vmeta_path in vmeta_paths:
                if vmeta_path not in self.processed_vmetas:
                    logging.debug(f"Starting processing of {vmeta_path}...")
                    self.processed_vmetas.add(vmeta_path)
                    config = self.__load_config(vmeta_path, output_path)

                    ######################################################Change Begin#####################################################
                    self.config = config  # Update the config
                    self._initialize_components(vmeta_path)  # Initialize components with the new config
                    self.metrics += self.__assess()
                    ######################################################Change End#####################################################

                    # self.__parse_messages(messages)
                else:
                    logging.debug(f"Vmeta file has previously been processed. Skipping: {vmeta_path}")

            logging.debug("Finished Processing. Ready for metric queries.")
            return 'ready'
        except Exception as e:
            logging.error("Unable to process the received vmeta files. Printing stack trace...")
            logging.error(e, exc_info=True)
            return 'error'

    def __parse_messages(self, messages):
        all_keys = set()
        fire_messages = []
        entity_state_messages = []

        for msg in messages:
            msg_dict = json.loads(msg)
            all_keys.update(msg_dict.keys())

            if "triggerSqueeze" in msg_dict:
                fire_messages.append(msg_dict)
            elif "appearance" in msg_dict:
                entity_state_messages.append(msg_dict)

        self.msg_keys = all_keys

    def __load_config(self, vmeta_path, output_path):
        config_path, video_path, start_time = load_vmeta(vmeta_path)
        config = load_config(config_path)
        config["start_time"] = start_time
        config["video_path"] = video_path
        config["output_path"] = output_path
        if self.force_transcode:
            video_path = transcode(video_path)
        fps = get_video_framerate(video_path)
        if fps is None:
            fps = 30
        config["frame_rate"] = fps
        config["frame_time"] = int((1/fps) * 1000)
        return config


    ######################################################Change Begin#####################################################
    def _is_in_entry_region(self, point):
        for entry_poly in self.entry_polys:  # assuming entry_polys is a list of polygons
            if entry_poly.contains(point):
                return True
        return False
    
    def __assess(self):
        config = self.config  # Use the initialized config

        # --------------------------------------------------------------
        # Visual angle config: store ONE value in config (full angle).
        # Some downstream functions expect a half-angle; compute it once.
        # --------------------------------------------------------------
        visual_angle_deg = float(config.get("visual_angle_degrees", 20.0))
        half_visual_angle_deg = visual_angle_deg / 2.0

        # --------------------------------------------------------------
        # Threat-clearance timing: use ONE config knob (seconds).
        # --------------------------------------------------------------
        min_threat_interaction_time_sec = float(config.get("min_threat_interaction_time_sec", 1.0))
        threat_interaction_frames = int(min_threat_interaction_time_sec * config.get("frame_rate", 30.0))

        # ------------------------------------------------------------------
        # Enemy IDs: allow multiple enemies as a list, falling back gracefully
        # ------------------------------------------------------------------
        enemy_ids = config.get("enemy_ids", [getattr(self.tracker, "ENEMY_FINAL_ID", 99)])

        # Initialize metrics
        metrics: list[AbstractMetric] = [
            IdentifyAndCapturePods_Metric(config),
            CapturePodTime_Metric(config),
            MoveAlongWall_Metric(config),
            EntranceVectors_Metric(config),
            EntranceHesitation_Metric(config),
            ThreatClearance_Metric(config),
            TeammateCoverage_Metric(config),
            ThreatCoverage_Metric(config),
            RoomCoverage_Metric(config),
            TotalRoomCoverageTime_Metric(config),
            #TotalEntryTime_Metric(config),
            #POD_Metric(config),
        ]

        vs = cv2.VideoCapture(config["video_path"])
        frame_total = count_video_frames(vs)

        # Prepare paths and data collectors for post‑hoc rendering
        self.output_directory = os.path.abspath(config["output_path"])
        self.video_basename = os.path.splitext(os.path.basename(config["video_path"]))[0]

        # Save empty map image once
        cv2.imwrite(os.path.join(self.output_directory, "EmptyMap.jpg"), config["Map Image"])
        logging.debug("Saved map cache to: {}".format(os.path.join(self.output_directory, "EmptyMap.jpg")))

        # Collectors
        raw_frames: list[np.ndarray] = []            # original frames for camera video
        tracker_output: list[dict] = []              # per‑frame serialised tracking info
        all_map_points: list[list] = []              # [frame, id, mapX, mapY]
        gaze_info: dict[tuple, tuple] = {}           # {(frame, id): (ox, oy, dx, dy)}
        all_frames: list[list] = []  # collect map_points per frame for batch metrics

        # Begin Processing
        for frame_num in tqdm(range(1, frame_total + 1), desc="Processing frames", unit="frame"):
            ret, frame = vs.read()
            if not ret or frame is None:
                break

            raw_frames.append(frame.copy())

            # Step 1-3: Run MMPose inferencer to get detections and keypoints
            infer_results = self.inferencer(
                frame,
                return_vis=False,
                bbox_thr=self.box_conf_threshold,
                kpt_thr=self.pose_conf_threshold,
                pose_based_nms=True
            )
            res = next(infer_results)
            instances = res['predictions'][0]

            dets = []
            matched_keypoints = []
            for inst in instances:
                # Unpack the returned bbox (list of lists)
                bbox = inst['bbox'][0]
                x1, y1, x2, y2 = bbox
                # Skip fallback full-frame bboxes (indicate no detection)
                frame_h, frame_w = frame.shape[:2]
                if x1 <= 0 and y1 <= 0 and x2 >= frame_w and y2 >= frame_h:
                    continue

                bbox_score = inst.get('bbox_score', 0.0)
                dets.append([x1, y1, x2, y2, bbox_score, 0])

                # Build keypoints array: shape (26,3)
                kps = np.array(inst['keypoints'])
                kp_scores = inst.get('keypoint_scores', None)
                if kp_scores is not None:
                    scores = np.array(kp_scores)
                else:
                    scores = np.ones((kps.shape[0],))
                kps3 = np.concatenate([kps, scores[:, None]], axis=1)
                matched_keypoints.append(kps3)

            dets = np.array(dets)
            if dets.size == 0:
                dets = np.empty((0, 6))
            if matched_keypoints:
                matched_keypoints = np.array(matched_keypoints)
            else:
                # No detections => empty array with shape (0, 26, 3)
                matched_keypoints = np.empty((0, 26, 3))

            # Step 4: Update tracker with detections and matched keypoints
            trackers = self.tracker.update(
                dets, frame,
                keypoints=matched_keypoints,
                keypoint_confidence_threshold=self.pose_conf_threshold,
                keypoint_indices=self.keypoint_indices
            )
            # Collect structured tracker output for this frame
            frame_objects = []
            for trk in trackers:
                trk_id = trk.get('track_id')
                x1 = int(trk.get('top_left_x', 0))
                y1 = int(trk.get('top_left_y', 0))
                w = int(trk.get('width', 0))
                h = int(trk.get('height', 0))
                bbox = [x1, y1, x1 + w, y1 + h]
                kps = trk.get('keypoints', None)
                if kps is not None:
                    keypoints = kps[:, :2].tolist()
                    keypoint_scores = kps[:, 2].tolist()
                else:
                    keypoints, keypoint_scores = [], []

                # Store gaze vector for this frame/id
                if kps is not None and kps.shape[0] == 26:
                    gvec = compute_gaze_vector(kps)
                    if gvec is not None:
                        origin, direction = gvec
                        gaze_info[(frame_num, trk_id)] = (
                            float(origin[0]), float(origin[1]),
                            float(direction[0]), float(direction[1])
                        )

                frame_objects.append({
                    "id": trk_id,
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "keypoint_scores": keypoint_scores
                })
            tracker_output.append({
                "frame": frame_num,
                "objects": frame_objects
            })

            # Step 5: Map tracked objects to map coordinates and update metrics
            map_points = []
            for trk in trackers:
                # Assuming 'current_map_pos' is provided by the tracker after mapping
                trk_id = trk.get('track_id')
                current_map_pos = trk.get('current_map_pos')
                if current_map_pos is not None:
                    mapX, mapY = current_map_pos
                    # Make sure that the point in map is inside the boundary polygon
                    if self.boundary is not None:
                        point = Point(mapX, mapY)
                        if (self.boundary.contains(point) or self._is_in_entry_region(point)):
                            # The point is inside the boundary, use it as is.
                            map_points.append([trk_id, mapX, mapY])
                        else:
                            # The point is outside, project it to the nearest point on the boundary.
                            nearest_point = nearest_points(self.boundary, point)[0]
                            map_points.append([trk_id, nearest_point.x, nearest_point.y])
            all_frames.append(map_points)
            # Save map tracking output
            for idx, mx, my in map_points:
                all_map_points.append([frame_num, idx, mx, my])

        # ---- Post‑processing: caches and videos ----
        fall_frames = detect_enemy_fall(tracker_output, enemy_ids=enemy_ids)

        save_position_cache(all_map_points, self.output_directory, self.video_basename)
        save_gaze_cache(gaze_info, self.output_directory, self.video_basename)
        save_enemy_fall_cache(fall_frames, self.output_directory, self.video_basename)

        # Disabled: not saving this artifact for now
#         annotate_camera_video(
#             raw_frames,
#             tracker_output,
#             config["frame_rate"],
#             self.output_directory,
#             self.video_basename,
#             enemy_ids=enemy_ids,
#             gaze_conf_threshold=self.pose_conf_threshold,
#         )
        annotate_camera_with_gaze_triangle(
            raw_frames,
            tracker_output,
            gaze_info,
            config["frame_rate"],
            self.output_directory,
            self.video_basename,
            enemy_ids=enemy_ids,
            half_angle_deg=half_visual_angle_deg,
            alpha=0.2,
            show_enemy_gaze=False
        )
        # Disabled: not saving this artifact for now
#         annotate_map_video(
#             config["Map Image"],
#             all_map_points,
#             config["frame_rate"],
#             self.output_directory,
#             self.video_basename,
#             total_frames=len(raw_frames)
#         )
        # --- Annotate map with gaze if boundary is available ---
        if self.boundary is not None:
            annotate_map_with_gaze(
                map_image=config["Map Image"],
                pixel_mapper=self.mapper,
                gaze_info=gaze_info,
                room_boundary_coords=list(self.boundary.exterior.coords),
                frame_rate=config["frame_rate"],
                output_directory=self.output_directory,
                video_basename=self.video_basename,
                enemy_ids=enemy_ids,
                show_enemy_gaze=False,
                half_angle_deg=half_visual_angle_deg,
                alpha=0.1,
                total_frames=len(raw_frames),
                accumulated_clear=True
            )
            # Compute and save room coverage cache
            coverage_data = compute_room_coverage(
                map_image=config["Map Image"],
                pixel_mapper=self.mapper,
                gaze_info=gaze_info,
                room_boundary_coords=list(self.boundary.exterior.coords),
                frame_rate=config["frame_rate"],
                total_frames=len(raw_frames),
                enemy_ids=enemy_ids,
                half_angle_deg=half_visual_angle_deg
            )
            save_room_coverage_cache(coverage_data, self.output_directory, self.video_basename)
        # Disabled: not saving this artifact for now
#         annotate_fall_video(
#             raw_frames,
#             tracker_output,
#             fall_frames,
#             config["frame_rate"],
#             self.output_directory,
#             self.video_basename,
#             enemy_ids=enemy_ids
#         )
        

        # Persist full tracker output for debugging / downstream analysis
        tracker_json_path = os.path.join(self.output_directory, f"{self.video_basename}_TrackerOutput.json")
        with open(tracker_json_path, "w") as f:
            json.dump(tracker_output, f, indent=4)
        logging.debug(f"Saved TrackerOutput to: {tracker_json_path}")

        logging.debug("Saved PositionCache to: {}".format(
            os.path.join(self.output_directory, f"{self.video_basename}_PositionCache.txt")))

        # Batch process metrics using collected frames and tracks
        # Build tracks_by_id: map each track ID to its full trajectory (or None) per frame
        all_ids = set()
        for frame_pts in all_frames:
            for trk_id, _, _ in frame_pts:
                all_ids.add(trk_id)
        tracks_by_id: dict[int, list] = {tid: [] for tid in all_ids}
        for frame_pts in all_frames:
            frame_dict = {tid: (x, y) for tid, x, y in frame_pts}
            for tid in all_ids:
                tracks_by_id[tid].append(frame_dict.get(tid))

        # ---- POD assignment & capture analysis (wrapper) --------------------
        pods_cfg = config.get("POD", [])
        working_radius     = config.get("pod_working_radius", 40.0)
        capture_threshold  = config.get("pod_capture_threshold_sec", 0.1)

        assignment, dynamic_work_areas, pod_capture_data = run_pod_analysis(
            tracks_by_id=tracks_by_id,
            tracker_output=tracker_output,
            pods_cfg=pods_cfg,
            pixel_mapper=self.mapper,
            boundary=self.boundary,
            enemy_ids=enemy_ids,
            working_radius=working_radius,
            frame_rate=config["frame_rate"],
            capture_threshold_sec=capture_threshold,
            save_cache=True,
            output_directory=self.output_directory,
            video_basename=self.video_basename
        )

        # ---- Render POD area video ----
        # Disabled: not saving this artifact for now
#         annotate_map_pod_video(
#             config["Map Image"],
#             all_map_points=all_map_points,
#             assignment=assignment,
#             dynamic_work_areas=dynamic_work_areas,
#             pod_capture_data=pod_capture_data,
#             frame_rate=config["frame_rate"],
#             output_directory=self.output_directory,
#             video_basename=self.video_basename,
#             total_frames=len(raw_frames),
#             enemy_ids=enemy_ids
#         )

        # ---- Render combined POD areas + trajectory trails video ----
        annotate_map_pod_with_paths_video(
            config["Map Image"],
            all_map_points=all_map_points,
            assignment=assignment,
            dynamic_work_areas=dynamic_work_areas,
            pod_capture_data=pod_capture_data,
            frame_rate=config["frame_rate"],
            output_directory=self.output_directory,
            video_basename=self.video_basename,
            total_frames=len(raw_frames),
            enemy_ids=enemy_ids
        )

        # Build structured context containing all inferences
        bbox_details = {}
        keypoint_details = {}
        for entry in tracker_output:
            frame_idx = entry["frame"]
            for obj in entry["objects"]:
                tid = obj["id"]
                bbox_details[(frame_idx, tid)] = tuple(obj["bbox"])
                keypoint_details[(frame_idx, tid)] = (obj["keypoints"], obj["keypoint_scores"])

        context = MetricContext(
            raw_frames=raw_frames,
            tracker_output=tracker_output,
            all_frames=all_frames,
            tracks_by_id=tracks_by_id,
            gaze_info=gaze_info,
            bbox_details=bbox_details,
            keypoint_details=keypoint_details,
            fall_frames=fall_frames,
            map_points=all_map_points,
            room_coverage=coverage_data,
            pod_capture=pod_capture_data
        )

        # ---- Threat‑clearance pre‑compute + cache ----
        clearance_map = compute_threat_clearance(
            tracker_output,
            keypoint_details,
            gaze_info,
            enemy_ids=config.get("enemy_ids", [99]),
            visual_angle_deg=visual_angle_deg,
            intersection_frames=threat_interaction_frames,
            wrist_frames=threat_interaction_frames*0.1,  # Finer‑grained threshold for wrist intersection time if desired
            gaze_frames=threat_interaction_frames*0.5,    # Finer‑grained threshold for gaze intersection time if desired
        )
        # Disabled: not saving this artifact for now
#         annotate_clearance_video(
#             raw_frames,
#             tracker_output,
#             clearance_map,
#             config["frame_rate"],
#             self.output_directory,
#             self.video_basename,
#             enemy_ids=enemy_ids
#         )

        # ---- Render combined tracking (boxes + skeletons) with clearance callouts ----
        annotate_camera_tracking_with_clearance(
            raw_frames,
            tracker_output,
            clearance_map,
            config["frame_rate"],
            self.output_directory,
            self.video_basename,
            enemy_ids=enemy_ids,
            gaze_conf_threshold=self.pose_conf_threshold,
            show_clearing_id=True
        )
        save_threat_clearance_cache(clearance_map, self.output_directory, self.video_basename)
        # Attach to context so metrics can reuse it
        context.threat_clearance = clearance_map

        metric_scores = []
        for m in metrics:
            m.process(context)
            # Handle POD_Metric returning (score, pod_assignment)
            score = m.getFinalScore()

            assessment = "below"
            if isinstance(score, (int, float)):
                if score > 0.9:
                    assessment = "above"
                elif score > 0.5:
                    assessment = "at"
            # Base metric output
            metric_entry = {
                "metric_name": m.metricName,
                "score": score,
                "assessment": assessment,
                "timestamp": config["start_time"]
            }
            metric_scores.append(metric_entry)
        # Save metrics cache to CSV
        save_metrics_cache(metric_scores, self.output_directory, self.video_basename)
        return metric_scores
    ######################################################Change End#####################################################


    def get_assessment(self, timestamp, metric_name):
        timestamp = int(timestamp)
        if self.writeXML:
            generate_vmeta(self.output_directory, self.video_basename, "_Tracking")
            generate_vmeta(self.output_directory, self.video_basename, "_MapTracking")
            self.writeXML = False

        desired_metrics = []
        for metric in self.metrics:
            if metric["metric_name"] == metric_name and metric["timestamp"] < timestamp and metric["timestamp"] > self.playback_time:
                desired_metrics.append(metric)

        self.playback_time = max(timestamp, self.playback_time)  # Enforce forward time progression
        if len(desired_metrics) > 0:
            logging.debug(f"Returning {len(desired_metrics)} results for query {metric_name} at time {timestamp}")
        else:
            logging.debug(f"Metric query for time {timestamp} received but no metrics match.")
        return json.dumps(desired_metrics, indent=4)

    def compare_expert(self, metric_name, session_folder, expert_folder, vmeta_path=None):
        """Compare a trainee (session_folder) against an expert (expert_folder) for a metric.

        This function is folder-based. Each metric's `expertCompare` is responsible for
        selecting the appropriate cache/artifacts from the two folders.

        NEW: `vmeta_path` can be provided (single vmeta for both trainee/expert) to load
        the drill configuration once and pass it into each metric's expertCompare.

        Expected metric interface (preferred):
            __init__(config: dict)
            expertCompare(session_folder, expert_folder, map_image=None, config=None) -> dict

        Note:
          - `expertCompare` may be implemented as an instance method to use `self.config`.
          - It may also remain a @staticmethod; both are supported.

        Backward compatibility:
          - If a metric does not accept `config`, we call it without that kwarg.
          - If a metric does not accept kwargs, we fall back to positional calling.
        """
        logging.debug("")
        logging.debug("Starting Expert Comparison for metric {}.".format(metric_name))

        # Validate folders
        if not session_folder or not os.path.isdir(session_folder):
            logging.debug("Error: Supplied session folder is missing or not a directory: %s", session_folder)
            return json.dumps({
                "Name": metric_name,
                "Type": "SideBySide",
                "ImgLocation": os.path.join(session_folder or "", "error_image.jpg"),
                "Text": "There was an error while processing this comparison. The supplied session folder is invalid."
            }, indent=4)

        if not expert_folder or not os.path.isdir(expert_folder):
            logging.debug("Error: Supplied expert folder is missing or not a directory: %s", expert_folder)
            return json.dumps({
                "Name": metric_name,
                "Type": "SideBySide",
                "ImgLocation": os.path.join(session_folder, "error_image.jpg"),
                "Text": "There was an error while processing this comparison. The supplied expert folder is invalid."
            }, indent=4)

        # Load drill config once from a single vmeta (same for trainee + expert)
        config = None
        if vmeta_path:
            try:
                config_path, _, _ = load_vmeta(vmeta_path)
                config = load_config(config_path)
            except Exception:
                logging.error("Error: Unable to load config from vmeta_path: %s", vmeta_path, exc_info=True)
                config = None

        # Load a map image if available (prefer session folder, then expert folder)
        map_path_session = os.path.join(session_folder, "EmptyMap.jpg")
        map_path_expert = os.path.join(expert_folder, "EmptyMap.jpg")
        map_path = map_path_session if os.path.exists(map_path_session) else (map_path_expert if os.path.exists(map_path_expert) else None)
        map_image = cv2.imread(map_path) if map_path is not None else None

        def _call_metric(metric_cls):
            """Call a metric's expertCompare with config-aware initialization.

            Preferred behavior:
              - Instantiate the metric with `config` (loaded from vmeta) so metrics can
                access `self.config` inside an instance `expertCompare` implementation.

            Backward compatibility:
              - If instantiation fails, call expertCompare on the class.
              - If a metric does not accept certain kwargs (e.g., config), fall back.
              - If a metric uses positional signatures, fall back.
            """
            if not hasattr(metric_cls, "expertCompare"):
                raise AttributeError(f"{metric_cls} has no expertCompare")

            # Try to initialize metric with config so expertCompare can use self.config
            metric_obj = None
            try:
                metric_obj = metric_cls(config or {})
            except TypeError:
                # Some legacy metrics may not accept config in __init__
                metric_obj = None

            target = metric_obj if metric_obj is not None else metric_cls

            # Prefer kwargs; try with config first, then without
            try:
                return target.expertCompare(
                    session_folder=session_folder,
                    expert_folder=expert_folder,
                    map_image=map_image,
                    config=config,
                )
            except TypeError:
                try:
                    return target.expertCompare(
                        session_folder=session_folder,
                        expert_folder=expert_folder,
                        map_image=map_image,
                    )
                except TypeError:
                    # Fallback to positional signatures
                    try:
                        return target.expertCompare(session_folder, expert_folder, map_image, config)
                    except TypeError:
                        return target.expertCompare(session_folder, expert_folder, map_image)

        # Expected metric interface (preferred):
        #   __init__(config: dict)
        #   expertCompare(session_folder, expert_folder, map_image=None, config=None) -> dict
        # Note: expertCompare can be instance method (uses self.config) or @staticmethod.
        # Metric calculation (metrics will decide which files to read from each folder)
        try:
            # Map metric_name -> metric class
            metric_map = {
                # ---- Existing / legacy ----
                "ENTRANCE_HESITATION": EntranceHesitation_Metric,
                "ENTRANCE_VECTORS": EntranceVectors_Metric,
                "STAY_ALONG_WALL": MoveAlongWall_Metric,
                "IDENTIFY_AND_HOLD_DESIGNATED_AREA": POD_Metric,
                "TOTAL_TIME_OF_ENTRY": TotalEntryTime_Metric,
                "IDENTIFY_AND_CAPTURE_POD": IdentifyAndCapturePods_Metric,
                "POD_CAPTURE_TIME": CapturePodTime_Metric,
                "THREAT_CLEARANCE": ThreatClearance_Metric,
                "TEAMMATE_COVERAGE": TeammateCoverage_Metric,
                "THREAT_COVERAGE": ThreatCoverage_Metric,
                "FLOOR_COVERAGE": RoomCoverage_Metric,
                "TOTAL_FLOOR_COVERAGE_TIME": TotalRoomCoverageTime_Metric,
            }

            metric_cls = metric_map.get(metric_name)
            if metric_cls is None:
                logging.debug("Metric %s not implemented in compare_expert mapping.", metric_name)
                out = {
                    "Name": metric_name,
                    "Type": "SideBySide",
                    "ImgLocation": os.path.join(session_folder, "error_image.jpg"),
                    "Text": "There was an error while processing this comparison. Most likely, this metric is not yet implemented."
                }
            else:
                logging.debug("Started computing %s", metric_name)
                out = _call_metric(metric_cls)

        except Exception:
            logging.error("Error while running expert comparison for %s", metric_name, exc_info=True)
            out = {
                "Name": metric_name,
                "Type": "SideBySide",
                "ImgLocation": os.path.join(session_folder, "error_image.jpg"),
                "Text": "There was an error while processing this comparison. Most likely, this metric is not yet implemented."
            }

        logging.debug("Completed analysis of metric {}.".format(metric_name))
        return json.dumps(out, indent=4)

    def make_path(self, raw_path):
        paths = []
        max_len = int(np.max(raw_path[:, 0]))
        for trk_idx in np.unique(raw_path[:, 1]):
            idx_path = raw_path[raw_path[:, 1] == trk_idx]
            idx_path_corrected = []
            for frame_num in range(1, max_len + 1):
                raw_frame = np.squeeze(idx_path[idx_path[:, 0] == frame_num])
                if raw_frame is not None and len(raw_frame) > 0:
                    idx_path_corrected.append((raw_frame[2], raw_frame[3]))
                else:
                    idx_path_corrected.append(None)
            paths.append(idx_path_corrected)
        return paths
