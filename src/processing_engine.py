import json
import logging
import os
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple

# torchvision's image extension imports a host libjpeg.9.dylib that the conda
# stack does not ship — we never use ``torchvision.io``, so silence its
# import-time warning before the first transitive import of torchvision.
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension.*",
    category=UserWarning,
)

import cv2
import numpy as np
from libs.giftpose.adapters import to_canonical
from libs.giftpose.inferencer import MMPoseInferencer
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from tqdm import tqdm

from libs.Track import OCSORT
from libs.Track.processing.utils import load_entry_polygons, load_pixel_mapper
from src.helper_functions import (
    annotate_camera_videos_combined,
    annotate_map_pod_video,
    annotate_map_pod_with_paths_video,
    annotate_map_video,
    annotate_map_with_gaze,
    camera_tracking_overlay_spec,
    clearance_overlay_spec,
    compute_gaze_vector,
    compute_room_coverage,
    compute_threat_clearance,
    gaze_triangle_overlay_spec,
    initialize_keypoint_indices,
    run_pod_analysis,
    save_gaze_cache,
    save_metrics_cache,
    save_position_cache,
    save_room_coverage_cache,
    save_threat_clearance_cache,
    tracking_with_clearance_overlay_spec,
)

from .metrics import *
from .metrics._shared import load_door_axes
from .metrics.context import MetricContext
from .metrics.metric import AbstractMetric
from .analysis import build_analysis_session
from .utils.audio import attach_audio_in_place, has_audio_stream
from .utils.config import load_config, load_vmeta
from .utils.run_info import load_run_info, make_run_id, save_run_info
from .utils.run_metadata import load_run_metadata, save_run_metadata
from .utils.drill_window import (
    DrillWindow,
    compute_drill_window,
    find_drill_start_frame,
    save_drill_window_sidecar,
)
from .utils.torch_memory import free_torch_memory
from .utils.transcode import transcode
from .utils.transcription import save_transcription_sidecar, transcribe_video
from .utils.video import count_video_frames, get_video_framerate
from .utils.vmeta import generate_vmeta

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("libs.Track.trackers.basetracker").setLevel(logging.ERROR)


def _extras_key(kps: np.ndarray) -> tuple:
    """Stable per-detection key for re-attaching extended-keypoint metadata
    after the tracker pass. Built from the canonical block's first rows,
    rounded so float32/float64 round-trips through the tracker still match.
    """
    arr = np.asarray(kps, dtype=np.float64)
    return tuple(np.round(arr[:5, :2], 2).ravel().tolist())


VIS_VMETA_SUFFIX: Dict[str, str] = {
    "annotate_camera_video": "_Tracking_Overlays",
    "annotate_camera_with_gaze_triangle": "_Gaze_Triangles",
    "annotate_clearance_video": "_Clearance_Callouts",
    "annotate_camera_tracking_with_clearance": "_Tracking_WithClearance",
    "annotate_map_video": "_Tracking_Map",
    "annotate_map_pod_video": "_Tracking_PodAreas",
    "annotate_map_pod_with_paths_video": "_Tracking_PodAreasWithTrails",
    "annotate_map_with_gaze": "_Gaze_MapCleared",
}

DEFAULT_ENABLED_VISUALIZATIONS: List[str] = [
    # Visualizations produced on every run, regardless of which metrics are
    # active. Each one declares its compute dependencies in
    # ``VIS_COMPUTE_DEPS`` below — enabling a viz here automatically pulls
    # in the helpers it needs. Plain camera/map tracking is dependency-free;
    # the gaze pair only needs the per-frame gaze vectors that the pose
    # loop already produces, so it's free to keep on.
    "annotate_camera_video",
    "annotate_map_video",
    "annotate_camera_with_gaze_triangle",
    "annotate_map_with_gaze",
]

# Master toggle for which metrics run during ``ProcessingEngine.__assess()``.
# Disabled entries keep their class definitions intact (still imported via
# ``from .metrics import *``) so re-enabling is a one-line flip here. The
# downstream heavy-compute helpers (POD analysis, threat clearance, room
# coverage) are gated below against METRIC_COMPUTE_DEPS so flipping a metric
# back on automatically restores its inputs.
ENABLED_METRICS: Dict[str, bool] = {
    "EntranceVectors_Metric": True,
    "EntranceHesitation_Metric": True,
    "TotalEntryTime_Metric": True,
    "MoveAlongWall_Metric": True,
    "IdentifyAndCapturePods_Metric": False,
    "CapturePodTime_Metric": False,
    "ThreatClearance_Metric": False,
    "TeammateCoverage_Metric": False,
    "ThreatCoverage_Metric": False,
    "RoomCoverage_Metric": False,
    "TotalRoomCoverageTime_Metric": False,
}

# Compute families — each name is a feature flag that, when True, makes the
# corresponding helper run inside ``__assess``. The dependency maps below
# resolve to a set of these names per active metric / visualization.
#
#   "pod"              → run_pod_analysis (assignment, dynamic_work_areas,
#                        pod_capture_data; populates context.pod_capture)
#   "threat_clearance" → compute_threat_clearance (clearance_map; populates
#                        context.threat_clearance)
#   "room_coverage"    → compute_room_coverage (coverage_data; populates
#                        context.room_coverage)
#
# Future compute families: declare the constant here, give it a gating
# branch in ``__assess``, and add it to whichever metrics / visualizations
# need it.
METRIC_COMPUTE_DEPS: Dict[str, frozenset] = {
    "EntranceVectors_Metric": frozenset(),
    "EntranceHesitation_Metric": frozenset(),
    "TotalEntryTime_Metric": frozenset(),
    # MoveAlongWall reads pod_capture for per-track end-frames but treats it
    # as optional (empty dict → full scoring window). Per product spec wall
    # adherence is scored from frame 1 → drill_end_frame, so we deliberately
    # don't pull POD in for it.
    "MoveAlongWall_Metric": frozenset(),
    "IdentifyAndCapturePods_Metric": frozenset({"pod"}),
    "CapturePodTime_Metric": frozenset({"pod"}),
    "ThreatClearance_Metric": frozenset({"threat_clearance"}),
    "TeammateCoverage_Metric": frozenset({"room_coverage"}),
    "ThreatCoverage_Metric": frozenset({"room_coverage"}),
    "RoomCoverage_Metric": frozenset({"room_coverage"}),
    "TotalRoomCoverageTime_Metric": frozenset({"room_coverage"}),
}

VIS_COMPUTE_DEPS: Dict[str, frozenset] = {
    "annotate_camera_video": frozenset(),
    "annotate_camera_with_gaze_triangle": frozenset(),
    "annotate_map_video": frozenset(),
    "annotate_map_with_gaze": frozenset(),
    "annotate_clearance_video": frozenset({"threat_clearance"}),
    "annotate_camera_tracking_with_clearance": frozenset({"threat_clearance"}),
    "annotate_map_pod_video": frozenset({"pod"}),
    "annotate_map_pod_with_paths_video": frozenset({"pod"}),
}


# Human-readable labels for the comparison payload. Keyed by upper-snake metric
# id as understood by ``compare_expert``.
_COMPARE_METRIC_LABELS: Dict[str, str] = {
    "ENTRANCE_HESITATION": "Entrance Hesitation",
    "ENTRANCE_VECTORS": "Entrance Vectors",
    "STAY_ALONG_WALL": "Stay Along Wall",
    "IDENTIFY_AND_HOLD_DESIGNATED_AREA": "Hold Designated Area",
    "TOTAL_TIME_OF_ENTRY": "Total Time of Entry",
    "IDENTIFY_AND_CAPTURE_POD": "Identify & Capture PODs",
    "POD_CAPTURE_TIME": "POD Capture Time",
    "THREAT_CLEARANCE": "Threat Clearance",
    "TEAMMATE_COVERAGE": "Teammate Coverage",
    "THREAT_COVERAGE": "Threat Coverage",
    "FLOOR_COVERAGE": "Floor Coverage",
    "TOTAL_FLOOR_COVERAGE_TIME": "Total Floor Coverage Time",
}


# Default visualization captions per metric id. Two-image metrics use
# ("current", "other") tuple ordering; one-image metrics use a single string.
# Layman-friendly, no jargon. Each caption is written from the trainee's POV.
_COMPARE_CAPTIONS: Dict[str, Tuple[str, ...]] = {
    "ENTRANCE_VECTORS": (
        "Your team's entry directions. Each arrow shows where one teammate was facing as they crossed the door.",
        "The reference team's entry directions. Compare how the arrows fan out across sides.",
    ),
    "ENTRANCE_HESITATION": (
        "Pause time at the door per entrant — shorter bars are better. Your run vs the reference are overlaid for direct comparison.",
    ),
    "TOTAL_TIME_OF_ENTRY": (
        "Total elapsed time from the first to the last entrant. Shorter is generally better.",
    ),
    "STAY_ALONG_WALL": (
        "Your team's paths from door to first position. Lines staying close to the walls (the shaded safe band) are preferred.",
        "The reference team's paths along the wall.",
    ),
    "IDENTIFY_AND_CAPTURE_POD": (
        "Which Points-Of-Domination (PODs) your team identified and reached.",
        "The reference team's POD coverage.",
    ),
    "IDENTIFY_AND_HOLD_DESIGNATED_AREA": (
        "Designated areas your team held over the drill.",
        "The reference team's designated-area coverage.",
    ),
    "POD_CAPTURE_TIME": (
        "Time-to-capture for each POD. Shorter bars are better.",
    ),
    "THREAT_CLEARANCE": (
        "Where and when threats were cleared during the drill.",
    ),
    "THREAT_COVERAGE": (
        "Visual coverage of threats over time.",
    ),
    "TEAMMATE_COVERAGE": (
        "How well teammates covered each other's sectors.",
    ),
    "FLOOR_COVERAGE": (
        "Heatmap of where the room was visually swept. Brighter areas were attended to more.",
    ),
    "TOTAL_FLOOR_COVERAGE_TIME": (
        "How long the floor was under visual coverage. Longer / fuller is better.",
    ),
}


def _read_metrics_csv(run_dir: str) -> Dict[str, float]:
    """Return ``{metric_name: score}`` from a run folder's ``*_Metrics.csv``.

    Used to populate the comparison payload with the score the live engine
    actually computed, without re-running ``metric.getFinalScore`` against
    cached data.
    """
    import csv as _csv
    import glob as _glob

    matches = _glob.glob(os.path.join(run_dir, "*_Metrics.csv"))
    if not matches:
        return {}
    matches.sort(key=os.path.getmtime, reverse=True)
    out: Dict[str, float] = {}
    try:
        with open(matches[0], "r", encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                name = (row.get("metric_name") or "").strip()
                if not name:
                    continue
                try:
                    out[name] = float(row.get("score") or 0.0)
                except (TypeError, ValueError):
                    pass
    except OSError:
        return {}
    return out


def _harvest_visualizations(
    legacy: Dict[str, object],
    metric_id: str,
    source_dir: str,
    output_dir: str,
) -> List[Dict[str, object]]:
    """Move legacy comparison images out of the run folder into ``output_dir``.

    The legacy ``expertCompare`` return dict carries image paths under various
    keys (``ImgLocation``, ``ReferenceImageLocation``, ``TraineeImageLocation``).
    We move each into ``output_dir`` and pair it with a per-metric caption so
    the viewer never sees the verbose legacy ``Text`` / ``TxtLocation`` fields.
    """
    import shutil as _shutil

    if not isinstance(legacy, dict):
        return []

    captions = _COMPARE_CAPTIONS.get(metric_id, ())

    def _emit(src: Optional[str], default_label: str, caption_idx: int) -> Optional[Dict[str, object]]:
        if not isinstance(src, str) or not src:
            return None
        if not os.path.isfile(src):
            return None
        dest = os.path.join(output_dir, os.path.basename(src))
        try:
            if os.path.abspath(src) != os.path.abspath(dest):
                _shutil.move(src, dest)
        except OSError:
            try:
                _shutil.copyfile(src, dest)
            except OSError:
                return None
        caption = captions[caption_idx] if caption_idx < len(captions) else ""
        return {
            "image_path": os.path.abspath(dest),
            "label": default_label,
            "caption": caption,
        }

    out: List[Dict[str, object]] = []
    trainee_img = legacy.get("TraineeImageLocation") or legacy.get("ImgLocation")
    reference_img = legacy.get("ReferenceImageLocation")

    if trainee_img and reference_img:
        a = _emit(trainee_img, "Your run", 0)
        b = _emit(reference_img, "Reference", 1)
        if a is not None:
            out.append(a)
        if b is not None:
            out.append(b)
    elif trainee_img:
        a = _emit(trainee_img, "Comparison", 0)
        if a is not None:
            out.append(a)

    # Discard any legacy TXT dumps to keep the run folder clean and avoid
    # leaking the verbose tabular content into the new payload.
    txt = legacy.get("TxtLocation")
    if isinstance(txt, str) and os.path.isfile(txt):
        try:
            os.remove(txt)
        except OSError:
            pass

    return out


class ProcessingEngine:
    def __init__(self, force_transcode: bool = False, config=None):
        self.force_transcode = force_transcode
        self.metrics = []
        self.processed_vmetas = set()
        self.playback_time = 0
        self.metric_names = set()
        self.output_directory = ""
        self.video_basename = ""
        self.writeXML = True
        self.config = config if config is not None else None
        logging.debug("Ready to receive requests.")

    @staticmethod
    def _resolve_pose_selection(config: dict) -> tuple[str, str]:
        """Resolve (pose_tag, pose_weights_path) from the pose-backend keys.

        Config surface (all optional — omitting every key reproduces the
        legacy behavior exactly):
          - ``pose_backend``: "body2d" (default) | "wholebody" | "pose3d"
          - ``pose_model_size``: "t"/"s"/"m"/"l"/"x" (meaning depends on the
            backend; body2d "x" = the project's fine-tuned checkpoint)
          - ``auto_download_models``: allow fetching official checkpoints
          - ``pose2d_config`` / ``pose2d_weights``: explicit architecture tag
            and weights — always win when the new keys are absent, and serve
            as the weights override when they are present (e.g. a future
            fine-tuned RTMW).
        """
        from libs.giftpose.registry import resolve_pose
        from libs.giftpose.weights.download import resolve_weights

        backend = str(config.get("pose_backend") or "body2d").lower()
        size = config.get("pose_model_size")
        auto_dl = bool(config.get("auto_download_models", False))

        legacy_tag = config.get("pose2d_config", "rtmpose-x-halpe26-384x288")
        legacy_weights = config.get("pose2d_weights", "models/pose.pth")

        if backend == "body2d" and size in (None, "", "x"):
            # Default path — identical to the pre-backend-selection engine.
            return legacy_tag, legacy_weights

        if backend == "body2d":
            tag = f"rtmpose-{size}-halpe26-256x192"
        elif backend == "wholebody":
            tag = f"rtmw-{size or 'l'}-cocktail14-133"
        elif backend == "pose3d":
            tag = f"rtmw3d-{size or 'l'}-cocktail14-133"
        else:
            raise ValueError(
                f"Unknown pose_backend {backend!r}; expected one of "
                f"'body2d', 'wholebody', 'pose3d'."
            )

        spec = resolve_pose(tag)  # raises with the supported-tag list
        # ``pose_backend_weights`` (dedicated key) overrides the registry
        # download for non-default backends — this is how a future fine-tuned
        # checkpoint for e.g. RTMW gets plugged in. ``pose2d_weights`` is
        # deliberately NOT reused here: it names the body2d default weights
        # (and is path-absolutized by load_config), so borrowing it would
        # feed a 26-kp checkpoint to a 133-kp architecture.
        weights = resolve_weights(
            spec,
            configured_path=config.get("pose_backend_weights"),
            models_dir="models",
            auto_download=auto_dl,
        )
        return tag, weights

    def _initialize_components(self, vmeta_path: Optional[str] = None):
        if vmeta_path is None:
            return

        # ``load_config`` already resolved point_mapping_path to absolute via
        # the central path-anchor registry; no further joining needed.
        self.mapper = load_pixel_mapper(
            self.config["point_mapping_path"],
            ransac_reproj_threshold=float(self.config.get("homography_ransac_thresh", 3.0)),
            confidence=float(self.config.get("homography_confidence", 0.999)),
            max_iters=int(self.config.get("homography_max_iters", 2000)),
        )

        # entry_polys_path is also pre-resolved by load_config.
        self.entry_polys = load_entry_polygons(self.config["entry_polys_path"])

        self.boundary = self.config.get("Boundary", None)
        if self.boundary is not None and not hasattr(self.boundary, "contains"):
            try:
                self.boundary = Polygon(self.boundary)
            except Exception:
                self.boundary = None

        initialize_keypoint_indices(self.config)

        # Single meaningful detection floor: the detector's NMS score_thr is tied
        # to ``box_conf_threshold`` (default 0.3, matching the OCSORT tracker's
        # det_thresh / entry_conf_threshold). The detector won't emit or
        # pose-estimate boxes weaker than the tracker would use anyway.
        box_conf_threshold = self.config.get("box_conf_threshold", 0.3)
        pose_tag, pose_weights = self._resolve_pose_selection(self.config)
        self.pose_tag = pose_tag
        # Keypoint layout produced by the pose backend. Everything downstream
        # of the canonical adapter always sees Halpe-26; this only controls
        # whether the adapter has any work to do ("halpe26" -> identity).
        from libs.giftpose.registry import resolve_pose as _resolve_pose_spec
        self.pose_meta = _resolve_pose_spec(pose_tag).meta
        self.inferencer = MMPoseInferencer(
            pose2d=pose_tag,
            pose2d_weights=pose_weights,
            device=self.config.get("device", "cpu"),
            det_model=self.config["det_model"],
            det_weights=self.config["det_weights"],
            det_cat_ids=self.config.get("det_cat_ids", (0,)),
            flip_test=self.config.get("flip_test"),
            compile_for_inference=self.config.get("compile_for_inference"),
            det_score_thr=box_conf_threshold,
            prefer_backend=self.config.get("prefer_backend"),
        )

        self.tracker = OCSORT(
            per_class=False,
            det_thresh=0.3,
            max_age=1000,
            min_hits=1,
            asso_threshold=0.6,
            delta_t=5,
            asso_func="diou",
            inertia=0.2,
            use_byte=True,
            pixel_mapper=self.mapper,
            limit_entry=True,
            entry_polys=self.entry_polys,
            entry_window_time=float("inf"),
            boundary=self.boundary,
            boundary_pad_pct=self.config.get("boundary_pad_pct", 0.05),
            track_enemy=self.config.get("track_enemy", True),
            entry_conf_threshold=0.3,
        )

        self.box_conf_threshold = self.config.get("box_conf_threshold", 0.3)
        self.pose_conf_threshold = self.config.get("pose_conf_threshold", 0.5)
        self.device = self.config.get("device", "cpu")
        self.map_img = self.config.get("Map Image", None)
        self.keypoint_indices = self.config.get("keypoint_indices", None)

    def _release_inferencer(self) -> None:
        """Drop the detection/pose backend once the frame loop is done.

        The backend (detector + pose weights, runtime sessions/engines) is
        only used inside the per-frame loop; releasing it before
        transcription / annotation / metrics returns its RAM (and GPU cache)
        to the rest of the run. ``_initialize_components`` recreates it for
        the next vmeta.
        """
        self.inferencer = None
        free_torch_memory(getattr(self, "device", None))

    def mt_initialize(self, messages, vmeta_paths, output_path):
        logging.debug(f"Received initialization message with {len(vmeta_paths)} vmeta files.")

        try:
            for vmeta_path in vmeta_paths:
                if vmeta_path not in self.processed_vmetas:
                    logging.debug(f"Starting processing of {vmeta_path}...")
                    self.processed_vmetas.add(vmeta_path)
                    config = self.__load_config(vmeta_path, output_path)

                    self.config = config
                    self._initialize_components(vmeta_path)
                    self.metrics += self.__assess()
                else:
                    logging.debug(f"Vmeta file has previously been processed. Skipping: {vmeta_path}")

            logging.debug("Finished Processing. Ready for metric queries.")
            return "ready"
        except Exception as e:
            logging.error("Unable to process the received vmeta files. Printing stack trace...")
            logging.error(e, exc_info=True)
            return "error"

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
        config_path, video_path, start_time, role, title = load_vmeta(vmeta_path)
        config = load_config(config_path)
        config["start_time"] = start_time
        config["video_path"] = video_path
        config["output_path"] = output_path
        config["role"] = role
        config["title"] = title
        config["vmeta_path"] = os.path.abspath(vmeta_path)
        if self.force_transcode:
            video_path = transcode(video_path)
        fps = get_video_framerate(video_path)
        if fps is None:
            fps = 30
        config["frame_rate"] = fps
        config["frame_time"] = int((1 / fps) * 1000)
        return config

    def _is_in_entry_region(self, point):
        if not getattr(self, "entry_polys", None):
            return False
        for entry_poly in self.entry_polys:
            if entry_poly.contains(point):
                return True
        return False

    def _resolve_inroom_ids(self, tracker_output, config):
        """Resolve confirmed in-room track IDs from tracker metadata."""
        inroom_ids = set()
        inroom_id_start = int(config.get("inroom_id_start", getattr(self.tracker, "INROOM_FINAL_ID_START", 99)))

        for frame_entry in tracker_output:
            for obj in frame_entry["objects"]:
                tid = obj.get("id")
                if tid is None:
                    continue

                role = obj.get("identity_role")
                is_inroom = bool(obj.get("is_inroom", False))
                birth_location = obj.get("birth_location")

                if role == "inroom" or is_inroom or birth_location == "inroom":
                    inroom_ids.add(int(tid))
                elif role is None and birth_location is None and int(tid) >= inroom_id_start:
                    inroom_ids.add(int(tid))

        return sorted(inroom_ids)

    def _resolve_entry_ids(self, tracker_output):
        """Resolve confirmed entry track IDs from tracker metadata."""
        entry_ids = set()
        for frame_entry in tracker_output:
            for obj in frame_entry["objects"]:
                tid = obj.get("id")
                if tid is None:
                    continue

                role = obj.get("identity_role")
                is_entry = bool(obj.get("is_entry", False))
                birth_location = obj.get("birth_location")

                if role == "entry" or is_entry or birth_location == "entry":
                    entry_ids.add(int(tid))
        return sorted(entry_ids)

    def __assess(self):
        config = self.config

        enabled = config.get("enabled_visualizations", None)
        if not isinstance(enabled, list) or not enabled:
            enabled = list(DEFAULT_ENABLED_VISUALIZATIONS)
        enabled = [name for name in enabled if name in VIS_VMETA_SUFFIX]
        self.enabled_visualizations = enabled

        visual_angle_deg = float(config.get("visual_angle_degrees", 20.0))
        half_visual_angle_deg = visual_angle_deg / 2.0

        min_threat_interaction_time_sec = float(config.get("min_threat_interaction_time_sec", 1.0))
        threat_interaction_frames = int(min_threat_interaction_time_sec * config.get("frame_rate", 30.0))

        # (name, class) pairs filtered by ``ENABLED_METRICS`` — disabled
        # entries are skipped here so their ``process(ctx)`` never runs and
        # no metric_flags / scores are emitted for them. Class imports stay
        # intact so flipping the toggle re-enables a metric without code
        # changes anywhere else.
        _candidate_metrics: List[Tuple[str, type]] = [
            ("EntranceVectors_Metric", EntranceVectors_Metric),
            ("EntranceHesitation_Metric", EntranceHesitation_Metric),
            ("TotalEntryTime_Metric", TotalEntryTime_Metric),
            ("MoveAlongWall_Metric", MoveAlongWall_Metric),
            ("IdentifyAndCapturePods_Metric", IdentifyAndCapturePods_Metric),
            ("CapturePodTime_Metric", CapturePodTime_Metric),
            ("ThreatClearance_Metric", ThreatClearance_Metric),
            ("TeammateCoverage_Metric", TeammateCoverage_Metric),
            ("ThreatCoverage_Metric", ThreatCoverage_Metric),
            ("RoomCoverage_Metric", RoomCoverage_Metric),
            ("TotalRoomCoverageTime_Metric", TotalRoomCoverageTime_Metric),
        ]
        metrics: List[AbstractMetric] = [
            cls(config)
            for name, cls in _candidate_metrics
            if ENABLED_METRICS.get(name, False)
        ]

        # Resolve which compute families need to run by taking the union of
        # the deps declared by each enabled metric and each enabled
        # visualization. New metrics / visualizations only need to update
        # the *_COMPUTE_DEPS maps above; the gating branches below stay
        # untouched.
        required_compute: set = set()
        for name, enabled in ENABLED_METRICS.items():
            if enabled:
                required_compute |= METRIC_COMPUTE_DEPS.get(name, frozenset())
        for viz_name in self.enabled_visualizations:
            required_compute |= VIS_COMPUTE_DEPS.get(viz_name, frozenset())

        needs_pod_processing = "pod" in required_compute
        needs_threat_clearance = "threat_clearance" in required_compute
        needs_room_coverage = "room_coverage" in required_compute

        vs = cv2.VideoCapture(config["video_path"])
        frame_total = count_video_frames(vs)

        # The provided output_path IS this run's folder — every artifact (and the
        # RunInfo.json manifest below) is written flat into it. Point one run at one
        # folder; Compare scans the parent root for sibling run folders.
        self.output_directory = os.path.abspath(config["output_path"])
        self.video_basename = os.path.splitext(os.path.basename(config["video_path"]))[0]

        role = config.get("role") or "trainee"
        title = config.get("title") or self.video_basename
        # run_id is a logical label stored in RunInfo.json (read back by Compare),
        # no longer the folder name.
        run_id = make_run_id(role, title)
        self.run_id = run_id
        config["run_id"] = run_id
        os.makedirs(self.output_directory, exist_ok=True)

        save_run_info(
            self.output_directory,
            run_id=run_id,
            role=role,
            title=title,
            video_basename=self.video_basename,
            vmeta_path=config.get("vmeta_path") or "",
            analysis_file=f"{self.video_basename}_Analysis.json",
        )

        # PNG is lossless — JPG would smear fine map lines (grid markings,
        # thin annotations) that the live tracking videos preserve because
        # they render straight from the in-memory ``config["Map Image"]``.
        cv2.imwrite(os.path.join(self.output_directory, "EmptyMap.png"), config["Map Image"])
        logging.debug(f"Saved map cache to: {os.path.join(self.output_directory, 'EmptyMap.png')}")

        # Drill-window detection deferes transcription until after the
        # pose+tracker loop has identified the first entry frame, then
        # transcribes only the audio slice from that frame onward (or skips
        # transcription entirely when no entry is detected). When
        # ``drill_window_enabled`` is False we keep the legacy behavior of
        # running transcription in parallel with the pose loop.
        assert config is not None  # narrowed for the closure below
        drill_window_enabled = bool(config.get("drill_window_enabled", True))
        enable_transcription = bool(config.get("enable_transcription", False))

        def _invoke_transcribe(
            audio_start_sec: Optional[float] = None,
            cfg: Dict = config,
            out_dir: str = self.output_directory,
            basename: str = self.video_basename,
        ) -> None:
            """Run whisperx with the project's standard args.

            Layer-1 toggles come from config; Layer-2 expert knobs are pinned
            here so all tuning happens in one place. ``audio_start_sec`` lets
            the deferred path transcribe only the post-entry slice. The dict-
            and string-typed bindings are captured via default args so that
            pyright preserves narrowing across the closure boundary.
            """
            try:
                transcribe_video(
                    source_video_path=cfg["video_path"],
                    output_dir=out_dir,
                    video_basename=basename,
                    # === Config-exposed (Layer 1) — only the toggles + device ===
                    device=cfg.get("transcription_device", "cpu"),
                    denoise_model=(
                        "dns48" if cfg.get("enable_denoise", False) else None
                    ),
                    denoise_device=cfg.get("denoise_device"),
                    # === Drill-window slice ===
                    audio_start_sec=audio_start_sec,
                    # === Hardcoded developer defaults (Layer 2) ===
                    model="large-v2",
                    language="en",
                    compute_type=None,
                    batch_size=16,
                    beam_size=5,
                    temperature=0.0,
                    temperature_increment_on_fallback=0.2,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    suppress_numerals=False,
                    initial_prompt=None,
                    hotwords=None,
                    vad_method="pyannote",
                    vad_onset=0.7,
                    vad_offset=0.363,
                    chunk_size=30,
                    run_alignment=True,
                    return_char_alignments=False,
                    interpolate_method="nearest",
                    threads=4,
                    denoise_dry=0.5,
                    keep_denoised_wav=True,
                )
            except Exception:
                logging.exception("Transcription failed; pipeline continuing.")

        # Legacy parallel-thread mode: drill-window detection is OFF so we
        # have no reason to defer transcription. Thread joins later, before
        # metric scoring.
        transcription_thread: Optional[threading.Thread] = None
        if enable_transcription and not drill_window_enabled:
            transcription_thread = threading.Thread(
                target=_invoke_transcribe,
                name="transcription",
                daemon=False,
            )
            transcription_thread.start()
            logging.info("Started transcription thread (parallel with pose pipeline)")

        tracker_output: List[dict] = []
        all_map_points: List[list] = []
        gaze_info: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        all_frames: List[list] = []
        processed_frames = 0
        tracks_by_id: Dict[int, List[Optional[Tuple[float, float]]]] = {}

        # Opt-in early stop: when ``max_frames_after_first_entry`` is set,
        # processing halts that many frames after the tracker's first
        # confirmed entry (same signal find_drill_start_frame uses). Lets
        # batch/validation runs skip long post-drill footage. Absent from
        # production configs -> no behavior change.
        max_after_entry = config.get("max_frames_after_first_entry")
        max_after_entry = int(max_after_entry) if max_after_entry else None
        first_entry_frame_live: Optional[int] = None

        # Stop-at-drill-end: once the tracker confirms the first entry, the
        # audio slice is fully determined, so transcription + drill-end
        # location run in a background thread while the frame loop continues.
        # As soon as the located end frame is reached (or already passed),
        # the loop stops — post-drill footage adds nothing downstream (all
        # artifacts and metrics are drill-window-scoped). If no end can be
        # identified (no transcript match), processing continues to the video
        # end exactly as before. Disable with ``stop_at_drill_end: false``.
        stop_at_end = (
            bool(config.get("stop_at_drill_end", True))
            and drill_window_enabled
            and enable_transcription
        )
        fps_cfg = float(config.get("frame_rate", 30.0) or 30.0)
        end_locator_thread: Optional[threading.Thread] = None
        end_locator_result: Dict[str, Any] = {}
        end_locator_done = threading.Event()

        use_streaming_asr = bool(config.get("asr_streaming", True))

        def _locate_drill_end(entry_frame: int) -> None:
            try:
                if use_streaming_asr:
                    _locate_drill_end_streaming(entry_frame)
                else:
                    _locate_drill_end_batch(entry_frame)
            except Exception:
                logging.exception(
                    "Drill-end location failed; processing continues to video end."
                )
            finally:
                end_locator_done.set()

        def _locate_drill_end_streaming(entry_frame: int) -> None:
            # Forward-chunked, stop-at-confirmed-end transcription. Loads the
            # ASR model once and transcribes only up to ~one confirmation chunk
            # past the drill end (never to EOF), then writes the standard
            # Transcription.json sidecar from the accumulated segments.
            from .utils.asr_session import build_session
            from .utils.drill_stream import locate_drill_end_streaming
            session = build_session(config.get("asr_backend", "whisperx"), config)
            denoise_session = None
            if bool(config.get("enable_denoise", False)):
                try:
                    from .utils.denoise import DenoiseSession
                    denoise_session = DenoiseSession(
                        model_name="dns48",
                        dry=float(config.get("denoise_dry", 0.5)),
                        device=config.get("denoise_device") or config.get("transcription_device", "cpu"),
                    )
                except Exception:
                    logging.warning("Denoise session init failed; continuing without denoise.", exc_info=True)
                    denoise_session = None
            try:
                res = locate_drill_end_streaming(
                    source_path=config["video_path"],
                    drill_start_frame=entry_frame,
                    fps=fps_cfg,
                    frame_total=frame_total,
                    config=config,
                    session=session,
                    denoise_session=denoise_session,
                )
            finally:
                session.close()
                if denoise_session is not None:
                    try:
                        denoise_session.close()
                    except Exception:
                        pass
            preroll = float(config.get("transcription_preroll_sec", 5.0) or 0.0)
            start_sec = (float(entry_frame) - 1.0) / fps_cfg if fps_cfg > 0 else 0.0
            save_transcription_sidecar(
                output_dir=self.output_directory,
                video_basename=self.video_basename,
                segments=res.segments,
                model=str(config.get("asr_backend", "whisperx")),
                language=config.get("transcription_language", "en"),
                aligned=True,
                audio_window={"start_sec": max(0.0, start_sec - max(0.0, preroll)), "end_sec": None},
            )
            end_locator_result["window"] = res.window
            logging.info(
                "Streaming drill-end: end_frame=%s reason=%s (%.1fs audio, %d passes, %.1fs).",
                res.window.end_frame, res.window.decision_reason,
                res.last_audio_sec, res.passes, res.elapsed_sec,
            )

        def _locate_drill_end_batch(entry_frame: int) -> None:
            start_sec = (float(entry_frame) - 1.0) / fps_cfg if fps_cfg > 0 else 0.0
            preroll = float(config.get("transcription_preroll_sec", 5.0) or 0.0)
            _invoke_transcribe(audio_start_sec=max(0.0, start_sec - max(0.0, preroll)))
            end_locator_result["window"] = compute_drill_window(
                transcription_path=os.path.join(
                    self.output_directory,
                    f"{self.video_basename}_Transcription.json",
                ),
                drill_start_frame=entry_frame,
                total_frames=frame_total,
                frame_rate=fps_cfg,
                config=config,
            )

        for frame_num in tqdm(range(1, frame_total + 1), desc="Processing frames", unit="frame"):
            ret, frame = vs.read()
            if not ret or frame is None:
                break

            processed_frames += 1

            infer_results = self.inferencer(
                frame,
                return_vis=False,
                bbox_thr=self.box_conf_threshold,
                kpt_thr=self.pose_conf_threshold,
                pose_based_nms=True,
            )
            res = next(infer_results)
            instances = res["predictions"][0]

            dets = []
            matched_keypoints = []
            inst_z = []  # per-instance metric-z (pose3d backends), else None

            for inst in instances:
                bbox = inst["bbox"][0]
                x1, y1, x2, y2 = bbox
                frame_h, frame_w = frame.shape[:2]
                if x1 <= 0 and y1 <= 0 and x2 >= frame_w and y2 >= frame_h:
                    continue

                bbox_score = inst.get("bbox_score", 0.0)
                dets.append([x1, y1, x2, y2, bbox_score, 0])

                kps = np.array(inst["keypoints"])
                kp_scores = inst.get("keypoint_scores", None)
                if kp_scores is not None:
                    scores = np.array(kp_scores)
                else:
                    scores = np.ones((kps.shape[0],))
                kps3 = np.concatenate([kps, scores[:, None]], axis=1)
                matched_keypoints.append(kps3)
                inst_z.append(inst.get("keypoint_z"))

            dets = np.array(dets)
            if dets.size == 0:
                dets = np.empty((0, 6))

            # Canonical adapter: non-default backends (wholebody / pose3d)
            # emit K != 26; convert to the Halpe-26 layout before the tracker
            # so every downstream consumer keeps its index semantics. The
            # default backend never enters this branch (identity path).
            extras_by_key: dict = {}
            if matched_keypoints and getattr(self, "pose_meta", "halpe26") != "halpe26":
                raw = np.stack(matched_keypoints, axis=0)  # (N, K, 3)
                kpts26, scores26, extras = to_canonical(
                    raw[:, :, :2].astype(np.float32),
                    raw[:, :, 2].astype(np.float32),
                    self.pose_meta,
                )
                matched_keypoints = [
                    np.concatenate([kpts26[i], scores26[i][:, None]], axis=1)
                    for i in range(raw.shape[0])
                ]
                if any(z is not None for z in inst_z):
                    extras["z_values"] = np.stack([
                        np.asarray(z, dtype=np.float32)
                        if z is not None else np.zeros(raw.shape[1], np.float32)
                        for z in inst_z
                    ])
                if extras:
                    wb_k = extras.get("wb_keypoints")
                    wb_s = extras.get("wb_scores")
                    zs = extras.get("z_values")
                    for i, row in enumerate(matched_keypoints):
                        extras_by_key[_extras_key(row)] = (
                            wb_k[i] if wb_k is not None else None,
                            wb_s[i] if wb_s is not None else None,
                            zs[i] if zs is not None else None,
                        )

            if matched_keypoints:
                matched_keypoints = np.array(matched_keypoints)
            else:
                matched_keypoints = np.empty((0, 26, 3))

            trackers = self.tracker.update(
                dets,
                frame,
                keypoints=matched_keypoints,
                keypoint_confidence_threshold=self.pose_conf_threshold,
                keypoint_indices=self.keypoint_indices,
            )

            frame_objects = []
            for trk in trackers:
                trk_id = trk.get("track_id")
                x1 = int(trk.get("top_left_x", 0))
                y1 = int(trk.get("top_left_y", 0))
                w = int(trk.get("width", 0))
                h = int(trk.get("height", 0))
                bbox = [x1, y1, x1 + w, y1 + h]
                kps = trk.get("keypoints", None)

                bbox_confidence = trk.get("confidence", None)
                track_class = trk.get("class", None)
                ankle_based_point = trk.get("ankle_based_point", None)
                current_map_pos = trk.get("current_map_pos", None)
                map_velocity = trk.get("map_velocity", None)

                if ankle_based_point is not None:
                    ankle_based_point = np.asarray(ankle_based_point, dtype=float).reshape(-1)
                    ankle_based_point = ankle_based_point.tolist() if ankle_based_point.size >= 2 else None

                if current_map_pos is not None:
                    current_map_pos = np.asarray(current_map_pos, dtype=float).reshape(-1)
                    current_map_pos = current_map_pos.tolist() if current_map_pos.size >= 2 else None

                if map_velocity is not None:
                    map_velocity = np.asarray(map_velocity, dtype=float).reshape(-1)
                    map_velocity = map_velocity.tolist() if map_velocity.size >= 2 else None

                if kps is not None:
                    # 0.01 px / 1e-3 score quantization — keeps the serialized
                    # TrackerOutput.json floats short (5-8x smaller file);
                    # downstream consumers only do distance/threshold math.
                    keypoints = np.round(kps[:, :2].astype(float), 2).tolist()
                    keypoint_scores = np.round(kps[:, 2].astype(float), 3).tolist()
                else:
                    keypoints, keypoint_scores = [], []

                if kps is not None and kps.shape[0] == 26:
                    gvec = compute_gaze_vector(kps)
                    if gvec is not None:
                        origin, direction = gvec
                        gaze_info[(frame_num, trk_id)] = (
                            float(origin[0]),
                            float(origin[1]),
                            float(direction[0]),
                            float(direction[1]),
                        )

                obj = {
                    "id": trk_id,
                    "bbox": bbox,
                    "confidence": float(bbox_confidence) if bbox_confidence is not None else None,
                    "class": int(track_class) if track_class is not None else None,
                    "ankle_based_point": ankle_based_point,
                    "current_map_pos": current_map_pos,
                    "map_velocity": map_velocity,
                    "keypoints": keypoints,
                    "keypoint_scores": keypoint_scores,
                    "identity_role": trk.get("identity_role", None),
                    "birth_location": trk.get("birth_location", None),
                    "is_inroom": bool(trk.get("is_inroom", False)),
                    "is_entry": bool(trk.get("is_entry", False)),
                }
                # Optional extended-keypoint metadata (non-default backends
                # only): matched back to the track via the canonical block the
                # tracker passes through. Absent for unmatched (coasting)
                # tracks — extras are strictly optional per frame.
                if extras_by_key and kps is not None and kps.shape[0] == 26:
                    ex = extras_by_key.get(_extras_key(np.asarray(kps)))
                    if ex is not None:
                        wb_k, wb_s, z_vals = ex
                        if wb_k is not None:
                            obj["keypoints_wb"] = np.round(
                                np.asarray(wb_k, dtype=float), 2
                            ).tolist()
                            obj["keypoint_scores_wb"] = np.round(
                                np.asarray(wb_s, dtype=float), 3
                            ).tolist()
                        if z_vals is not None:
                            obj["keypoints_z"] = np.round(
                                np.asarray(z_vals, dtype=float), 4
                            ).tolist()
                frame_objects.append(obj)

            tracker_output.append({"frame": frame_num, "objects": frame_objects})

            map_points = []
            for trk in trackers:
                trk_id = trk.get("track_id")
                current_map_pos = trk.get("current_map_pos")
                if current_map_pos is not None:
                    cur = np.asarray(current_map_pos, dtype=float).reshape(-1)
                    if cur.size < 2 or not np.isfinite(cur[:2]).all():
                        continue

                    mapX, mapY = float(cur[0]), float(cur[1])
                    if self.boundary is not None:
                        point = Point(mapX, mapY)
                        if self.boundary.contains(point) or self._is_in_entry_region(point):
                            map_points.append([trk_id, mapX, mapY])
                        else:
                            nearest_point = nearest_points(self.boundary, point)[0]
                            map_points.append([trk_id, nearest_point.x, nearest_point.y])

            for tid in tracks_by_id.keys():
                tracks_by_id[tid].append(None)

            for tid, mx, my in map_points:
                if tid not in tracks_by_id:
                    tracks_by_id[tid] = [None] * (processed_frames - 1)
                    tracks_by_id[tid].append((float(mx), float(my)))
                else:
                    tracks_by_id[tid][-1] = (float(mx), float(my))

            all_frames.append(map_points)
            for idx, mx, my in map_points:
                all_map_points.append([frame_num, idx, mx, my])

            if ((max_after_entry is not None or stop_at_end)
                    and first_entry_frame_live is None):
                if any(o.get("is_entry") or o.get("birth_location") == "entry"
                       for o in frame_objects):
                    first_entry_frame_live = frame_num
                    if stop_at_end and end_locator_thread is None:
                        end_locator_thread = threading.Thread(
                            target=_locate_drill_end,
                            args=(frame_num,),
                            name="drill-end-locator",
                            daemon=False,
                        )
                        end_locator_thread.start()
                        logging.info(
                            "First entry at frame %d — transcription started in "
                            "background to locate the drill end.", frame_num,
                        )
            if (max_after_entry is not None
                    and first_entry_frame_live is not None
                    and frame_num >= first_entry_frame_live + max_after_entry):
                logging.info(
                    "Early stop: %d frames processed after first entry at frame %d "
                    "(max_frames_after_first_entry=%d).",
                    frame_num - first_entry_frame_live, first_entry_frame_live,
                    max_after_entry,
                )
                break
            if stop_at_end and end_locator_done.is_set():
                _w = end_locator_result.get("window")
                if (_w is not None and not _w.end_uncertain
                        and frame_num >= int(_w.end_frame)):
                    logging.info(
                        "Stopping at frame %d: drill end identified at frame %d (%s).",
                        frame_num, int(_w.end_frame), _w.decision_reason,
                    )
                    break

        vs.release()
        self._release_inferencer()

        inroom_ids = self._resolve_inroom_ids(tracker_output, config)
        entry_ids = self._resolve_entry_ids(tracker_output)
        logging.debug(f"Resolved entry IDs from tracker metadata: {entry_ids}")
        logging.debug(f"Resolved in-room IDs from tracker metadata: {inroom_ids}")

        fall_frames = None

        # ------------------------------------------------------------------
        # Drill-window detection
        # ------------------------------------------------------------------
        # When enabled, derive ``drill_start_frame`` from the tracker output
        # and (synchronously) transcribe only the post-entry audio slice. If
        # no entry was detected, transcription is skipped entirely. The
        # detected window is later threaded into MetricContext, the coverage/
        # POD/clearance helpers, and the annotation calls so all downstream
        # work operates on just the actual drill segment.
        fps = float(config.get("frame_rate", 30.0) or 30.0)

        if drill_window_enabled and end_locator_thread is not None:
            # Stop-at-drill-end path: transcription + window location already
            # ran (or are finishing) in the background locator thread. Join it
            # and reuse its result rather than transcribing a second time.
            end_locator_thread.join()
            drill_start_frame_detected = find_drill_start_frame(tracker_output)
            drill_window = end_locator_result.get("window")
            if drill_window is None:
                # Locator failed (exception) — fall back to a synchronous
                # compute over whatever transcript exists, so the run still
                # produces a window instead of crashing.
                drill_window = compute_drill_window(
                    transcription_path=os.path.join(
                        self.output_directory,
                        f"{self.video_basename}_Transcription.json",
                    ),
                    drill_start_frame=drill_start_frame_detected,
                    total_frames=processed_frames,
                    frame_rate=fps,
                    config=config,
                )
        elif drill_window_enabled:
            drill_start_frame_detected = find_drill_start_frame(tracker_output)
            if drill_start_frame_detected is not None and enable_transcription:
                # Time convention: starts use (frame-1)/fps — frame N begins
                # at (N-1)/fps for 1-indexed frames — matching the metrics'
                # frame→seconds math; ends use frame/fps (end of frame).
                drill_start_sec = (
                    (float(drill_start_frame_detected) - 1.0) / fps if fps > 0 else 0.0
                )
                # Decouple the ASR slice from the exact detected start: a
                # constant pre-roll makes WhisperX's VAD/segmentation
                # independent of few-frame shifts in entry detection (which
                # otherwise flip segment boundaries and move the matched
                # drill-end segment by seconds). End-candidate filtering in
                # compute_drill_window still uses drill_start_sec, so
                # pre-roll audio can never select an end before the start.
                preroll = float(config.get("transcription_preroll_sec", 5.0) or 0.0)
                audio_start = max(0.0, drill_start_sec - max(0.0, preroll))
                logging.info(
                    "Running deferred transcription on audio slice from %.3fs "
                    "(drill start %.3fs / frame %d, pre-roll %.1fs).",
                    audio_start, drill_start_sec, drill_start_frame_detected, preroll,
                )
                _invoke_transcribe(audio_start_sec=audio_start)
            elif drill_start_frame_detected is None:
                logging.info(
                    "Skipping transcription: no entry crossing detected by tracker."
                )

            transcription_path = os.path.join(
                self.output_directory, f"{self.video_basename}_Transcription.json"
            )
            drill_window = compute_drill_window(
                transcription_path=transcription_path if enable_transcription else None,
                drill_start_frame=drill_start_frame_detected,
                total_frames=processed_frames,
                frame_rate=fps,
                config=config,
            )
        else:
            # Legacy path: parallel transcription thread is still running.
            if transcription_thread is not None:
                transcription_thread.join()
                logging.info("Transcription thread joined (legacy parallel mode).")
            drill_window = DrillWindow(
                start_frame=1,
                end_frame=processed_frames,
                end_uncertain=False,
                decision_reason="drill_window_disabled",
                start_time_sec=0.0,
                end_time_sec=processed_frames / fps if fps > 0 else None,
            )

        save_drill_window_sidecar(
            drill_window,
            output_directory=self.output_directory,
            video_basename=self.video_basename,
        )

        drill_start_frame = int(drill_window.start_frame)
        drill_end_frame = int(drill_window.end_frame)
        drill_start_sec = drill_window.start_time_sec or 0.0
        drill_end_sec = drill_window.end_time_sec or (processed_frames / fps if fps > 0 else None)

        save_position_cache(all_map_points, self.output_directory, self.video_basename)
        save_gaze_cache(gaze_info, self.output_directory, self.video_basename)

        preserve_audio_enabled = (
            bool(config.get("preserve_audio", True))
            and has_audio_stream(config["video_path"])
        )

        def _attach_audio(viz_key: str) -> None:
            # Always muxes from the ORIGINAL source. Even when denoising is
            # enabled, only the transcription consumes the denoised WAV;
            # saved annotated videos retain the un-denoised audio. When the
            # annotated video is trimmed to the drill window, the source
            # audio is sliced to the same range so the two stay in sync.
            if not preserve_audio_enabled:
                return
            suffix = VIS_VMETA_SUFFIX.get(viz_key)
            if not suffix:
                return
            annotated_path = os.path.join(
                self.output_directory, f"{self.video_basename}{suffix}.mp4"
            )
            attach_audio_in_place(
                annotated_path,
                config["video_path"],
                audio_start_sec=drill_start_sec if drill_start_sec > 0 else None,
                audio_end_sec=drill_end_sec,
            )

        # ------------------------------------------------------------------
        # Camera-view overlays — all enabled overlays are drawn in ONE decode
        # pass over the source video (decode dominates annotation cost).
        # ``keypoint_details``/``bbox_details`` and the threat-clearance map
        # are computed up front: clearance overlays need the map at draw
        # time, and the metric context reuses all three later.
        # ------------------------------------------------------------------
        bbox_details = {}
        keypoint_details = {}
        for entry in tracker_output:
            frame_idx = entry["frame"]
            for obj in entry["objects"]:
                tid = obj["id"]
                bbox_details[(frame_idx, tid)] = tuple(obj["bbox"])
                keypoint_details[(frame_idx, tid)] = (obj["keypoints"], obj["keypoint_scores"])

        clearance_map = None
        if needs_threat_clearance:
            clearance_map = compute_threat_clearance(
                tracker_output,
                keypoint_details,
                gaze_info,
                inroom_ids=inroom_ids,
                visual_angle_deg=visual_angle_deg,
                intersection_frames=threat_interaction_frames,
                wrist_frames=max(1, int(threat_interaction_frames * 0.1)),
                gaze_frames=max(1, int(threat_interaction_frames * 0.5)),
                start_frame=drill_start_frame,
                end_frame=drill_end_frame,
            )

        camera_overlay_specs = []
        camera_overlay_keys = []
        if "annotate_camera_video" in self.enabled_visualizations:
            camera_overlay_specs.append(
                camera_tracking_overlay_spec(
                    self.output_directory,
                    self.video_basename,
                    inroom_ids,
                    gaze_conf_threshold=self.pose_conf_threshold,
                )
            )
            camera_overlay_keys.append("annotate_camera_video")
        if "annotate_camera_with_gaze_triangle" in self.enabled_visualizations:
            camera_overlay_specs.append(
                gaze_triangle_overlay_spec(
                    gaze_info,
                    self.output_directory,
                    self.video_basename,
                    inroom_ids,
                    half_angle_deg=half_visual_angle_deg,
                    alpha=0.2,
                    show_inroom_gaze=False,
                )
            )
            camera_overlay_keys.append("annotate_camera_with_gaze_triangle")
        if clearance_map is not None and "annotate_clearance_video" in self.enabled_visualizations:
            camera_overlay_specs.append(
                clearance_overlay_spec(
                    clearance_map,
                    self.output_directory,
                    self.video_basename,
                    inroom_ids,
                )
            )
            camera_overlay_keys.append("annotate_clearance_video")
        if clearance_map is not None and "annotate_camera_tracking_with_clearance" in self.enabled_visualizations:
            camera_overlay_specs.append(
                tracking_with_clearance_overlay_spec(
                    clearance_map,
                    self.output_directory,
                    self.video_basename,
                    inroom_ids,
                    gaze_conf_threshold=self.pose_conf_threshold,
                    show_clearing_id=True,
                )
            )
            camera_overlay_keys.append("annotate_camera_tracking_with_clearance")

        if camera_overlay_specs:
            annotate_camera_videos_combined(
                tracker_output,
                config["frame_rate"],
                camera_overlay_specs,
                video_path=config["video_path"],
                start_frame=drill_start_frame,
                end_frame=drill_end_frame,
            )
            for viz_key in camera_overlay_keys:
                _attach_audio(viz_key)

        if "annotate_map_video" in self.enabled_visualizations:
            annotate_map_video(
                config["Map Image"],
                all_map_points,
                config["frame_rate"],
                self.output_directory,
                self.video_basename,
                total_frames=processed_frames,
                inroom_ids=inroom_ids,
                start_frame=drill_start_frame,
                end_frame=drill_end_frame,
            )

        coverage_data = None
        if self.boundary is not None:
            if "annotate_map_with_gaze" in self.enabled_visualizations:
                annotate_map_with_gaze(
                    map_image=config["Map Image"],
                    pixel_mapper=self.mapper,
                    gaze_info=gaze_info,
                    room_boundary_coords=list(self.boundary.exterior.coords),
                    frame_rate=config["frame_rate"],
                    output_directory=self.output_directory,
                    video_basename=self.video_basename,
                    inroom_ids=inroom_ids,
                    show_inroom_gaze=False,
                    half_angle_deg=half_visual_angle_deg,
                    alpha=0.1,
                    total_frames=processed_frames,
                    accumulated_clear=True,
                    start_frame=drill_start_frame,
                    end_frame=drill_end_frame,
                )

            if needs_room_coverage:
                coverage_data = compute_room_coverage(
                    map_image=config["Map Image"],
                    pixel_mapper=self.mapper,
                    gaze_info=gaze_info,
                    room_boundary_coords=list(self.boundary.exterior.coords),
                    frame_rate=config["frame_rate"],
                    total_frames=processed_frames,
                    inroom_ids=inroom_ids,
                    half_angle_deg=half_visual_angle_deg,
                    start_frame=drill_start_frame,
                    end_frame=drill_end_frame,
                )
                save_room_coverage_cache(coverage_data, self.output_directory, self.video_basename)

        # 3D skeleton plot video (pose3d backend only; inspection artifact,
        # color-coded by track id). No-ops when the run carries no z data.
        if bool(config.get("annotate_pose3d_plot", True)):
            try:
                from src.pose3d_viz import render_pose3d_plot_video
                render_pose3d_plot_video(
                    tracker_output,
                    frame_rate=config["frame_rate"],
                    output_directory=self.output_directory,
                    video_basename=self.video_basename,
                    video_path=config.get("video_path"),
                    start_frame=drill_start_frame,
                    end_frame=drill_end_frame,
                )
            except Exception:
                logging.warning("Pose3D plot rendering failed; continuing.", exc_info=True)

        tracker_json_path = os.path.join(self.output_directory, f"{self.video_basename}_TrackerOutput.json")
        with open(tracker_json_path, "w") as f:
            # Compact separators: this is the largest sidecar (per-frame,
            # per-person keypoints) and machine-read only — indenting it
            # multiplies file size and write time for no reader.
            json.dump(tracker_output, f, separators=(",", ":"))
        logging.debug(f"Saved TrackerOutput to: {tracker_json_path}")
        logging.debug(
            f"Saved PositionCache to: {os.path.join(self.output_directory, f'{self.video_basename}_PositionCache.txt')}"
        )

        # Persist the runtime fps (read from the actual video) so offline
        # comparisons against this session's caches can recover the same
        # value rather than trusting the map JSON, which can drift.
        try:
            save_run_metadata(
                self.output_directory,
                self.video_basename,
                fps=float(config.get("frame_rate") or 0.0),
                video_path=config.get("video_path"),
            )
        except Exception:
            logging.warning("Failed to write run-metadata sidecar.", exc_info=True)

        assignment = {}
        dynamic_work_areas = {}
        pod_capture_data: Dict[int, Dict[str, Optional[float]]] = {}
        if needs_pod_processing:
            pods_cfg = config.get("POD", [])
            pod_groups = config.get("pod_groups", None)
            working_radius = config.get("pod_working_radius", 40.0)
            capture_threshold = config.get("pod_capture_threshold_sec", 0.1)

            # Build door axes (per-door (p̂_A, p̂_B, n_in, type)) so POD-by-entry
            # assignment uses the same direction frame as the entrance-vectors metric.
            door_axes = []
            if self.boundary is not None and getattr(self, "entry_polys", None):
                try:
                    boundary_pts = list(self.boundary.exterior.coords)
                    door_polys = [list(p.exterior.coords) for p in self.entry_polys]
                    door_axes = load_door_axes(boundary_pts, door_polys)
                except Exception:
                    door_axes = []

            assignment, dynamic_work_areas, pod_capture_data = run_pod_analysis(
                tracks_by_id=tracks_by_id,
                tracker_output=tracker_output,
                pods_cfg=pods_cfg,
                pod_groups=pod_groups,
                pixel_mapper=self.mapper,
                boundary=self.boundary,
                inroom_ids=inroom_ids,
                working_radius=working_radius,
                frame_rate=config["frame_rate"],
                capture_threshold_sec=capture_threshold,
                save_cache=True,
                output_directory=self.output_directory,
                video_basename=self.video_basename,
                start_frame=drill_start_frame,
                end_frame=drill_end_frame,
                door_axes=door_axes,
            )

            if "annotate_map_pod_video" in self.enabled_visualizations:
                annotate_map_pod_video(
                    config["Map Image"],
                    all_map_points=all_map_points,
                    assignment=assignment,
                    dynamic_work_areas=dynamic_work_areas,
                    pod_capture_data=pod_capture_data,
                    frame_rate=config["frame_rate"],
                    output_directory=self.output_directory,
                    video_basename=self.video_basename,
                    total_frames=processed_frames,
                    inroom_ids=inroom_ids,
                    start_frame=drill_start_frame,
                    end_frame=drill_end_frame,
                )

            if "annotate_map_pod_with_paths_video" in self.enabled_visualizations:
                annotate_map_pod_with_paths_video(
                    config["Map Image"],
                    all_map_points=all_map_points,
                    assignment=assignment,
                    dynamic_work_areas=dynamic_work_areas,
                    pod_capture_data=pod_capture_data,
                    frame_rate=config["frame_rate"],
                    output_directory=self.output_directory,
                    video_basename=self.video_basename,
                    total_frames=processed_frames,
                    inroom_ids=inroom_ids,
                    start_frame=drill_start_frame,
                    end_frame=drill_end_frame,
                )

        context = MetricContext(
            tracker_output=tracker_output,
            all_frames=all_frames,
            tracks_by_id=tracks_by_id,
            gaze_info=gaze_info,
            bbox_details=bbox_details,
            keypoint_details=keypoint_details,
            fall_frames=fall_frames,
            map_points=all_map_points,
            entry_ids=entry_ids,
            inroom_ids=inroom_ids,
            room_coverage=coverage_data,
            pod_capture=pod_capture_data,
            drill_start_frame=drill_start_frame,
            drill_end_frame=drill_end_frame,
            drill_window_meta=drill_window.to_dict(),
            pixel_mapper=self.mapper,
        )

        # Clearance map + overlays were produced in the combined camera pass
        # above; here we only persist the cache and expose it to metrics.
        if clearance_map is not None:
            save_threat_clearance_cache(clearance_map, self.output_directory, self.video_basename)
            context.threat_clearance = clearance_map

        metric_scores = []
        # Stash select metric instances so build_analysis_session can pull
        # per-metric extras (excursion summaries, etc.) without us having to
        # plumb every metric class through the function signature.
        move_along_wall_metric = None
        for m in metrics:
            # Wall-adherence is measured across the full pre-drill + drill
            # span (frame 1 → drill_end_frame), not just the in-drill window.
            # Mutate the context's drill_start_frame only for this metric and
            # restore it after so other metrics still see the detected start.
            if getattr(m, "metricName", "") == "STAY_ALONG_WALL":
                saved_drill_start = context.drill_start_frame
                context.drill_start_frame = 1
                try:
                    m.process(context)
                finally:
                    context.drill_start_frame = saved_drill_start
            else:
                m.process(context)
            score = m.getFinalScore()

            metric_entry = {
                "metric_name": m.metricName,
                "score": score,
                "timestamp": config["start_time"],
            }
            metric_scores.append(metric_entry)

            if getattr(m, "metricName", "") == "STAY_ALONG_WALL":
                move_along_wall_metric = m

        save_metrics_cache(metric_scores, self.output_directory, self.video_basename)

        # When drill-window detection is enabled the transcription has
        # already run synchronously after the pose loop, so there is no
        # background thread to join here. The legacy parallel-thread path
        # (drill_window_enabled=False) joins earlier — right after the pose
        # loop, before drill-window decisions are made — so the sidecar is on
        # disk before downstream consumers run.

        position_cache_path = os.path.join(
            self.output_directory,
            f"{self.video_basename}_PositionCache.txt",
        )
        gaze_cache_path = os.path.join(
            self.output_directory,
            f"{self.video_basename}_GazeCache.txt",
        )
        empty_map_path = os.path.join(self.output_directory, "EmptyMap.png")
        metrics_cache_path = os.path.join(
            self.output_directory,
            f"{self.video_basename}_MetricsCache.txt",
        )
        analysis_json_path = os.path.join(
            self.output_directory,
            f"{self.video_basename}_Analysis.json",
        )

        try:
            move_summary = None
            if move_along_wall_metric is not None:
                move_summary = {
                    "score": float(move_along_wall_metric.getFinalScore()),
                    "per_entrant": list(
                        getattr(move_along_wall_metric, "_per_entrant_summary", []) or []
                    ),
                }
            analysis_session = build_analysis_session(
                tracks_by_id=tracks_by_id,
                inroom_ids=inroom_ids,
                frame_rate=config["frame_rate"],
                config=config,
                video_basename=self.video_basename,
                video_path=config.get("video_path"),
                total_frames=processed_frames,
                duration_sec=(processed_frames / config["frame_rate"]) if config.get("frame_rate") else None,
                start_time=config.get("start_time"),
                map_img=config.get("Map Image"),
                output_directory=self.output_directory,
                tracker_output_json_path=tracker_json_path,
                position_cache_path=position_cache_path,
                gaze_cache_path=gaze_cache_path,
                empty_map_path=empty_map_path,
                drill_window=drill_window,
                metric_flags=list(getattr(context, "metric_flags", []) or []),
                move_along_wall=move_summary,
            )
            analysis_session.setdefault("artifacts", {})
            analysis_session["artifacts"]["metrics_cache"] = {
                "path": metrics_cache_path,
                "label": "Metrics Cache",
                "type": "text",
                "exists": os.path.exists(metrics_cache_path),
            }

            with open(analysis_json_path, "w") as f:
                json.dump(analysis_session, f, indent=4)
            logging.debug(f"Saved analysis session JSON to: {analysis_json_path}")
        except Exception:
            logging.error("Failed to build or save analysis session JSON.", exc_info=True)

        return metric_scores

    def get_assessment(self, timestamp, metric_name):
        timestamp = int(timestamp)
        if self.writeXML:
            for viz_name in getattr(self, "enabled_visualizations", []) or []:
                suffix = VIS_VMETA_SUFFIX.get(viz_name)
                if not suffix:
                    continue
                generate_vmeta(self.output_directory, self.video_basename, suffix)
            self.writeXML = False

        desired_metrics = []
        for metric in self.metrics:
            if metric["metric_name"] == metric_name:
                desired_metrics.append(metric)

        self.playback_time = max(timestamp, self.playback_time)
        if len(desired_metrics) > 0:
            logging.debug(f"Returning {len(desired_metrics)} results for query {metric_name} at time {timestamp}")
        else:
            logging.debug(f"Metric query for time {timestamp} received but no metrics match.")
        return json.dumps(desired_metrics, indent=4)

    def compare_expert(self, metric_name, current_run, other_run, *, output_dir):
        """Compare two runs on a single metric. Returns a JSON-encoded simplified payload.

        ``current_run`` and ``other_run`` must both be per-run directories that
        contain a ``RunInfo.json`` manifest (see ``src.utils.run_info``).
        Comparison artifacts (side-by-side images) are written to
        ``output_dir`` — never into the run folders themselves. Callers own
        the temp directory and its lifecycle.

        Return shape::

            {
              "metric_id": "ENTRANCE_VECTORS",
              "label": "Entrance Vectors",
              "current": {"score": ..., "role": ..., "run_id": ...},
              "other":   {"score": ..., "role": ..., "run_id": ...},
              "visualizations": [
                {"image_path": ..., "label": ..., "caption": ...},
                ...
              ],
              "explanation": "...one-sentence summary..."
            }

        The relative grading (above / at / near / below) is computed in the
        viewer from the two scores. We don't persist or transmit a
        context-independent assessment because the meaningful thresholds vary
        room-to-room.
        """
        logging.debug(f"Starting comparison for metric {metric_name}.")

        def _err(message: str, *, label: Optional[str] = None) -> str:
            return json.dumps(
                {
                    "metric_id": metric_name,
                    "label": label or _COMPARE_METRIC_LABELS.get(metric_name, metric_name),
                    "current": {"score": None, "role": None, "run_id": None},
                    "other": {"score": None, "role": None, "run_id": None},
                    "visualizations": [],
                    "explanation": message,
                },
                indent=4,
            )

        current_info = load_run_info(current_run) if current_run else None
        if current_info is None:
            return _err("Current run folder is missing or has no RunInfo.json manifest.")
        other_info = load_run_info(other_run) if other_run else None
        if other_info is None:
            return _err("Comparison run folder is missing or has no RunInfo.json manifest.")

        if not output_dir:
            return _err("No output_dir supplied for comparison artifacts.")
        os.makedirs(output_dir, exist_ok=True)

        config = None
        vmeta_path = current_info.get("vmeta_path")
        if vmeta_path and os.path.isfile(vmeta_path):
            try:
                config_path, _, _, _, _ = load_vmeta(vmeta_path)
                config = load_config(config_path)
            except Exception:
                logging.error("Unable to load config from vmeta_path: %s", vmeta_path, exc_info=True)
                config = None

        if config is not None:
            md = load_run_metadata(current_run) or load_run_metadata(other_run)
            md_fps = (md or {}).get("fps")
            try:
                md_fps_f = float(md_fps) if md_fps is not None else None
            except (TypeError, ValueError):
                md_fps_f = None
            if md_fps_f and md_fps_f > 0:
                config["frame_rate"] = md_fps_f

        # Prefer the lossless PNG, but accept the legacy JPG name as a
        # fallback so older run folders still work without re-processing.
        map_path = None
        for candidate in (
            os.path.join(current_run, "EmptyMap.png"),
            os.path.join(other_run, "EmptyMap.png"),
            os.path.join(current_run, "EmptyMap.jpg"),
            os.path.join(other_run, "EmptyMap.jpg"),
        ):
            if os.path.exists(candidate):
                map_path = candidate
                break
        map_image = cv2.imread(map_path) if map_path else None

        current_scores = _read_metrics_csv(current_run)
        other_scores = _read_metrics_csv(other_run)
        current_score = current_scores.get(metric_name)
        other_score = other_scores.get(metric_name)

        metric_label = _COMPARE_METRIC_LABELS.get(metric_name, metric_name)

        metric_map = {
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
            return _err(f"Metric {metric_name} is not implemented for reference comparison.", label=metric_label)

        try:
            metric_obj = metric_cls(config or {})
        except TypeError:
            metric_obj = None
        target = metric_obj if metric_obj is not None else metric_cls

        try:
            legacy = target.expertCompare(
                session_folder=current_run,
                expert_folder=other_run,
                map_image=map_image,
            )
        except TypeError:
            try:
                legacy = target.expertCompare(current_run, other_run, map_image, config)
            except TypeError:
                try:
                    legacy = target.expertCompare(current_run, other_run, map_image)
                except Exception:
                    logging.error("Error while running reference comparison for %s", metric_name, exc_info=True)
                    return _err("Comparison failed unexpectedly. Check the engine logs.", label=metric_label)
        except Exception:
            logging.error("Error while running reference comparison for %s", metric_name, exc_info=True)
            return _err("Comparison failed unexpectedly. Check the engine logs.", label=metric_label)

        visualizations = _harvest_visualizations(legacy, metric_name, current_run, output_dir)

        explanation = ""
        if metric_obj is not None:
            try:
                explanation = metric_obj.explainComparison(current_score, other_score)
            except Exception:
                explanation = ""
        if not explanation:
            explanation = AbstractMetric.explainComparison(  # type: ignore[call-arg]
                metric_obj if metric_obj is not None else AbstractMetric({}),
                current_score,
                other_score,
            )

        payload = {
            "metric_id": metric_name,
            "label": metric_label,
            "current": {
                "score": current_score,
                "role": current_info.get("role"),
                "run_id": current_info.get("run_id"),
            },
            "other": {
                "score": other_score,
                "role": other_info.get("role"),
                "run_id": other_info.get("run_id"),
            },
            "visualizations": visualizations,
            "explanation": explanation,
        }
        return json.dumps(payload, indent=4)

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