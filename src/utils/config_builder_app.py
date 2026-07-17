"""
Config Builder UI (PyQt5) for Army ECR Battle Drill pipeline

Features
- Main page: POD + Boundary loaders (txt), point mapping / entry polys / map image pickers
- Main knobs: visual angle, threat interaction time, entry time threshold, hesitation threshold,
              hesitation threshold (2→3 entrants), POD working radius, POD capture threshold,
              per-POD time limits (dynamic), coverage time threshold, stay-along-wall
- Live preview: map image + boundary polygon + POD points (labeled) + working-radius circles + wall band
- Advanced page: all other config elements with defaults pre-filled
- Root folder picker: used for saving config.json and for storing relative paths when possible

IMPORTANT:
- Boundary, MapPath, point_mapping_path, entry_polys_path are REQUIRED and start empty. (POD is optional and may be empty.)
- Save is disabled until all required items + project root are selected.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

from PyQt5.QtCore import Qt, QPointF, QRectF, QSize
from PyQt5.QtGui import (
    QPixmap, QPen, QBrush, QColor, QPolygonF, QFont, QPainter, QPainterPath
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit, QGroupBox,
    QScrollArea, QSplitter, QFrame, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsEllipseItem,
    QGraphicsTextItem, QSizePolicy, QComboBox, QSpacerItem, QStyle,
    QGraphicsPathItem
)

from shapely.geometry import Polygon, MultiPolygon

# ----------------------------
# Defaults (canonical baseline)
# ----------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # REQUIRED core inputs (start empty)
    "POD": [],
    "Boundary": [],
    "point_mapping_path": "",
    "entry_polys_path": "",
    "MapPath": "",

    # models (advanced defaults). Tags resolve to in-repo giftpose builders;
    # the legacy ``libs/mmpose/...`` config-string paths are also accepted.
    "det_model": "rtmdet-m-person-640",
    "det_weights": "models/detect-best-mAP.pth",
    "det_cat_ids": [0],
    "pose2d_config": "rtmpose-x-halpe26-384x288",
    "pose2d_weights": "models/pose.pth",

    # pose backend selection (advanced defaults). See MODELS.md.
    "pose_backend": "body2d",
    "pose_model_size": "x",
    "auto_download_models": False,

    # thresholds (advanced defaults)
    "box_conf_threshold": 0.3,
    "pose_conf_threshold": 0.3,
    "flip_test": False,
    "compile_for_inference": False,

    # tracker / runtime (advanced defaults)
    "keypoint_indices": [15, 16],
    "device": "cpu",
    "boundary_pad_pct": 0.05,
    "track_enemy": True,
    "enemy_ids": [99],

    # unified knobs (main defaults)
    "visual_angle_degrees": 20.0,
    "min_threat_interaction_time_sec": 1.0,
    "entry_time_threshold_sec": 2.0,
    "team_size": 4,
    "HESITATION_THRESHOLD": 1.0,
    "HESITATION_THRESHOLD_SECOND": 2.0,

    # pod knobs (main defaults)
    "pod_working_radius": 40.0,
    "pod_capture_threshold_sec": 0.1,
    "pod_time_limits": [5.0],
    "pod_groups": ["A"],

    # coverage (main defaults)
    "coverage_time_threshold": 3.0,

    # gaze keypoints (advanced defaults)
    "gaze_keypoint_map": {"NOSE": 0, "LEYE": 1, "REYE": 2, "LEAR": 3, "REAR": 4},

    # optional comparison override
    "frame_rate": 30.0,

    # audio + transcription (advanced defaults)
    "preserve_audio": True,
    "enable_transcription": True,
    "transcription_device": "cpu",
    "enable_denoise": False,
    "denoise_device": "cpu",
    "asr_backend": "parakeet",
    "asr_stream_chunk_sec": 8.0,
    "asr_stream_confirm_sec": 3.0,
    # drill window auto-detection (advanced defaults)
    "drill_window_enabled": True,
    "drill_window_required_words": "room,clear",
    "stop_at_drill_end": True,
}

COMMENTS: Dict[str, str] = {
    "POD": "Designated areas/points used for POD assignment + capture analysis and POD map videos.",
    "Boundary": "Room boundary polygon; used to clamp/project mapped positions and for coverage/gaze computations.",
    "point_mapping_path": "Path to pixel→map coordinate mapping file (PixelMapper).",
    "entry_polys_path": "Path to entry-region polygons file (allows entry points near doors even if outside boundary).",
    "MapPath": "Path to the static room map image used for map-based overlays and coverage.",

    "det_model": "Person detector model config (MMDetection) used by the pose inferencer.",
    "det_weights": "Detector checkpoint weights file.",
    "det_cat_ids": "Detector category IDs to keep (typically [0] for person).",
    "pose2d_config": "2D pose model config (MMPose).",
    "pose2d_weights": "2D pose checkpoint weights file.",

    "pose_backend": "Pose stage backend: 'body2d' (default, 26-kp Halpe26), 'wholebody' (133-kp RTMW incl. face+hands), or 'pose3d' (RTMW3D, adds metric z per keypoint). Downstream tracking/metrics/overlays are identical for all backends via the canonical Halpe26 adapter. See MODELS.md.",
    "pose_model_size": "Model size for the selected backend. body2d: t/s/m/l/x ('x' = the project's fine-tuned models/pose.pth; other sizes are official checkpoints). wholebody: m/l/x. pose3d: l.",
    "auto_download_models": "When true, official checkpoints for non-default sizes/backends are downloaded automatically into models/ on first use. The fine-tuned default weights are never downloaded.",
    "pose_backend_weights": "Optional explicit weights file for a non-default backend (e.g. a future fine-tuned RTMW checkpoint). Overrides the registry download.",
    "transcription_preroll_sec": "Seconds of audio included BEFORE the detected drill start when the streaming transcription begins (default 5.0). Gives the ASR speech context ahead of the entry; end candidates are still restricted to segments starting at/after the drill start.",
    "box_conf_threshold": "Minimum bbox confidence to accept a detection.",
    "pose_conf_threshold": "Minimum keypoint confidence to accept pose keypoints and render gaze/triangles.",
    "flip_test": "When true, the pose model also runs on a horizontally-flipped copy of each crop and averages the two outputs (~0.5-1 px better keypoint accuracy on hard / occluded cases). Doubles the per-frame pose batch and roughly doubles pose forward time. Default false for speed; enable if accuracy matters more than fps.",
    "compile_for_inference": "When true, runs a one-time graph optimizer at backend construction (PyTorch: torch.compile; TorchScript: torch.jit.optimize_for_inference). Adds 30-120s of startup cost in exchange for faster steady-state inference. No effect on ONNX / TensorRT backends — those graphs are already optimized at session/engine build time. Default false; enable for long-running sessions where startup cost is amortized.",
    "keypoint_indices": "Which keypoints the tracker uses for keypoint-based positioning logic.",
    "device": "Compute device for inference (e.g., 'cpu', 'cuda', 'mps').",

    "boundary_pad_pct": "Extra padding around boundary used by tracker when validating positions.",
    "track_enemy": "Enable enemy tracking behaviors in the tracker.",

    "enemy_ids": "Track IDs considered enemies (used for fall detection, gaze/coverage filtering, threat clearance).",
    "visual_angle_degrees": "Full field-of-view angle (degrees) used for gaze triangles, map gaze/coverage, and threat-clearance.",
    "min_threat_interaction_time_sec": "Minimum interaction time (seconds) required to count a threat as cleared.",
    "entry_time_threshold_sec": "Max allowed team entry span (seconds) for full score in TOTAL_TIME_OF_ENTRY.",
    "team_size": "Number of friendly entrants the team-entry metrics evaluate (entrance hesitation, total entry time, entrance vectors). Independent of POD count: a room may have N PODs but a team of M people. When omitted, falls back to the number of POD points for backward compatibility with older saved configs.",
    "HESITATION_THRESHOLD": "No-penalty hesitation gap (seconds) used for all consecutive entrant pairs except the second pair. Doctrine allows some delay between entrants 2 and 3 in case of any adverse action in the room.",
    "HESITATION_THRESHOLD_SECOND": "Special no-penalty hesitation gap (seconds) used only for the gap between entrants 2→3. This second threshold is intentionally more permissive because doctrine allows some delay there in case of any adverse action in the room.",

    "pod_working_radius": "Radius (map pixels) around each POD used to compute work areas for POD capture analysis.",
    "pod_capture_threshold_sec": "Seconds required inside a POD work area to count as captured.",
    "pod_time_limits": "Per-POD time limit (seconds) to capture each designated area after entry. Interpreted relative to when the assigned entrant first enters/appears in the room.",
    "pod_groups": "Per-POD grouping label (A/B). Group PODs that belong to the same side/segment of the room to allow assignment to people moving either left or right from the entry",

    "coverage_time_threshold": "Seconds of sustained coverage needed for full score in TOTAL_FLOOR_COVERAGE_TIME.",

    "gaze_keypoint_map": "Keypoint indices (Halpe26) used to compute gaze direction (nose/eyes/ears).",
    "frame_rate": "(Optional) Override FPS used in comparisons; normally set automatically from video during processing.",

    # audio + transcription
    "preserve_audio": "When true, camera-view annotated videos keep the original audio track (muxed via ffmpeg). Map-view videos remain silent. Harmless if the source has no audio.",
    "enable_transcription": "When true (default), NeMo Parakeet-TDT transcribes the drill audio and saves a _Transcription.json sidecar with word-level timestamps. The first run downloads the Parakeet checkpoint (~2.4 GB) to the HuggingFace cache; subsequent runs reuse it.",
    "transcription_device": "Compute device for the Parakeet ASR. NeMo runs on cpu (or cuda where available); MPS is not a NeMo device, so on Apple Silicon the ASR runs on cpu while the pose pipeline uses MPS.",
    "enable_denoise": "When true, runs Facebook Denoiser (dns48 model, streaming DemucsStreamer) on the audio before ASR as an optional speech-enhancement pass. Improves ASR accuracy on noisy field audio. Only the transcription consumes the denoised audio — saved annotated videos always keep the original track. First use downloads a ~128 MB checkpoint to ~/.cache/torch/hub/checkpoints/.",
    "denoise_device": "Compute device for FB Denoiser. cpu or cuda only — Demucs's internal conv1d exceeds the MPS 65536-output-channel kernel limit; if mps is set the denoiser auto-falls-back to cpu with a warning. Independent of transcription_device. Ignored when enable_denoise is false.",
    "drill_window_enabled": "When true, enables auto-detection of drill start (first tracker entry crossing) and drill end (first confirmed transcript segment containing every word in drill_window_required_words). All metrics, artifact videos, and audio are trimmed to the detected window; transcription streams forward from the entry and stops once the end is confirmed. Default true.",
    "stop_at_drill_end": "When true (default), transcription runs in a background thread as soon as the first entry is detected; the frame loop stops as soon as the located drill-end frame is reached, skipping post-drill footage (faster, and prevents post-drill track fragments from being mistaken for entrants). If no drill end can be identified (no matching transcript segment), processing continues to the video end unchanged. Requires drill_window_enabled and enable_transcription; ignored otherwise. Set false to always process the full video.",
    "asr_backend": "Speech-to-text engine (NeMo Parakeet-TDT). Transcription is forward-chunked/streaming: it stops as soon as the 'room clear' phrase is confirmed (stream-compatible, no wasted tail, robust to a mid-conversation cut).",
    "asr_stream_chunk_sec": "Streaming ASR: seconds of audio added per forward chunk before re-checking for the drill-end phrase (default 8.0). Smaller = finer granularity, more transcription passes.",
    "asr_stream_confirm_sec": "Streaming ASR: shorter chunk (default 3.0s) read once a tentative 'room clear' appears, to confirm it via a second consecutive pass (LocalAgreement) while minimizing how far past the end we transcribe.",
    "drill_window_required_words": "Comma-separated list of words that must all appear in a transcript segment for it to qualify as the drill-end callout. Order doesn't matter, fillers between are fine, and matching is lowercase + punctuation-stripped. Default 'room,clear'.",

    # UI-only
    "project_root": "Project root folder used to save config.json and make selected paths relative when possible.",
}

_num_re = re.compile(r"[-+]?\d*\.?\d+")


def _to_int(x: float) -> int:
    return int(round(float(x)))


def parse_points_file_xy_lines(path: str) -> List[List[int]]:
    """Parses file with one point per line: 'x, y' or 'x y'. Ignores # comments and blanks."""
    pts: List[List[int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            nums = _num_re.findall(s)
            if len(nums) < 2:
                continue
            x, y = float(nums[0]), float(nums[1])
            pts.append([_to_int(x), _to_int(y)])
    if len(pts) < 1:
        raise ValueError("No valid points found.")
    return pts


def parse_boundary_file(path: str) -> List[List[int]]:
    """
    Accepts either:
    - one long line: x1,y1,x2,y2,...
    - or multiple lines with x,y
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    nums = _num_re.findall(text)
    if len(nums) >= 6 and len(nums) % 2 == 0:
        pts = []
        for i in range(0, len(nums), 2):
            pts.append([_to_int(float(nums[i])), _to_int(float(nums[i + 1]))])
        if len(pts) < 3:
            raise ValueError("Boundary must have at least 3 points.")
        return pts

    pts = parse_points_file_xy_lines(path)
    if len(pts) < 3:
        raise ValueError("Boundary must have at least 3 points.")
    return pts


# ----------------------------
# Model
# ----------------------------

@dataclass
class ConfigModel:
    data: Dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CONFIG)))
    project_root: str = ""

    def merge_from(self, other: Dict[str, Any]) -> None:
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        for k, v in other.items():
            if k == "_comments":
                continue
            merged[k] = v
        self.data = merged
        self._normalize()

    def _normalize(self) -> None:
        # POD + Boundary should be list[list[int]]
        for key in ("POD", "Boundary"):
            if key in self.data and isinstance(self.data[key], list):
                norm = []
                for pt in self.data[key]:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        norm.append([_to_int(pt[0]), _to_int(pt[1])])
                self.data[key] = norm

        # enemy_ids list[int]
        if "enemy_ids" in self.data:
            if not isinstance(self.data["enemy_ids"], list):
                self.data["enemy_ids"] = [99]
            else:
                ids = []
                for x in self.data["enemy_ids"]:
                    try:
                        ids.append(int(x))
                    except Exception:
                        pass
                self.data["enemy_ids"] = ids or [99]

        # pod_time_limits list[float] - auto-extend to POD count
        n_pods = len(self.data.get("POD", []))
        limits = self.data.get("pod_time_limits", [])
        if not isinstance(limits, list) or not limits:
            limits = [30.0]
        limits_f = []
        for x in limits:
            try:
                limits_f.append(float(x))
            except Exception:
                pass
        if not limits_f:
            limits_f = [30.0]
        if n_pods > 0:
            while len(limits_f) < n_pods:
                limits_f.append(limits_f[-1])
            if len(limits_f) > n_pods:
                limits_f = limits_f[:n_pods]
        self.data["pod_time_limits"] = limits_f

        # pod_groups list[str] - auto-extend to POD count
        groups = self.data.get("pod_groups", [])
        if not isinstance(groups, list) or not groups:
            groups = ["A"]
        groups_s: List[str] = []
        for g in groups:
            try:
                gs = str(g).strip().upper()
            except Exception:
                continue
            if gs not in ("A", "B"):
                gs = "A"
            groups_s.append(gs)
        if not groups_s:
            groups_s = ["A"]
        if n_pods > 0:
            while len(groups_s) < n_pods:
                groups_s.append(groups_s[-1])
            if len(groups_s) > n_pods:
                groups_s = groups_s[:n_pods]
        self.data["pod_groups"] = groups_s

        # gaze_keypoint_map
        gkm = self.data.get("gaze_keypoint_map", {})
        if not isinstance(gkm, dict):
            gkm = {}
        out = {"NOSE": 0, "LEYE": 1, "REYE": 2, "LEAR": 3, "REAR": 4}
        for k in list(out.keys()):
            if k in gkm:
                try:
                    out[k] = int(gkm[k])
                except Exception:
                    pass
        self.data["gaze_keypoint_map"] = out

    def to_json_dict(self, include_comments: bool = True) -> Dict[str, Any]:
        out = json.loads(json.dumps(self.data))
        if include_comments:
            out["_comments"] = COMMENTS
        return out

    def set_path(self, key: str, file_path: str) -> None:
        """Store a path in portable form using its anchor (see :data:`PATH_ANCHORS`).

        Config-anchored paths are stored relative to the chosen project root
        (the directory where the JSON will be saved). Project-anchored paths
        are stored relative to the repository root computed at import time.
        Anything that can't be made relative (different drive, no anchor set)
        is stored as the absolute form with forward-slash separators.
        """
        from .paths import PATH_ANCHORS, PROJECT_ROOT, relativize_path

        if not isinstance(file_path, str) or not file_path:
            self.data[key] = ""
            return

        anchor = PATH_ANCHORS.get(key, "config")
        if anchor == "project":
            base = PROJECT_ROOT
        else:
            base = self.project_root or PROJECT_ROOT

        if not base:
            self.data[key] = file_path.replace(os.sep, "/")
            return
        self.data[key] = relativize_path(file_path, base_dir=base)

    def resolve_path(self, key: str) -> str:
        """Inverse of :meth:`set_path`: returns an absolute path for display
        and on-disk lookups."""
        from .paths import PATH_ANCHORS, PROJECT_ROOT, resolve_config_path

        v = self.data.get(key, "")
        if not isinstance(v, str) or not v:
            return ""

        anchor = PATH_ANCHORS.get(key, "config")
        config_dir = self.project_root or PROJECT_ROOT
        return resolve_config_path(v, config_dir=config_dir, anchor=anchor)


# ----------------------------
# Preview widget (map + overlays)
# ----------------------------


class MapPreview(QGraphicsView):
    def __init__(self, model: ConfigModel, on_pod_moved: Optional[Callable[[int, float, float], None]] = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # Zoom config (view-only; does NOT affect saved scene coordinates)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._user_zoom: float = 1.0   # multiplier applied on top of fitInView
        self._min_zoom: float = 0.2
        self._max_zoom: float = 8.0

        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._boundary_item: Optional[QGraphicsPolygonItem] = None
        self._pod_items: List[QGraphicsEllipseItem] = []
        self._pod_labels: List[QGraphicsTextItem] = []
        self._radius_items: List[QGraphicsEllipseItem] = []
        self._pod_label_bgs: List[QGraphicsEllipseItem] = []
        self._on_pod_moved_cb = on_pod_moved

        self.refresh()

    def sizeHint(self) -> QSize:
        return QSize(700, 600)

    def wheelEvent(self, event):
        """Zoom the map/scene with mouse wheel or trackpad scroll.

        This only changes the view transform (camera) and does not modify any scene
        coordinates, so saved POD/Boundary values are unaffected.
        """
        # Use the vertical delta; on trackpads this can be smooth.
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        # Smooth zoom factor. (angleDelta is typically 120 per wheel notch)
        factor = 1.0015 ** float(delta)

        old = float(getattr(self, "_user_zoom", 1.0))
        new = max(self._min_zoom, min(self._max_zoom, old * factor))
        apply = new / old if old != 0 else 1.0

        self._user_zoom = new
        self.scale(apply, apply)
        event.accept()

    def reset_zoom(self) -> None:
        """Reset view zoom/pan to a clean fit-to-view."""
        self._user_zoom = 1.0
        # Clear any accumulated transforms (including pan) then refit.
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def refresh(self) -> None:
        self.scene.clear()
        self._pod_items.clear()
        self._pod_labels.clear()
        self._pod_label_bgs.clear()
        self._radius_items.clear()
        self._pix_item = None
        self._boundary_item = None

        # Map
        map_path = self.model.resolve_path("MapPath")
        pix = QPixmap(map_path) if map_path and os.path.exists(map_path) else QPixmap()
        if not pix.isNull():
            self._pix_item = self.scene.addPixmap(pix)
            self._pix_item.setZValue(0)
            self.scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        else:
            self.scene.setSceneRect(QRectF(0, 0, 700, 500))

        # Boundary
        boundary = self.model.data.get("Boundary", [])
        if isinstance(boundary, list) and len(boundary) >= 3:
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in boundary])

            # Strong boundary highlight (glow + main)
            glow_pen = QPen(QColor(0, 220, 255, 160), 6)
            glow_pen.setCosmetic(True)
            main_pen = QPen(QColor(255, 255, 255), 2)
            main_pen.setCosmetic(True)

            glow_item = QGraphicsPolygonItem(poly)
            glow_item.setPen(glow_pen)
            glow_item.setBrush(QBrush(Qt.NoBrush))
            glow_item.setZValue(8)
            self.scene.addItem(glow_item)

            self._boundary_item = QGraphicsPolygonItem(poly)
            self._boundary_item.setPen(main_pen)
            self._boundary_item.setBrush(QBrush(Qt.NoBrush))
            self._boundary_item.setZValue(10)
            self.scene.addItem(self._boundary_item)
            # Note: STAY_ALONG_WALL no longer uses a global band drawn here.
            # The runtime band is per-person (shoulder-to-elbow length scaled
            # to map units), so a static preview overlay would be misleading.

        # POD points + labels + radius circles
        pods = self.model.data.get("POD", [])
        r = float(self.model.data.get("pod_working_radius", 40.0))
        for i, pt in enumerate(pods if isinstance(pods, list) else []):
            if not (isinstance(pt, list) and len(pt) >= 2):
                continue
            x, y = float(pt[0]), float(pt[1])

            # radius circle
            circ = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            circ_pen = QPen(QColor(255, 215, 0), 2)
            circ_pen.setCosmetic(True)
            circ.setPen(circ_pen)
            circ.setBrush(QBrush(QColor(255, 215, 0, 20)))
            circ.setZValue(6)
            self.scene.addItem(circ)
            self._radius_items.append(circ)

            # point (draggable)
            def _moved(idx: int, nx: float, ny: float):
                # Update model
                pods_local = self.model.data.get("POD", [])
                if isinstance(pods_local, list) and 0 <= idx < len(pods_local):
                    pods_local[idx] = [int(round(nx)), int(round(ny))]
                    self.model.data["POD"] = pods_local

                # Update linked visuals: radius circle + label
                rr = float(self.model.data.get("pod_working_radius", 40.0))
                if 0 <= idx < len(self._radius_items):
                    self._radius_items[idx].setRect(nx - rr, ny - rr, 2 * rr, 2 * rr)
                if 0 <= idx < len(self._pod_labels):
                    self._pod_labels[idx].setPos(nx + 6, ny - 14)
                    # Keep label background rectangle aligned with text
                    if 0 <= idx < len(self._pod_label_bgs):
                        br = self._pod_labels[idx].boundingRect()
                        pad_x, pad_y = 3.0, 1.5
                        self._pod_label_bgs[idx].setRect(
                            (nx + 6) + br.x() - pad_x,
                            (ny - 14) + br.y() - pad_y,
                            br.width() + 2 * pad_x,
                            br.height() + 2 * pad_y,
                        )

                # Bubble up to window so it can rewrite the POD txt file
                if self._on_pod_moved_cb is not None:
                    self._on_pod_moved_cb(idx, nx, ny)

            dot = DraggablePodItem(i, QPointF(x, y), 4.0, _moved)
            self.scene.addItem(dot)
            self._pod_items.append(dot)

            # label (more visible: background box + high-contrast text)
            t = QGraphicsTextItem(str(i))
            t.setDefaultTextColor(QColor(0, 0, 0))
            t.setFont(QFont("Arial", 12, QFont.Bold))
            t.setPos(x + 6, y - 16)
            t.setZValue(9)
            self.scene.addItem(t)
            self._pod_labels.append(t)

            # Background rectangle behind label text for readability on light maps
            br = t.boundingRect()
            pad_x, pad_y = 3.0, 1.5
            bg = QGraphicsEllipseItem(
                (x + 6) + br.x() - pad_x,
                (y - 16) + br.y() - pad_y,
                br.width() + 2 * pad_x,
                br.height() + 2 * pad_y,
            )
            bg.setPen(QPen(QColor(0, 0, 0, 180), 1))
            bg.setBrush(QBrush(QColor(255, 255, 255, 200)))
            bg.setZValue(8)  # behind the text label
            self.scene.addItem(bg)
            self._pod_label_bgs.append(bg)

        # Always fit first, then re-apply any user zoom so refreshes don't reset zoom.
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        if getattr(self, "_user_zoom", 1.0) != 1.0:
            self.scale(self._user_zoom, self._user_zoom)

# --- Draggable POD item ---

from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPen, QBrush, QColor

class DraggablePodItem(QGraphicsEllipseItem):
    """A draggable POD point dot that updates the underlying model when moved."""

    def __init__(self, pod_index: int, center: QPointF, radius_px: float, on_moved: Callable[[int, float, float], None]):
        super().__init__(center.x() - radius_px, center.y() - radius_px, 2 * radius_px, 2 * radius_px)
        self.pod_index = pod_index
        self._r = radius_px
        self._on_moved = on_moved
        self.setZValue(7)
        self.setPen(QPen(Qt.NoPen))
        self.setBrush(QBrush(QColor(255, 140, 0)))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.OpenHandCursor)

    def _center(self) -> QPointF:
        # Use scene coordinates so it reflects the dragged position
        return self.sceneBoundingRect().center()

    def mousePressEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setCursor(Qt.OpenHandCursor)
        c = self._center()
        try:
            self._on_moved(self.pod_index, float(c.x()), float(c.y()))
        except Exception:
            pass

    def itemChange(self, change, value):
        return super().itemChange(change, value)


# ----------------------------
# Utility widgets
# ----------------------------

class FilePicker(QWidget):
    def __init__(self, title: str, on_pick, filter_str: str = "All Files (*)", required: bool = False, parent=None):
        super().__init__(parent)
        self.on_pick = on_pick
        self.filter_str = filter_str
        self.required = required
        self.base_title = title

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(title)
        self.label.setMinimumWidth(140)
        self.path = QLineEdit()
        self.path.setReadOnly(True)
        self.btn = QPushButton("Select…")
        self.btn.clicked.connect(self._pick)

        layout.addWidget(self.label)
        layout.addWidget(self.path, 1)
        layout.addWidget(self.btn)

        self.set_required_missing(self.required)

    def _pick(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select file", "", self.filter_str)
        if fp:
            self.on_pick(fp)

    def set_required_missing(self, missing: bool):
        if self.required and missing:
            self.label.setText(f"{self.base_title} *")
            self.label.setStyleSheet("QLabel { color: #ff6666; font-weight: 600; }")
        else:
            self.label.setText(self.base_title)
            self.label.setStyleSheet("")

    def set_path(self, p: str):
        self.path.setText(p)
        self.set_required_missing(not bool(p))


# ----------------------------
# Main Window
# ----------------------------

class ConfigBuilderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Battle Drill Config Builder")
        self.resize(1200, 750)

        self.model = ConfigModel()
        self._pod_points_src_path = ""

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)

        # Top bar: project root + load/save
        top = QHBoxLayout()
        root_layout.addLayout(top)

        self.root_path = QLineEdit()
        self.root_path.setPlaceholderText("Select project root folder (used for saving + relative paths)…")
        self.root_path.setReadOnly(True)

        self.btn_root = QPushButton("Set Root…")
        self.btn_root.clicked.connect(self.pick_root)

        self.btn_load = QPushButton("Load config.json…")
        self.btn_load.clicked.connect(self.load_config)

        self.btn_save = QPushButton("Save config.json…")
        self.btn_save.clicked.connect(self.save_config)
        self.btn_save.setEnabled(False)  # enabled only when required fields are ready
        
        self.btn_reset = QPushButton("New / Reset")
        self.btn_reset.setToolTip("Reset all fields to start a new config")
        self.btn_reset.clicked.connect(self.reset_config)

        top.addWidget(QLabel("Project Root:"))
        top.addWidget(self.root_path, 1)
        top.addWidget(self.btn_root)
        top.addSpacing(10)
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_reset)
        top.addWidget(self.btn_save)

        # Tabs
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)

        self.main_tab = QWidget()
        self.adv_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.adv_tab, "Advanced")

        self._build_main_tab()
        self._build_advanced_tab()

        # initial populate
        self.refresh_all()

        self.statusBar().showMessage("Select required core inputs to enable saving.")
        self._update_ready_state()

    # ---------- Root / Load / Save ----------

    def pick_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Project Root")
        if d:
            self.model.project_root = d
            self.root_path.setText(d)
            self.refresh_paths()
            self._update_ready_state()

    def load_config(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Load config.json", "", "JSON Files (*.json);;All Files (*)")
        if not fp:
            return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.model.merge_from(data)
            if not self.model.project_root:
                self.model.project_root = os.path.dirname(fp)
                self.root_path.setText(self.model.project_root)
            self.refresh_all()
            self._update_ready_state()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load config:\n\n{e}")

    def save_config(self):
        if not self.model.project_root:
            QMessageBox.warning(self, "Missing Root", "Please set a Project Root folder first.")
            return
        if len(self.model.data.get("Boundary", [])) < 3:
            QMessageBox.warning(self, "Invalid Boundary", "Boundary must have at least 3 points.")
            return
        if not self.model.data.get("MapPath", ""):
            QMessageBox.warning(self, "Missing MapPath", "Please select a Map image.")
            return
        if not self.model.data.get("point_mapping_path", ""):
            QMessageBox.warning(self, "Missing point_mapping_path", "Please select point mapping file.")
            return
        if not self.model.data.get("entry_polys_path", ""):
            QMessageBox.warning(self, "Missing entry_polys_path", "Please select entry polys file.")
            return

        default_name = os.path.join(self.model.project_root, "config.json")
        fp, _ = QFileDialog.getSaveFileName(self, "Save config.json", default_name, "JSON Files (*.json)")
        if not fp:
            return

        try:
            from .paths import PATH_ANCHORS, PROJECT_ROOT, relativize_path

            save_dir = os.path.dirname(os.path.abspath(fp))
            out = self.model.to_json_dict(include_comments=True)

            # Two-step: resolve every path key to absolute (uses the model's
            # current anchor knowledge), then re-relativize against the actual
            # save location. This guarantees portability regardless of where
            # the user pointed `project_root` while editing.
            for key, anchor in PATH_ANCHORS.items():
                if key not in out:
                    continue
                abs_p = self.model.resolve_path(key)
                if not abs_p:
                    continue
                base = save_dir if anchor == "config" else PROJECT_ROOT
                out[key] = relativize_path(abs_p, base_dir=base)

            with open(fp, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            QMessageBox.information(self, "Saved", f"Saved config to:\n{fp}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save config:\n\n{e}")
            
    def reset_config(self):
        # Preserve project root (so you can keep saving into the same project)
        keep_root = self.model.project_root

        # Reset model to defaults
        self.model.data = json.loads(json.dumps(DEFAULT_CONFIG))
        self.model._normalize()
        self.model.project_root = keep_root

        # Clear POD file source path so drags don't rewrite an old file
        self._pod_points_src_path = ""

        # Clear visible file picker paths
        self.pod_file_picker.set_path("")
        self.boundary_file_picker.set_path("")
        self.map_picker.set_path("")
        self.point_map_picker.set_path("")
        self.entry_polys_picker.set_path("")

        # Rebuild UI from model defaults
        self.refresh_all()
        self._update_ready_state()
        self.statusBar().showMessage("Reset to defaults. Select required inputs to save.")

    # ---------- Readiness state ----------

    def _update_ready_state(self):
        has_pod = isinstance(self.model.data.get("POD"), list) and len(self.model.data.get("POD")) > 0
        has_boundary = isinstance(self.model.data.get("Boundary"), list) and len(self.model.data.get("Boundary")) >= 3

        map_ok = bool(self.model.data.get("MapPath", ""))
        pm_ok = bool(self.model.data.get("point_mapping_path", ""))
        ep_ok = bool(self.model.data.get("entry_polys_path", ""))

        self.pod_file_picker.set_required_missing(False)
        self.boundary_file_picker.set_required_missing(not has_boundary)
        self.map_picker.set_required_missing(not map_ok)
        self.point_map_picker.set_required_missing(not pm_ok)
        self.entry_polys_picker.set_required_missing(not ep_ok)

        ready = has_boundary and map_ok and pm_ok and ep_ok and bool(self.model.project_root)
        self.btn_save.setEnabled(ready)

        if not self.model.project_root:
            msg = "Set Project Root to enable saving."
        elif not has_boundary:
            msg = "Load Boundary file (*.txt)."
        elif not map_ok:
            msg = "Select Map image."
        elif not pm_ok:
            msg = "Select point_mapping_path."
        elif not ep_ok:
            msg = "Select entry_polys_path."
        else:
            msg = "Ready to save config.json"
        self.statusBar().showMessage(msg)

    # ---------- Main tab UI ----------

    def _build_main_tab(self):
        layout = QHBoxLayout(self.main_tab)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        core_group = QGroupBox("Core Inputs (Required)")
        core_form = QVBoxLayout(core_group)

        self.pod_file_picker = FilePicker(
            "POD points file:",
            self.pick_pod_points,
            filter_str="Text Files (*.txt);;All Files (*)",
            required=False
        )
        core_form.addWidget(self.pod_file_picker)

        self.boundary_file_picker = FilePicker(
            "Boundary file:",
            self.pick_boundary,
            filter_str="Text Files (*.txt);;All Files (*)",
            required=True
        )
        core_form.addWidget(self.boundary_file_picker)

        self.map_picker = FilePicker(
            "Map image:",
            lambda p: self._set_path_and_refresh("MapPath", p),
            filter_str="Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            required=True
        )
        core_form.addWidget(self.map_picker)

        self.point_map_picker = FilePicker(
            "Point mapping:",
            lambda p: self._set_path_and_refresh("point_mapping_path", p),
            filter_str="Text Files (*.txt);;All Files (*)",
            required=True
        )
        core_form.addWidget(self.point_map_picker)

        self.entry_polys_picker = FilePicker(
            "Entry polys:",
            lambda p: self._set_path_and_refresh("entry_polys_path", p),
            filter_str="Text Files (*.txt);;All Files (*)",
            required=True
        )
        core_form.addWidget(self.entry_polys_picker)

        left_layout.addWidget(core_group)

        knobs_group = QGroupBox("Main Metric & Behavior Settings")
        knobs_layout = QGridLayout(knobs_group)
        knobs_layout.setHorizontalSpacing(12)
        knobs_layout.setVerticalSpacing(8)

        row = 0

        self.spin_visual_angle = self._mk_dspin("visual_angle_degrees", 0.0, 180.0, 0.5, "deg")
        knobs_layout.addWidget(self._mk_label_btn("Visual angle", "visual_angle_degrees"), row, 0)
        knobs_layout.addWidget(self.spin_visual_angle, row, 1)
        knobs_layout.addWidget(QLabel("degrees"), row, 2)
        row += 1

        self.spin_threat_time = self._mk_dspin("min_threat_interaction_time_sec", 0.0, 30.0, 0.05, "sec")
        knobs_layout.addWidget(self._mk_label_btn("Threat min time", "min_threat_interaction_time_sec"), row, 0)
        knobs_layout.addWidget(self.spin_threat_time, row, 1)
        knobs_layout.addWidget(QLabel("seconds"), row, 2)
        row += 1

        self.spin_entry_time = self._mk_dspin("entry_time_threshold_sec", 0.0, 30.0, 0.1, "sec")
        knobs_layout.addWidget(self._mk_label_btn("Entry time threshold", "entry_time_threshold_sec"), row, 0)
        knobs_layout.addWidget(self.spin_entry_time, row, 1)
        knobs_layout.addWidget(QLabel("seconds"), row, 2)
        row += 1

        self.spin_team_size = QSpinBox()
        self.spin_team_size.setRange(1, 32)
        self.spin_team_size.setSingleStep(1)
        self.spin_team_size.valueChanged.connect(
            lambda v: self._on_value_changed("team_size", int(v))
        )
        self.spin_team_size.editingFinished.connect(lambda: self.show_description("team_size"))
        self.spin_team_size.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        knobs_layout.addWidget(self._mk_label_btn("Team size", "team_size"), row, 0)
        knobs_layout.addWidget(self.spin_team_size, row, 1)
        knobs_layout.addWidget(QLabel("entrants"), row, 2)
        row += 1

        self.spin_hesitation = self._mk_dspin("HESITATION_THRESHOLD", 0.0, 30.0, 0.1, "sec")
        knobs_layout.addWidget(self._mk_label_btn("Hesitation threshold", "HESITATION_THRESHOLD"), row, 0)
        knobs_layout.addWidget(self.spin_hesitation, row, 1)
        knobs_layout.addWidget(QLabel("seconds"), row, 2)
        row += 1

        self.spin_hesitation_second = self._mk_dspin("HESITATION_THRESHOLD_SECOND", 0.0, 30.0, 0.1, "sec")
        knobs_layout.addWidget(self._mk_label_btn("Hesitation threshold (2→3 entrants)", "HESITATION_THRESHOLD_SECOND"), row, 0)
        knobs_layout.addWidget(self.spin_hesitation_second, row, 1)
        knobs_layout.addWidget(QLabel("seconds"), row, 2)
        row += 1

        self.spin_pod_radius = self._mk_dspin("pod_working_radius", 0.0, 5000.0, 1.0, "px")
        self.spin_pod_radius.valueChanged.connect(lambda _: self.preview.refresh())
        knobs_layout.addWidget(self._mk_label_btn("POD working radius", "pod_working_radius"), row, 0)
        knobs_layout.addWidget(self.spin_pod_radius, row, 1)
        knobs_layout.addWidget(QLabel("map pixels"), row, 2)
        row += 1

        self.spin_pod_capture = self._mk_dspin("pod_capture_threshold_sec", 0.0, 30.0, 0.05, "sec")
        knobs_layout.addWidget(self._mk_label_btn("POD capture threshold", "pod_capture_threshold_sec"), row, 0)
        knobs_layout.addWidget(self.spin_pod_capture, row, 1)
        knobs_layout.addWidget(QLabel("seconds"), row, 2)
        row += 1

        self.spin_coverage_time = self._mk_dspin("coverage_time_threshold", 0.0, 120.0, 0.1, "sec")
        knobs_layout.addWidget(self._mk_label_btn("Coverage time threshold", "coverage_time_threshold"), row, 0)
        knobs_layout.addWidget(self.spin_coverage_time, row, 1)
        knobs_layout.addWidget(QLabel("seconds"), row, 2)
        row += 1

        # STAY_ALONG_WALL no longer exposes a knob: the band is per-person,
        # derived from each entrant's own shoulder-to-elbow length at runtime.

        limits_box = QGroupBox("Per-POD Time Limits (seconds)")
        limits_v = QVBoxLayout(limits_box)

        # Header row with Info button for the entire Per-POD component
        limits_header = QWidget()
        lh = QHBoxLayout(limits_header)
        lh.setContentsMargins(0, 0, 0, 0)
        lh.setSpacing(6)
        header_lbl = QLabel("Time limits & grouping")
        header_lbl.setStyleSheet("QLabel { font-weight: 600; }")
        info_btn = QPushButton()
        info_btn.setToolTip("Info")
        info_btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        info_btn.setIconSize(QSize(14, 14))
        info_btn.setFixedSize(26, 22)
        info_btn.clicked.connect(lambda: self.show_description("pod_time_limits"))
        lh.addWidget(header_lbl)
        lh.addStretch(1)
        lh.addWidget(info_btn)
        limits_v.addWidget(limits_header)

        self.pod_limits_container = QWidget()
        self.pod_limits_layout = QVBoxLayout(self.pod_limits_container)
        self.pod_limits_layout.setContentsMargins(0, 0, 0, 0)
        self.pod_limits_layout.setSpacing(6)

        limits_scroll = QScrollArea()
        limits_scroll.setWidgetResizable(True)
        limits_scroll.setFrameShape(QFrame.NoFrame)
        limits_scroll.setWidget(self.pod_limits_container)
        limits_scroll.setMinimumHeight(160)
        limits_v.addWidget(limits_scroll)

        left_layout.addWidget(knobs_group)
        left_layout.addWidget(limits_box)
        left_layout.addStretch(1)

        self.preview = MapPreview(self.model, on_pod_moved=self._on_pod_moved)
        # Preview controls
        preview_controls = QWidget()
        pc = QHBoxLayout(preview_controls)
        pc.setContentsMargins(0, 0, 0, 0)
        pc.setSpacing(6)
        pc.addStretch(1)
        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.setToolTip("Fit map to view and reset zoom")
        btn_reset_zoom.clicked.connect(self.preview.reset_zoom)
        pc.addWidget(btn_reset_zoom)

        right_layout.addWidget(preview_controls)
        right_layout.addWidget(self.preview, 1)

        desc_group = QGroupBox("Description (what this setting does)")
        desc_layout = QVBoxLayout(desc_group)
        self.desc = QTextEdit()
        self.desc.setReadOnly(True)
        self.desc.setMinimumHeight(140)
        self.desc.setPlaceholderText("Select a setting to see its purpose…")
        desc_layout.addWidget(self.desc)
        right_layout.addWidget(desc_group)

        self._rebuild_pod_time_limits()

    def _mk_label_btn(self, label: str, key: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label + ":")
        lbl.setStyleSheet("QLabel { font-weight: 600; }")
        btn = QPushButton()
        btn.setToolTip("Info")
        btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        btn.setIconSize(QSize(14, 14))
        btn.setFixedSize(26, 22)
        btn.clicked.connect(lambda: self.show_description(key))
        h.addWidget(lbl)
        h.addStretch(1)
        h.addWidget(btn)
        return w

    def show_description(self, key: str):
        txt = COMMENTS.get(key, "(No description available)")
        if key == "pod_time_limits":
            grp_txt = COMMENTS.get("pod_groups", "")
            if grp_txt:
                txt = f"{txt}\n\nGrouping:\n{grp_txt}"
        self.desc.setPlainText(f"{key}\n\n{txt}")

    def _mk_dspin(self, key: str, mn: float, mx: float, step: float, suffix: str) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(mn, mx)
        sp.setDecimals(3 if step < 0.1 else 2 if step < 1 else 1)
        sp.setSingleStep(step)
        sp.setSuffix(f" {suffix}" if suffix else "")
        sp.valueChanged.connect(lambda v, k=key: self._on_value_changed(k, float(v)))
        sp.editingFinished.connect(lambda k=key: self.show_description(k))
        sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return sp

    def _on_value_changed(self, key: str, value: Any):
        self.model.data[key] = value
        if key == "pod_working_radius":
            self.preview.refresh()

    def pick_pod_points(self, file_path: str):
        try:
            pts = parse_points_file_xy_lines(file_path)
            self.model.data["POD"] = pts
            self.model._normalize()
            self.pod_file_picker.set_path(file_path)
            self._rebuild_pod_time_limits()
            self.preview.refresh()
            self.show_description("POD")
            self._pod_points_src_path = file_path
            self._update_ready_state()
        except Exception as e:
            QMessageBox.critical(self, "POD Parse Error", f"Failed to parse POD points:\n\n{e}")

    def pick_boundary(self, file_path: str):
        try:
            pts = parse_boundary_file(file_path)
            self.model.data["Boundary"] = pts
            self.boundary_file_picker.set_path(file_path)
            self.preview.refresh()
            self.show_description("Boundary")
            self._update_ready_state()
        except Exception as e:
            QMessageBox.critical(self, "Boundary Parse Error", f"Failed to parse Boundary:\n\n{e}")

    def _set_path_and_refresh(self, key: str, file_path: str):
        self.model.set_path(key, file_path)
        self.refresh_paths()
        if key == "MapPath":
            self.preview.refresh()
        self.show_description(key)
        self._update_ready_state()

    def _rebuild_pod_time_limits(self):
        while self.pod_limits_layout.count():
            item = self.pod_limits_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        pods = self.model.data.get("POD", [])
        n = len(pods) if isinstance(pods, list) else 0

        limits = self.model.data.get("pod_time_limits", [])
        if not isinstance(limits, list) or not limits:
            limits = [30.0]

        if n > 0:
            lf = []
            for x in limits:
                try:
                    lf.append(float(x))
                except Exception:
                    pass
            if not lf:
                lf = [30.0]
            while len(lf) < n:
                lf.append(lf[-1])
            lf = lf[:n]
            self.model.data["pod_time_limits"] = lf
            limits = lf

        # Ensure pod_groups exists and matches POD count
        groups = self.model.data.get("pod_groups", [])
        if not isinstance(groups, list) or not groups:
            groups = ["A"]
        gf: List[str] = []
        for g in groups:
            gs = str(g).strip().upper()
            if gs not in ("A", "B"):
                gs = "A"
            gf.append(gs)
        if not gf:
            gf = ["A"]
        if n > 0:
            while len(gf) < n:
                gf.append(gf[-1])
            gf = gf[:n]
        self.model.data["pod_groups"] = gf
        groups = gf

        self._pod_limit_spins: List[QDoubleSpinBox] = []
        self._pod_group_combos: List[QComboBox] = []
        for i in range(max(n, 1)):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            label = QLabel(f"POD {i} limit:")
            label.setMinimumWidth(120)
            grp_label = QLabel("Group:")
            grp_label.setMinimumWidth(55)
            grp = QComboBox()
            grp.addItems(["A", "B"])
            gval = str(groups[i]) if (i < len(groups)) else (str(groups[-1]) if groups else "A")
            gidx = grp.findText(gval)
            if gidx >= 0:
                grp.setCurrentIndex(gidx)
            grp.currentTextChanged.connect(lambda t, idx=i: self._on_pod_group_changed(idx, t))
            grp.activated.connect(lambda _=None, k="pod_time_limits": self.show_description(k))
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 600.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.1)
            sp.setSuffix(" sec")
            val = float(limits[i]) if i < len(limits) else float(limits[-1]) if limits else 30.0
            sp.setValue(val)
            sp.valueChanged.connect(lambda v, idx=i: self._on_pod_limit_changed(idx, float(v)))
            sp.editingFinished.connect(lambda k="pod_time_limits": self.show_description(k))
            h.addWidget(label)
            h.addWidget(grp_label)
            h.addWidget(grp)
            h.addWidget(sp, 1)
            self.pod_limits_layout.addWidget(row)
            self._pod_limit_spins.append(sp)
            self._pod_group_combos.append(grp)

        self.pod_limits_layout.addStretch(1)

    def _on_pod_limit_changed(self, idx: int, value: float):
        limits = self.model.data.get("pod_time_limits", [])
        if not isinstance(limits, list):
            limits = []
        while len(limits) <= idx:
            limits.append(limits[-1] if limits else 30.0)
        limits[idx] = float(value)
        self.model.data["pod_time_limits"] = limits

    def _on_pod_group_changed(self, idx: int, value: str):
        groups = self.model.data.get("pod_groups", [])
        if not isinstance(groups, list):
            groups = []
        while len(groups) <= idx:
            groups.append(groups[-1] if groups else "A")
        v = str(value).strip().upper()
        if v not in ("A", "B"):
            v = "A"
        groups[idx] = v
        self.model.data["pod_groups"] = groups
        # Keep model consistent
        self.model._normalize()

    def refresh_paths(self):
        self.map_picker.set_path(self.model.resolve_path("MapPath"))
        self.point_map_picker.set_path(self.model.resolve_path("point_mapping_path"))
        self.entry_polys_picker.set_path(self.model.resolve_path("entry_polys_path"))

    def refresh_all(self):
        self.spin_visual_angle.setValue(float(self.model.data.get("visual_angle_degrees", 20.0)))
        self.spin_threat_time.setValue(float(self.model.data.get("min_threat_interaction_time_sec", 1.0)))
        self.spin_entry_time.setValue(float(self.model.data.get("entry_time_threshold_sec", 2.0)))
        # team_size falls back to len(POD) on the fly so legacy configs (which
        # never wrote the key) still populate something sensible.
        ts_default = len(self.model.data.get("POD", []) or []) or 4
        self.spin_team_size.setValue(int(self.model.data.get("team_size", ts_default) or ts_default))
        self.spin_hesitation.setValue(float(self.model.data.get("HESITATION_THRESHOLD", 1.0)))
        self.spin_hesitation_second.setValue(float(self.model.data.get("HESITATION_THRESHOLD_SECOND", 2.0)))
        self.spin_pod_radius.setValue(float(self.model.data.get("pod_working_radius", 40.0)))
        self.spin_pod_capture.setValue(float(self.model.data.get("pod_capture_threshold_sec", 0.1)))
        self.spin_coverage_time.setValue(float(self.model.data.get("coverage_time_threshold", 3.0)))

        self.refresh_paths()
        self._rebuild_pod_time_limits()
        self.preview.refresh()
        self._populate_advanced_fields()
        self._update_ready_state()

    # ---------- Advanced tab UI (unchanged behavior; defaults prefilled) ----------

    def _build_advanced_tab(self):
        adv_layout = QVBoxLayout(self.adv_tab)
        adv_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        adv_layout.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        self.adv_form = QFormLayout(content)
        self.adv_form.setLabelAlignment(Qt.AlignRight)
        self.adv_form.setHorizontalSpacing(16)
        self.adv_form.setVerticalSpacing(10)

        self.adv_widgets: Dict[str, QWidget] = {}

        self.adv_widgets["det_model"] = self._adv_file_row("det_model", "Python Files (*.py);;All Files (*)")
        self.adv_widgets["det_weights"] = self._adv_file_row("det_weights", "Model Files (*.pth *.pt);;All Files (*)")
        self.adv_widgets["pose2d_config"] = self._adv_file_row("pose2d_config", "Python Files (*.py);;All Files (*)")
        self.adv_widgets["pose2d_weights"] = self._adv_file_row("pose2d_weights", "Model Files (*.pth *.pt);;All Files (*)")

        self.adv_widgets["det_cat_ids"] = self._adv_text_row("det_cat_ids", placeholder="e.g., 0 or 0,1")
        self.adv_widgets["box_conf_threshold"] = self._adv_dspin_row("box_conf_threshold", 0.0, 1.0, 0.01)
        self.adv_widgets["pose_conf_threshold"] = self._adv_dspin_row("pose_conf_threshold", 0.0, 1.0, 0.01)
        self.adv_widgets["keypoint_indices"] = self._adv_text_row("keypoint_indices", placeholder="e.g., 15,16")

        chk_flip = QCheckBox("Enable")
        chk_flip.stateChanged.connect(
            lambda s: self._set_adv_value("flip_test", bool(s == Qt.Checked))
        )
        chk_flip.clicked.connect(lambda: self.show_description("flip_test"))
        self.adv_widgets["flip_test"] = chk_flip
        self._add_adv_row("flip_test", chk_flip)

        chk_compile = QCheckBox("Enable")
        chk_compile.stateChanged.connect(
            lambda s: self._set_adv_value("compile_for_inference", bool(s == Qt.Checked))
        )
        chk_compile.clicked.connect(lambda: self.show_description("compile_for_inference"))
        self.adv_widgets["compile_for_inference"] = chk_compile
        self._add_adv_row("compile_for_inference", chk_compile)

        dev = QComboBox()
        dev.addItems(["cpu", "cuda", "mps"])
        dev.currentTextChanged.connect(lambda t: self._set_adv_value("device", t))
        dev.activated.connect(lambda _: self.show_description("device"))
        self.adv_widgets["device"] = dev
        self._add_adv_row("device", dev)

        self.adv_widgets["boundary_pad_pct"] = self._adv_dspin_row("boundary_pad_pct", 0.0, 0.5, 0.01)

        chk = QCheckBox("Enable")
        chk.stateChanged.connect(lambda s: self._set_adv_value("track_enemy", bool(s == Qt.Checked)))
        chk.clicked.connect(lambda: self.show_description("track_enemy"))
        self.adv_widgets["track_enemy"] = chk
        self._add_adv_row("track_enemy", chk)

        self.adv_widgets["enemy_ids"] = self._adv_text_row("enemy_ids", placeholder="e.g., 99 or 99,100")
        self.adv_widgets["gaze_keypoint_map"] = self._adv_gaze_map_row()
        self.adv_widgets["frame_rate"] = self._adv_dspin_row("frame_rate", 1.0, 240.0, 1.0)

        # --- Audio & Transcription (Phase 3 / 5 additions) ---
        chk_audio = QCheckBox("Preserve audio in camera-view videos")
        chk_audio.stateChanged.connect(
            lambda s: self._set_adv_value("preserve_audio", bool(s == Qt.Checked))
        )
        chk_audio.clicked.connect(lambda: self.show_description("preserve_audio"))
        self.adv_widgets["preserve_audio"] = chk_audio
        self._add_adv_row("preserve_audio", chk_audio)

        chk_txn = QCheckBox("Run speech transcription (drill-end detection)")
        chk_txn.stateChanged.connect(
            lambda s: self._set_adv_value("enable_transcription", bool(s == Qt.Checked))
        )
        chk_txn.clicked.connect(lambda: self.show_description("enable_transcription"))
        self.adv_widgets["enable_transcription"] = chk_txn
        self._add_adv_row("enable_transcription", chk_txn)

        txn_device = QComboBox()
        txn_device.addItems(["cpu", "cuda"])
        txn_device.currentTextChanged.connect(
            lambda t: self._set_adv_value("transcription_device", t)
        )
        txn_device.activated.connect(lambda _: self.show_description("transcription_device"))
        self.adv_widgets["transcription_device"] = txn_device
        self._add_adv_row("transcription_device", txn_device)

        chk_denoise = QCheckBox("Run FB Denoiser before transcription (dns64)")
        chk_denoise.stateChanged.connect(
            lambda s: self._set_adv_value("enable_denoise", bool(s == Qt.Checked))
        )
        chk_denoise.clicked.connect(lambda: self.show_description("enable_denoise"))
        self.adv_widgets["enable_denoise"] = chk_denoise
        self._add_adv_row("enable_denoise", chk_denoise)

        denoise_device = QComboBox()
        denoise_device.addItems(["cpu", "cuda"])
        denoise_device.currentTextChanged.connect(
            lambda t: self._set_adv_value("denoise_device", t)
        )
        denoise_device.activated.connect(lambda _: self.show_description("denoise_device"))
        self.adv_widgets["denoise_device"] = denoise_device
        self._add_adv_row("denoise_device", denoise_device)

        chk_drill = QCheckBox("Auto-detect drill window (start = first entry, end = clearance callout)")
        chk_drill.stateChanged.connect(
            lambda s: self._set_adv_value("drill_window_enabled", bool(s == Qt.Checked))
        )
        chk_drill.clicked.connect(lambda: self.show_description("drill_window_enabled"))
        self.adv_widgets["drill_window_enabled"] = chk_drill
        self._add_adv_row("drill_window_enabled", chk_drill)

        drill_words = QLineEdit()
        drill_words.setPlaceholderText("comma-separated; default room,clear")
        drill_words.editingFinished.connect(
            lambda: self._set_adv_value(
                "drill_window_required_words", drill_words.text().strip() or "room,clear"
            )
        )
        drill_words.textChanged.connect(lambda _: self.show_description("drill_window_required_words"))
        self.adv_widgets["drill_window_required_words"] = drill_words
        self._add_adv_row("drill_window_required_words", drill_words)

        desc_group = QGroupBox("Description")
        v = QVBoxLayout(desc_group)
        self.adv_desc = QTextEdit()
        self.adv_desc.setReadOnly(True)
        self.adv_desc.setMinimumHeight(120)
        v.addWidget(self.adv_desc)
        adv_layout.addWidget(desc_group)

        self.tabs.currentChanged.connect(self._tab_changed)

    def _tab_changed(self, idx: int):
        if self.tabs.tabText(idx) == "Advanced":
            self.adv_desc.setPlainText(self.desc.toPlainText())

    def _add_adv_row(self, key: str, widget: QWidget):
        label = QLabel(key)
        label.setToolTip(COMMENTS.get(key, ""))
        btn = QPushButton()
        btn.setToolTip("Info")
        btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        btn.setIconSize(QSize(14, 14))
        btn.setFixedSize(26, 22)
        btn.clicked.connect(lambda: self._show_adv_description(key))

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(widget, 1)
        h.addWidget(btn)
        self.adv_form.addRow(label, row)

    def _show_adv_description(self, key: str):
        txt = COMMENTS.get(key, "(No description available)")
        self.adv_desc.setPlainText(f"{key}\n\n{txt}")

    def _adv_file_row(self, key: str, filter_str: str):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        le = QLineEdit()
        le.setReadOnly(True)
        btn_pick = QPushButton("Select…")
        btn_pick.clicked.connect(lambda: self._pick_adv_file(key, le, filter_str))
        h.addWidget(le, 1)
        h.addWidget(btn_pick)
        self._add_adv_row(key, row)
        self.adv_widgets[key + "_lineedit"] = le
        return row

    def _pick_adv_file(self, key: str, lineedit: QLineEdit, filter_str: str):
        fp, _ = QFileDialog.getOpenFileName(self, f"Select {key}", "", filter_str)
        if fp:
            self.model.set_path(key, fp)
            lineedit.setText(self.model.resolve_path(key))
            self._show_adv_description(key)

    def _adv_text_row(self, key: str, placeholder: str = ""):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        le = QLineEdit()
        le.setPlaceholderText(placeholder)
        le.editingFinished.connect(lambda k=key, w=le: self._on_adv_text_commit(k, w.text()))
        h.addWidget(le, 1)
        self._add_adv_row(key, row)
        self.adv_widgets[key + "_lineedit"] = le
        return row

    def _adv_dspin_row(self, key: str, mn: float, mx: float, step: float):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        sp = QDoubleSpinBox()
        sp.setRange(mn, mx)
        sp.setSingleStep(step)
        sp.setDecimals(3 if step < 0.1 else 2)
        sp.valueChanged.connect(lambda v, k=key: self._set_adv_value(k, float(v)))
        sp.editingFinished.connect(lambda k=key: self._show_adv_description(k))
        h.addWidget(sp, 1)
        self._add_adv_row(key, row)
        self.adv_widgets[key + "_spin"] = sp
        return row

    def _adv_gaze_map_row(self):
        box = QWidget()
        g = QGridLayout(box)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)

        fields = ["NOSE", "LEYE", "REYE", "LEAR", "REAR"]
        self._gaze_spins: Dict[str, QSpinBox] = {}
        for r, name in enumerate(fields):
            lab = QLabel(name)
            sp = QSpinBox()
            sp.setRange(0, 100)
            sp.valueChanged.connect(lambda v, n=name: self._on_gaze_map_changed(n, int(v)))
            g.addWidget(lab, r, 0)
            g.addWidget(sp, r, 1)
            self._gaze_spins[name] = sp

        self._add_adv_row("gaze_keypoint_map", box)
        return box

    def _on_gaze_map_changed(self, name: str, value: int):
        gkm = self.model.data.get("gaze_keypoint_map", {})
        if not isinstance(gkm, dict):
            gkm = {}
        gkm[name] = int(value)
        self.model.data["gaze_keypoint_map"] = gkm

    def _on_adv_text_commit(self, key: str, text: str):
        text = text.strip()
        if key in ("det_cat_ids", "keypoint_indices", "enemy_ids"):
            arr = []
            if text:
                for token in re.split(r"[,\s]+", text):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        arr.append(int(token))
                    except Exception:
                        pass
            if key == "det_cat_ids":
                self.model.data[key] = arr or [0]
            elif key == "keypoint_indices":
                self.model.data[key] = arr or [15, 16]
            elif key == "enemy_ids":
                self.model.data[key] = arr or [99]
        else:
            self.model.data[key] = text
        self.model._normalize()
        self._show_adv_description(key)

    def _set_adv_value(self, key: str, value: Any):
        self.model.data[key] = value

    def _populate_advanced_fields(self):
        for k in ("det_model", "det_weights", "pose2d_config", "pose2d_weights"):
            le = self.adv_widgets.get(k + "_lineedit")
            if isinstance(le, QLineEdit):
                le.setText(self.model.resolve_path(k))

        for k in ("det_cat_ids", "keypoint_indices", "enemy_ids"):
            le = self.adv_widgets.get(k + "_lineedit")
            if isinstance(le, QLineEdit):
                v = self.model.data.get(k, [])
                if isinstance(v, list):
                    le.setText(",".join(str(x) for x in v))
                else:
                    le.setText(str(v))

        for k in ("box_conf_threshold", "pose_conf_threshold", "boundary_pad_pct", "frame_rate"):
            sp = self.adv_widgets.get(k + "_spin")
            if isinstance(sp, QDoubleSpinBox):
                sp.blockSignals(True)
                sp.setValue(float(self.model.data.get(k, DEFAULT_CONFIG.get(k, 0.0))))
                sp.blockSignals(False)

        dev = self.adv_widgets.get("device")
        if isinstance(dev, QComboBox):
            cur = str(self.model.data.get("device", "cpu"))
            idx = dev.findText(cur)
            if idx >= 0:
                dev.setCurrentIndex(idx)

        chk = self.adv_widgets.get("track_enemy")
        if isinstance(chk, QCheckBox):
            chk.blockSignals(True)
            chk.setChecked(bool(self.model.data.get("track_enemy", True)))
            chk.blockSignals(False)

        for key in ("flip_test", "compile_for_inference"):
            chk = self.adv_widgets.get(key)
            if isinstance(chk, QCheckBox):
                chk.blockSignals(True)
                chk.setChecked(bool(self.model.data.get(key, DEFAULT_CONFIG[key])))
                chk.blockSignals(False)

        gkm = self.model.data.get("gaze_keypoint_map", {})
        if isinstance(gkm, dict):
            for name, sp in self._gaze_spins.items():
                sp.blockSignals(True)
                sp.setValue(int(gkm.get(name, DEFAULT_CONFIG["gaze_keypoint_map"][name])))
                sp.blockSignals(False)

        # --- Audio & Transcription ---
        for key in ("preserve_audio", "enable_transcription", "enable_denoise"):
            chk = self.adv_widgets.get(key)
            if isinstance(chk, QCheckBox):
                chk.blockSignals(True)
                chk.setChecked(bool(self.model.data.get(key, DEFAULT_CONFIG[key])))
                chk.blockSignals(False)

        txn_device = self.adv_widgets.get("transcription_device")
        if isinstance(txn_device, QComboBox):
            cur = str(self.model.data.get("transcription_device", DEFAULT_CONFIG["transcription_device"]))
            idx = txn_device.findText(cur)
            txn_device.blockSignals(True)
            if idx >= 0:
                txn_device.setCurrentIndex(idx)
            txn_device.blockSignals(False)

        denoise_device = self.adv_widgets.get("denoise_device")
        if isinstance(denoise_device, QComboBox):
            cur = str(self.model.data.get("denoise_device", DEFAULT_CONFIG["denoise_device"]))
            idx = denoise_device.findText(cur)
            denoise_device.blockSignals(True)
            if idx >= 0:
                denoise_device.setCurrentIndex(idx)
            denoise_device.blockSignals(False)

        # --- Drill window detection ---
        chk_drill = self.adv_widgets.get("drill_window_enabled")
        if isinstance(chk_drill, QCheckBox):
            chk_drill.blockSignals(True)
            chk_drill.setChecked(bool(self.model.data.get(
                "drill_window_enabled", DEFAULT_CONFIG["drill_window_enabled"]
            )))
            chk_drill.blockSignals(False)

        drill_words = self.adv_widgets.get("drill_window_required_words")
        if isinstance(drill_words, QLineEdit):
            cur = str(self.model.data.get(
                "drill_window_required_words",
                DEFAULT_CONFIG["drill_window_required_words"],
            ))
            drill_words.blockSignals(True)
            drill_words.setText(cur)
            drill_words.blockSignals(False)

    def _on_pod_moved(self, idx: int, x: float, y: float):
        # Ensure model is normalized
        self.model._normalize()

        # Rewrite the POD points file if we have one
        if self._pod_points_src_path and os.path.exists(self._pod_points_src_path):
            try:
                pods = self.model.data.get("POD", [])
                if isinstance(pods, list):
                    with open(self._pod_points_src_path, "w", encoding="utf-8") as f:
                        for px, py in pods:
                            f.write(f"{int(px)}, {int(py)}\n")
            except Exception as e:
                # Non-fatal; show in status bar
                self.statusBar().showMessage(f"Warning: could not write POD file: {e}")

        # Keep preview consistent (radius circles already adjusted; this ensures any other dependent UI stays updated)
        self.preview.refresh()
        self._update_ready_state()


# ----------------------------
# Run
# ----------------------------

def main():
    import sys
    app = QApplication(sys.argv)
    w = ConfigBuilderWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    