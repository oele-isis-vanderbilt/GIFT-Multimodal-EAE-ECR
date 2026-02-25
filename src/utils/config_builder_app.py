"""
Config Builder UI (PyQt5) for Army ECR Battle Drill pipeline

Features
- Main page: POD + Boundary loaders (txt), point mapping / entry polys / map image pickers
- Main knobs: visual angle, threat interaction time, entry time threshold, POD working radius,
              POD capture threshold, per-POD time limits (dynamic), coverage time threshold, stay-along-wall
- Live preview: map image + boundary polygon + POD points (labeled) + working-radius circles + wall band
- Advanced page: all other config elements with defaults pre-filled
- Root folder picker: used for saving config.json and for storing relative paths when possible

IMPORTANT:
- POD, Boundary, MapPath, point_mapping_path, entry_polys_path are REQUIRED and start empty.
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

    # models (advanced defaults)
    "det_model": "libs/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",
    "det_weights": "models/detect.pth",
    "det_cat_ids": [0],
    "pose2d_config": "libs/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py",
    "pose2d_weights": "models/pose.pth",

    # thresholds (advanced defaults)
    "box_conf_threshold": 0.3,
    "pose_conf_threshold": 0.3,

    # tracker / runtime (advanced defaults)
    "keypoint_indices": [15, 16],
    "device": "mps",
    "boundary_pad_pct": 0.05,
    "track_enemy": True,
    "enemy_ids": [99],

    # unified knobs (main defaults)
    "visual_angle_degrees": 20.0,
    "min_threat_interaction_time_sec": 1.0,
    "entry_time_threshold_sec": 2.0,

    # pod knobs (main defaults)
    "pod_working_radius": 40.0,
    "pod_capture_threshold_sec": 0.1,
    "pod_time_limits": [30.0],

    # coverage / wall (main defaults)
    "coverage_time_threshold": 3.0,
    "stay_along_wall_pWall": 0.2,

    # gaze keypoints (advanced defaults)
    "gaze_keypoint_map": {"NOSE": 0, "LEYE": 1, "REYE": 2, "LEAR": 3, "REAR": 4},

    # optional comparison override
    "frame_rate": 30.0,
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

    "box_conf_threshold": "Minimum bbox confidence to accept a detection.",
    "pose_conf_threshold": "Minimum keypoint confidence to accept pose keypoints and render gaze/triangles.",
    "keypoint_indices": "Which keypoints the tracker uses for keypoint-based positioning logic.",
    "device": "Compute device for inference (e.g., 'cpu', 'cuda', 'mps').",

    "boundary_pad_pct": "Extra padding around boundary used by tracker when validating positions.",
    "track_enemy": "Enable enemy tracking behaviors in the tracker.",

    "enemy_ids": "Track IDs considered enemies (used for fall detection, gaze/coverage filtering, threat clearance).",
    "visual_angle_degrees": "Full field-of-view angle (degrees) used for gaze triangles, map gaze/coverage, and threat-clearance.",
    "min_threat_interaction_time_sec": "Minimum interaction time (seconds) required to count a threat as cleared.",
    "entry_time_threshold_sec": "Max allowed team entry span (seconds) for full score in TOTAL_TIME_OF_ENTRY.",

    "pod_working_radius": "Radius (map pixels) around each POD used to compute work areas for POD capture analysis.",
    "pod_capture_threshold_sec": "Seconds required inside a POD work area to count as captured.",
    "pod_time_limits": "Per-POD time limits (seconds) for POD_CAPTURE_TIME scoring (auto-extends if fewer than POD count).",

    "coverage_time_threshold": "Seconds of sustained coverage needed for full score in TOTAL_FLOOR_COVERAGE_TIME.",
    "stay_along_wall_pWall": "Sensitivity/threshold for STAY_ALONG_WALL metric (higher usually means stricter wall adherence).",

    "gaze_keypoint_map": "Keypoint indices (Halpe26) used to compute gaze direction (nose/eyes/ears).",
    "frame_rate": "(Optional) Override FPS used in comparisons; normally set automatically from video during processing.",

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
        """Store relative-to-root if possible."""
        if self.project_root:
            try:
                rel = os.path.relpath(file_path, self.project_root)
                if not rel.startswith(".."):
                    self.data[key] = rel.replace("\\", "/")
                    return
            except Exception:
                pass
        self.data[key] = file_path

    def resolve_path(self, key: str) -> str:
        v = self.data.get(key, "")
        if not isinstance(v, str) or not v:
            return ""
        if self.project_root and not os.path.isabs(v):
            return os.path.join(self.project_root, v)
        return v


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

        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._boundary_item: Optional[QGraphicsPolygonItem] = None
        self._wall_band_item: Optional[QGraphicsPathItem] = None
        self._pod_items: List[QGraphicsEllipseItem] = []
        self._pod_labels: List[QGraphicsTextItem] = []
        self._radius_items: List[QGraphicsEllipseItem] = []
        self._on_pod_moved_cb = on_pod_moved

        self.refresh()

    def sizeHint(self) -> QSize:
        return QSize(700, 600)

    def refresh(self) -> None:
        self.scene.clear()
        self._pod_items.clear()
        self._pod_labels.clear()
        self._radius_items.clear()
        self._pix_item = None
        self._boundary_item = None
        self._wall_band_item = None

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

            # Wall band: outer - inner(buffered inward). Works for concave boundaries.
            pWall = float(self.model.data.get("stay_along_wall_pWall", 0.2))
            band_path = self._compute_wall_band_path(boundary, pWall)
            if band_path is not None:
                self._wall_band_item = QGraphicsPathItem(band_path)
                self._wall_band_item.setPen(QPen(QColor(0, 0, 0, 0)))
                self._wall_band_item.setBrush(QBrush(QColor(0, 200, 255, 55)))
                self._wall_band_item.setZValue(9)
                self.scene.addItem(self._wall_band_item)

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

                # Bubble up to window so it can rewrite the POD txt file
                if self._on_pod_moved_cb is not None:
                    self._on_pod_moved_cb(idx, nx, ny)

            dot = DraggablePodItem(i, QPointF(x, y), 4.0, _moved)
            self.scene.addItem(dot)
            self._pod_items.append(dot)

            # label
            t = QGraphicsTextItem(str(i))
            t.setDefaultTextColor(QColor(255, 255, 255))
            t.setFont(QFont("Arial", 10, QFont.Bold))
            t.setPos(x + 6, y - 14)
            t.setZValue(8)
            self.scene.addItem(t)
            self._pod_labels.append(t)

        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _compute_wall_band_path(self, boundary_pts: List[List[int]], pWall: float) -> Optional[QPainterPath]:
        pts = [(float(x), float(y)) for x, y in boundary_pts]
        if len(pts) < 3:
            return None

        outer = Polygon(pts)
        if outer.is_empty or not outer.is_valid:
            outer = outer.buffer(0)
        if outer.is_empty:
            return None

        minx, miny, maxx, maxy = outer.bounds
        w = maxx - minx
        h = maxy - miny
        if w <= 1 or h <= 1:
            return None

        pWall = max(0.0, min(1.0, float(pWall)))
        inset = (0.02 + 0.18 * pWall) * min(w, h)

        inner = outer.buffer(-inset)
        if inner.is_empty:
            return None

        if isinstance(inner, MultiPolygon):
            inner_poly = max(list(inner.geoms), key=lambda g: g.area)
        else:
            inner_poly = inner

        band = outer.difference(inner_poly)
        if band.is_empty:
            return None

        def add_poly(poly: Polygon, path: QPainterPath):
            ext = list(poly.exterior.coords)
            if len(ext) >= 3:
                path.moveTo(ext[0][0], ext[0][1])
                for x, y in ext[1:]:
                    path.lineTo(x, y)
                path.closeSubpath()

            for ring in poly.interiors:
                coords = list(ring.coords)
                if len(coords) >= 3:
                    hole = QPainterPath()
                    hole.moveTo(coords[0][0], coords[0][1])
                    for x, y in coords[1:]:
                        hole.lineTo(x, y)
                    hole.closeSubpath()
                    path.addPath(hole)

        path = QPainterPath()
        if isinstance(band, MultiPolygon):
            for g in band.geoms:
                if isinstance(g, Polygon):
                    add_poly(g, path)
        elif isinstance(band, Polygon):
            add_poly(band, path)
        else:
            return None

        path.setFillRule(Qt.OddEvenFill)
        return path


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

        top.addWidget(QLabel("Project Root:"))
        top.addWidget(self.root_path, 1)
        top.addWidget(self.btn_root)
        top.addSpacing(10)
        top.addWidget(self.btn_load)
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

        if len(self.model.data.get("POD", [])) < 1:
            QMessageBox.warning(self, "Invalid POD", "POD list is empty. Load a POD file.")
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
            out = self.model.to_json_dict(include_comments=True)
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            QMessageBox.information(self, "Saved", f"Saved config to:\n{fp}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save config:\n\n{e}")

    # ---------- Readiness state ----------

    def _update_ready_state(self):
        has_pod = isinstance(self.model.data.get("POD"), list) and len(self.model.data.get("POD")) > 0
        has_boundary = isinstance(self.model.data.get("Boundary"), list) and len(self.model.data.get("Boundary")) >= 3

        map_ok = bool(self.model.data.get("MapPath", ""))
        pm_ok = bool(self.model.data.get("point_mapping_path", ""))
        ep_ok = bool(self.model.data.get("entry_polys_path", ""))

        self.pod_file_picker.set_required_missing(not has_pod)
        self.boundary_file_picker.set_required_missing(not has_boundary)
        self.map_picker.set_required_missing(not map_ok)
        self.point_map_picker.set_required_missing(not pm_ok)
        self.entry_polys_picker.set_required_missing(not ep_ok)

        ready = has_pod and has_boundary and map_ok and pm_ok and ep_ok and bool(self.model.project_root)
        self.btn_save.setEnabled(ready)

        if not self.model.project_root:
            msg = "Set Project Root to enable saving."
        elif not has_pod:
            msg = "Load POD points file (*.txt)."
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
            required=True
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

        self.spin_pwall = self._mk_dspin("stay_along_wall_pWall", 0.0, 1.0, 0.01, "")
        self.spin_pwall.valueChanged.connect(lambda _: self.preview.refresh())
        knobs_layout.addWidget(self._mk_label_btn("Stay-along-wall", "stay_along_wall_pWall"), row, 0)
        knobs_layout.addWidget(self.spin_pwall, row, 1)
        knobs_layout.addWidget(QLabel("0..1"), row, 2)
        row += 1

        limits_box = QGroupBox("Per-POD Time Limits (seconds)")
        limits_v = QVBoxLayout(limits_box)
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
        if key in ("pod_working_radius", "stay_along_wall_pWall"):
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

        self._pod_limit_spins: List[QDoubleSpinBox] = []
        for i in range(max(n, 1)):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            label = QLabel(f"POD {i} limit:")
            label.setMinimumWidth(120)
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
            h.addWidget(sp, 1)
            self.pod_limits_layout.addWidget(row)
            self._pod_limit_spins.append(sp)

        self.pod_limits_layout.addStretch(1)
        self.show_description("pod_time_limits")

    def _on_pod_limit_changed(self, idx: int, value: float):
        limits = self.model.data.get("pod_time_limits", [])
        if not isinstance(limits, list):
            limits = []
        while len(limits) <= idx:
            limits.append(limits[-1] if limits else 30.0)
        limits[idx] = float(value)
        self.model.data["pod_time_limits"] = limits

    def refresh_paths(self):
        self.map_picker.set_path(self.model.resolve_path("MapPath"))
        self.point_map_picker.set_path(self.model.resolve_path("point_mapping_path"))
        self.entry_polys_picker.set_path(self.model.resolve_path("entry_polys_path"))

    def refresh_all(self):
        self.spin_visual_angle.setValue(float(self.model.data.get("visual_angle_degrees", 20.0)))
        self.spin_threat_time.setValue(float(self.model.data.get("min_threat_interaction_time_sec", 1.0)))
        self.spin_entry_time.setValue(float(self.model.data.get("entry_time_threshold_sec", 2.0)))
        self.spin_pod_radius.setValue(float(self.model.data.get("pod_working_radius", 40.0)))
        self.spin_pod_capture.setValue(float(self.model.data.get("pod_capture_threshold_sec", 0.1)))
        self.spin_coverage_time.setValue(float(self.model.data.get("coverage_time_threshold", 3.0)))
        self.spin_pwall.setValue(float(self.model.data.get("stay_along_wall_pWall", 0.2)))

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

        gkm = self.model.data.get("gaze_keypoint_map", {})
        if isinstance(gkm, dict):
            for name, sp in self._gaze_spins.items():
                sp.blockSignals(True)
                sp.setValue(int(gkm.get(name, DEFAULT_CONFIG["gaze_keypoint_map"][name])))
                sp.blockSignals(False)
                
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
    