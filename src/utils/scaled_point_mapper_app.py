# scaled_point_mapper_tool.py
# PyQt5 tool:
# - Load a floorplan image
# - Two modes:
#   1) Add Walls: click 2 points to create a wall segment (endpoints snap to existing corners)
#   2) Add Scaled Point: pick two adjacent walls (share a corner), enter REAL distances to each wall
#      and REAL wall lengths for A & B, then the tool places the point on the map.
# - Points are always visible (including in wall mode).
# - Visible corner dots at wall endpoints:
#     • Click a corner dot to use it exactly (no “near” clicking required)
#     • Corner dots do NOT select walls (walls only select on line click)
#     • While drawing a wall, the chosen endpoint dot highlights
#     • While drawing a wall, the first endpoint is shown as a temporary cyan ring
# - Save output image: base map + points only (no walls)
#
# Dependencies: PyQt5
#
# Run:
#   python scaled_point_mapper_app.py

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Set

from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QSplitter, QGroupBox, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QAbstractItemView, QFormLayout,
    QDoubleSpinBox, QLineEdit, QSizePolicy, QStackedWidget,
    QScrollArea, QTextEdit
)
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsEllipseItem


APP_NAME = "scaled_point_mapper"


# ---------------- Geometry helpers ----------------
def dist2(a: QPointF, b: QPointF) -> float:
    dx = float(a.x() - b.x())
    dy = float(a.y() - b.y())
    return dx * dx + dy * dy


def close_pt(a: QPointF, b: QPointF, eps: float) -> bool:
    return dist2(a, b) <= eps * eps


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------- Data models ----------------
@dataclass
class Wall:
    wall_id: int
    p1: QPointF
    p2: QPointF


@dataclass
class SPoint:
    point_id: int
    p: QPointF


# ---------------- Graphics items ----------------
class WallItem(QGraphicsLineItem):
    def __init__(self, wall_id: int, p1: QPointF, p2: QPointF):
        super().__init__(p1.x(), p1.y(), p2.x(), p2.y())
        self.wall_id = wall_id
        self.setFlag(QGraphicsLineItem.ItemIsSelectable, True)
        self.setZValue(10)


class PointItem(QGraphicsEllipseItem):
    def __init__(self, point_id: int, p: QPointF, r: float = 2.0):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.point_id = point_id
        self.setPos(p)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setZValue(30)


class CornerMarker(QGraphicsEllipseItem):
    """A small marker to show the shared corner between Wall A and Wall B."""
    def __init__(self, p: QPointF, r: float = 6.0):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.setPos(p)
        self.setZValue(25)
        pen = QPen(QColor(180, 0, 255))
        pen.setWidth(3)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)


class CornerPointItem(QGraphicsEllipseItem):
    """Visible corner point for wall endpoints. Click to use as an exact wall endpoint.
    NOTE: Not selectable (so wall selection only happens when the wall is clicked).
    """
    def __init__(self, key_xy: Tuple[int, int], p: QPointF, r: float = 5.0):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.key_xy = key_xy
        self.setPos(p)
        self.setZValue(22)
        pen = QPen(QColor(90, 90, 90))
        pen.setWidth(2)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)


# ---------------- Canvas ----------------
class MapCanvas(QGraphicsView):
    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)

        self._has_image = False
        self._img_w = 0
        self._img_h = 0

        # panning
        self._panning = False
        self._pan_start = None
        self._press_pos = None

        # When clicking a corner dot, we emit that exact dot center
        self._forced_click_point: Optional[QPointF] = None
        self._forced_click_corner_key: Optional[Tuple[int, int]] = None

        self._user_scaled = False

        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)

    def has_image(self) -> bool:
        return self._has_image

    def image_size(self) -> Tuple[int, int]:
        return self._img_w, self._img_h

    def setImage(self, img: QImage):
        if img.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            self._has_image = False
            self._img_w = self._img_h = 0
            self.scene().setSceneRect(0, 0, 1, 1)
            self.resetZoom()
            return

        pix = QPixmap.fromImage(img)
        self._pixmap_item.setPixmap(pix)
        self._img_w, self._img_h = img.width(), img.height()
        self.scene().setSceneRect(0, 0, self._img_w, self._img_h)
        self._has_image = True
        self.resetZoom()

    def resetZoom(self):
        self._user_scaled = False
        self.resetTransform()
        if self._has_image:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self._has_image:
            super().wheelEvent(event)
            return
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.04 if delta > 0 else 1 / 1.04
            self._user_scaled = True
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._has_image and not self._user_scaled:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if not self._has_image:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton:
            self._press_pos = event.pos()
            item = self.itemAt(event.pos())

            # If clicking a wall or a selectable point, let Qt handle selection.
            if isinstance(item, (WallItem, PointItem)):
                self._forced_click_point = None
                self._forced_click_corner_key = None
                super().mousePressEvent(event)
                return

            # If clicking an endpoint corner marker, do NOT pan; emit exact corner on release.
            if isinstance(item, CornerPointItem):
                self._panning = False
                self._pan_start = None
                self._forced_click_point = QPointF(item.pos().x(), item.pos().y())
                self._forced_click_corner_key = item.key_xy
                event.accept()
                return

            # background: start panning
            self._forced_click_point = None
            self._forced_click_corner_key = None
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._has_image and self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._has_image:
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and (self._panning or self._forced_click_point is not None):
            self.unsetCursor()
            moved = 999
            if self._press_pos is not None:
                moved = (event.pos() - self._press_pos).manhattanLength()

            self._panning = False
            self._pan_start = None

            # treat as click if tiny movement
            if moved <= 3:
                if self._forced_click_point is not None:
                    pos = self._forced_click_point
                else:
                    pos = self.mapToScene(event.pos())
                x = max(0.0, min(float(self._img_w - 1), float(pos.x())))
                y = max(0.0, min(float(self._img_h - 1), float(pos.y())))
                self.clicked.emit(x, y)

            self._forced_click_point = None
            self._forced_click_corner_key = None
            self._press_pos = None
            event.accept()
            return

        super().mouseReleaseEvent(event)


# ---------------- Main Window ----------------
class MainWindow(QMainWindow):
    MODE_WALLS = 0
    MODE_POINT = 1

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 860)

        self.canvas = MapCanvas()

        self.map_qimage: Optional[QImage] = None

        self._wall_id_seq = 1
        self._pt_id_seq = 1

        self.walls: Dict[int, Wall] = {}
        self.wall_items: Dict[int, WallItem] = {}

        self.points: Dict[int, SPoint] = {}
        self.point_items: Dict[int, PointItem] = {}

        self.corner_items: Dict[Tuple[int, int], CornerPointItem] = {}

        # Mode + wall drawing state
        self.mode = self.MODE_WALLS
        self._wall_draw_first: Optional[QPointF] = None

        # A/B selection
        self.wallA_id: Optional[int] = None
        self.wallB_id: Optional[int] = None
        self._corner_marker: Optional[CornerMarker] = None

        # Snap behavior
        self.snap_eps_px = 8.0

        # NEW: temp marker for first wall endpoint and highlight set for active endpoints
        self._temp_wall_first_marker: Optional[QGraphicsEllipseItem] = None
        self._active_wall_corner_keys: Set[Tuple[int, int]] = set()

        self._build_ui()
        self._wire()
        self._refresh_all()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        # Left panel
        left = QWidget()
        left.setMinimumWidth(380)
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(10, 10, 10, 10)
        left_l.setSpacing(10)

        # Project group
        g_proj = QGroupBox("1) Project")
        proj_l = QVBoxLayout(g_proj)
        proj_l.setSpacing(8)

        self.btn_load = QPushButton("Load Map Image")
        self.btn_save_img = QPushButton("Save Output Image (Points Only)")
        self.btn_reset_zoom = QPushButton("Reset Zoom")

        self.le_loaded = QLineEdit()
        self.le_loaded.setReadOnly(True)
        self.le_loaded.setPlaceholderText("No image loaded")

        proj_l.addWidget(self.btn_load)
        proj_l.addWidget(self.btn_save_img)
        proj_l.addWidget(self.btn_reset_zoom)
        proj_l.addWidget(QLabel("Loaded file:"))
        proj_l.addWidget(self.le_loaded)

        left_l.addWidget(g_proj)

        # Mode group
        g_mode = QGroupBox("2) Mode")
        mode_l = QVBoxLayout(g_mode)
        mode_l.setSpacing(8)

        self.btn_mode_walls = QPushButton("Add Walls")
        self.btn_mode_points = QPushButton("Add Scaled Point")

        for b in (self.btn_mode_walls, self.btn_mode_points):
            b.setCheckable(True)
            b.setMinimumHeight(34)

        mode_l.addWidget(QLabel("Choose what you're doing right now:"))
        mode_l.addWidget(self.btn_mode_walls)
        mode_l.addWidget(self.btn_mode_points)

        left_l.addWidget(g_mode)

        # Stacked mode panels
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # --- Walls panel ---
        wall_panel = QWidget()
        wp_l = QVBoxLayout(wall_panel)
        wp_l.setContentsMargins(0, 0, 0, 0)
        wp_l.setSpacing(10)

        g_w_instr = QGroupBox("3A) Walls Instructions")
        wi_l = QVBoxLayout(g_w_instr)
        self.txt_walls_instr = QTextEdit()
        self.txt_walls_instr.setReadOnly(True)
        self.txt_walls_instr.setMinimumHeight(120)
        self.txt_walls_instr.setText(
            "Add Walls:\n"
            "• Click the map twice to create one wall segment.\n"
            "• Existing wall endpoints show as small corner dots. Click a dot to use it exactly.\n"
            "• Endpoints also SNAP if you click close to a corner.\n"
            "• Walls are selected only by clicking the wall line (corner dots do not select walls).\n"
            "• While drawing a wall, the chosen corner dot (if any) will highlight.\n\n"
            "Tips:\n"
            "• Draw walls around each room/hallway edge.\n"
            "• Reuse corners by clicking the visible dots for clean adjacency.\n"
        )
        wi_l.addWidget(self.txt_walls_instr)

        g_walls = QGroupBox("Walls")
        walls_l = QVBoxLayout(g_walls)
        walls_l.setSpacing(8)

        self.lbl_wall_state = QLabel("Ready: click 1st wall point")
        self.lbl_wall_state.setStyleSheet("font-weight:600;")

        self.list_walls = QListWidget()
        self.list_walls.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_walls.setMinimumHeight(160)

        self.btn_wall_cancel = QPushButton("Cancel Wall In-Progress")
        self.btn_wall_delete = QPushButton("Delete Selected Wall(s)")
        self.btn_wall_clear = QPushButton("Clear All Walls")

        walls_l.addWidget(self.lbl_wall_state)
        walls_l.addWidget(self.list_walls, 1)
        walls_l.addWidget(self.btn_wall_cancel)
        walls_l.addWidget(self.btn_wall_delete)
        walls_l.addWidget(self.btn_wall_clear)

        wp_l.addWidget(g_w_instr)
        wp_l.addWidget(g_walls, 1)

        # --- Point panel ---
        point_panel = QWidget()
        pp_l = QVBoxLayout(point_panel)
        pp_l.setContentsMargins(0, 0, 0, 0)
        pp_l.setSpacing(10)

        g_p_instr = QGroupBox("3B) Point Instructions")
        pi_l = QVBoxLayout(g_p_instr)
        self.txt_point_instr = QTextEdit()
        self.txt_point_instr.setReadOnly(True)
        self.txt_point_instr.setMinimumHeight(140)
        self.txt_point_instr.setText(
            "Add Scaled Point (from two adjacent walls):\n"
            "1) Select TWO walls that meet at a corner.\n"
            "2) Mark one as Wall A and one as Wall B.\n"
            "3) Enter the REAL wall lengths for A and B.\n"
            "4) Enter REAL distances to the walls:\n"
            "   • Distance to Wall B (measured parallel to Wall A)\n"
            "   • Distance to Wall A (measured parallel to Wall B)\n"
            "5) Click 'Add Point' to place the point on the map.\n\n"
            "Units note:\n"
            "• Lengths and distances must be in the SAME units. Units don’t matter (feet/meters/etc) as long as consistent.\n\n"
            "Note:\n"
            "• The corner is the shared endpoint of A and B (highlighted in purple).\n"
        )
        pi_l.addWidget(self.txt_point_instr)

        g_pf = QGroupBox("Scaled Point Setup")
        pf_l = QVBoxLayout(g_pf)
        pf_l.setSpacing(8)

        # A/B header with color swatches
        ab_row = QWidget()
        ab_row_l = QHBoxLayout(ab_row)
        ab_row_l.setContentsMargins(0, 0, 0, 0)
        ab_row_l.setSpacing(8)

        self.swatch_A = self._make_color_swatch(QColor(0, 200, 0))      # green
        self.swatch_B = self._make_color_swatch(QColor(255, 140, 0))    # orange

        self.lbl_ab_status = QLabel("Wall A: (not set)   |   Wall B: (not set)")
        self.lbl_ab_status.setStyleSheet("font-weight:600;")
        self.lbl_ab_status.setWordWrap(True)

        ab_row_l.addWidget(self.swatch_A)
        ab_row_l.addWidget(QLabel("A"))
        ab_row_l.addSpacing(10)
        ab_row_l.addWidget(self.swatch_B)
        ab_row_l.addWidget(QLabel("B"))
        ab_row_l.addSpacing(10)
        ab_row_l.addWidget(self.lbl_ab_status, 1)

        pf_l.addWidget(ab_row)

        self.btn_mark_A = QPushButton("Mark Selected Wall as A")
        self.btn_mark_B = QPushButton("Mark Selected Wall as B")

        form = QFormLayout()
        form.setVerticalSpacing(8)

        self.spin_lenA = QDoubleSpinBox()
        self.spin_lenB = QDoubleSpinBox()
        self.spin_d_to_B = QDoubleSpinBox()
        self.spin_d_to_A = QDoubleSpinBox()

        for sp in (self.spin_lenA, self.spin_lenB, self.spin_d_to_B, self.spin_d_to_A):
            sp.setDecimals(4)
            sp.setRange(0.0, 1_000_000.0)
            sp.setSingleStep(0.1)
            sp.setValue(0.0)

        self.spin_lenA.setValue(1.0)
        self.spin_lenB.setValue(1.0)

        form.addRow("Wall A real length:", self.spin_lenA)
        form.addRow("Wall B real length:", self.spin_lenB)
        form.addRow("Distance to Wall B (along A):", self.spin_d_to_B)
        form.addRow("Distance to Wall A (along B):", self.spin_d_to_A)

        self.btn_add_point = QPushButton("Add Point")
        self.lbl_pf_feedback = QLabel("")
        self.lbl_pf_feedback.setWordWrap(True)

        pf_l.addWidget(self.lbl_ab_status)
        pf_l.addWidget(self.btn_mark_A)
        pf_l.addWidget(self.btn_mark_B)
        pf_l.addLayout(form)
        pf_l.addWidget(self.btn_add_point)
        pf_l.addWidget(self.lbl_pf_feedback)

        pp_l.addWidget(g_p_instr)
        pp_l.addWidget(g_pf, 1)

        self.stack.addWidget(wall_panel)
        self.stack.addWidget(point_panel)

        left_l.addWidget(self.stack, 2)

        # Points group (always visible)
        g_pts = QGroupBox("4) Points (Always Visible)")
        pts_l = QVBoxLayout(g_pts)
        pts_l.setSpacing(8)

        self.list_points = QListWidget()
        self.list_points.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_points.setMinimumHeight(140)

        self.btn_pt_delete = QPushButton("Delete Selected Point(s)")
        self.btn_pt_clear = QPushButton("Clear All Points")

        pts_l.addWidget(self.list_points, 1)
        pts_l.addWidget(self.btn_pt_delete)
        pts_l.addWidget(self.btn_pt_clear)

        left_l.addWidget(g_pts, 1)

        split.addWidget(left)

        # Right panel: canvas
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 8, 8, 8)
        right_l.setSpacing(6)

        title = QLabel("Map (Ctrl + Wheel = Zoom, Drag background = Pan, Click items to select)")
        title.setStyleSheet("font-weight:600;")
        right_l.addWidget(title)
        right_l.addWidget(self.canvas, 1)

        split.addWidget(right)
        split.setSizes([420, 1080])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        # Initial mode selection
        self.btn_mode_walls.setChecked(True)
        self.btn_mode_points.setChecked(False)
        self.stack.setCurrentIndex(self.MODE_WALLS)

    def _wire(self):
        self.btn_load.clicked.connect(self.on_load)
        self.btn_save_img.clicked.connect(self.on_save_image)
        self.btn_reset_zoom.clicked.connect(self.canvas.resetZoom)

        self.btn_mode_walls.clicked.connect(lambda: self.set_mode(self.MODE_WALLS))
        self.btn_mode_points.clicked.connect(lambda: self.set_mode(self.MODE_POINT))

        self.canvas.clicked.connect(self.on_canvas_clicked)

        self.btn_wall_cancel.clicked.connect(self.on_wall_cancel)
        self.btn_wall_delete.clicked.connect(self.on_wall_delete_selected)
        self.btn_wall_clear.clicked.connect(self.on_wall_clear_all)
        self.list_walls.itemSelectionChanged.connect(self.on_wall_list_selection_changed)

        self.btn_mark_A.clicked.connect(self.on_mark_A)
        self.btn_mark_B.clicked.connect(self.on_mark_B)
        self.btn_add_point.clicked.connect(self.on_add_point)

        self.list_points.itemSelectionChanged.connect(self.on_point_list_selection_changed)
        self.btn_pt_delete.clicked.connect(self.on_point_delete_selected)
        self.btn_pt_clear.clicked.connect(self.on_point_clear_all)

        self.canvas.scene().selectionChanged.connect(self.on_scene_selection_changed)

    def _make_color_swatch(self, color: QColor) -> QLabel:
        sw = QLabel()
        sw.setFixedSize(14, 14)
        sw.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            "border: 1px solid #444;"
        )
        return sw


    # ---------------- UX helpers ----------------
    def _warn(self, title: str, msg: str):
        QMessageBox.warning(self, title, msg)

    def _info(self, title: str, msg: str):
        QMessageBox.information(self, title, msg)

    def _confirm(self, title: str, msg: str) -> bool:
        return QMessageBox.question(self, title, msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes

    def set_mode(self, mode: int):
        if mode == self.MODE_WALLS:
            self.mode = self.MODE_WALLS
            self.btn_mode_walls.setChecked(True)
            self.btn_mode_points.setChecked(False)
            self.stack.setCurrentIndex(self.MODE_WALLS)
            self.lbl_pf_feedback.setText("")
        else:
            self.mode = self.MODE_POINT
            self.btn_mode_walls.setChecked(False)
            self.btn_mode_points.setChecked(True)
            self.stack.setCurrentIndex(self.MODE_POINT)
            # cancel half-drawn wall if switching
            self._wall_draw_first = None
            self._active_wall_corner_keys = set()
            self._clear_temp_wall_first_marker()
        self._refresh_all()

    # ---------------- Temp marker helpers ----------------
    def _set_temp_wall_first_marker(self, p: QPointF):
        """Show/position a temporary marker for the first wall endpoint."""
        r = 6.0
        if self._temp_wall_first_marker is None:
            it = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            it.setZValue(28)
            pen = QPen(QColor(0, 200, 255))
            pen.setWidth(3)
            it.setPen(pen)
            it.setBrush(QBrush(Qt.NoBrush))
            it.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
            self._temp_wall_first_marker = it
            self.canvas.scene().addItem(it)
        self._temp_wall_first_marker.setPos(p)

    def _clear_temp_wall_first_marker(self):
        if self._temp_wall_first_marker is not None:
            try:
                self.canvas.scene().removeItem(self._temp_wall_first_marker)
            except Exception:
                pass
        self._temp_wall_first_marker = None

    # ---------------- Core refresh ----------------
    def _refresh_all(self):
        self._refresh_wall_list()
        self._refresh_point_list()
        self._rebuild_corner_items()
        self._update_corner_marker()
        self._apply_styles()
        self._refresh_status_texts()

    def _refresh_status_texts(self):
        if not self.canvas.has_image():
            self.lbl_wall_state.setText("Load a map image first.")
            self.lbl_pf_feedback.setText("Load a map image to begin.")
            self._set_ab_status()
            return

        if self.mode == self.MODE_WALLS:
            if self._wall_draw_first is None:
                self.lbl_wall_state.setText("Ready: click 1st wall point")
            else:
                x, y = int(round(self._wall_draw_first.x())), int(round(self._wall_draw_first.y()))
                self.lbl_wall_state.setText(f"Wall in progress: first point = ({x}, {y}) → click 2nd point")
        else:
            self.lbl_wall_state.setText("Wall mode is OFF (switch to 'Add Walls' to draw).")

        self._set_ab_status()

    def _set_ab_status(self):
        a_txt = f"Wall A: {self.wallA_id}" if self.wallA_id else "Wall A: (not set)"
        b_txt = f"Wall B: {self.wallB_id}" if self.wallB_id else "Wall B: (not set)"
        self.lbl_ab_status.setText(f"{a_txt}   |   {b_txt}")

    def _refresh_wall_list(self):
        self.list_walls.blockSignals(True)
        self.list_walls.clear()
        for wid in sorted(self.walls.keys()):
            w = self.walls[wid]
            item = QListWidgetItem(
                f"Wall {wid}   ({int(w.p1.x())},{int(w.p1.y())}) → ({int(w.p2.x())},{int(w.p2.y())})"
            )
            item.setData(Qt.UserRole, wid)
            self.list_walls.addItem(item)
        self.list_walls.blockSignals(False)

    def _refresh_point_list(self):
        self.list_points.blockSignals(True)
        self.list_points.clear()
        for pid in sorted(self.points.keys()):
            p = self.points[pid].p
            item = QListWidgetItem(f"Point {pid}   ({int(p.x())},{int(p.y())})")
            item.setData(Qt.UserRole, pid)
            self.list_points.addItem(item)
        self.list_points.blockSignals(False)

    def _selected_wall_ids_from_list(self) -> List[int]:
        ids: List[int] = []
        for it in self.list_walls.selectedItems():
            wid = it.data(Qt.UserRole)
            if isinstance(wid, int):
                ids.append(wid)
        return ids

    def _selected_point_ids_from_list(self) -> List[int]:
        ids: List[int] = []
        for it in self.list_points.selectedItems():
            pid = it.data(Qt.UserRole)
            if isinstance(pid, int):
                ids.append(pid)
        return ids

    # ---------------- Corner + vectors ----------------
    def _walls_shared_corner(self, wid1: int, wid2: int) -> Optional[QPointF]:
        if wid1 not in self.walls or wid2 not in self.walls:
            return None
        w1 = self.walls[wid1]
        w2 = self.walls[wid2]
        for c in (w1.p1, w1.p2):
            if close_pt(c, w2.p1, eps=2.0) or close_pt(c, w2.p2, eps=2.0):
                return QPointF(c.x(), c.y())
        return None

    def _are_walls_adjacent(self, wid1: Optional[int], wid2: Optional[int]) -> bool:
        if wid1 is None or wid2 is None:
            return False
        if wid1 == wid2:
            return False
        return self._walls_shared_corner(wid1, wid2) is not None

    def _vector_from_corner(self, w: Wall, corner: QPointF) -> Optional[QPointF]:
        if close_pt(w.p1, corner, eps=2.0):
            return QPointF(w.p2.x() - corner.x(), w.p2.y() - corner.y())
        if close_pt(w.p2, corner, eps=2.0):
            return QPointF(w.p1.x() - corner.x(), w.p1.y() - corner.y())
        return None

    def _update_corner_marker(self):
        if self._corner_marker is not None:
            try:
                self.canvas.scene().removeItem(self._corner_marker)
            except Exception:
                pass
            self._corner_marker = None

        if self.wallA_id is None or self.wallB_id is None:
            return
        if self.wallA_id == self.wallB_id:
            return

        corner = self._walls_shared_corner(self.wallA_id, self.wallB_id)
        if corner is None:
            return

        self._corner_marker = CornerMarker(corner, r=7.0)
        self.canvas.scene().addItem(self._corner_marker)

    # ---------------- Snap for walls ----------------
    def _all_wall_endpoints(self) -> List[QPointF]:
        pts: List[QPointF] = []
        for w in self.walls.values():
            pts.append(w.p1)
            pts.append(w.p2)
        return pts

    def _snap_point(self, p: QPointF) -> QPointF:
        best = None
        best_d2 = 1e18
        for e in self._all_wall_endpoints():
            d2 = dist2(p, e)
            if d2 < best_d2:
                best_d2 = d2
                best = e
        if best is not None and best_d2 <= (self.snap_eps_px * self.snap_eps_px):
            return QPointF(best.x(), best.y())
        return p

    # ---------------- Corner items ----------------
    def _clear_corner_items(self):
        for _, it in list(self.corner_items.items()):
            try:
                self.canvas.scene().removeItem(it)
            except Exception:
                pass
        self.corner_items = {}

    def _rebuild_corner_items(self):
        self._clear_corner_items()
        for w in self.walls.values():
            for p in (w.p1, w.p2):
                key = (int(round(p.x())), int(round(p.y())))
                if key in self.corner_items:
                    continue
                item = CornerPointItem(key, QPointF(key[0], key[1]), r=5.0)
                self.corner_items[key] = item
                self.canvas.scene().addItem(item)

    # ---------------- Styles ----------------
    def _apply_styles(self):
        # Corner dots base + active endpoint highlight
        for key, it in self.corner_items.items():
            pen = QPen(QColor(90, 90, 90))
            pen.setWidth(2)
            it.setPen(pen)
            it.setBrush(QBrush(QColor(255, 255, 255)))

        for key in self._active_wall_corner_keys:
            if key in self.corner_items:
                pen = QPen(QColor(0, 200, 255))
                pen.setWidth(4)
                self.corner_items[key].setPen(pen)
                self.corner_items[key].setBrush(QBrush(QColor(230, 255, 255)))

        # Walls base
        for wid, item in self.wall_items.items():
            pen = QPen(QColor(180, 0, 0))
            pen.setWidth(3)
            item.setPen(pen)

        # Points base (small + minimal)
        for pid, item in self.point_items.items():
            pen = QPen(QColor(0, 0, 0))
            pen.setWidth(2)
            item.setPen(pen)
            item.setBrush(QBrush(QColor(0, 0, 0)))

        # Selected walls (list selection)
        selected_walls = set(self._selected_wall_ids_from_list())
        for wid in selected_walls:
            if wid in self.wall_items:
                pen = QPen(QColor(40, 120, 255))
                pen.setWidth(5)
                self.wall_items[wid].setPen(pen)

        # A/B highlight
        if self.wallA_id in self.wall_items:
            pen = QPen(QColor(0, 200, 0))
            pen.setWidth(6)
            self.wall_items[self.wallA_id].setPen(pen)

        if self.wallB_id in self.wall_items:
            pen = QPen(QColor(255, 140, 0))
            pen.setWidth(6)
            self.wall_items[self.wallB_id].setPen(pen)

        # Selected points (list selection)
        selected_pts = set(self._selected_point_ids_from_list())
        for pid in selected_pts:
            if pid in self.point_items:
                pen = QPen(QColor(255, 220, 0))
                pen.setWidth(3)
                self.point_items[pid].setPen(pen)
                self.point_items[pid].setBrush(QBrush(QColor(255, 220, 0)))

    # ---------------- Actions ----------------
    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load map image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        img = QImage(path)
        if img.isNull():
            self._warn("Load", "Failed to load image.")
            return

        self.map_qimage = img
        self.canvas.setImage(img)
        self.le_loaded.setText(os.path.basename(path))

        # reset all data
        self._clear_all_walls(ask=False)
        self._clear_corner_items()
        self._clear_all_points(ask=False)

        self.wallA_id = None
        self.wallB_id = None
        self._wall_draw_first = None
        self._active_wall_corner_keys = set()
        self._clear_temp_wall_first_marker()

        self._wall_id_seq = 1
        self._pt_id_seq = 1

        self.spin_lenA.setValue(1.0)
        self.spin_lenB.setValue(1.0)
        self.spin_d_to_A.setValue(0.0)
        self.spin_d_to_B.setValue(0.0)
        self.lbl_pf_feedback.setText("")

        self._refresh_all()

    def on_canvas_clicked(self, x: float, y: float):
        if not self.canvas.has_image():
            return
        if self.mode != self.MODE_WALLS:
            return

        p = QPointF(x, y)
        p = self._snap_point(p)

        if self._wall_draw_first is None:
            self._wall_draw_first = p

            # show temporary first-point marker
            self._set_temp_wall_first_marker(p)

            # highlight if it matches an existing corner dot (after snapping)
            k = (int(round(p.x())), int(round(p.y())))
            self._active_wall_corner_keys = set()
            if k in self.corner_items:
                self._active_wall_corner_keys.add(k)

            self._apply_styles()
            self._refresh_status_texts()
            return

        p1 = self._wall_draw_first
        p2 = p
        self._wall_draw_first = None

        # add second endpoint highlight if it is a corner dot
        k2 = (int(round(p2.x())), int(round(p2.y())))
        if k2 in self.corner_items:
            self._active_wall_corner_keys.add(k2)

        # clear temp marker when finishing
        self._clear_temp_wall_first_marker()

        # Ignore too short
        if dist2(p1, p2) < 9.0:
            self._active_wall_corner_keys = set()
            self._refresh_all()
            return

        wid = self._wall_id_seq
        self._wall_id_seq += 1

        w = Wall(wall_id=wid, p1=p1, p2=p2)
        item = WallItem(wid, p1, p2)
        self.canvas.scene().addItem(item)

        self.walls[wid] = w
        self.wall_items[wid] = item

        # active highlights only while in-progress
        self._active_wall_corner_keys = set()
        self._refresh_all()

    def on_wall_cancel(self):
        self._wall_draw_first = None
        self._active_wall_corner_keys = set()
        self._clear_temp_wall_first_marker()
        self._refresh_status_texts()
        self._apply_styles()

    def on_wall_delete_selected(self):
        ids = self._selected_wall_ids_from_list()
        if not ids:
            return
        if not self._confirm("Delete Walls", f"Delete {len(ids)} selected wall(s)?"):
            return
        for wid in ids:
            self._remove_wall(wid)

        if self.wallA_id not in self.walls:
            self.wallA_id = None
        if self.wallB_id not in self.walls:
            self.wallB_id = None

        self._refresh_all()

    def on_wall_clear_all(self):
        if not self.walls:
            return
        if not self._confirm("Clear Walls", "Clear all walls?"):
            return
        self._clear_all_walls(ask=False)
        self.wallA_id = None
        self.wallB_id = None
        self._refresh_all()

    def _remove_wall(self, wid: int):
        if wid in self.wall_items:
            try:
                self.canvas.scene().removeItem(self.wall_items[wid])
            except Exception:
                pass
            del self.wall_items[wid]
        if wid in self.walls:
            del self.walls[wid]

    def _clear_all_walls(self, ask: bool = True):
        if ask and self.walls and not self._confirm("Clear Walls", "Clear all walls?"):
            return
        for wid in list(self.walls.keys()):
            self._remove_wall(wid)
        self._wall_draw_first = None
        self._clear_corner_items()
        self._active_wall_corner_keys = set()
        self._clear_temp_wall_first_marker()

    def on_wall_list_selection_changed(self):
        selected = set(self._selected_wall_ids_from_list())
        self.canvas.scene().blockSignals(True)
        for wid, item in self.wall_items.items():
            item.setSelected(wid in selected)
        self.canvas.scene().blockSignals(False)
        self._apply_styles()

    def on_scene_selection_changed(self):
        sel_wall_ids = []
        sel_pt_ids = []
        for it in self.canvas.scene().selectedItems():
            if isinstance(it, WallItem):
                sel_wall_ids.append(it.wall_id)
            elif isinstance(it, PointItem):
                sel_pt_ids.append(it.point_id)

        self.list_walls.blockSignals(True)
        self.list_walls.clearSelection()
        if sel_wall_ids:
            sel_set = set(sel_wall_ids)
            for i in range(self.list_walls.count()):
                item = self.list_walls.item(i)
                wid = item.data(Qt.UserRole)
                if wid in sel_set:
                    item.setSelected(True)
        self.list_walls.blockSignals(False)

        self.list_points.blockSignals(True)
        self.list_points.clearSelection()
        if sel_pt_ids:
            sel_set = set(sel_pt_ids)
            for i in range(self.list_points.count()):
                item = self.list_points.item(i)
                pid = item.data(Qt.UserRole)
                if pid in sel_set:
                    item.setSelected(True)
        self.list_points.blockSignals(False)

        self._apply_styles()

    def on_mark_A(self):
        ids = self._selected_wall_ids_from_list()
        if len(ids) != 1:
            self._warn("Mark A", "Select exactly 1 wall to mark as A.")
            return

        new_a = ids[0]
        if self.wallB_id is not None:
            if not self._are_walls_adjacent(new_a, self.wallB_id):
                self._warn("Mark A", "Wall A must be adjacent to Wall B (they must share a corner). Select a different wall.")
                return

        self.wallA_id = new_a
        if self.wallB_id == self.wallA_id:
            self.wallB_id = None
        self._refresh_all()

    def on_mark_B(self):
        ids = self._selected_wall_ids_from_list()
        if len(ids) != 1:
            self._warn("Mark B", "Select exactly 1 wall to mark as B.")
            return

        new_b = ids[0]
        if self.wallA_id is not None:
            if not self._are_walls_adjacent(self.wallA_id, new_b):
                self._warn("Mark B", "Wall B must be adjacent to Wall A (they must share a corner). Select a different wall.")
                return

        self.wallB_id = new_b
        if self.wallA_id == self.wallB_id:
            self.wallA_id = None
        self._refresh_all()

    def on_add_point(self):
        if not self.canvas.has_image():
            return
        if self.wallA_id is None or self.wallB_id is None:
            self._warn("Add Point", "Set both Wall A and Wall B first.")
            return
        if self.wallA_id == self.wallB_id:
            self._warn("Add Point", "Wall A and Wall B must be different.")
            return

        corner = self._walls_shared_corner(self.wallA_id, self.wallB_id)
        if corner is None:
            self._warn("Add Point", "Walls A and B are not adjacent (they must share a corner endpoint).")
            return

        wA = self.walls[self.wallA_id]
        wB = self.walls[self.wallB_id]
        vA = self._vector_from_corner(wA, corner)
        vB = self._vector_from_corner(wB, corner)
        if vA is None or vB is None:
            self._warn("Add Point", "Failed to compute vectors from the shared corner.")
            return

        lenA_real = float(self.spin_lenA.value())
        lenB_real = float(self.spin_lenB.value())
        d_to_B = float(self.spin_d_to_B.value())
        d_to_A = float(self.spin_d_to_A.value())

        if lenA_real <= 0 or lenB_real <= 0:
            self._warn("Add Point", "Wall A and Wall B real lengths must be > 0.")
            return

        rA = d_to_B / lenA_real
        rB = d_to_A / lenB_real

        px = float(corner.x()) + rA * float(vA.x()) + rB * float(vB.x())
        py = float(corner.y()) + rA * float(vA.y()) + rB * float(vB.y())

        w, h = self.canvas.image_size()
        px = clamp(px, 0.0, float(w - 1))
        py = clamp(py, 0.0, float(h - 1))

        pid = self._pt_id_seq
        self._pt_id_seq += 1

        pt = SPoint(point_id=pid, p=QPointF(px, py))
        item = PointItem(pid, pt.p, r=3.0)

        self.points[pid] = pt
        self.point_items[pid] = item
        self.canvas.scene().addItem(item)

        self.lbl_pf_feedback.setText(f"Added Point {pid} at ({int(px)}, {int(py)}).")
        self._refresh_all()

    def on_point_list_selection_changed(self):
        selected = set(self._selected_point_ids_from_list())
        self.canvas.scene().blockSignals(True)
        for pid, item in self.point_items.items():
            item.setSelected(pid in selected)
        self.canvas.scene().blockSignals(False)
        self._apply_styles()

    def on_point_delete_selected(self):
        ids = self._selected_point_ids_from_list()
        if not ids:
            return
        if not self._confirm("Delete Points", f"Delete {len(ids)} selected point(s)?"):
            return
        for pid in ids:
            self._remove_point(pid)
        self._refresh_all()

    def on_point_clear_all(self):
        if not self.points:
            return
        if not self._confirm("Clear Points", "Clear all points?"):
            return
        self._clear_all_points(ask=False)
        self._refresh_all()

    def _remove_point(self, pid: int):
        if pid in self.point_items:
            try:
                self.canvas.scene().removeItem(self.point_items[pid])
            except Exception:
                pass
            del self.point_items[pid]
        if pid in self.points:
            del self.points[pid]

    def _clear_all_points(self, ask: bool = True):
        if ask and self.points and not self._confirm("Clear Points", "Clear all points?"):
            return
        for pid in list(self.points.keys()):
            self._remove_point(pid)

    def on_save_image(self):
        if self.map_qimage is None or self.map_qimage.isNull():
            self._warn("Save", "Load a map image first.")
            return
        if not self.points:
            self._warn("Save", "No points to save. Add at least one point.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save output image", "output_points.png", "PNG (*.png);;JPG (*.jpg *.jpeg)"
        )
        if not out_path:
            return

        base = QImage(self.map_qimage)
        painter = QPainter(base)
        painter.setRenderHint(QPainter.Antialiasing, True)

        for pid in sorted(self.points.keys()):
            p = self.points[pid].p
            pen = QPen(QColor(0, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            r = 3
            painter.drawEllipse(QPointF(p.x(), p.y()), r, r)

        painter.end()

        ok = base.save(out_path)
        if not ok:
            self._warn("Save", "Failed to save image.")
            return
        self._info("Saved", f"Saved:\n{out_path}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()