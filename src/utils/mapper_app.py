# mapper.py
# PyQt5 app:
# 1) Homography mapping (CAMERA point -> MAP point pairs)
# 2) Entry Regions (multiple polygons on MAP)
# 3) Map Points (single points on MAP, one per line)
# 4) Room Boundary (single polygon on MAP)
#
# Dependencies: PyQt5, opencv-python, numpy
#
# Run:
#   python -m src.utils.mapper_app

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QPen, QBrush, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, QLabel, QLineEdit,
    QTextEdit, QTabWidget, QGroupBox, QFormLayout, QSlider,
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QListWidget, QListWidgetItem, QSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsTextItem,
    QGraphicsEllipseItem,
    QSizePolicy
)
class DraggablePointItem(QGraphicsEllipseItem):
    """A draggable point that notifies the owning ImageCanvas when moved."""

    def __init__(self, x: float, y: float, r: float, kind: str, index: int, canvas_ref, color: QColor):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.setPos(QPointF(x, y))
        self.kind = kind
        self.index = index
        self.canvas_ref = canvas_ref

        pen = QPen(color)
        pen.setWidth(2)
        self.setPen(pen)
        self.setBrush(QBrush(color))

        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsScenePositionChanges, True)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange and self.canvas_ref is not None:
            # Clamp to image bounds if known
            if self.canvas_ref._has_image:
                x = max(0, min(self.canvas_ref._image_w - 1, float(value.x())))
                y = max(0, min(self.canvas_ref._image_h - 1, float(value.y())))
                return QPointF(x, y)
        if change == QGraphicsEllipseItem.ItemPositionHasChanged and self.canvas_ref is not None:
            p = self.pos()
            self.canvas_ref._notify_point_moved(self.kind, self.index, int(round(p.x())), int(round(p.y())))
        return super().itemChange(change, value)


APP_NAME = "mapper"


def cv_bgr_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr is None:
        return QImage()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def ensure_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        return False


def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


@dataclass
class MappingPair:
    fx: int
    fy: int
    mx: Optional[int] = None
    my: Optional[int] = None

    @property
    def complete(self) -> bool:
        return self.mx is not None and self.my is not None


@dataclass
class RegionPolygon:
    points: List[Tuple[int, int]]


class ImageCanvas(QGraphicsView):
    pointClicked = pyqtSignal(int, int)
    pointMoved = pyqtSignal(str, int, int, int)  # kind, index, x, y

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)

        self._image_w = 0
        self._image_h = 0
        self._overlay_items = []
        self._user_scaled = False

        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._has_image = False

        # Click/drag state
        self._press_pos_view: Optional[QPoint] = None
        self._panning = False
        self._pan_start: Optional[QPoint] = None

    def has_image(self) -> bool:
        return self._has_image

    def setImage(self, qimg: QImage):
        if qimg.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            self._image_w, self._image_h = 0, 0
            self._has_image = False
            self._clear_overlays()
            self._user_scaled = False
            return

        pix = QPixmap.fromImage(qimg)
        self._pixmap_item.setPixmap(pix)
        self._image_w, self._image_h = qimg.width(), qimg.height()
        self.scene().setSceneRect(0, 0, self._image_w, self._image_h)
        self._has_image = True
        self.resetZoom()

    def resetZoom(self):
        self._user_scaled = False
        self.resetTransform()
        if self._has_image:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self._has_image:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.04 if delta > 0 else 1 / 1.04
        self._user_scaled = True
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._has_image and not self._user_scaled:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def _notify_point_moved(self, kind: str, index: int, x: int, y: int):
        self.pointMoved.emit(kind, index, x, y)

    def mousePressEvent(self, event):
        if not self._has_image:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton:
            self._press_pos_view = event.pos()

            # If clicking on a draggable point, let Qt handle moving it.
            item = self.itemAt(event.pos())
            if isinstance(item, DraggablePointItem):
                self._panning = False
                self._pan_start = None
                super().mousePressEvent(event)
                return

            # Otherwise start potential panning (on background/pixmap)
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._has_image and self._panning and self._pan_start is not None:
            # Pan the view by moving scrollbars
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

        if event.button() == Qt.LeftButton:
            # End panning
            if self._panning:
                self.unsetCursor()
                # Treat as a click only if movement was tiny
                if self._press_pos_view is not None:
                    moved = (event.pos() - self._press_pos_view).manhattanLength()
                else:
                    moved = 999

                self._panning = False
                self._pan_start = None

                if moved <= 3:
                    pos = self.mapToScene(event.pos())
                    x = clamp_int(pos.x(), 0, max(0, self._image_w - 1))
                    y = clamp_int(pos.y(), 0, max(0, self._image_h - 1))
                    self.pointClicked.emit(x, y)
                self._press_pos_view = None
                event.accept()
                return

        super().mouseReleaseEvent(event)

    def _clear_overlays(self):
        for it in self._overlay_items:
            try:
                self.scene().removeItem(it)
            except Exception:
                pass
        self._overlay_items = []

    def draw_points_with_labels(self, points: List[Tuple]):
        """Draw points. Each entry can be either:
        (x, y, label, circle_color, text_color)
        or
        (x, y, label, circle_color, text_color, kind, index)
        """
        self._clear_overlays()
        if not self._has_image:
            return

        for p in points:
            if len(p) == 5:
                x, y, label, circle_color, text_color = p
                kind = None
                idx = None
            else:
                x, y, label, circle_color, text_color, kind, idx = p

            r = 5

            if kind is not None and idx is not None:
                item = DraggablePointItem(x, y, r, str(kind), int(idx), self, circle_color)
                self.scene().addItem(item)
                self._overlay_items.append(item)
            else:
                path = QPainterPath()
                path.addEllipse(QPointF(x, y), r, r)
                item = QGraphicsPathItem(path)
                pen = QPen(circle_color)
                pen.setWidth(2)
                item.setPen(pen)
                item.setBrush(QBrush(circle_color))
                self.scene().addItem(item)
                self._overlay_items.append(item)

            txt = QGraphicsTextItem(str(label))
            txt.setDefaultTextColor(text_color)
            txt.setPos(x + 8, y - 12)
            self.scene().addItem(txt)
            self._overlay_items.append(txt)


    def draw_polylines(self, polylines: List[Tuple[List[Tuple[int, int]], bool, QColor, int]]):
        """Draw polylines/polygons without clearing existing overlays."""
        if not self._has_image:
            return

        for pts, closed, color, width in polylines:
            if len(pts) < 2:
                continue
            path = QPainterPath(QPointF(pts[0][0], pts[0][1]))
            for (x, y) in pts[1:]:
                path.lineTo(QPointF(x, y))
            if closed and len(pts) >= 3:
                path.closeSubpath()

            item = QGraphicsPathItem(path)
            pen = QPen(color)
            pen.setWidth(width)
            item.setPen(pen)
            item.setBrush(QBrush(Qt.NoBrush))
            self.scene().addItem(item)
            self._overlay_items.append(item)


class VideoFrameProvider:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.path: str = ""
        self.current_index: int = 0

    def open(self, path: str) -> bool:
        self.close()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self.cap = cap
        self.path = path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.current_index = 0
        return True

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.total_frames = 0
        self.path = ""
        self.current_index = 0

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        if self.cap is None or self.total_frames <= 0:
            return None
        index = max(0, min(self.total_frames - 1, int(index)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.current_index = index
        return frame


class MapperMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 850)

        self.map_image_bgr: Optional[np.ndarray] = None
        self.camera_frame_bgr: Optional[np.ndarray] = None
        self.video = VideoFrameProvider()

        self.mapping_pairs: List[MappingPair] = []
        self.mapping_expected: str = "frame"

        self.region_polys: List[RegionPolygon] = []
        self.current_poly: List[Tuple[int, int]] = []
        self.current_poly_confirmed: bool = False

        self.map_points: List[Tuple[int, int]] = []

        self.room_boundary: List[Tuple[int, int]] = []
        self.room_boundary_confirmed: bool = False

        self.mapping_dirty = False
        self.regions_dirty = False
        self.points_dirty = False
        self.boundary_dirty = False

        self._build_ui()
        self._wire_signals()
        self._refresh_all_ui()
        self._update_video_controls_compactness()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)

        main_split = QSplitter(Qt.Horizontal)
        self.main_split = main_split
        # NOTE: setCollapsible must be called after widgets are added to the splitter
        root_layout.addWidget(main_split)

        # LEFT
        left = QWidget()
        self.left_panel = left
        left.setMinimumWidth(170)  # allow real squish
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        setup_group = QGroupBox("Project Setup")
        setup_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        setup_form = QFormLayout(setup_group)
        setup_form.setVerticalSpacing(8)
        setup_form.setHorizontalSpacing(10)

        self.btn_load_map = QPushButton("Load Map Image")
        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_frame = QPushButton("Load Frame Image")
        self.btn_clear_camera = QPushButton("Clear Camera Source")

        self.le_output_dir = QLineEdit()
        self.le_output_dir.setReadOnly(True)
        self.btn_choose_dir = QPushButton("Choose...")

        dir_row = QWidget()
        dir_row_l = QHBoxLayout(dir_row)
        dir_row_l.setContentsMargins(0, 0, 0, 0)
        dir_row_l.setSpacing(6)
        dir_row_l.addWidget(self.le_output_dir, 1)
        dir_row_l.addWidget(self.btn_choose_dir)

        self.le_project_name = QLineEdit("project")

        setup_form.addRow("Map:", self.btn_load_map)
        cam_row = QWidget()
        cam_row_l = QHBoxLayout(cam_row)
        cam_row_l.setContentsMargins(0, 0, 0, 0)
        cam_row_l.setSpacing(6)
        cam_row_l.addWidget(self.btn_load_video)
        cam_row_l.addWidget(self.btn_load_frame)
        cam_row_l.addWidget(self.btn_clear_camera)
        setup_form.addRow("Camera:", cam_row)
        setup_form.addRow("Save Root Dir:", dir_row)
        setup_form.addRow("Project Name:", self.le_project_name)
        left_layout.addWidget(setup_group)

        # VIDEO CONTROLS
        video_group = QGroupBox("Video Controls")
        self.video_group = video_group
        video_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(10, 10, 10, 10)
        video_layout.setSpacing(8)

        self.lbl_video_info = QLabel("No video loaded.")
        self.lbl_video_info.setWordWrap(True)
        self.lbl_video_info.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setEnabled(False)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        nav_row = QWidget()
        nav_l = QHBoxLayout(nav_row)
        nav_l.setContentsMargins(0, 0, 0, 0)
        nav_l.setSpacing(6)

        self.btn_first = QPushButton("<<")
        self.btn_prev = QPushButton("<")
        self.btn_next = QPushButton(">")
        self.btn_last = QPushButton(">>")

        for b in (self.btn_first, self.btn_prev, self.btn_next, self.btn_last):
            b.setEnabled(False)
            # allow squish hard
            b.setMinimumWidth(1)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            nav_l.addWidget(b)

        self.video_nav_buttons = [self.btn_first, self.btn_prev, self.btn_next, self.btn_last]

        video_layout.addWidget(self.lbl_video_info)
        video_layout.addWidget(self.slider_frame)
        video_layout.addWidget(nav_row)
        left_layout.addWidget(video_group)

        # TABS
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs, 1)

        # Mapping tab
        map_tab = QWidget()
        map_tab_layout = QVBoxLayout(map_tab)
        map_tab_layout.setContentsMargins(8, 8, 8, 8)
        map_tab_layout.setSpacing(8)

        self.lbl_status = QLabel("Load images to begin.")
        self.lbl_status.setStyleSheet("font-weight: 600;")

        self.txt_instructions = QTextEdit()
        self.txt_instructions.setReadOnly(True)
        self.txt_instructions.setMinimumHeight(90)

        self.table_pairs = QTableWidget(0, 4)
        self.table_pairs.setHorizontalHeaderLabels(["#", "Frame (x,y)", "Map (x,y)", "Status"])
        self.table_pairs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_pairs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_pairs.horizontalHeader().setStretchLastSection(True)

        btn_row = QWidget()
        btn_l = QHBoxLayout(btn_row)
        btn_l.setContentsMargins(0, 0, 0, 0)
        btn_l.setSpacing(6)
        self.btn_map_undo = QPushButton("Undo")
        self.btn_map_delete = QPushButton("Delete Selected")
        self.btn_map_clear = QPushButton("Clear All")
        self.btn_map_save = QPushButton("Save Mapping TXT")
        for b in (self.btn_map_undo, self.btn_map_delete, self.btn_map_clear, self.btn_map_save):
            btn_l.addWidget(b)

        map_tab_layout.addWidget(self.lbl_status)
        map_tab_layout.addWidget(self.txt_instructions)
        map_tab_layout.addWidget(self.table_pairs, 1)
        map_tab_layout.addWidget(btn_row)

        # Regions tab
        reg_tab = QWidget()
        reg_layout = QVBoxLayout(reg_tab)
        reg_layout.setContentsMargins(8, 8, 8, 8)
        reg_layout.setSpacing(8)

        reg_top = QWidget()
        reg_top_l = QHBoxLayout(reg_top)
        reg_top_l.setContentsMargins(0, 0, 0, 0)
        reg_top_l.setSpacing(8)

        self.spin_min_points = QSpinBox()
        self.spin_min_points.setMinimum(3)
        self.spin_min_points.setMaximum(20)
        self.spin_min_points.setValue(3)

        reg_top_l.addWidget(QLabel("Min points:"))
        reg_top_l.addWidget(self.spin_min_points)
        reg_top_l.addStretch(1)

        self.txt_reg_help = QTextEdit()
        self.txt_reg_help.setReadOnly(True)
        self.txt_reg_help.setMinimumHeight(90)

        self.list_polys = QListWidget()
        self.list_polys.setMinimumHeight(140)

        reg_btn_row = QWidget()
        reg_btn_l = QHBoxLayout(reg_btn_row)
        reg_btn_l.setContentsMargins(0, 0, 0, 0)
        reg_btn_l.setSpacing(6)
        self.btn_reg_undo = QPushButton("Undo Point")
        self.btn_reg_confirm = QPushButton("Confirm Polygon")
        self.btn_reg_new = QPushButton("New Polygon")
        self.btn_reg_delete = QPushButton("Delete Selected Polygon")
        self.btn_reg_clear = QPushButton("Clear All")
        self.btn_reg_save = QPushButton("Save Regions TXT")
        for b in (self.btn_reg_undo, self.btn_reg_confirm, self.btn_reg_new,
                  self.btn_reg_delete, self.btn_reg_clear, self.btn_reg_save):
            reg_btn_l.addWidget(b)

        reg_layout.addWidget(reg_top)
        reg_layout.addWidget(self.txt_reg_help)
        reg_layout.addWidget(QLabel("Saved Polygons:"))
        reg_layout.addWidget(self.list_polys, 1)
        reg_layout.addWidget(reg_btn_row)

        # Map Points tab
        pts_tab = QWidget()
        pts_layout = QVBoxLayout(pts_tab)
        pts_layout.setContentsMargins(8, 8, 8, 8)
        pts_layout.setSpacing(8)

        self.txt_pts_help = QTextEdit()
        self.txt_pts_help.setReadOnly(True)
        self.txt_pts_help.setMinimumHeight(90)

        self.table_pts = QTableWidget(0, 2)
        self.table_pts.setHorizontalHeaderLabels(["x", "y"])
        self.table_pts.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_pts.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_pts.horizontalHeader().setStretchLastSection(True)

        pts_btn_row = QWidget()
        pts_btn_l = QHBoxLayout(pts_btn_row)
        pts_btn_l.setContentsMargins(0, 0, 0, 0)
        pts_btn_l.setSpacing(6)
        self.btn_pts_undo = QPushButton("Undo")
        self.btn_pts_delete = QPushButton("Delete Selected")
        self.btn_pts_clear = QPushButton("Clear All")
        self.btn_pts_save = QPushButton("Save Points TXT")
        for b in (self.btn_pts_undo, self.btn_pts_delete, self.btn_pts_clear, self.btn_pts_save):
            pts_btn_l.addWidget(b)

        pts_layout.addWidget(self.txt_pts_help)
        pts_layout.addWidget(self.table_pts, 1)
        pts_layout.addWidget(pts_btn_row)

        # Boundary tab
        bnd_tab = QWidget()
        bnd_layout = QVBoxLayout(bnd_tab)
        bnd_layout.setContentsMargins(8, 8, 8, 8)
        bnd_layout.setSpacing(8)

        bnd_top = QWidget()
        bnd_top_l = QHBoxLayout(bnd_top)
        bnd_top_l.setContentsMargins(0, 0, 0, 0)
        bnd_top_l.setSpacing(8)

        self.spin_bnd_min_points = QSpinBox()
        self.spin_bnd_min_points.setMinimum(3)
        self.spin_bnd_min_points.setMaximum(50)
        self.spin_bnd_min_points.setValue(4)

        bnd_top_l.addWidget(QLabel("Min points:"))
        bnd_top_l.addWidget(self.spin_bnd_min_points)
        bnd_top_l.addStretch(1)

        self.txt_bnd_help = QTextEdit()
        self.txt_bnd_help.setReadOnly(True)
        self.txt_bnd_help.setMinimumHeight(90)

        bnd_btn_row = QWidget()
        bnd_btn_l = QHBoxLayout(bnd_btn_row)
        bnd_btn_l.setContentsMargins(0, 0, 0, 0)
        bnd_btn_l.setSpacing(6)
        self.btn_bnd_undo = QPushButton("Undo Point")
        self.btn_bnd_confirm = QPushButton("Confirm Boundary")
        self.btn_bnd_clear = QPushButton("Clear")
        self.btn_bnd_save = QPushButton("Save Boundary TXT")
        for b in (self.btn_bnd_undo, self.btn_bnd_confirm, self.btn_bnd_clear, self.btn_bnd_save):
            bnd_btn_l.addWidget(b)

        bnd_layout.addWidget(bnd_top)
        bnd_layout.addWidget(self.txt_bnd_help)
        bnd_layout.addWidget(bnd_btn_row)

        self.tabs.addTab(map_tab, "Mapping")
        self.tabs.addTab(reg_tab, "Entry Regions")
        self.tabs.addTab(pts_tab, "Map Points")
        self.tabs.addTab(bnd_tab, "Room Boundary")

        main_split.addWidget(left)

        # RIGHT
        right_split = QSplitter(Qt.Horizontal)

        cam_wrap = QWidget()
        cam_l = QVBoxLayout(cam_wrap)
        cam_l.setContentsMargins(8, 8, 4, 8)
        cam_l.setSpacing(6)

        cam_header = QWidget()
        cam_header_l = QHBoxLayout(cam_header)
        cam_header_l.setContentsMargins(0, 0, 0, 0)
        cam_header_l.setSpacing(8)
        cam_header_l.addWidget(QLabel("Camera / Frame"), 1)
        self.btn_cam_reset_zoom = QPushButton("Reset Zoom")
        self.btn_cam_reset_zoom.setFixedWidth(110)
        cam_header_l.addWidget(self.btn_cam_reset_zoom)

        self.canvas_cam = ImageCanvas("camera")
        self.canvas_cam.setMinimumSize(320, 320)
        cam_l.addWidget(cam_header)
        cam_l.addWidget(self.canvas_cam, 1)

        map_wrap = QWidget()
        map_l = QVBoxLayout(map_wrap)
        map_l.setContentsMargins(4, 8, 8, 8)
        map_l.setSpacing(6)

        map_header = QWidget()
        map_header_l = QHBoxLayout(map_header)
        map_header_l.setContentsMargins(0, 0, 0, 0)
        map_header_l.setSpacing(8)
        map_header_l.addWidget(QLabel("Map"), 1)
        self.btn_map_reset_zoom = QPushButton("Reset Zoom")
        self.btn_map_reset_zoom.setFixedWidth(110)
        map_header_l.addWidget(self.btn_map_reset_zoom)

        self.canvas_map = ImageCanvas("map")
        self.canvas_map.setMinimumSize(320, 320)
        map_l.addWidget(map_header)
        map_l.addWidget(self.canvas_map, 1)

        right_split.addWidget(cam_wrap)
        right_split.addWidget(map_wrap)
        right_split.setSizes([700, 700])

        main_split.addWidget(right_split)
        # Now that widgets are added, we can allow collapsing the left panel
        main_split.setCollapsible(0, True)
        main_split.setChildrenCollapsible(False)
        main_split.setSizes([360, 1140])
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)

        self._set_mapping_instructions()
        self._set_regions_instructions()
        self._set_points_instructions()
        self._set_boundary_instructions()

    def _wire_signals(self):
        # Load/save setup
        self.btn_load_map.clicked.connect(self.on_load_map)
        self.btn_load_video.clicked.connect(self.on_load_video)
        self.btn_load_frame.clicked.connect(self.on_load_frame_image)
        self.btn_clear_camera.clicked.connect(self.on_clear_camera)
        self.btn_choose_dir.clicked.connect(self.on_choose_dir)

        # Video controls
        self.slider_frame.valueChanged.connect(self.on_frame_slider_changed)
        self.btn_first.clicked.connect(lambda: self._jump_frame(0))
        self.btn_prev.clicked.connect(lambda: self._jump_frame(self.video.current_index - 1))
        self.btn_next.clicked.connect(lambda: self._jump_frame(self.video.current_index + 1))
        self.btn_last.clicked.connect(lambda: self._jump_frame(max(0, self.video.total_frames - 1)))

        # Canvases
        self.canvas_cam.pointClicked.connect(self.on_camera_clicked)
        self.canvas_map.pointClicked.connect(self.on_map_clicked)
        self.canvas_cam.pointMoved.connect(self.on_point_moved)
        self.canvas_map.pointMoved.connect(self.on_point_moved)

        # Zoom reset
        self.btn_cam_reset_zoom.clicked.connect(self.canvas_cam.resetZoom)
        self.btn_map_reset_zoom.clicked.connect(self.canvas_map.resetZoom)

        # Tabs
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Adaptive collapsing of Video Controls as left panel shrinks
        self.main_split.splitterMoved.connect(lambda _pos, _idx: self._update_video_controls_compactness())

        # Mapping actions
        self.btn_map_undo.clicked.connect(self.on_mapping_undo)
        self.btn_map_delete.clicked.connect(self.on_mapping_delete_selected)
        self.btn_map_clear.clicked.connect(self.on_mapping_clear_all)
        self.btn_map_save.clicked.connect(self.on_mapping_save)

        # Regions actions
        self.btn_reg_undo.clicked.connect(self.on_regions_undo)
        self.btn_reg_confirm.clicked.connect(self.on_regions_confirm)
        self.btn_reg_new.clicked.connect(self.on_regions_new_polygon)
        self.btn_reg_delete.clicked.connect(self.on_regions_delete_selected)
        self.btn_reg_clear.clicked.connect(self.on_regions_clear_all)
        self.btn_reg_save.clicked.connect(self.on_regions_save)

        # Map Points actions
        self.btn_pts_undo.clicked.connect(self.on_points_undo)
        self.btn_pts_delete.clicked.connect(self.on_points_delete_selected)
        self.btn_pts_clear.clicked.connect(self.on_points_clear_all)
        self.btn_pts_save.clicked.connect(self.on_points_save)

        # Room Boundary actions
        self.btn_bnd_undo.clicked.connect(self.on_boundary_undo)
        self.btn_bnd_confirm.clicked.connect(self.on_boundary_confirm)
        self.btn_bnd_clear.clicked.connect(self.on_boundary_clear)
        self.btn_bnd_save.clicked.connect(self.on_boundary_save)

    def on_point_moved(self, kind: str, index: int, x: int, y: int):
        """Handle drag-moving of existing points."""
        try:
            kind = str(kind)
            index = int(index)
        except Exception:
            return

        if kind == 'mapping_frame':
            if 0 <= index < len(self.mapping_pairs):
                self.mapping_pairs[index].fx = x
                self.mapping_pairs[index].fy = y
                self.mapping_dirty = True
                self._refresh_all_ui()
            return

        if kind == 'mapping_map':
            if 0 <= index < len(self.mapping_pairs):
                self.mapping_pairs[index].mx = x
                self.mapping_pairs[index].my = y
                self.mapping_dirty = True
                self._refresh_all_ui()
            return

        if kind == 'map_points':
            if 0 <= index < len(self.map_points):
                self.map_points[index] = (x, y)
                self.points_dirty = True
                self._refresh_all_ui()
            return

        if kind == 'boundary':
            if 0 <= index < len(self.room_boundary) and not self.room_boundary_confirmed:
                self.room_boundary[index] = (x, y)
                self.boundary_dirty = True
                self._refresh_all_ui()
            return

        if kind == 'entry_current':
            if 0 <= index < len(self.current_poly) and not self.current_poly_confirmed:
                self.current_poly[index] = (x, y)
                self.regions_dirty = True
                self._refresh_all_ui()
            return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_video_controls_compactness()

    def _update_video_controls_compactness(self):
        """Squish first (slider stretches), then collapse buttons progressively as panel narrows."""
        if not hasattr(self, "left_panel"):
            return
        w = self.left_panel.width()

        # If panel is extremely narrow, hide the entire Video Controls group
        if hasattr(self, 'video_group'):
            if w < 180:
                self.video_group.hide()
            else:
                self.video_group.show()

        # tuned thresholds
        wide = 360
        medium = 320
        narrow = 280
        tiny = 240
        micro = 210

        # always let slider eat available space
        self.slider_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        if w >= wide:
            self.lbl_video_info.show()
            self.slider_frame.show()
            for b in self.video_nav_buttons:
                b.show()
            return

        if medium <= w < wide:
            # squish stage: keep everything visible
            self.lbl_video_info.show()
            self.slider_frame.show()
            for b in self.video_nav_buttons:
                b.show()
            return

        if narrow <= w < medium:
            # collapse first/last
            self.lbl_video_info.show()
            self.slider_frame.show()
            self.btn_first.hide()
            self.btn_last.hide()
            self.btn_prev.show()
            self.btn_next.show()
            return

        if tiny <= w < narrow:
            # collapse all buttons
            self.lbl_video_info.show()
            self.slider_frame.show()
            for b in self.video_nav_buttons:
                b.hide()
            return

        if micro <= w < tiny:
            # hide label too
            self.lbl_video_info.hide()
            self.slider_frame.show()
            for b in self.video_nav_buttons:
                b.hide()
            return

        # extreme: hide everything in video group (pure collapse)
        self.lbl_video_info.hide()
        self.slider_frame.hide()
        for b in self.video_nav_buttons:
            b.hide()

    def _set_mapping_instructions(self):
        self.txt_instructions.setText(
            "Mapping (Homography):\n"
            "1) Load Map.\n"
            "2) Load Video or Frame.\n"
            "3) Click CAMERA point (green).\n"
            "4) Click MAP point (red).\n"
            "5) Repeat.\n\n"
            "Rule: cannot switch/save if last pair incomplete.\n"
        )

    def _set_regions_instructions(self):
        self.txt_reg_help.setText(
            "Entry Regions (Map only):\n"
            "- Click MAP to add vertices.\n"
            "- Confirm Polygon, then New Polygon.\n"
            "- Save writes one polygon per line.\n"
        )

    def _set_points_instructions(self):
        self.txt_pts_help.setText(
            "Map Points (Map only):\n"
            "- Click MAP to add points.\n"
            "- Save writes one point per line: 'x, y'.\n"
        )

    def _set_boundary_instructions(self):
        self.txt_bnd_help.setText(
            "Room Boundary (Map only):\n"
            "- Click MAP to add boundary vertices.\n"
            "- Confirm Boundary then Save.\n"
            "- Save writes one line: 'x1,y1, x2,y2, ...'.\n"
        )

    def _have_map_loaded(self) -> bool:
        return self.map_image_bgr is not None and self.canvas_map.has_image()

    def _have_camera_loaded(self) -> bool:
        return self.camera_frame_bgr is not None and self.canvas_cam.has_image()

    def _output_dir_ok(self) -> bool:
        p = self.le_output_dir.text().strip()
        return bool(p) and os.path.isdir(p)

    def _project_name(self) -> str:
        name = self.le_project_name.text().strip()
        return name if name else "project"

    def _mapping_incomplete(self) -> bool:
        return (
            self.mapping_expected == "map"
            and len(self.mapping_pairs) > 0
            and (not self.mapping_pairs[-1].complete)
        )

    def _warn(self, title: str, msg: str):
        QMessageBox.warning(self, title, msg)

    def _info(self, title: str, msg: str):
        QMessageBox.information(self, title, msg)

    def _confirm(self, title: str, msg: str) -> bool:
        return QMessageBox.question(self, title, msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes

    def _refresh_all_ui(self):
        self._refresh_status()
        self._refresh_mapping_table()
        self._refresh_polys_list()
        self._refresh_points_table()
        self._redraw_overlays()

    def _refresh_status(self):
        tab = self.tabs.currentIndex()
        if tab == 0:
            if not self._have_map_loaded():
                self.lbl_status.setText("Mapping: Load MAP.")
            elif not self._have_camera_loaded():
                self.lbl_status.setText("Mapping: Load CAMERA source.")
            else:
                self.lbl_status.setText(
                    "Mapping: Click CAMERA point." if self.mapping_expected == "frame"
                    else "Mapping: Click MAP point."
                )
            return

        if not self._have_map_loaded():
            self.lbl_status.setText("Map-only tool: Load MAP.")
            return

        if tab == 1:
            self.lbl_status.setText("Entry Regions: click MAP to add vertices.")
        elif tab == 2:
            self.lbl_status.setText("Map Points: click MAP to add points.")
        elif tab == 3:
            self.lbl_status.setText("Room Boundary: click MAP to add boundary.")

    def _refresh_mapping_table(self):
        self.table_pairs.setRowCount(0)
        for i, p in enumerate(self.mapping_pairs, start=1):
            r = self.table_pairs.rowCount()
            self.table_pairs.insertRow(r)
            self.table_pairs.setItem(r, 0, QTableWidgetItem(str(i)))
            self.table_pairs.setItem(r, 1, QTableWidgetItem(f"{p.fx}, {p.fy}"))
            if p.complete:
                self.table_pairs.setItem(r, 2, QTableWidgetItem(f"{p.mx}, {p.my}"))
                self.table_pairs.setItem(r, 3, QTableWidgetItem("OK"))
            else:
                self.table_pairs.setItem(r, 2, QTableWidgetItem("-"))
                self.table_pairs.setItem(r, 3, QTableWidgetItem("Waiting MAP"))
        self.table_pairs.resizeColumnsToContents()

    def _refresh_polys_list(self):
        self.list_polys.clear()
        for idx, poly in enumerate(self.region_polys, start=1):
            self.list_polys.addItem(QListWidgetItem(f"Polygon {idx} ({len(poly.points)} pts)"))

    def _refresh_points_table(self):
        self.table_pts.setRowCount(0)
        for (x, y) in self.map_points:
            r = self.table_pts.rowCount()
            self.table_pts.insertRow(r)
            self.table_pts.setItem(r, 0, QTableWidgetItem(str(x)))
            self.table_pts.setItem(r, 1, QTableWidgetItem(str(y)))
        self.table_pts.resizeColumnsToContents()

    def _redraw_overlays(self):
        tab = self.tabs.currentIndex()

        if tab == 0:
            cam_pts = []
            map_pts = []
            for idx, pair in enumerate(self.mapping_pairs, start=1):
                cam_pts.append((pair.fx, pair.fy, str(idx), QColor(0, 200, 0), QColor(0, 200, 0), 'mapping_frame', idx-1))
                if pair.complete:
                    map_pts.append((pair.mx, pair.my, str(idx), QColor(220, 0, 0), QColor(220, 0, 0), 'mapping_map', idx-1))
            self.canvas_cam.draw_points_with_labels(cam_pts)
            self.canvas_map.draw_points_with_labels(map_pts)
            return

        # map-only: clear camera overlays
        self.canvas_cam.draw_points_with_labels([])

        if tab == 1:
            polylines = [(poly.points, True, QColor(0, 200, 0), 3) for poly in self.region_polys]
            if self.current_poly:
                polylines.append((self.current_poly, self.current_poly_confirmed,
                                  QColor(0, 200, 0) if self.current_poly_confirmed else QColor(220, 0, 0), 3))
            pts = []
            if self.current_poly:
                for i, (x, y) in enumerate(self.current_poly, start=1):
                    c = QColor(0, 200, 0) if self.current_poly_confirmed else QColor(220, 0, 0)
                    pts.append((x, y, str(i), c, c, 'entry_current', i-1))
            self.canvas_map.draw_points_with_labels(pts)
            self.canvas_map.draw_polylines(polylines)
            return

        if tab == 2:
            pts = [(x, y, str(i), QColor(0, 140, 255), QColor(0, 140, 255), 'map_points', i-1)
                   for i, (x, y) in enumerate(self.map_points, start=1)]
            self.canvas_map.draw_points_with_labels(pts)
            return

        if tab == 3:
            polylines = []
            if self.room_boundary:
                color = QColor(0, 200, 0) if self.room_boundary_confirmed else QColor(220, 0, 0)
                polylines.append((self.room_boundary, self.room_boundary_confirmed, color, 3))
            pts = []
            if self.room_boundary:
                for i, (x, y) in enumerate(self.room_boundary, start=1):
                    c = QColor(0, 200, 0) if self.room_boundary_confirmed else QColor(220, 0, 0)
                    pts.append((x, y, str(i), c, c, 'boundary', i-1))
            self.canvas_map.draw_points_with_labels(pts)
            self.canvas_map.draw_polylines(polylines)

    def _set_video_controls_enabled(self, enabled: bool):
        self.slider_frame.setEnabled(enabled)
        for b in self.video_nav_buttons:
            b.setEnabled(enabled)

    def _jump_frame(self, idx: int):
        if self.video.cap is None:
            return
        idx = max(0, min(self.video.total_frames - 1, int(idx)))
        self.slider_frame.blockSignals(True)
        self.slider_frame.setValue(idx)
        self.slider_frame.blockSignals(False)
        self._load_video_frame(idx)

    def _load_video_frame(self, idx: int):
        frame = self.video.get_frame(idx)
        if frame is None:
            self._warn("Video", "Failed to read that frame.")
            return
        self.camera_frame_bgr = frame
        self.canvas_cam.setImage(cv_bgr_to_qimage(frame))
        self.lbl_video_info.setText(f"Video: {os.path.basename(self.video.path)} | Frame {idx+1}/{self.video.total_frames}")
        self.mapping_dirty = True
        self._redraw_overlays()

    def on_choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output root directory")
        if d and ensure_dir(d):
            self.le_output_dir.setText(d)

    def on_load_map(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load map image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            self._warn("Map", "Failed to load map image.")
            return
        self.map_image_bgr = bgr
        self.canvas_map.setImage(cv_bgr_to_qimage(bgr))
        self._refresh_all_ui()
        self._update_video_controls_compactness()

    def on_load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)")
        if not path:
            return
        ok = self.video.open(path)
        if not ok or self.video.total_frames <= 0:
            self._warn("Video", "Failed to open video (or 0 frames).")
            self.video.close()
            return
        self._set_video_controls_enabled(True)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(max(0, self.video.total_frames - 1))
        self.slider_frame.setValue(0)
        self._load_video_frame(0)
        self._refresh_all_ui()
        self._update_video_controls_compactness()

    def on_load_frame_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load frame image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            self._warn("Frame", "Failed to load frame image.")
            return
        self.video.close()
        self._set_video_controls_enabled(False)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.lbl_video_info.setText("Frame image loaded (no video).")
        self.camera_frame_bgr = bgr
        self.canvas_cam.setImage(cv_bgr_to_qimage(bgr))
        self._refresh_all_ui()
        self._update_video_controls_compactness()

    def on_clear_camera(self):
        if self._mapping_incomplete():
            self._warn("Mapping", "Finish selecting MAP point for last pair before clearing camera.")
            return
        self.video.close()
        self._set_video_controls_enabled(False)
        self.lbl_video_info.setText("No video loaded.")
        self.camera_frame_bgr = None
        self.canvas_cam.setImage(QImage())
        self._refresh_all_ui()
        self._update_video_controls_compactness()

    def on_frame_slider_changed(self, value: int):
        if self.video.cap is None:
            return
        self._load_video_frame(value)

    def on_tab_changed(self, idx: int):
        if idx != 0 and self._mapping_incomplete():
            self._warn("Mapping", "Select the corresponding MAP point before switching tasks.")
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(0)
            self.tabs.blockSignals(False)
            return
        self._redraw_overlays()

    def on_camera_clicked(self, x: int, y: int):
        if self.tabs.currentIndex() != 0:
            self._warn("Map-only mode", "Camera clicks are disabled for this task. Please click on the MAP image.")
            return
        if not self._have_camera_loaded() or not self._have_map_loaded():
            return
        if self.mapping_expected != "frame":
            self._warn("Mapping", "Click the MAP image for the corresponding point first.")
            return
        self.mapping_pairs.append(MappingPair(fx=x, fy=y))
        self.mapping_expected = "map"
        self.mapping_dirty = True
        self._refresh_all_ui()

    def on_map_clicked(self, x: int, y: int):
        if not self._have_map_loaded():
            return
        tab = self.tabs.currentIndex()

        if tab == 0:
            if not self._have_camera_loaded():
                self._warn("Mapping", "Load a camera source (video/frame) first.")
                return
            if self.mapping_expected != "map":
                self._warn("Mapping", "Please click the CAMERA image first.")
                return
            last = self.mapping_pairs[-1]
            if last.complete:
                self.mapping_expected = "frame"
                return
            last.mx, last.my = x, y
            self.mapping_expected = "frame"
            self.mapping_dirty = True
            self._refresh_all_ui()
            return

        if self._mapping_incomplete():
            self._warn("Mapping", "Finish mapping pair before using map-only tools.")
            return

        if tab == 1:
            if self.current_poly_confirmed:
                self._warn("Entry Regions", "Polygon confirmed. Use 'New Polygon' to start another.")
                return
            self.current_poly.append((x, y))
            self.regions_dirty = True
            self._refresh_all_ui()
            return

        if tab == 2:
            self.map_points.append((x, y))
            self.points_dirty = True
            self._refresh_all_ui()
            return

        if tab == 3:
            if self.room_boundary_confirmed:
                self._warn("Room Boundary", "Boundary confirmed. Clear to redraw.")
                return
            self.room_boundary.append((x, y))
            self.boundary_dirty = True
            self._refresh_all_ui()

    # --- Mapping actions ---
    def on_mapping_undo(self):
        if not self.mapping_pairs:
            return
        self.mapping_pairs.pop()
        self.mapping_expected = "frame"
        self.mapping_dirty = True
        self._refresh_all_ui()

    def on_mapping_delete_selected(self):
        rows = {idx.row() for idx in self.table_pairs.selectionModel().selectedRows()}
        if not rows:
            return
        if self.mapping_expected == "map":
            self._warn("Mapping", "Finish selecting MAP point for last pair before deleting.")
            return
        for r in sorted(rows, reverse=True):
            if 0 <= r < len(self.mapping_pairs):
                self.mapping_pairs.pop(r)
        self.mapping_dirty = True
        self._refresh_all_ui()

    def on_mapping_clear_all(self):
        if self._mapping_incomplete():
            self._warn("Mapping", "Finish selecting MAP point for last pair before clearing.")
            return
        if self.mapping_pairs and not self._confirm("Clear Mapping", "Clear all mapping pairs?"):
            return
        self.mapping_pairs = []
        self.mapping_expected = "frame"
        self.mapping_dirty = True
        self._refresh_all_ui()

    def on_mapping_save(self):
        if self._mapping_incomplete():
            self._warn("Mapping", "Cannot save: last mapping pair missing MAP point.")
            return
        if not self.mapping_pairs:
            self._warn("Mapping", "No mapping points to save.")
            return
        if not self._output_dir_ok():
            self._warn("Save", "Choose a valid Save Root Directory first.")
            return
        name = self._project_name()
        out_path = os.path.join(self.le_output_dir.text().strip(), f"{name}_mapping.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for p in self.mapping_pairs:
                f.write(f"{p.fx}, {p.fy}, {p.mx}, {p.my}\n")
        self.mapping_dirty = False
        self._info("Saved", f"Mapping saved:\n{out_path}")

    # --- Regions actions ---
    def on_regions_undo(self):
        if self.current_poly_confirmed:
            self._warn("Entry Regions", "Polygon confirmed. Use New Polygon or delete saved polygons.")
            return
        if self.current_poly:
            self.current_poly.pop()
            self.regions_dirty = True
            self._refresh_all_ui()

    def on_regions_confirm(self):
        if not self._have_map_loaded():
            self._warn("Entry Regions", "Load a MAP image first.")
            return
        if self.current_poly_confirmed:
            return
        min_pts = int(self.spin_min_points.value())
        if len(self.current_poly) < min_pts:
            self._warn("Entry Regions", f"Polygon needs at least {min_pts} points.")
            return
        self.current_poly_confirmed = True
        self.regions_dirty = True
        self._refresh_all_ui()

    def on_regions_new_polygon(self):
        min_pts = int(self.spin_min_points.value())
        if not self.current_poly:
            return
        if not self.current_poly_confirmed:
            self._warn("Entry Regions", "Confirm the polygon before starting a new one.")
            return
        if len(self.current_poly) < min_pts:
            self._warn("Entry Regions", f"Polygon needs at least {min_pts} points.")
            return
        self.region_polys.append(RegionPolygon(points=list(self.current_poly)))
        self.current_poly = []
        self.current_poly_confirmed = False
        self.regions_dirty = True
        self._refresh_all_ui()

    def on_regions_delete_selected(self):
        row = self.list_polys.currentRow()
        if row < 0 or row >= len(self.region_polys):
            return
        if not self._confirm("Delete Polygon", "Delete selected polygon?"):
            return
        self.region_polys.pop(row)
        self.regions_dirty = True
        self._refresh_all_ui()

    def on_regions_clear_all(self):
        if (self.region_polys or self.current_poly) and not self._confirm("Clear Regions", "Clear all polygons?"):
            return
        self.region_polys = []
        self.current_poly = []
        self.current_poly_confirmed = False
        self.regions_dirty = True
        self._refresh_all_ui()

    def on_regions_save(self):
        if not self._have_map_loaded():
            self._warn("Entry Regions", "Load a MAP image first.")
            return
        if not self._output_dir_ok():
            self._warn("Save", "Choose a valid Save Root Directory first.")
            return
        min_pts = int(self.spin_min_points.value())
        if self.current_poly and not self.current_poly_confirmed:
            self._warn("Entry Regions", "Cannot save: current polygon not confirmed.")
            return
        all_polys = list(self.region_polys)
        if self.current_poly_confirmed and len(self.current_poly) >= min_pts:
            all_polys.append(RegionPolygon(points=list(self.current_poly)))
        if not all_polys:
            self._warn("Entry Regions", "No polygons to save.")
            return
        name = self._project_name()
        out_path = os.path.join(self.le_output_dir.text().strip(), f"{name}_entry_polygons.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for poly in all_polys:
                f.write(", ".join([f"{x},{y}" for (x, y) in poly.points]) + "\n")
        self.regions_dirty = False
        self._info("Saved", f"Entry regions saved:\n{out_path}")

    # --- Map points ---
    def on_points_undo(self):
        if self.map_points:
            self.map_points.pop()
            self.points_dirty = True
            self._refresh_all_ui()

    def on_points_delete_selected(self):
        rows = {idx.row() for idx in self.table_pts.selectionModel().selectedRows()}
        if not rows:
            return
        for r in sorted(rows, reverse=True):
            if 0 <= r < len(self.map_points):
                self.map_points.pop(r)
        self.points_dirty = True
        self._refresh_all_ui()

    def on_points_clear_all(self):
        if self.map_points and not self._confirm("Clear Points", "Clear all map points?"):
            return
        self.map_points = []
        self.points_dirty = True
        self._refresh_all_ui()

    def on_points_save(self):
        if not self._have_map_loaded():
            self._warn("Map Points", "Load a MAP image first.")
            return
        if not self._output_dir_ok():
            self._warn("Save", "Choose a valid Save Root Directory first.")
            return
        if not self.map_points:
            self._warn("Map Points", "No points to save.")
            return
        name = self._project_name()
        out_path = os.path.join(self.le_output_dir.text().strip(), f"{name}_map_points.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for (x, y) in self.map_points:
                f.write(f"{x}, {y}\n")
        self.points_dirty = False
        self._info("Saved", f"Map points saved:\n{out_path}")

    # --- Boundary ---
    def on_boundary_undo(self):
        if self.room_boundary_confirmed:
            self._warn("Room Boundary", "Boundary confirmed. Clear to redraw.")
            return
        if self.room_boundary:
            self.room_boundary.pop()
            self.boundary_dirty = True
            self._refresh_all_ui()

    def on_boundary_confirm(self):
        if not self._have_map_loaded():
            self._warn("Room Boundary", "Load a MAP image first.")
            return
        if self.room_boundary_confirmed:
            return
        min_pts = int(self.spin_bnd_min_points.value())
        if len(self.room_boundary) < min_pts:
            self._warn("Room Boundary", f"Boundary needs at least {min_pts} points.")
            return
        self.room_boundary_confirmed = True
        self.boundary_dirty = True
        self._refresh_all_ui()

    def on_boundary_clear(self):
        if self.room_boundary and not self._confirm("Clear Boundary", "Clear room boundary?"):
            return
        self.room_boundary = []
        self.room_boundary_confirmed = False
        self.boundary_dirty = True
        self._refresh_all_ui()

    def on_boundary_save(self):
        if not self._have_map_loaded():
            self._warn("Room Boundary", "Load a MAP image first.")
            return
        if not self._output_dir_ok():
            self._warn("Save", "Choose a valid Save Root Directory first.")
            return
        if not self.room_boundary:
            self._warn("Room Boundary", "No boundary to save.")
            return
        if not self.room_boundary_confirmed:
            self._warn("Room Boundary", "Confirm boundary before saving.")
            return
        name = self._project_name()
        out_path = os.path.join(self.le_output_dir.text().strip(), f"{name}_room_boundary.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(", ".join([f"{x},{y}" for (x, y) in self.room_boundary]) + "\n")
        self.boundary_dirty = False
        self._info("Saved", f"Room boundary saved:\n{out_path}")

    def closeEvent(self, event):
        if self._mapping_incomplete():
            self._warn("Mapping", "Select corresponding MAP point before exiting.")
            event.ignore()
            return
        if self.mapping_dirty or self.regions_dirty or self.points_dirty or self.boundary_dirty:
            if not self._confirm("Exit", "You have unsaved changes. Exit anyway?"):
                event.ignore()
                return
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MapperMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()