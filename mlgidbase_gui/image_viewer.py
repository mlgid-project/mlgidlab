from __future__ import annotations

import os

# Pin pyqtgraph to PySide6 before it auto-detects.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from mlgidbase_gui.file_model import EntryStack, MatchedStructure, PeakTable
from mlgidbase_gui.polar import stack_to_polar

OVERLAY_KINDS = ("detected", "fitted", "manual")
MODE_CARTESIAN = "cartesian"
MODE_POLAR = "polar"

LABEL_MODIFIERS = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier

# Subdivisions along the angular edge for the full 0–90° range; narrower
# segments scale down proportionally (with a small minimum for sharp corners).
ANGULAR_SUBDIV_FULL = 90
ANGULAR_SUBDIV_MIN = 4

# The polar grid used by the viewer spans this angle range. A peak whose
# angle_width is infinite, NaN, or exceeds the range is clipped to it.
ANGLE_MIN_DEG = 0.0
ANGLE_MAX_DEG = 90.0

# Visual style for each overlay kind. Dashed for "raw" detection output,
# solid for the refined fit, dotted yellow for user-drawn manual labels.
OVERLAY_STYLE: dict[str, dict] = {
    "detected": {"color": "#ff5c5c", "style": Qt.PenStyle.DashLine, "width": 1.2},
    "fitted":   {"color": "#26d0ce", "style": Qt.PenStyle.SolidLine, "width": 1.2},
    "manual":   {"color": "#ffeb3b", "style": Qt.PenStyle.SolidLine, "width": 1.6},
}

SELECTION_STYLE = {"color": "#ffffff", "style": Qt.PenStyle.SolidLine, "width": 2.5}

# Distinct, dark-mode-legible palette for matched structures. Cycled by
# insertion order so multiple structures in one frame are easy to tell apart.
# Avoids the existing detected/fitted/manual hues to prevent confusion.
MATCHED_PALETTE: tuple[str, ...] = (
    "#1f77ff",  # azure
    "#bf5af2",  # violet
    "#30d158",  # green
    "#ff9f0a",  # orange
    "#64d2ff",  # light cyan
    "#ff375f",  # rose
    "#a8e10c",  # lime
    "#ffd60a",  # amber
    "#5ac8fa",  # sky
    "#ac8e68",  # taupe
)
MATCHED_STYLE = {"style": Qt.PenStyle.SolidLine, "width": 1.6}

# Curated list of colormaps. Names are matplotlib's; pg.colormap.get falls
# back to matplotlib's registry, which is always available since matplotlib
# is a transitive dep via silx.
COLORMAPS = ("viridis", "inferno", "plasma", "magma", "cividis", "gray")
DEFAULT_COLORMAP = "viridis"


def _robust_levels(frame: np.ndarray) -> tuple[float, float]:
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, (1.0, 99.5))
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


@dataclass
class _DisplayParams:
    image_pg: np.ndarray
    pos: tuple[float, float]
    scale: tuple[float, float]
    levels: tuple[float, float]
    x_label: tuple[str, str]
    y_label: tuple[str, str]


@dataclass
class ManualPeak:
    """A user-drawn polar peak box. In-memory only; phase 4c persists these."""

    radius: float
    angle: float
    radius_width: float
    angle_width: float
    is_ring: bool = False
    temp_id: int = 0


def _peaks_from_manual(manual: list[ManualPeak]) -> PeakTable:
    """Adapt a list of ManualPeak to the PeakTable shape so the existing
    rendering helpers can draw them without special-casing."""
    if not manual:
        empty = np.zeros(0, dtype=float)
        return PeakTable(
            q_xy=empty, q_z=empty, angle=empty, radius=empty,
            angle_width=empty, radius_width=empty,
            is_ring=np.zeros(0, dtype=bool),
            ids=np.zeros(0, dtype=int),
        )
    return PeakTable(
        q_xy=np.array([m.radius * np.cos(np.deg2rad(m.angle)) for m in manual]),
        q_z=np.array([m.radius * np.sin(np.deg2rad(m.angle)) for m in manual]),
        angle=np.array([m.angle for m in manual], dtype=float),
        radius=np.array([m.radius for m in manual], dtype=float),
        angle_width=np.array([m.angle_width for m in manual], dtype=float),
        radius_width=np.array([m.radius_width for m in manual], dtype=float),
        is_ring=np.array([m.is_ring for m in manual], dtype=bool),
        ids=np.array([m.temp_id for m in manual], dtype=int),
    )


class _LabelEventFilter(QObject):
    """Qt event filter that emits high-level labelling signals from raw mouse
    events on a graphics-view's viewport.

    Installed instead of subclassing ``pg.ViewBox`` because pyqtgraph's drag
    dispatch only fires ``mouseDragEvent`` when the press was accepted at the
    QGraphicsItem layer — which it isn't for plain LMB on the image area, so a
    ViewBox subclass never sees the drag.
    """

    drawStarted = Signal(QPointF)
    drawUpdated = Signal(QPointF, QPointF)
    drawFinished = Signal(QPointF, QPointF)
    selectAt = Signal(QPointF)

    # Pixel tolerance below which a press+release counts as a click, not a drag.
    CLICK_TOLERANCE_PX = 4

    def __init__(
        self, graphics_view, viewbox: pg.ViewBox, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._gv = graphics_view
        self._vb = viewbox
        self._drawing = False
        self._origin: QPointF | None = None
        self._press_pos: QPoint | None = None
        self._press_mods: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier

    def install(self) -> None:
        self._gv.viewport().installEventFilter(self)

    def eventFilter(self, _obj: QObject, ev: QEvent) -> bool:  # type: ignore[override]
        et = ev.type()
        if et == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
            mods = ev.modifiers()
            if _has_label_modifiers(mods):
                pos = self._viewport_to_data(ev.position().toPoint())
                self._origin = pos
                self._drawing = True
                self.drawStarted.emit(pos)
                return True  # consume so pan doesn't engage
            self._press_pos = ev.position().toPoint()
            self._press_mods = mods
            return False
        if et == QEvent.Type.MouseMove and self._drawing and self._origin is not None:
            pos = self._viewport_to_data(ev.position().toPoint())
            self.drawUpdated.emit(self._origin, pos)
            return True
        if et == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
            if self._drawing and self._origin is not None:
                end = self._viewport_to_data(ev.position().toPoint())
                self.drawFinished.emit(self._origin, end)
                self._drawing = False
                self._origin = None
                return True
            if self._press_pos is not None:
                delta = ev.position().toPoint() - self._press_pos
                bare_click = (
                    delta.manhattanLength() <= self.CLICK_TOLERANCE_PX
                    and self._press_mods == Qt.KeyboardModifier.NoModifier
                )
                self._press_pos = None
                self._press_mods = Qt.KeyboardModifier.NoModifier
                if bare_click:
                    pos = self._viewport_to_data(ev.position().toPoint())
                    self.selectAt.emit(pos)
                    # Do not consume — pyqtgraph still emits a click for menus, etc.
        return False

    def _viewport_to_data(self, viewport_pt: QPoint) -> QPointF:
        scene_pt = self._gv.mapToScene(viewport_pt)
        return self._vb.mapSceneToView(scene_pt)


def _has_label_modifiers(mods: Qt.KeyboardModifier) -> bool:
    return bool(
        mods & Qt.KeyboardModifier.ControlModifier
        and mods & Qt.KeyboardModifier.AltModifier
    )


class _PeakShapeItem(pg.GraphicsObject):
    """Draws a collection of peak shapes from a single QPainterPath.

    In polar mode every peak is an axis-aligned rectangle. In Cartesian mode,
    rings become quarter-circle arcs at the central radius and segments become
    polygons formed by tessellating the polar rectangle's angular edges.
    """

    def __init__(self, color: str, style: Qt.PenStyle, width: float) -> None:
        super().__init__()
        pen = pg.mkPen(QColor(color), width=width)
        pen.setStyle(style)
        pen.setCosmetic(True)  # constant pixel width regardless of zoom
        self._pen = pen
        self._path = QPainterPath()
        self._bounding = QRectF()

    def set_polar(self, peaks: PeakTable | None) -> None:
        path = QPainterPath()
        if peaks is not None and len(peaks) > 0:
            for i in range(len(peaks)):
                clip = _clip_angle(float(peaks.angle[i]), float(peaks.angle_width[i]))
                if clip is None:
                    continue
                a_lo, a_hi = clip
                r = float(peaks.radius[i])
                dr = float(peaks.radius_width[i])
                path.addRect(QRectF(r - dr / 2, a_lo, dr, a_hi - a_lo))
        self._update_path(path)

    def set_cartesian(self, peaks: PeakTable | None) -> None:
        path = QPainterPath()
        if peaks is not None and len(peaks) > 0:
            for i in range(len(peaks)):
                clip = _clip_angle(float(peaks.angle[i]), float(peaks.angle_width[i]))
                if clip is None:
                    continue
                a_lo, a_hi = clip
                path.addPath(
                    _polar_rect_polygon(
                        float(peaks.radius[i]),
                        float(peaks.radius_width[i]),
                        a_lo,
                        a_hi,
                    )
                )
        self._update_path(path)

    def clear_path(self) -> None:
        self._update_path(QPainterPath())

    def _update_path(self, path: QPainterPath) -> None:
        self.prepareGeometryChange()
        self._path = path
        self._bounding = path.boundingRect()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounding

    def paint(self, painter, *_args) -> None:
        painter.setPen(self._pen)
        painter.drawPath(self._path)


def _clip_angle(a_deg: float, da_deg: float) -> tuple[float, float] | None:
    """Clip a polar angular box to the viewer's visible range.

    Treats infinite or non-finite angle_width as 'spans the whole quadrant',
    so rings (whose angle_width is sometimes inf) still draw correctly.
    Returns (lo, hi) in degrees, or None if the box is empty/invalid.
    """
    if not np.isfinite(a_deg) or not np.isfinite(da_deg):
        a_lo, a_hi = ANGLE_MIN_DEG, ANGLE_MAX_DEG
    else:
        a_lo = a_deg - da_deg / 2.0
        a_hi = a_deg + da_deg / 2.0
    a_lo = max(a_lo, ANGLE_MIN_DEG)
    a_hi = min(a_hi, ANGLE_MAX_DEG)
    if a_hi <= a_lo:
        return None
    return a_lo, a_hi


def _polar_rect_polygon(
    radius: float, dr: float, a_lo_deg: float, a_hi_deg: float
) -> QPainterPath:
    """Render a polar rectangle (already clipped) as a closed polygon in q-space.

    For full-quadrant rings this becomes a proper quarter-annulus that closes
    along the q_xy and q_z axes. For narrow segments, a thin curved trapezoid.
    """
    a_lo = np.deg2rad(a_lo_deg)
    a_hi = np.deg2rad(a_hi_deg)
    inner = max(radius - dr / 2.0, 0.0)
    outer = radius + dr / 2.0

    span = a_hi_deg - a_lo_deg
    n_sub = max(
        int(np.ceil(span / 90.0 * ANGULAR_SUBDIV_FULL)),
        ANGULAR_SUBDIV_MIN,
    )
    angs = np.linspace(a_lo, a_hi, n_sub)

    path = QPainterPath()
    path.moveTo(QPointF(float(outer * np.cos(angs[0])), float(outer * np.sin(angs[0]))))
    for ang in angs[1:]:
        path.lineTo(QPointF(float(outer * np.cos(ang)), float(outer * np.sin(ang))))
    for ang in angs[::-1]:
        path.lineTo(QPointF(float(inner * np.cos(ang)), float(inner * np.sin(ang))))
    path.closeSubpath()
    return path


class GIWAXSImageViewer(QWidget):
    """Image viewer with Cartesian ↔ polar mode toggle + peak-box overlays."""

    frameChanged = Signal(int)
    modeChanged = Signal(str)
    manualPeakAdded = Signal(int, object)     # frame, ManualPeak
    manualPeakRemoved = Signal(int, object)   # frame, ManualPeak
    selectionChanged = Signal(object)         # ManualPeak | None
    peakGeometryChanged = Signal(object)      # ManualPeak whose r/dr/a/da changed
    # Emitted whenever the *current* frame's matched-structure list might be
    # different from what the UI showed last (frame change, fresh load,
    # re-render after pipeline run). Args: (frame, list[MatchedStructure]).
    matchedStructuresChanged = Signal(int, list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 4, 8, 4)
        bar.addWidget(QLabel("View:"))
        self._radio_cart = QRadioButton("Cartesian")
        self._radio_polar = QRadioButton("Polar")
        self._radio_polar.setChecked(True)
        self._radio_group = QButtonGroup(self)
        self._radio_group.addButton(self._radio_cart)
        self._radio_group.addButton(self._radio_polar)
        self._radio_cart.toggled.connect(self._on_radio_toggled)
        bar.addWidget(self._radio_cart)
        bar.addWidget(self._radio_polar)
        bar.addSpacing(16)
        bar.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        for name in COLORMAPS:
            self._cmap_combo.addItem(name)
        self._cmap_combo.setCurrentText(DEFAULT_COLORMAP)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        bar.addWidget(self._cmap_combo)
        bar.addStretch(1)
        bar_widget = QWidget(self)
        bar_widget.setLayout(bar)
        outer.addWidget(bar_widget)

        self._plot = pg.PlotItem()
        self._view = pg.ImageView(self, view=self._plot)
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        outer.addWidget(self._view)

        self._plot.invertY(False)
        self._plot.setAspectLocked(False)

        self._detected = _PeakShapeItem(**OVERLAY_STYLE["detected"])
        self._fitted = _PeakShapeItem(**OVERLAY_STYLE["fitted"])
        self._manual = _PeakShapeItem(**OVERLAY_STYLE["manual"])
        self._selection = _PeakShapeItem(**SELECTION_STYLE)
        vb = self._plot.getViewBox()
        vb.addItem(self._detected, ignoreBounds=True)
        vb.addItem(self._fitted, ignoreBounds=True)
        vb.addItem(self._manual, ignoreBounds=True)
        vb.addItem(self._selection, ignoreBounds=True)

        self._preview_item = pg.QtWidgets.QGraphicsRectItem()
        preview_pen = pg.mkPen(QColor("#ffeb3b"), width=1.0)
        preview_pen.setStyle(Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)
        self._preview_item.setPen(preview_pen)
        self._preview_item.setBrush(QColor(255, 235, 59, 40))
        self._preview_item.setZValue(50)
        self._preview_item.setVisible(False)
        vb.addItem(self._preview_item, ignoreBounds=True)

        # Mouse handling lives in a Qt event filter on the graphics-view's
        # viewport — see _LabelEventFilter for why a ViewBox subclass doesn't work.
        self._label_filter = _LabelEventFilter(self._view.ui.graphicsView, vb, self)
        self._label_filter.install()
        self._label_filter.drawStarted.connect(self._on_draw_started)
        self._label_filter.drawUpdated.connect(self._on_draw_updated)
        self._label_filter.drawFinished.connect(self._on_draw_finished)
        self._label_filter.selectAt.connect(self._on_select_at)

        # Apply default colormap immediately.
        self._apply_cmap(DEFAULT_COLORMAP)

        # Need keyboard focus for the Delete shortcut to fire.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._frame_peaks: dict[int, dict[str, PeakTable | None]] = {}
        self._manual_peaks: dict[int, list[ManualPeak]] = {}
        self._visibility: dict[str, bool] = {kind: True for kind in OVERLAY_KINDS}

        # Matched-structure overlays. Variable count per frame (one per row in
        # each matched_* dataset), each with its own color and visibility.
        # Items in ``_matched_items`` belong to the *currently rendered* frame
        # only — they're torn down and rebuilt on frame change.
        self._matched_per_frame: dict[int, list[MatchedStructure]] = {}
        # Per-(frame, unique_id) state lets us preserve user toggles when the
        # frame changes back. Defaults to True on first sight.
        self._matched_visibility: dict[tuple[int, str], bool] = {}
        self._matched_master_visible: bool = True
        self._matched_items: list[tuple[str, _PeakShapeItem]] = []

        self._mode = MODE_POLAR
        self._stack: EntryStack | None = None
        self._polar_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._next_manual_id = -1  # negative IDs distinguish manual from detected
        self._selected: ManualPeak | None = None
        self._roi_item: pg.ROI | None = None
        # Undo stack of (action, peak, frame). `action` is "add" or "remove".
        self._undo_stack: list[tuple[str, ManualPeak, int]] = []

        self._view.sigTimeChanged.connect(self._on_time_changed)

    # -- Public API --

    def show_stack(self, stack: EntryStack) -> None:
        self._stack = stack
        self._polar_cache = None
        self._frame_peaks.clear()
        self._render_active_mode()

    def set_peaks(self, frame: int, peaks: dict[str, PeakTable | None]) -> None:
        self._frame_peaks[frame] = peaks
        if frame == self.current_frame:
            self._render_overlays(frame)

    def set_overlay_visible(self, kind: str, visible: bool) -> None:
        if kind not in OVERLAY_KINDS:
            return
        self._visibility[kind] = visible
        item = self._overlay_item(kind)
        if item is not None:
            item.setVisible(visible)
        if kind == "manual" and not visible:
            # Hiding manual overlay also clears any active selection highlight.
            self._selected = None
            self._selection.clear_path()

    # -- Matched-structure API --

    def set_matched_structures(
        self, frame: int, structures: list[MatchedStructure]
    ) -> None:
        """Replace the list of matched structures for ``frame``.

        Visibility flags for previously-seen ``unique_id``s on this frame are
        preserved; new structures default to visible. Re-renders if ``frame``
        is the one currently shown.
        """
        self._matched_per_frame[frame] = list(structures)
        # Drop visibility entries for structures no longer present.
        present_ids = {(frame, s.unique_id) for s in structures}
        existing = {k for k in self._matched_visibility if k[0] == frame}
        for stale in existing - present_ids:
            self._matched_visibility.pop(stale, None)
        for s in structures:
            self._matched_visibility.setdefault((frame, s.unique_id), True)
        if frame == self.current_frame:
            self._render_overlays(frame)
            self.matchedStructuresChanged.emit(frame, list(structures))

    def matched_structures(self, frame: int) -> list[MatchedStructure]:
        return list(self._matched_per_frame.get(frame, []))

    def matched_color(self, structure: MatchedStructure) -> str:
        """Return the hex color assigned to a structure on the current frame.
        Color is deterministic per insertion order within the frame so the
        Display panel and the overlay agree without extra plumbing.
        """
        frame = self.current_frame
        lst = self._matched_per_frame.get(frame, [])
        for i, s in enumerate(lst):
            if s.unique_id == structure.unique_id:
                return MATCHED_PALETTE[i % len(MATCHED_PALETTE)]
        return MATCHED_PALETTE[0]

    def matched_visibility(self, frame: int, unique_id: str) -> bool:
        return self._matched_visibility.get((frame, unique_id), True)

    def set_matched_master_visible(self, visible: bool) -> None:
        self._matched_master_visible = visible
        for _uid, item in self._matched_items:
            item.setVisible(self._is_matched_item_visible(_uid))

    def set_matched_structure_visible(self, unique_id: str, visible: bool) -> None:
        frame = self.current_frame
        self._matched_visibility[(frame, unique_id)] = visible
        for uid, item in self._matched_items:
            if uid == unique_id:
                item.setVisible(self._is_matched_item_visible(uid))

    def _is_matched_item_visible(self, unique_id: str) -> bool:
        if not self._matched_master_visible:
            return False
        return self._matched_visibility.get(
            (self.current_frame, unique_id), True
        )

    def _overlay_item(self, kind: str) -> _PeakShapeItem | None:
        return {
            "detected": self._detected,
            "fitted":   self._fitted,
            "manual":   self._manual,
        }.get(kind)

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_CARTESIAN, MODE_POLAR) or mode == self._mode:
            return
        self._mode = mode
        if mode == MODE_POLAR:
            self._radio_polar.setChecked(True)
        else:
            self._radio_cart.setChecked(True)
        self._sync_roi()  # ROI exists only in polar mode
        self._render_active_mode()
        self.modeChanged.emit(mode)

    @property
    def mode(self) -> str:
        return self._mode

    def clear(self) -> None:
        self._view.clear()
        self._detected.clear_path()
        self._fitted.clear_path()
        self._manual.clear_path()
        self._selection.clear_path()
        self._frame_peaks.clear()
        self._manual_peaks.clear()
        self._undo_stack.clear()
        # Tear down all matched items and forget per-frame state.
        self._teardown_matched_items()
        self._matched_per_frame.clear()
        self._matched_visibility.clear()
        had_selection = self._selected is not None
        self._selected = None
        self._sync_roi()
        self._stack = None
        self._polar_cache = None
        if had_selection:
            self.selectionChanged.emit(None)

    # -- Manual peaks --

    def manual_peaks(self, frame: int) -> list[ManualPeak]:
        return list(self._manual_peaks.get(frame, []))

    def add_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        self._manual_peaks.setdefault(frame, []).append(peak)
        self._undo_stack.append(("add", peak, frame))
        if frame == self.current_frame:
            self._render_overlays(frame)
        self.manualPeakAdded.emit(frame, peak)

    def remove_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        peaks = self._manual_peaks.get(frame, [])
        if peak in peaks:
            peaks.remove(peak)
            self._undo_stack.append(("remove", peak, frame))
            was_selected = peak is self._selected
            if was_selected:
                self._selected = None
                self._sync_roi()
            if frame == self.current_frame:
                self._render_overlays(frame)
            if was_selected:
                self.selectionChanged.emit(None)
            self.manualPeakRemoved.emit(frame, peak)

    def commit_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        """Drop a manual peak that has been persisted to the NeXus file.

        Like ``remove_manual_peak`` but does not push to the undo stack — the
        peak now lives in the detected/fitted overlay, so undoing back to its
        manual state would resurrect a duplicate. Any existing undo entries
        referencing this peak are scrubbed for the same reason.
        """
        peaks = self._manual_peaks.get(frame, [])
        if peak in peaks:
            peaks.remove(peak)
        self._undo_stack = [
            entry for entry in self._undo_stack if entry[1] is not peak
        ]
        was_selected = peak is self._selected
        if was_selected:
            self._selected = None
            self._sync_roi()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if was_selected:
            self.selectionChanged.emit(None)
        self.manualPeakRemoved.emit(frame, peak)

    def undo_last_action(self) -> None:
        """Reverse the most recent add or remove. Silently no-ops if empty."""
        if not self._undo_stack:
            return
        action, peak, frame = self._undo_stack.pop()
        if action == "add":
            peaks = self._manual_peaks.get(frame, [])
            if peak in peaks:
                peaks.remove(peak)
            was_selected = peak is self._selected
            if was_selected:
                self._selected = None
                self._sync_roi()
            if frame == self.current_frame:
                self._render_overlays(frame)
            if was_selected:
                self.selectionChanged.emit(None)
            self.manualPeakRemoved.emit(frame, peak)
        elif action == "remove":
            self._manual_peaks.setdefault(frame, []).append(peak)
            if frame == self.current_frame:
                self._render_overlays(frame)
            self.manualPeakAdded.emit(frame, peak)

    @property
    def current_frame(self) -> int:
        return int(self._view.currentIndex)

    @property
    def selected_peak(self) -> ManualPeak | None:
        return self._selected

    def polar_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (polar_stack, radius, angle), computing the cache if needed.

        Returns None if no stack is currently loaded. Used by the profile
        viewer to share a single polar transform across panels.
        """
        if self._stack is None:
            return None
        if self._polar_cache is None:
            self._polar_cache = stack_to_polar(
                self._stack.image_stack, self._stack.q_xy, self._stack.q_z
            )
        return self._polar_cache

    # -- Rendering --

    def _render_active_mode(self) -> None:
        if self._stack is None:
            return
        if self._mode == MODE_POLAR:
            params = self._build_polar_params()
        else:
            params = self._build_cartesian_params()
        self._apply_params(params)
        self._render_overlays(self.current_frame)

    def _build_cartesian_params(self) -> _DisplayParams:
        assert self._stack is not None
        # File order is (frames, q_z, q_xy); pyqtgraph wants (t, x, y).
        img_pg = np.transpose(self._stack.image_stack, (0, 2, 1))
        x0 = float(self._stack.q_xy[0])
        y0 = float(self._stack.q_z[0])
        sx = (
            float(self._stack.q_xy[-1] - self._stack.q_xy[0])
            / max(len(self._stack.q_xy) - 1, 1)
        )
        sy = (
            float(self._stack.q_z[-1] - self._stack.q_z[0])
            / max(len(self._stack.q_z) - 1, 1)
        )
        levels = _robust_levels(self._stack.image_stack[0])
        return _DisplayParams(
            image_pg=img_pg,
            pos=(x0, y0),
            scale=(sx, sy),
            levels=levels,
            x_label=("q_xy", "Å⁻¹"),
            y_label=("q_z", "Å⁻¹"),
        )

    def _build_polar_params(self) -> _DisplayParams:
        assert self._stack is not None
        if self._polar_cache is None:
            self._polar_cache = stack_to_polar(
                self._stack.image_stack, self._stack.q_xy, self._stack.q_z
            )
        polar_stack, radius, angle = self._polar_cache  # (frames, n_r, n_ang)
        # We want radius along x, angle along y. polar_stack is already
        # (frame, radius, angle) — that maps directly to pyqtgraph's (t, x, y).
        img_pg = polar_stack
        x0 = float(radius[0])
        y0 = float(angle[0])
        sx = float(radius[-1] - radius[0]) / max(len(radius) - 1, 1)
        sy = float(angle[-1] - angle[0]) / max(len(angle) - 1, 1)
        levels = _robust_levels(polar_stack[0])
        return _DisplayParams(
            image_pg=img_pg,
            pos=(x0, y0),
            scale=(sx, sy),
            levels=levels,
            x_label=("radius", "Å⁻¹"),
            y_label=("angle", "deg"),
        )

    def _apply_params(self, p: _DisplayParams) -> None:
        self._plot.setLabel("bottom", p.x_label[0], units=p.x_label[1])
        self._plot.setLabel("left", p.y_label[0], units=p.y_label[1])
        self._view.setImage(
            p.image_pg,
            autoRange=True,
            autoLevels=False,
            levels=p.levels,
            pos=p.pos,
            scale=p.scale,
        )
        # The roiPlot is pyqtgraph's frame-timeline strip; only meaningful
        # when the stack has more than one frame.
        multi_frame = p.image_pg.shape[0] > 1
        self._view.ui.roiPlot.setVisible(multi_frame)
        if multi_frame:
            self._view.ui.splitter.setSizes([4, 1])
        else:
            self._view.ui.splitter.setSizes([1, 0])

    def _on_time_changed(self, index: int, _time: float) -> None:
        idx = int(index)
        self._render_overlays(idx)
        self.frameChanged.emit(idx)
        # The panel rebuilds its matched-structure rows from this signal —
        # different frames can have a different set of solutions.
        self.matchedStructuresChanged.emit(idx, self.matched_structures(idx))

    def _render_overlays(self, frame: int) -> None:
        peaks = self._frame_peaks.get(frame, {})
        det = peaks.get("detected")
        fit = peaks.get("fitted")

        manual_list = list(self._manual_peaks.get(frame, []))
        # When an ROI is active the selected peak is shown via the ROI handles —
        # exclude it from the path overlay so it doesn't render twice.
        roi_active = self._roi_item is not None and self._selected is not None
        if roi_active:
            manual_list = [m for m in manual_list if m is not self._selected]
        manual_table = _peaks_from_manual(manual_list)
        sel_table = (
            _peaks_from_manual([self._selected])
            if self._selected and not roi_active
            else None
        )

        if self._mode == MODE_POLAR:
            self._detected.set_polar(det)
            self._fitted.set_polar(fit)
            self._manual.set_polar(manual_table)
            if sel_table is not None:
                self._selection.set_polar(sel_table)
            else:
                self._selection.clear_path()
        else:
            self._detected.set_cartesian(det)
            self._fitted.set_cartesian(fit)
            self._manual.set_cartesian(manual_table)
            if sel_table is not None:
                self._selection.set_cartesian(sel_table)
            else:
                self._selection.clear_path()

        self._detected.setVisible(self._visibility["detected"])
        self._fitted.setVisible(self._visibility["fitted"])
        self._manual.setVisible(self._visibility["manual"])

        # Matched overlays: rebuild items for whatever the current frame has.
        self._render_matched_overlays(frame)

    def _render_matched_overlays(self, frame: int) -> None:
        """Tear down the previous frame's matched items and rebuild for this
        frame. Each structure becomes one ``_PeakShapeItem`` in its assigned
        color, painted in the current display mode (polar / Cartesian).
        """
        self._teardown_matched_items()
        structures = self._matched_per_frame.get(frame, [])
        if not structures:
            return
        vb = self._plot.getViewBox()
        for i, s in enumerate(structures):
            color = MATCHED_PALETTE[i % len(MATCHED_PALETTE)]
            item = _PeakShapeItem(color=color, **MATCHED_STYLE)
            if self._mode == MODE_POLAR:
                item.set_polar(s.peaks)
            else:
                item.set_cartesian(s.peaks)
            item.setVisible(self._is_matched_item_visible(s.unique_id))
            vb.addItem(item, ignoreBounds=True)
            self._matched_items.append((s.unique_id, item))

    def _teardown_matched_items(self) -> None:
        if not self._matched_items:
            return
        vb = self._plot.getViewBox()
        for _uid, item in self._matched_items:
            vb.removeItem(item)
        self._matched_items.clear()

    # -- Internals --

    def _on_radio_toggled(self) -> None:
        new = MODE_CARTESIAN if self._radio_cart.isChecked() else MODE_POLAR
        if new != self._mode:
            self.set_mode(new)

    def _on_cmap_changed(self, name: str) -> None:
        self._apply_cmap(name)

    def _apply_cmap(self, name: str) -> None:
        # Try matplotlib first (always present via silx); fall back to the
        # internal pyqtgraph maps if the user picked something not in mpl.
        cmap = None
        for source in ("matplotlib", None):
            try:
                cmap = pg.colormap.get(name, source=source) if source else pg.colormap.get(name)
            except Exception:
                cmap = None
            if cmap is not None:
                break
        if cmap is not None:
            self._view.setColorMap(cmap)

    # -- Labelling event handlers (polar mode only for now) --

    def _on_draw_started(self, origin: QPointF) -> None:
        if self._mode != MODE_POLAR:
            return
        rect = QRectF(origin, origin)
        self._preview_item.setRect(rect.normalized())
        self._preview_item.setVisible(True)

    def _on_draw_updated(self, origin: QPointF, current: QPointF) -> None:
        if self._mode != MODE_POLAR or not self._preview_item.isVisible():
            return
        self._preview_item.setRect(QRectF(origin, current).normalized())

    def _on_draw_finished(self, origin: QPointF, end: QPointF) -> None:
        self._preview_item.setVisible(False)
        if self._mode != MODE_POLAR:
            return
        # In polar mode: x = radius (Å⁻¹), y = angle (deg).
        rect = QRectF(origin, end).normalized()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        peak = ManualPeak(
            radius=float(rect.center().x()),
            angle=float(rect.center().y()),
            radius_width=float(rect.width()),
            angle_width=float(rect.height()),
            is_ring=False,
            temp_id=self._next_manual_id,
        )
        self._next_manual_id -= 1
        self.add_manual_peak(self.current_frame, peak)
        # Auto-select the freshly drawn peak so the user can adjust edges.
        self._set_selected(peak)

    def _on_select_at(self, pos: QPointF) -> None:
        if self._mode != MODE_POLAR:
            return
        x, y = float(pos.x()), float(pos.y())
        for peak in reversed(self._manual_peaks.get(self.current_frame, [])):
            r_lo = peak.radius - peak.radius_width / 2
            r_hi = peak.radius + peak.radius_width / 2
            a_lo = peak.angle - peak.angle_width / 2
            a_hi = peak.angle + peak.angle_width / 2
            if r_lo <= x <= r_hi and a_lo <= y <= a_hi:
                self._set_selected(peak)
                return
        # Click on empty space → deselect
        if self._selected is not None:
            self._set_selected(None)

    def _set_selected(self, peak: ManualPeak | None) -> None:
        """Update the selection and sync the ROI + emit selectionChanged once."""
        if peak is self._selected:
            return
        self._selected = peak
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(peak)

    def keyPressEvent(self, ev) -> None:  # type: ignore[override]
        if ev.key() == Qt.Key.Key_Delete and self._selected is not None:
            self.remove_manual_peak(self.current_frame, self._selected)
            ev.accept()
            return
        super().keyPressEvent(ev)

    # -- Resizable ROI on the selected peak --

    def _sync_roi(self) -> None:
        """Create / update / destroy the resize ROI to match the selection.

        Polar mode only: a ``pg.ROI`` with edge handles wraps the selected
        ``ManualPeak`` so the user can drag any edge to adjust radius/angle.
        """
        if self._roi_item is not None:
            try:
                self._roi_item.sigRegionChanged.disconnect(self._on_roi_changed)
            except (RuntimeError, TypeError):
                pass
            self._plot.getViewBox().removeItem(self._roi_item)
            self._roi_item = None

        if self._selected is None or self._mode != MODE_POLAR:
            return

        peak = self._selected
        pos = (
            peak.radius - peak.radius_width / 2.0,
            peak.angle - peak.angle_width / 2.0,
        )
        size = (peak.radius_width, peak.angle_width)

        style = OVERLAY_STYLE["manual"]
        pen = pg.mkPen(QColor(style["color"]), width=style["width"] + 0.4)
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        hover_pen = pg.mkPen(QColor(SELECTION_STYLE["color"]), width=SELECTION_STYLE["width"])
        hover_pen.setCosmetic(True)

        roi = pg.ROI(pos=pos, size=size, pen=pen, hoverPen=hover_pen, movable=True)
        # Edge-only handles (no corners): each handle drags one edge while the
        # opposite edge stays anchored.
        roi.addScaleHandle([1.0, 0.5], [0.0, 0.5])  # right
        roi.addScaleHandle([0.0, 0.5], [1.0, 0.5])  # left
        roi.addScaleHandle([0.5, 1.0], [0.5, 0.0])  # top
        roi.addScaleHandle([0.5, 0.0], [0.5, 1.0])  # bottom
        roi.setZValue(60)
        roi.sigRegionChanged.connect(self._on_roi_changed)

        self._plot.getViewBox().addItem(roi, ignoreBounds=True)
        self._roi_item = roi

    def _on_roi_changed(self) -> None:
        if self._selected is None or self._roi_item is None:
            return
        roi = self._roi_item
        pos = roi.pos()
        size = roi.size()
        w = abs(float(size[0]))
        h = abs(float(size[1]))
        # ROI sizes can go negative if dragged past the opposite edge — take abs
        # then derive the new center from the (possibly flipped) bottom-left.
        x0 = float(pos[0]) + min(float(size[0]), 0.0)
        y0 = float(pos[1]) + min(float(size[1]), 0.0)
        self._selected.radius_width = w
        self._selected.angle_width = h
        self._selected.radius = x0 + w / 2.0
        self._selected.angle = y0 + h / 2.0
        # Manual overlay path skips the selected peak (the ROI shows it),
        # but other overlays may need a refresh — keep current frame's render synced.
        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(self._selected)

    def update_peak_geometry_external(self, peak: ManualPeak) -> None:
        """Sync the ROI to a peak whose geometry was changed elsewhere
        (e.g. by dragging a profile region). Suppresses ROI signals so this
        doesn't loop back into ``_on_roi_changed``.
        """
        if peak is not self._selected or self._roi_item is None:
            return
        roi = self._roi_item
        roi.blockSignals(True)
        try:
            roi.setPos(
                [
                    peak.radius - peak.radius_width / 2.0,
                    peak.angle - peak.angle_width / 2.0,
                ],
                update=False,
            )
            roi.setSize([peak.radius_width, peak.angle_width])
        finally:
            roi.blockSignals(False)
        self._render_overlays(self.current_frame)
