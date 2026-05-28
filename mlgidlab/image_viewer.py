from __future__ import annotations

import os

# Pin pyqtgraph to PySide6 before it auto-detects.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainterPath
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from mlgidlab.file_model import (
    EntryStack,
    FrameSource,
    MatchedStructure,
    PeakTable,
    _LazyImageStack,
    _LazyPolarStack,
)
from mlgidlab.polar import polar_to_qxyz, stack_to_polar  # noqa: F401 (stack_to_polar retained as a reference impl)

import logging
logger = logging.getLogger(__name__)

OVERLAY_KINDS = ("detected", "fitted", "manual")
MODE_CARTESIAN = "cartesian"
MODE_POLAR = "polar"
# Raw detector data preview — pixel coordinates, no overlays. Reached only
# when a RawSession is active; converted-NeXus sessions never visit this
# mode and their existing Cartesian / Polar paths are unchanged.
MODE_RAW = "raw"

LABEL_MODIFIERS = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier

# Subdivisions along the angular edge for the full 0–90° range; narrower
# segments scale down proportionally (with a small minimum for sharp corners).
ANGULAR_SUBDIV_FULL = 90
ANGULAR_SUBDIV_MIN = 4

# Outer bounds for clipping a peak's angular extent before drawing it as
# a polygon. Set to atan2's full range so peaks produced by converted
# images that span multiple quadrants (e.g. ``vert_positive=False`` →
# angles in [0°, 180°]) still draw correctly. Peaks whose angle / width
# are non-finite are treated as "ring" — their polygon spans the full
# range below.
ANGLE_MIN_DEG = -180.0
ANGLE_MAX_DEG = 180.0

# Visual style for each overlay kind. Dashed for "raw" detection output,
# solid for the refined fit, dotted yellow for user-drawn manual labels.
OVERLAY_STYLE: dict[str, dict] = {
    "detected": {"color": "#ff5c5c", "style": Qt.PenStyle.DashLine, "width": 1.2},
    "fitted":   {"color": "#26d0ce", "style": Qt.PenStyle.SolidLine, "width": 1.2},
    "manual":   {"color": "#ffeb3b", "style": Qt.PenStyle.SolidLine, "width": 1.6},
}

SELECTION_STYLE = {"color": "#ffffff", "style": Qt.PenStyle.SolidLine, "width": 2.5}

# Faint preview of the would-be fitted_peaks box for the currently selected
# manual peak. Same hue as the fitted overlay so the user reads the
# relationship at a glance, but dashed + reduced opacity so it's clearly a
# preview rather than a stored peak.
FITTED_PREVIEW_STYLE = {
    "color": OVERLAY_STYLE["fitted"]["color"],
    "style": Qt.PenStyle.DashLine,
    "width": 1.4,
}
FITTED_PREVIEW_OPACITY = 0.45

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
# Line styles cycled after the palette wraps. Combined with
# MATCHED_PALETTE this yields ``len(palette) * len(styles)`` unique
# pens before any (colour, style) pair repeats — enough headroom for
# the 28-row deduped solutions on real datasets without resorting to
# colour-only disambiguation.
MATCHED_LINE_STYLES: tuple[Qt.PenStyle, ...] = (
    Qt.PenStyle.SolidLine,
    Qt.PenStyle.DashLine,
    Qt.PenStyle.DashDotLine,
    Qt.PenStyle.DotLine,
)
MATCHED_LINE_WIDTH = 1.6
# Backwards-compat: callers that still want the default line style
# can keep using this dict. New code should prefer ``matched_pen_for``
# which combines the palette + line-style cycle.
MATCHED_STYLE = {"style": MATCHED_LINE_STYLES[0], "width": MATCHED_LINE_WIDTH}


def matched_pen_for(index: int) -> dict:
    """Return ``{color, style, width}`` for the ``index``-th structure.

    Colour cycles first so adjacent rows pick up a different hue at
    the same line style — the palette gives the strongest visual
    contrast and is enough for files with up to ``len(MATCHED_PALETTE)``
    matched structures. Once the palette wraps, the line style steps
    to the next (dashed → dash-dot → dotted) so the next 10 rows are
    still distinguishable from the first 10 even when their colours
    repeat. With 10 colours × 4 styles the palette runs out only past
    40 simultaneous structures.
    """
    n_colors = len(MATCHED_PALETTE)
    n_styles = len(MATCHED_LINE_STYLES)
    color = MATCHED_PALETTE[index % n_colors]
    style = MATCHED_LINE_STYLES[(index // n_colors) % n_styles]
    return {"color": color, "style": style, "width": MATCHED_LINE_WIDTH}

# Curated list of colormaps. Names are matplotlib's; pg.colormap.get falls
# back to matplotlib's registry, which is always available since matplotlib
# is a transitive dep via silx.
COLORMAPS = ("viridis", "inferno", "plasma", "magma", "cividis", "gray")
DEFAULT_COLORMAP = "magma"


def _disable_viewport_scroll(widget) -> None:
    """Disable QAbstractScrollArea-level scrolling on a pyqtgraph widget.

    Pyqtgraph's GraphicsView / PlotWidget inherit from QAbstractScrollArea,
    so even with the scrollbars hidden the viewport can still slide
    when the scene rect is slightly bigger than the visible area
    (typically by a few pixels of axis-label padding). Overriding
    ``scrollContentsBy`` to a no-op blocks every scroll path —
    scrollbar drag, wheel-on-bar, two-finger gesture, programmatic
    `setValue` — without touching the inner ViewBox's pan / zoom,
    which lives one Qt level deeper as a graphics-item event.

    Implemented by reparenting the instance to a dynamically-created
    subclass; safer than instance-level monkey-patching since Qt
    dispatches virtual methods through the C++ vtable.
    """
    cls = type(widget)
    if cls.__name__.endswith("_NoScroll"):
        return
    new_cls = type(
        cls.__name__ + "_NoScroll",
        (cls,),
        {"scrollContentsBy": lambda self, dx, dy: None},
    )
    widget.__class__ = new_cls


def _bin_index(axis: np.ndarray, value: float) -> int:
    """Floor-based bin index for an evenly-spaced axis.

    The image-display routines call ``setImage(pos=axis[0], scale=step)``,
    so axis[0] is the LOWER edge of pixel 0 and pixel ``i`` covers
    ``[axis[0] + i*step, axis[0] + (i+1)*step)``. Returning the bin
    index by ``floor`` instead of ``argmin(|axis - v|)`` keeps the
    cursor readout constant within a displayed pixel — argmin
    transitions at axis-midpoints, which is half a pixel off from
    where the user sees the boundary.
    """
    n = len(axis)
    if n == 0:
        return 0
    if n == 1:
        return 0
    step = (float(axis[-1]) - float(axis[0])) / (n - 1)
    if step == 0.0:
        return 0
    idx = int(np.floor((float(value) - float(axis[0])) / step))
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


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
    """Per-mode display state. ``image_pg`` is the *single 2D frame*
    for the current frame index — we never feed pyqtgraph a 3D stack
    anymore (see lazy-loading milestone). The viewer keeps the most
    recent ``_DisplayParams`` so per-frame scrubs can re-use ``pos`` /
    ``scale`` / ``levels`` without re-deriving them.
    """
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


@dataclass
class SelectedPeak:
    """Snapshot of the currently-selected peak, regardless of source.

    Carries enough geometry for the ROI and parameter panel without forcing
    callers to know which overlay holds the underlying data. ``manual_ref``
    is set only when ``kind == "manual"`` and is the same instance held by
    ``_manual_peaks`` — mutating it propagates to the manual overlay.
    """

    kind: str  # "manual" | "detected" | "fitted" | "matched"
    frame: int
    peak_id: int
    radius: float
    angle: float
    radius_width: float
    angle_width: float
    is_ring: bool = False
    structure_uid: str | None = None
    # Human-readable structure label + overlay color, populated only for
    # matched selections so the parameter panel can render the source row
    # without re-deriving these from the viewer's matched bookkeeping.
    structure_label: str | None = None
    structure_color: str | None = None
    manual_ref: ManualPeak | None = None
    # mlgidDETECT confidence score for the underlying row. Populated for
    # detected / fitted / matched selections; left as None for manual
    # peaks (which have no model provenance) so the parameter panel can
    # skip the row instead of showing a misleading zero.
    score: float | None = None
    # Peak amplitude (2D-Gaussian peak height). Populated for detected /
    # fitted / matched selections; the profile viewer uses it to render
    # the projection of the persisted 2D Gaussian onto the radial /
    # angular axis (physics-audit F-06: "profile must reflect 2D fit").
    # None for manual peaks (no fit yet), which fall back to the live
    # 1D scipy fit on the integrated profile data.
    amplitude: float | None = None
    # When non-None, the selection represents a *matched structure*
    # rather than a single peak. The list holds every fitted-peak id
    # that belongs to the structure; the overlay highlight is drawn
    # around all of them at once. The single peak_id field still
    # carries a representative id (typically the first one) so the
    # ParameterPanel can render the structure_label + first-peak
    # geometry. Always None for detected / fitted / manual selections.
    multi_peak_ids: list[int] | None = None

    @classmethod
    def from_manual(cls, peak: ManualPeak, frame: int) -> SelectedPeak:
        return cls(
            kind="manual",
            frame=frame,
            peak_id=peak.temp_id,
            radius=peak.radius,
            angle=peak.angle,
            radius_width=peak.radius_width,
            angle_width=peak.angle_width,
            is_ring=peak.is_ring,
            manual_ref=peak,
        )

    def polar_tuple(self) -> tuple[float, float, float, float]:
        return (self.radius, self.angle, self.radius_width, self.angle_width)


# -- Undo/redo actions -----------------------------------------------------
#
# Each action carries the data needed to flip the viewer + (for FileGeom)
# the file. ``undo`` and ``redo`` mirror each other so we can move back and
# forth on the stack without special-casing.


class _Action(Protocol):
    def undo(self, viewer: "GIWAXSImageViewer") -> None: ...
    def redo(self, viewer: "GIWAXSImageViewer") -> None: ...


@dataclass
class ManualAddAction:
    frame: int
    peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_add_manual(self.frame, self.peak)


@dataclass
class ManualRemoveAction:
    frame: int
    peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_add_manual(self.frame, self.peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.peak)


@dataclass
class ManualGeomAction:
    frame: int
    peak: ManualPeak
    before: tuple[float, float, float, float]  # (r, a, dr, da)
    after: tuple[float, float, float, float]

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_manual_geom(self.frame, self.peak, self.before)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_manual_geom(self.frame, self.peak, self.after)


@dataclass
class ManualReplaceAction:
    """Atomic swap of the single manual peak on a frame.

    With the new "at most one manual box per frame" policy, drawing a
    new box replaces any existing one. We model that as a single undo
    entry (instead of separate add + remove entries) so a single
    Ctrl+Z rewinds the whole replace cleanly. ``old_peak`` may be None
    when the user drew the very first manual peak on this frame —
    redo then just adds the new one without removing anything.
    """

    frame: int
    old_peak: ManualPeak | None
    new_peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.new_peak)
        if self.old_peak is not None:
            viewer._undoable_add_manual(self.frame, self.old_peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        if self.old_peak is not None:
            viewer._undoable_remove_manual(self.frame, self.old_peak)
        viewer._undoable_add_manual(self.frame, self.new_peak)


@dataclass
class FileGeomAction:
    frame: int
    kind: str  # "detected" | "fitted"
    peak_id: int
    before: tuple[float, float, float, float]
    after: tuple[float, float, float, float]

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_file_geom(self.frame, self.kind, self.peak_id, self.before)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_file_geom(self.frame, self.kind, self.peak_id, self.after)


def _action_targets_manual(action: _Action, peak: ManualPeak) -> bool:
    """True when ``action`` references the ManualPeak ``peak`` by identity.

    Used by ``commit_manual_peak`` to scrub any stale stack entries that
    point at a peak we just persisted to the file.
    """
    target = getattr(action, "peak", None)
    return target is peak


def _peaks_subset(table: PeakTable, ids: list[int]) -> PeakTable:
    """Return a row-subset of ``table`` keyed by ``ids``.

    Used to build a multi-peak selection overlay from a matched
    structure's full id list. Rows whose id isn't in ``table.ids``
    are silently skipped (stale matched references — same tolerance
    applied in ``file_model.list_matched_structures``).
    """
    if len(table) == 0 or not ids:
        empty = np.zeros(0, dtype=float)
        return PeakTable(
            q_xy=empty, q_z=empty, angle=empty, radius=empty,
            angle_width=empty, radius_width=empty,
            is_ring=np.zeros(0, dtype=bool),
            ids=np.zeros(0, dtype=int),
            score=empty, amplitude=empty,
        )
    want = set(int(x) for x in ids)
    idx = np.array(
        [i for i in range(len(table)) if int(table.ids[i]) in want],
        dtype=int,
    )
    if idx.size == 0:
        return _peaks_subset(table, [])
    return PeakTable(
        q_xy=table.q_xy[idx],
        q_z=table.q_z[idx],
        angle=table.angle[idx],
        radius=table.radius[idx],
        angle_width=table.angle_width[idx],
        radius_width=table.radius_width[idx],
        is_ring=table.is_ring[idx],
        ids=table.ids[idx],
        score=table.score[idx],
        amplitude=table.amplitude[idx],
    )


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
            score=empty,
            amplitude=empty,
        )
    _radius = np.array([m.radius for m in manual], dtype=float)
    _angle = np.array([m.angle for m in manual], dtype=float)
    _qxy, _qz = polar_to_qxyz(_radius, _angle)
    return PeakTable(
        q_xy=_qxy,
        q_z=_qz,
        angle=_angle,
        radius=_radius,
        angle_width=np.array([m.angle_width for m in manual], dtype=float),
        radius_width=np.array([m.radius_width for m in manual], dtype=float),
        is_ring=np.array([m.is_ring for m in manual], dtype=bool),
        ids=np.array([m.temp_id for m in manual], dtype=int),
        score=np.zeros(len(manual), dtype=float),
        amplitude=np.zeros(len(manual), dtype=float),
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
    # Bare LMB double-click (no modifiers, no drag) — wired to reset zoom.
    doubleClicked = Signal()
    # Hover-aware cursor tracking — fires on every mouse move (with or
    # without a button held). The viewer translates the data-space point
    # into the public ``cursorMoved`` payload consumed by the status bar.
    cursorPos = Signal(QPointF)
    cursorLeft = Signal()

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
        # MouseMove only fires with a button held unless tracking is on.
        # The status-bar cursor readout needs hover updates, so force it.
        self._gv.viewport().setMouseTracking(True)

    def eventFilter(self, _obj: QObject, ev: QEvent) -> bool:  # type: ignore[override]
        et = ev.type()
        if (
            et == QEvent.Type.MouseButtonDblClick
            and ev.button() == Qt.MouseButton.LeftButton
            and ev.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            # Bare LMB double-click anywhere on the image resets the zoom.
            # Modifier+double-click and other-button double-click fall
            # through so pyqtgraph's default handlers (e.g. ROI editing)
            # still see the event.
            self.doubleClicked.emit()
            return True
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
        if et == QEvent.Type.MouseMove:
            # Always emit cursor position for the status-bar readout —
            # independent of whether a draw drag is in progress.
            data_pos = self._viewport_to_data(ev.position().toPoint())
            self.cursorPos.emit(data_pos)
            if self._drawing and self._origin is not None:
                self.drawUpdated.emit(self._origin, data_pos)
                return True
        if et == QEvent.Type.Leave:
            self.cursorLeft.emit()
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

    def set_polar(
        self,
        peaks: PeakTable | None,
        extent: tuple[float, float] | None = None,
    ) -> None:
        path = QPainterPath()
        if peaks is not None and len(peaks) > 0:
            for i in range(len(peaks)):
                clip = _clip_angle(
                    float(peaks.angle[i]), float(peaks.angle_width[i]),
                    extent=extent,
                )
                if clip is None:
                    continue
                a_lo, a_hi = clip
                r = float(peaks.radius[i])
                dr = float(peaks.radius_width[i])
                path.addRect(QRectF(r - dr / 2, a_lo, dr, a_hi - a_lo))
        self._update_path(path)

    def set_cartesian(
        self,
        peaks: PeakTable | None,
        extent: tuple[float, float] | None = None,
    ) -> None:
        path = QPainterPath()
        if peaks is not None and len(peaks) > 0:
            for i in range(len(peaks)):
                clip = _clip_angle(
                    float(peaks.angle[i]), float(peaks.angle_width[i]),
                    extent=extent,
                )
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


def _polar_box_contains(peak: ManualPeak, x: float, y: float) -> bool:
    """Hit-test the polar bounding box of a ManualPeak."""
    r_lo = peak.radius - peak.radius_width / 2.0
    r_hi = peak.radius + peak.radius_width / 2.0
    a_lo = peak.angle - peak.angle_width / 2.0
    a_hi = peak.angle + peak.angle_width / 2.0
    return r_lo <= x <= r_hi and a_lo <= y <= a_hi


def _polar_table_row_contains(table: PeakTable, i: int, x: float, y: float) -> bool:
    """Hit-test row ``i`` of a PeakTable in polar coordinates."""
    r = float(table.radius[i])
    dr = float(table.radius_width[i])
    a = float(table.angle[i])
    da = float(table.angle_width[i])
    r_lo, r_hi = r - dr / 2.0, r + dr / 2.0
    a_lo, a_hi = a - da / 2.0, a + da / 2.0
    return r_lo <= x <= r_hi and a_lo <= y <= a_hi


def _cart_to_polar(q_xy: float, q_z: float) -> tuple[float, float]:
    """Map a Cartesian click ``(q_xy, q_z)`` back to ``(radius, angle_deg)``.

    Matches the convention used everywhere else in the pipeline: the
    Cartesian projections of a polar peak are written as
    ``q_xy = r * cos(a)``, ``q_z = r * sin(a)`` (see
    ``file_model.add_fitted_peak_row``), so inverting with
    ``hypot`` + ``atan2(q_z, q_xy)`` reproduces the original
    ``(radius, angle)``. The polar containment checks downstream
    can then run unchanged.
    """
    r = float(np.hypot(q_xy, q_z))
    a = float(np.degrees(np.arctan2(q_z, q_xy)))
    return r, a


def _cart_box_contains(peak: ManualPeak, q_xy: float, q_z: float) -> bool:
    """Cartesian hit-test for a ManualPeak's polar box."""
    r, a = _cart_to_polar(q_xy, q_z)
    return _polar_box_contains(peak, r, a)


def _cart_table_row_contains(
    table: PeakTable, i: int, q_xy: float, q_z: float,
) -> bool:
    """Cartesian hit-test for row ``i`` of a PeakTable."""
    r, a = _cart_to_polar(q_xy, q_z)
    return _polar_table_row_contains(table, i, r, a)


def _clip_angle(
    a_deg: float,
    da_deg: float,
    extent: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Clip a polar angular box to the viewer's visible range.

    Treats infinite or non-finite angle_width as 'spans the whole quadrant',
    so rings (whose angle_width is sometimes inf) still draw correctly.
    ``extent`` is the actual displayed angular axis of the active polar
    stack — pass it so ring overlays stop at the image edge instead of
    extending to the global ``[-180°, 180°]`` clipping bounds. When
    ``extent`` is None the global bounds are used as a fallback (raw-
    mode renders, unit tests).

    Returns (lo, hi) in degrees, or None if the box is empty/invalid.
    """
    if extent is None:
        ext_lo, ext_hi = ANGLE_MIN_DEG, ANGLE_MAX_DEG
    else:
        ext_lo, ext_hi = float(extent[0]), float(extent[1])
        if ext_hi < ext_lo:
            ext_lo, ext_hi = ext_hi, ext_lo
    if not np.isfinite(a_deg) or not np.isfinite(da_deg):
        a_lo, a_hi = ext_lo, ext_hi
    else:
        a_lo = a_deg - da_deg / 2.0
        a_hi = a_deg + da_deg / 2.0
    a_lo = max(a_lo, ext_lo)
    a_hi = min(a_hi, ext_hi)
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
    # Cursor readout — emits a dict describing the data point under the
    # cursor (q-mode vs pixel-mode), or None when the pointer leaves
    # the viewport. Consumers (status bar) format the dict for display.
    cursorMoved = Signal(object)
    manualPeakAdded = Signal(int, object)     # frame, ManualPeak
    manualPeakRemoved = Signal(int, object)   # frame, ManualPeak
    selectionChanged = Signal(object)         # SelectedPeak | None
    peakGeometryChanged = Signal(object)      # SelectedPeak whose r/dr/a/da changed
    # Drag-end variant of ``peakGeometryChanged``. Fires once when the
    # user releases the ROI handle (or the user-side undo/redo path
    # settles a geometry mutation). Consumers that want to react only
    # to the final position — not every drag tick — subscribe here.
    # MainWindow's 2D-preview refresh listens to this one because each
    # pygidfit call is ~100-500 ms and per-tick refits would freeze
    # the drag.
    peakGeometryDragFinished = Signal(object)  # SelectedPeak (post-drag)
    # Emitted on drag-end for non-manual peaks: (frame, kind, peak_id,
    # polar_kwargs). MainWindow drives the actual h5py mutation since it
    # owns the silx tree handle that needs releasing first.
    peakRowWriteRequested = Signal(int, str, int, dict)
    # Emitted when the user presses Delete on a non-manual peak.
    deletePeakRequested = Signal(object)      # SelectedPeak
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
        bar.addSpacing(16)
        # Log/linear contrast toggle. When checked, the displayed image
        # is log10(clip(data, floor, inf)) and the histogram levels are
        # recomputed on the transformed array so the LUT stays sensible.
        # Coordinates and overlays are unaffected — only the intensity
        # mapping changes.
        self._log_check = QCheckBox("Log scale")
        self._log_check.setChecked(False)
        self._log_check.setToolTip(
            "Display log10(intensity) instead of linear intensity. "
            "Useful for GIWAXS data with wide dynamic range; coordinate "
            "axes and overlays are unchanged."
        )
        self._log_check.toggled.connect(self._on_log_toggled)
        bar.addWidget(self._log_check)
        # The frame-navigation controls (prev / play / next / slider /
        # label) used to live in the Display dock's Frame row. They
        # now sit here in the toolbar so the user can scrub frames
        # regardless of which right-dock tab is in front. The host
        # injects them via ``insert_frame_controls`` after the
        # Display dock builds them — see MainWindow.
        bar.addStretch(1)
        # Keep a handle on the layout so the host can splice the
        # frame-navigation widgets in just before the trailing stretch.
        self._toolbar_layout = bar
        bar_widget = QWidget(self)
        bar_widget.setLayout(bar)
        outer.addWidget(bar_widget)

        self._plot = pg.PlotItem()
        self._view = pg.ImageView(self, view=self._plot)
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        # Hide pyqtgraph's bottom timeline strip — redundant with the
        # Display-dock frame slider. ``setImage`` re-shows it via
        # ``roiClicked`` for any multi-frame stack, so we re-apply our
        # toggle state in ``_apply_params`` after every render.
        self._view.ui.roiPlot.hide()
        # Set the splitter handle width to 0 so even when the strip is
        # hidden there's no grey separator line eating into the image's
        # x-axis label area.
        self._view.ui.splitter.setHandleWidth(0)
        self._view.ui.splitter.setSizes([1, 0])
        # Pyqtgraph's GraphicsView occasionally lets the scene scroll
        # by a few pixels when its sceneRect has drifted from the
        # viewport size — kill scrollbars + frame so the plot is
        # unconditionally pinned inside its tab.
        gv = self._view.ui.graphicsView
        gv.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setFrameStyle(QFrame.Shape.NoFrame)
        # Block QAbstractScrollArea-level scrolling without touching
        # ViewBox pan / zoom — see _disable_viewport_scroll docstring.
        _disable_viewport_scroll(gv)
        outer.addWidget(self._view)

        self._plot.invertY(False)
        self._plot.setAspectLocked(False)
        # PyQtGraph occasionally underestimates the bottom axis cell so
        # the axis label ("radius" / "q_xy") gets clipped by the
        # viewport's lower edge — and that clip is what creates the
        # small scrollable region. A small bottom layout margin gives
        # the label guaranteed clearance and keeps the plot fitted
        # inside its tab.
        self._plot.layout.setContentsMargins(0, 0, 0, 12)

        self._detected = _PeakShapeItem(**OVERLAY_STYLE["detected"])
        self._fitted = _PeakShapeItem(**OVERLAY_STYLE["fitted"])
        self._manual = _PeakShapeItem(**OVERLAY_STYLE["manual"])
        self._selection = _PeakShapeItem(**SELECTION_STYLE)
        self._fitted_preview = _PeakShapeItem(**FITTED_PREVIEW_STYLE)
        self._fitted_preview.setOpacity(FITTED_PREVIEW_OPACITY)
        vb = self._plot.getViewBox()
        vb.addItem(self._detected, ignoreBounds=True)
        vb.addItem(self._fitted, ignoreBounds=True)
        vb.addItem(self._manual, ignoreBounds=True)
        vb.addItem(self._selection, ignoreBounds=True)
        vb.addItem(self._fitted_preview, ignoreBounds=True)

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
        self._label_filter.doubleClicked.connect(self.reset_zoom)
        self._label_filter.cursorPos.connect(self._on_cursor_pos)
        self._label_filter.cursorLeft.connect(self._on_cursor_left)

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
        # ``unique_id``s currently hidden by the Display-dock search
        # filter. Independent of ``_matched_visibility`` (which
        # tracks checkbox state); the filter set overrides checkbox
        # state but doesn't mutate it, so clearing the filter
        # restores the previous user selection.
        self._matched_filter_hidden: set[str] = set()
        # Minimum detection score to render on the Detected overlay.
        # Driven by the Display dock's Min-score slider. 0.0 (the
        # default) lets every detected peak through; raising it
        # hides weak detections without mutating the file.
        self._detected_score_cutoff: float = 0.0
        self._matched_items: list[tuple[str, _PeakShapeItem]] = []

        self._mode = MODE_POLAR
        self._log_scale: bool = False
        self._stack: EntryStack | None = None
        # The FrameSource backing the active EntryStack. Owns the live
        # h5py handle + per-frame LRU; released across the silx detach
        # / reattach dance via release_frame_source / acquire_frame_source.
        self._frame_source: "FrameSource | None" = None  # type: ignore[name-defined]
        # The viewer drives frame indexing itself (the Display-dock slider
        # is the single source of truth). pyqtgraph used to track this via
        # ImageView.currentIndex, but we stopped feeding it 3D stacks once
        # large-file support landed — see the lazy-loading milestone.
        self._frame_index: int = 0
        # Last applied _DisplayParams so per-frame scrubs reuse pos/scale/
        # levels without rebuilding them.
        self._display_params: _DisplayParams | None = None
        # Raw-mode preview state. Held separately from ``_stack`` because raw
        # detector frames have no q-axes — they're rendered in pixel
        # coordinates and carry no overlays.
        self._raw_image_stack: np.ndarray | None = None
        # Polar grid axes derived from the FrameSource. Stored as a
        # (lazy_polar_stack, radius, angle) tuple so consumers
        # (profile viewer, cursor readout) get a single-frame index
        # without precomputing the full polar stack.
        self._polar_cache: "tuple[_LazyPolarStack, np.ndarray, np.ndarray] | None" = None  # type: ignore[name-defined]
        self._next_manual_id = -1  # negative IDs distinguish manual from detected
        self._selected: SelectedPeak | None = None
        self._roi_item: pg.ROI | None = None
        # Stacks of `_Action` objects. Pushing to undo clears redo. ROI drags
        # populate _roi_drag_before on sigRegionChangeStarted and consume it
        # on sigRegionChangeFinished — partial state never lands on the stack.
        self._undo_stack: list[_Action] = []
        self._redo_stack: list[_Action] = []
        self._roi_drag_before: tuple[float, float, float, float] | None = None
        # Set during a pipeline run so we don't allow concurrent ROI edits or
        # Delete keypresses while mlgidbase has the file open for writes.
        self._busy: bool = False

        # Geometry of the fitted-preview box for the current selection
        # (radius_center, fwhm_radial, angle_center, fwhm_angular). Cleared
        # whenever the selection isn't a manual / detected peak with valid
        # 1D fits.
        self._fitted_preview_geom: tuple[float, float, float, float] | None = None
        # When True, the preview is rendered as a ring (full angular sweep
        # at angle = 45°, angle_width = ∞) regardless of the angular fit —
        # mirrors what Add-to-fitted will write when the "Save fitted as
        # ring" toggle is on.
        self._fitted_preview_is_ring: bool = False

        # NB: pyqtgraph's sigTimeChanged is no longer connected — we
        # stopped feeding ImageView 3D stacks once lazy frame loading
        # landed (see the lazy-loading milestone in
        # Documentation/). Frame changes now flow exclusively through
        # MainWindow → viewer.set_frame, and ``frameChanged`` is emitted
        # from there.

        # Add a "Reset zoom" action to the viewbox right-click menu so the
        # user can undo a manual zoom without leaving the keyboard / mouse.
        # pyqtgraph's default "View All" does the same thing under a less
        # discoverable label; this just adds the explicitly-named entry.
        self._install_reset_zoom_action()

    # -- Public API --

    def show_stack(self, stack: EntryStack, *, preserve_view: bool = False) -> None:
        """Render ``stack`` in the active mode.

        ``preserve_view`` keeps the current viewbox range and frame
        index across the re-render — used after pipeline ops and
        direct h5py edits where the underlying stack is identical and
        only the peak overlays changed. Default ``False`` (autorange)
        for the entry-switch / file-open paths, where the new stack
        typically has different axes.

        ``stack.image_stack`` must be a ``_LazyImageStack`` (i.e. the
        return value of ``file_model.load_entry``). The viewer extracts
        its backing ``FrameSource`` and stashes it on
        ``self._frame_source`` so frame reads can stream from disk
        without the full stack ever entering RAM.
        """
        # Capture before resetting cached state so the saved range is the
        # one the user is actually looking at right now.
        saved_xrange: tuple[float, float] | None = None
        saved_yrange: tuple[float, float] | None = None
        saved_frame: int | None = None
        if preserve_view and self._stack is not None:
            try:
                xr, yr = self._plot.getViewBox().viewRange()
                saved_xrange = (float(xr[0]), float(xr[1]))
                saved_yrange = (float(yr[0]), float(yr[1]))
            except Exception:
                logger.debug("suppressed exception in GIWAXSImageViewer.show_stack", exc_info=True)
                pass
            saved_frame = self.current_frame

        # Release any prior FrameSource before adopting the new one —
        # an open h5py handle on the previous temp file would block its
        # cleanup during session close on Windows.
        if (
            self._frame_source is not None
            and not isinstance(stack.image_stack, _LazyImageStack)
        ):
            self._frame_source.release()
            self._frame_source = None
        elif (
            self._frame_source is not None
            and isinstance(stack.image_stack, _LazyImageStack)
            and stack.image_stack.source is not self._frame_source
        ):
            self._frame_source.release()
            self._frame_source = None

        self._stack = stack
        if isinstance(stack.image_stack, _LazyImageStack):
            self._frame_source = stack.image_stack.source
        else:
            # Defensive fallback for tests that hand-build an EntryStack
            # with a raw ndarray. The viewer's hot path always goes
            # through _frame_source, so reject this case loudly.
            raise TypeError(
                "show_stack expects an EntryStack whose image_stack is "
                "a _LazyImageStack (i.e. produced by load_entry). Got: "
                f"{type(stack.image_stack).__name__}"
            )
        # ``_raw_image_stack`` belongs to a prior RawSession; clearing it
        # here ensures _render_active_mode never tries to re-render raw
        # pixel data over a NeXus stack.
        self._raw_image_stack = None
        # If the previous session was raw, the mode flag is still
        # ``MODE_RAW`` even though raw rendering ignored the radios.
        # Snap back to whichever Cartesian / Polar radio is checked
        # (Polar is the default at startup) so ``_render_active_mode``
        # takes the converted-data branch.
        if self._mode == MODE_RAW:
            self._mode = (
                MODE_CARTESIAN if self._radio_cart.isChecked() else MODE_POLAR
            )
        self._polar_cache = None
        self._frame_peaks.clear()
        # Clamp frame index to the new stack's range. Preserved when
        # the caller asked for it and the prior frame is still valid;
        # otherwise default to 0 so a fresh entry starts at the first
        # frame.
        if preserve_view and saved_frame is not None:
            self._frame_index = max(
                0, min(int(saved_frame), self._stack.n_frames - 1)
            )
        else:
            self._frame_index = 0
        # Skip the setImage autoRange entirely when we're preserving
        # the user's zoom — otherwise the viewbox flashes to the
        # full extent before the post-render setRange snaps it back,
        # which is visually jarring (and breaks if Qt schedules the
        # autoRange via a deferred signal that fires after setRange).
        # The post-render setRange below is the belt to this braces.
        self._render_active_mode(auto_range=not preserve_view)
        if preserve_view and saved_xrange is not None and saved_yrange is not None:
            self._plot.getViewBox().setRange(
                xRange=saved_xrange, yRange=saved_yrange, padding=0
            )

    def set_mode_radios_visible(self, visible: bool) -> None:
        """Show / hide the Cartesian / Polar radios in the top toolbar.

        Used by the host to remove mode controls when a raw session is
        active — raw frames don't carry q-axes, so the toggles would be
        nonsensical. The toolbar's "Colormap" + "Timeline" widgets stay
        visible because they apply equally to raw and converted data.
        """
        # Find the leading "View:" label by walking up from the radio's
        # parent layout — the label was added directly before the radios.
        for w in (self._radio_cart, self._radio_polar):
            w.setVisible(visible)
        # Hide the "View:" prefix label too. It lives in the same toolbar
        # row built by __init__; locate it by text rather than caching a
        # reference at construction time so existing layout code stays put.
        for label in self.findChildren(QLabel):
            if label.text() == "View:":
                label.setVisible(visible)
                break

    def show_raw_stack(self, arr_3d: np.ndarray) -> None:
        """Render a raw detector stack in pixel coordinates.

        Used only for raw-mode (pre-conversion) preview. Wipes any prior
        NeXus-mode state — overlays, peaks, undo history — because none
        of it applies to a raw detector frame. The viewer's frame slider
        and timeline still drive frame navigation across the stack.
        """
        if arr_3d.ndim != 3:
            raise ValueError(
                f"show_raw_stack expects a 3D (N, H, W) array, got shape {arr_3d.shape}"
            )
        # Drop NeXus-mode state (peaks / matched / undo / cached polar).
        # ``clear()`` already covers everything except the raw-stack field
        # itself.
        self.clear()
        self._mode = MODE_RAW
        self._raw_image_stack = np.ascontiguousarray(arr_3d)
        self._render_active_mode()

    def reset_zoom(self) -> None:
        """Auto-fit the viewbox to the current image."""
        try:
            self._plot.getViewBox().autoRange()
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer.reset_zoom", exc_info=True)
            pass

    def _install_reset_zoom_action(self) -> None:
        vb = self._plot.getViewBox()
        menu = getattr(vb, "menu", None)
        if menu is None:
            return
        action = QAction("Reset zoom", menu)
        action.triggered.connect(self.reset_zoom)
        # Insert at the top so it lands above pyqtgraph's default entries.
        first = menu.actions()[0] if menu.actions() else None
        if first is not None:
            menu.insertAction(first, action)
            menu.insertSeparator(first)
        else:
            menu.addAction(action)
        self._reset_zoom_action = action  # keep a reference

    def set_peaks(self, frame: int, peaks: dict[str, PeakTable | None]) -> None:
        self._frame_peaks[frame] = peaks
        if frame == self.current_frame:
            self._render_overlays(frame)

    def insert_frame_controls(self, widgets: list[QWidget]) -> None:
        """Splice ``widgets`` into the toolbar just before the
        trailing stretch.

        The host (MainWindow) owns the frame-navigation widgets
        (previous, play, next, slider, label) so it can keep their
        signal wiring intact when re-parenting them. The trailing
        stretch added in ``__init__`` is **preserved** here — when
        the slider is hidden (single-frame stack) the stretch is
        the only expanding item left, which keeps the other toolbar
        controls clustered to the left instead of being spread out
        across the image width. When the slider is visible it
        carries a much larger stretch factor so it dominates and
        consumes the leftover horizontal space; the trailing
        stretch only kicks in when the slider's stretch
        contribution is zero (hidden widget).
        """
        bar = self._toolbar_layout
        # Insert in front of the trailing stretch (always the last
        # item from __init__). ``insert_at`` is updated after each
        # call so widgets land in the order given.
        insert_at = bar.count() - 1
        if insert_at < 0:
            insert_at = 0
        bar.insertSpacing(insert_at, 16)
        insert_at += 1
        for w in widgets:
            if isinstance(w, QSlider):
                # Slider gets a much larger stretch factor than the
                # trailing stretch so it eats the leftover width
                # when visible. When the slider is hidden, the
                # trailing stretch (factor 1) absorbs the space
                # alone and the other toolbar items stay packed
                # against the left edge.
                bar.insertWidget(insert_at, w, 100)
            else:
                bar.insertWidget(insert_at, w)
            insert_at += 1

    def set_overlay_visible(self, kind: str, visible: bool) -> None:
        if kind not in OVERLAY_KINDS:
            return
        self._visibility[kind] = visible
        item = self._overlay_item(kind)
        if item is not None:
            item.setVisible(visible)
        # Hiding the overlay that owns the current selection also clears the
        # selection highlight so it doesn't dangle.
        if (
            not visible
            and self._selected is not None
            and self._selected.kind == kind
        ):
            self.clear_selection()

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
        Deterministic per insertion order within the frame so the Display
        panel and the overlay agree without extra plumbing.
        """
        return self.matched_pen(structure)["color"]

    def matched_pen(self, structure: MatchedStructure) -> dict:
        """Return the full ``{color, style, width}`` pen for ``structure``.

        Pairs with ``matched_pen_for(index)`` — the panel uses this so
        each row's swatch reproduces the exact line style on screen.
        """
        frame = self.current_frame
        lst = self._matched_per_frame.get(frame, [])
        for i, s in enumerate(lst):
            if s.unique_id == structure.unique_id:
                return matched_pen_for(i)
        return matched_pen_for(0)

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

    def set_detected_score_cutoff(self, value: float) -> None:
        """Hide detected-overlay rows whose score is below ``value``.

        Re-renders overlays for the current frame so the change
        takes effect immediately. The cutoff is applied with a
        small float-epsilon so a slider at exactly the maximum
        score still shows that row (otherwise FP roundoff on stored
        scores can make ``score >= cutoff`` spuriously False).
        """
        self._detected_score_cutoff = float(value)
        self._render_overlays(self.current_frame)

    def set_matched_filter_hidden(self, hidden_ids) -> None:
        """Hide overlays for structures filtered out of the
        Display-dock search.

        Independent of ``set_matched_structure_visible`` (the
        checkbox-driven path): an empty filter set restores
        whatever the user's per-structure checkboxes say, while a
        non-empty set forces those ``unique_id``s off regardless of
        checkbox state. This lets the user search by CIF substring
        without losing their checkbox selections when the filter
        clears.
        """
        self._matched_filter_hidden = set(hidden_ids)
        for uid, item in self._matched_items:
            item.setVisible(self._is_matched_item_visible(uid))

    def _is_matched_item_visible(self, unique_id: str) -> bool:
        if not self._matched_master_visible:
            return False
        if unique_id in self._matched_filter_hidden:
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
        self._fitted_preview.clear_path()
        self._fitted_preview_geom = None
        self._frame_peaks.clear()
        self._manual_peaks.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        # Tear down all matched items and forget per-frame state.
        self._teardown_matched_items()
        self._matched_per_frame.clear()
        self._matched_visibility.clear()
        had_selection = self._selected is not None
        self._selected = None
        self._roi_drag_before = None
        self._sync_roi()
        # Release the long-lived h5py read handle before dropping the
        # stack reference — temp-dir cleanup at session close otherwise
        # fails on Windows because the file is still open.
        if self._frame_source is not None:
            self._frame_source.release()
            self._frame_source = None
        self._stack = None
        self._raw_image_stack = None
        self._polar_cache = None
        self._display_params = None
        self._frame_index = 0
        if had_selection:
            self.selectionChanged.emit(None)

    # -- File-handle coordination with the silx detach/reattach dance --

    def release_frame_source(self) -> None:
        """Close the FrameSource's h5py handle + clear its LRUs.

        Called by MainWindow._detach_silx_tree before any path that
        opens the same temp file ``r+`` (pipeline runs, ROI commit
        write-throughs, Add-to-fitted, clear-peaks, save-as).
        Idempotent — safe to call when no FrameSource is active.

        Also clears ``self._polar_cache`` so the cursor-readout
        handler's ``self._polar_cache is None`` guard fires while
        the FrameSource is closed. Without this drop, the cached
        ``_LazyPolarStack`` would still satisfy that None-check but
        its ``__getitem__`` would call into a released
        ``FrameSource`` and raise ``RuntimeError("FrameSource not
        acquired")`` every time the user moved the mouse across the
        polar plot during a pipeline run.
        """
        if self._frame_source is not None:
            self._frame_source.release()
        self._polar_cache = None

    def acquire_frame_source(self) -> None:
        """Reopen the FrameSource's h5py handle after a write completes.

        Pairs with ``release_frame_source`` via the silx reattach path.
        The polar cache is invalidated because per-frame polar arrays
        depend on the underlying Cartesian frames — and after a pipeline
        run those may have been overwritten. The polar grid axes
        (radius / angle) stay valid since they're functions of q_xy /
        q_z only, which don't change.

        Re-render runs with ``auto_range=False`` so the user's zoom
        survives the detach/reattach dance. This is the only call
        site for ``acquire_frame_source`` and it's always paired
        with ``release_frame_source`` for an op that the user
        triggered while looking at a specific area (pipeline ops,
        Add-to-fitted, clear-peaks…). Resetting the viewbox there
        is precisely the "zoom jumps to full extent after every
        commit" bug.
        """
        if self._frame_source is not None and not self._frame_source.is_open:
            self._frame_source.acquire()
            # Force a refresh of the polar cache wrapper so consumers
            # holding the previous tuple drop their reference.
            self._polar_cache = None
            # Re-render the current frame so the on-screen image
            # reflects any post-write changes to the underlying data.
            # _render_active_mode rebuilds the display params + LUT
            # but skips setImage's autoRange so the zoom survives.
            if self._stack is not None:
                self._render_active_mode(auto_range=False)

    # -- Manual peaks --

    def manual_peaks(self, frame: int) -> list[ManualPeak]:
        return list(self._manual_peaks.get(frame, []))

    def add_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        self._undoable_add_manual(frame, peak)
        self._push_undo(ManualAddAction(frame=frame, peak=peak))

    def remove_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        if peak not in self._manual_peaks.get(frame, []):
            return
        self._undoable_remove_manual(frame, peak)
        self._push_undo(ManualRemoveAction(frame=frame, peak=peak))

    def angular_extent(self) -> tuple[float, float] | None:
        """Return ``(angle_min_deg, angle_max_deg)`` for the active polar
        stack. Used by the host to size ring-mode expansions to the
        actual displayed angular range — converted files vary
        (``[0, 90]`` for the upper-right quadrant; ``[-180, 180]`` for
        full-quadrant data). Returns None when no polar stack is
        currently rendered (raw mode, no file open).
        """
        if self._polar_cache is None:
            return None
        _, _, angle = self._polar_cache
        if angle.size == 0:
            return None
        return float(angle[0]), float(angle[-1])

    def set_manual_geometry(
        self,
        peak: ManualPeak,
        radius: float,
        angle: float,
        radius_width: float,
        angle_width: float,
        is_ring: bool,
    ) -> None:
        """Mutate every geometry field on ``peak`` (including ``is_ring``)
        and trigger the standard refresh path. Skips the undo stack —
        this is for transient state changes driven by UI toggles
        (e.g. the ring checkbox), not user-initiated edits that should
        be reversible via Ctrl+Z. The host stashes its own pre-state
        when it needs to revert.

        Mirrors `_apply_manual_geom` but adds `is_ring` so we can flip
        the ring/segment kind in lockstep with the angular sweep.
        """
        peak.radius = radius
        peak.angle = angle
        peak.radius_width = radius_width
        peak.angle_width = angle_width
        peak.is_ring = is_ring
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self._selected.radius = radius
            self._selected.angle = angle
            self._selected.radius_width = radius_width
            self._selected.angle_width = angle_width
            self._selected.is_ring = is_ring
            self._sync_roi_geometry()
        # Find the frame this peak lives on so the overlay refreshes
        # against the right bucket.
        for fr, peaks in self._manual_peaks.items():
            if peak in peaks:
                if fr == self.current_frame:
                    self._render_overlays(fr)
                break
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self.peakGeometryChanged.emit(self._selected)

    def commit_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        """Drop a manual peak that has been persisted to the NeXus file.

        Like ``remove_manual_peak`` but does not push to the undo stack — the
        peak now lives in the detected/fitted overlay, so undoing back to its
        manual state would resurrect a duplicate. Any existing undo/redo
        entries referencing this peak are scrubbed for the same reason.
        """
        peaks = self._manual_peaks.get(frame, [])
        if peak in peaks:
            peaks.remove(peak)
        self._undo_stack = [a for a in self._undo_stack if not _action_targets_manual(a, peak)]
        self._redo_stack = [a for a in self._redo_stack if not _action_targets_manual(a, peak)]
        was_selected = (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        )
        if was_selected:
            self._selected = None
            self._sync_roi()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if was_selected:
            self.selectionChanged.emit(None)
        self.manualPeakRemoved.emit(frame, peak)

    def undo_last_action(self) -> None:
        """Reverse the most recent action. No-ops if empty."""
        if self._busy or not self._undo_stack:
            return
        action = self._undo_stack.pop()
        action.undo(self)
        self._redo_stack.append(action)

    def redo_last_action(self) -> None:
        """Re-apply the most recently undone action."""
        if self._busy or not self._redo_stack:
            return
        action = self._redo_stack.pop()
        action.redo(self)
        self._undo_stack.append(action)

    def clear_history(self) -> None:
        """Drop both undo and redo stacks. Called after pipeline ops that
        reshuffle peak ids — pending FileGeomActions would key off stale ids.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()

    def clear_selection(self) -> None:
        if self._selected is None:
            return
        self._selected = None
        self._fitted_preview_geom = None
        self._fitted_preview_is_ring = False
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(None)

    def clear_all_manual_peaks(self) -> None:
        """Drop every manual peak across all frames + the undo history.

        Matches Tools → Clear all manual peaks. Manual peaks are
        in-memory only, so no file write is involved. The selection is
        also cleared if it pointed at a manual peak.
        """
        if not self._manual_peaks:
            # Still clear undo history of any orphaned ManualGeomActions
            # and refresh in case overlays drift.
            self._undo_stack.clear()
            self._redo_stack.clear()
            return
        self._manual_peaks.clear()
        if self._selected is not None and self._selected.kind == "manual":
            self._selected = None
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
            self._sync_roi()
            self.selectionChanged.emit(None)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._render_overlays(self.current_frame)

    def set_fitted_preview(
        self,
        center_r: float | None,
        width_r: float | None,
        center_a: float | None,
        width_a: float | None,
        *,
        is_ring: bool = False,
    ) -> None:
        """Paint the dashed cyan preview of the would-be fitted_peaks box.

        Pure painter: the box is drawn at ``(width_r × width_a)``
        verbatim, centred at ``(center_r, center_a)``. All convention
        math (``2σ_r × 2σ_a`` for both modes; fallback to the user's
        drawn box when scipy's 1D fit hasn't converged) lives in the
        caller (``MainWindow._update_fitted_preview``) so a single
        place owns the rules.

        Pass any of ``center_r`` / ``width_r`` as ``None`` to hide.
        Pass ``is_ring=True`` to draw a full-angular-sweep ring at the
        canonical ``angle = 45°, angle_width = ∞`` — the caller still
        passes a sentinel for ``center_a`` / ``width_a`` (they're
        ignored).
        """
        if (
            center_r is None or width_r is None
            or not (np.isfinite(center_r) and np.isfinite(width_r) and width_r > 0)
        ):
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
        elif is_ring:
            # Ring path: angular fit isn't required (or even meaningful)
            # — store sentinel angular values that _render_overlays
            # rewrites to (45°, ∞) when it builds the preview row.
            self._fitted_preview_geom = (
                float(center_r), float(width_r), 45.0, 0.0,
            )
            self._fitted_preview_is_ring = True
        elif (
            center_a is None or width_a is None
            or not (np.isfinite(center_a) and np.isfinite(width_a) and width_a > 0)
        ):
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
        else:
            self._fitted_preview_geom = (
                float(center_r), float(width_r),
                float(center_a), float(width_a),
            )
            self._fitted_preview_is_ring = False
        self._render_overlays(self.current_frame)

    def set_busy(self, busy: bool) -> None:
        """Disable interactive editing while a pipeline run is in flight."""
        self._busy = busy
        self._sync_roi()

    @property
    def current_frame(self) -> int:
        return self._frame_index

    @property
    def n_frames(self) -> int:
        """Number of frames in the active stack (0 if no stack loaded)."""
        if self._mode == MODE_RAW and self._raw_image_stack is not None:
            return int(self._raw_image_stack.shape[0])
        return 0 if self._stack is None else int(self._stack.n_frames)

    def set_frame(self, frame: int) -> None:
        """Seek to ``frame``.

        The single source of truth for frame changes. Reads the new
        frame from the active source (FrameSource for NeXus, the
        in-memory raw stack for raw mode), pushes the 2D image to
        pyqtgraph via ``setImage`` (with the cached pos / scale /
        levels so the LUT and viewbox stay stable), updates the
        per-frame overlays, and emits ``frameChanged`` plus
        ``matchedStructuresChanged`` exactly once.

        No-op when ``frame`` is already current or out of range.
        """
        n = self.n_frames
        if n == 0:
            return
        idx = max(0, min(int(frame), n - 1))
        if idx == self._frame_index:
            return
        self._frame_index = idx
        self._render_frame(idx, auto_range=False)
        self._render_overlays(idx)
        self.frameChanged.emit(idx)
        # The Display dock rebuilds its matched-structure rows from this
        # signal — different frames can have different solutions.
        self.matchedStructuresChanged.emit(idx, self.matched_structures(idx))

    @property
    def selected_peak(self) -> SelectedPeak | None:
        return self._selected

    @property
    def is_dragging(self) -> bool:
        """True while an image-side ROI handle drag is in progress.

        Exposed so the host can decide whether per-tick
        ``peakGeometryChanged`` signals come from a live drag (and
        deserve debounced expensive recompute) or a settled
        programmatic update.
        """
        return self._roi_drag_before is not None

    # -- Action helpers (used by both public API and undo/redo) --

    def _push_undo(self, action: _Action) -> None:
        self._undo_stack.append(action)
        self._redo_stack.clear()

    def _undoable_add_manual(self, frame: int, peak: ManualPeak) -> None:
        """Insert a manual peak without touching the undo stack.

        Single-box policy invariant: whenever a manual peak is on
        screen it is also the active selection. This applies to every
        add path — the user-draw flow already auto-selected before;
        now undo of a remove (which restores the manual box) and redo
        of an add do too. Skipped when the peak is added on a non-
        current frame because the user can't see / interact with it.
        """
        bucket = self._manual_peaks.setdefault(frame, [])
        if peak not in bucket:
            bucket.append(peak)
        if frame == self.current_frame:
            self._render_overlays(frame)
        self.manualPeakAdded.emit(frame, peak)
        if frame == self.current_frame:
            self._set_selected(SelectedPeak.from_manual(peak, frame))

    def _undoable_remove_manual(self, frame: int, peak: ManualPeak) -> None:
        """Remove a manual peak without touching the undo stack."""
        bucket = self._manual_peaks.get(frame, [])
        if peak in bucket:
            bucket.remove(peak)
        was_selected = (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        )
        if was_selected:
            self._selected = None
            self._sync_roi()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if was_selected:
            self.selectionChanged.emit(None)
        self.manualPeakRemoved.emit(frame, peak)

    def _apply_manual_geom(
        self, frame: int, peak: ManualPeak,
        polar: tuple[float, float, float, float],
    ) -> None:
        r, a, dr, da = polar
        peak.radius = r
        peak.angle = a
        peak.radius_width = dr
        peak.angle_width = da
        # If this peak is the active selection, mirror it on the SelectedPeak
        # snapshot and refresh the ROI without retriggering its signals.
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self._selected.radius = r
            self._selected.angle = a
            self._selected.radius_width = dr
            self._selected.angle_width = da
            self._sync_roi_geometry()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self.peakGeometryChanged.emit(self._selected)

    def _apply_file_geom(
        self, frame: int, kind: str, peak_id: int,
        polar: tuple[float, float, float, float],
    ) -> None:
        r, a, dr, da = polar
        # Update the in-memory PeakTable so overlays paint the new box right
        # away; the disk write is fired separately via peakRowWriteRequested.
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        table = peaks_for_frame.get(kind)
        if table is not None and len(table) > 0:
            matches = np.where(table.ids == peak_id)[0]
            if matches.size > 0:
                idx = int(matches[0])
                table.radius[idx] = r
                table.angle[idx] = a
                table.radius_width[idx] = dr
                table.angle_width[idx] = da
                table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(r, a)
        # If the user edited a fitted peak, every matched solution that
        # references it must re-slice from the updated fitted table.
        if kind == "fitted":
            self._refresh_matched_for(frame)
        # Reflect the change on the SelectedPeak if it's the active selection.
        if (
            self._selected is not None
            and self._selected.kind in (kind, "matched")
            and self._selected.frame == frame
            and self._selected.peak_id == peak_id
        ):
            self._selected.radius = r
            self._selected.angle = a
            self._selected.radius_width = dr
            self._selected.angle_width = da
            self._sync_roi_geometry()
        if frame == self.current_frame:
            self._render_overlays(frame)
        # Fire the file-write so undo/redo also persist.
        self.peakRowWriteRequested.emit(
            frame, kind, int(peak_id),
            {"radius": r, "angle": a, "radius_width": dr, "angle_width": da},
        )
        if (
            self._selected is not None
            and self._selected.peak_id == peak_id
            and self._selected.frame == frame
        ):
            self.peakGeometryChanged.emit(self._selected)

    def _refresh_matched_for(self, frame: int) -> None:
        """Re-slice the frame's fitted PeakTable into each MatchedStructure
        using its cached ``peak_list`` indices. Cheap (numpy fancy index).
        """
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        fitted = peaks_for_frame.get("fitted")
        structures = self._matched_per_frame.get(frame, [])
        if fitted is None or not structures:
            return
        n_fit = len(fitted)
        for s in structures:
            idx = s.peak_list
            idx = idx[(idx >= 0) & (idx < n_fit)]
            s.peaks = PeakTable(
                q_xy=fitted.q_xy[idx],
                q_z=fitted.q_z[idx],
                angle=fitted.angle[idx],
                radius=fitted.radius[idx],
                angle_width=fitted.angle_width[idx],
                radius_width=fitted.radius_width[idx],
                is_ring=fitted.is_ring[idx],
                ids=fitted.ids[idx],
                score=fitted.score[idx],
                amplitude=fitted.amplitude[idx],
            )

    def polar_data(
        self,
    ) -> "tuple[_LazyPolarStack, np.ndarray, np.ndarray] | None":
        """Return ``(polar_stack, radius, angle)`` for the active entry.

        ``polar_stack`` is a ``_LazyPolarStack`` — supports
        ``polar_stack[i]`` (one polar frame) and
        ``polar_stack[frame, r, a]`` (single-pixel cursor lookup).
        Per-frame resampling happens on demand inside the
        ``FrameSource``; nothing is precomputed eagerly.

        Returns None if no stack is currently loaded or the
        FrameSource is currently released (e.g. mid-pipeline silx
        detach). Callers should re-call after the reattach.
        """
        if self._stack is None or self._frame_source is None:
            return None
        if not self._frame_source.is_open:
            return None
        if self._polar_cache is None:
            radius, angle = self._frame_source.polar_axes()
            self._polar_cache = (
                _LazyPolarStack(self._frame_source), radius, angle,
            )
        return self._polar_cache

    # -- Rendering --

    def _render_active_mode(self, *, auto_range: bool = True) -> None:
        """Build per-mode axis state + first-frame levels, then render.

        Called on stack swap (entry change, file open), mode toggle
        (Cartesian ↔ polar), and log/linear toggle. The actual per-
        frame pixel push happens in ``_render_frame`` which is also
        the path ``set_frame`` takes on every scrub.

        ``auto_range`` defaults to True so cold-open / mode-swap
        renders fit the new image into the viewbox. Callers that
        preserve the user's prior zoom (``show_stack(preserve_view=True)``,
        the log-scale toggle) pass ``False`` so ``setImage`` never
        clobbers the viewbox in the first place.
        """
        if self._mode == MODE_RAW:
            if self._raw_image_stack is None:
                return
            self._display_params = self._build_raw_params()
            self._render_frame(self._frame_index, auto_range=auto_range)
            # No overlays in raw mode — nothing to render past _render_frame.
            return
        if self._stack is None or self._frame_source is None:
            return
        if self._mode == MODE_POLAR:
            self._display_params = self._build_polar_params()
        else:
            self._display_params = self._build_cartesian_params()
        self._render_frame(self._frame_index, auto_range=auto_range)
        self._render_overlays(self.current_frame)

    def _build_raw_params(self) -> _DisplayParams:
        """Pixel-coordinate axis state for a raw detector stack.

        File order is (frames, H, W). For pyqtgraph we want each frame
        displayed as (W, H) — see ``_render_frame``'s raw branch for
        the per-frame transpose. Axes are labelled in pixels; q
        coordinates aren't meaningful before conversion.
        """
        assert self._raw_image_stack is not None
        # ``image_pg`` here is just the *first* frame, used for level
        # computation. Per-frame pushes happen in ``_render_frame``.
        first = np.asarray(self._raw_image_stack[0]).T  # (W, H)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(0.0, 0.0),
            scale=(1.0, 1.0),
            levels=levels,
            x_label=("x", "px"),
            y_label=("y", "px"),
        )

    def _build_cartesian_params(self) -> _DisplayParams:
        """Cartesian axis state. Reads frame 0 via the FrameSource to
        compute robust intensity levels; per-frame rendering is lazy.
        """
        assert self._stack is not None and self._frame_source is not None
        q_xy = self._frame_source.q_xy
        q_z = self._frame_source.q_z
        x0 = float(q_xy[0]); y0 = float(q_z[0])
        sx = float(q_xy[-1] - q_xy[0]) / max(len(q_xy) - 1, 1)
        sy = float(q_z[-1] - q_z[0]) / max(len(q_z) - 1, 1)
        first = self._frame_source.get_cartesian(0).T  # (n_qxy, n_qz)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(x0, y0),
            scale=(sx, sy),
            levels=levels,
            x_label=("q_xy", "Å⁻¹"),
            y_label=("q_z", "Å⁻¹"),
        )

    def _build_polar_params(self) -> _DisplayParams:
        """Polar axis state + lazy polar wrapper.

        The polar grid (``radius`` / ``angle``) is derived once from
        ``q_xy`` / ``q_z`` and reused across frames; per-frame polar
        resampling happens on demand in ``FrameSource.get_polar``.
        """
        assert self._stack is not None and self._frame_source is not None
        radius, angle = self._frame_source.polar_axes()
        # Cache a (lazy_polar_stack, radius, angle) tuple so consumers
        # (profile viewer, cursor readout) share the same FrameSource-
        # backed view without re-creating wrappers.
        if self._polar_cache is None:
            self._polar_cache = (
                _LazyPolarStack(self._frame_source), radius, angle,
            )
        x0 = float(radius[0]); y0 = float(angle[0])
        sx = float(radius[-1] - radius[0]) / max(len(radius) - 1, 1)
        sy = float(angle[-1] - angle[0]) / max(len(angle) - 1, 1)
        # polar_stack frame layout is (n_radius, n_angle); pyqtgraph
        # wants (x=radius, y=angle) per frame, which already matches.
        first = self._frame_source.get_polar(0)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(x0, y0),
            scale=(sx, sy),
            levels=levels,
            x_label=("radius", "Å⁻¹"),
            y_label=("angle", "deg"),
        )

    def _render_frame(self, idx: int, *, auto_range: bool) -> None:
        """Push the 2D frame at ``idx`` to pyqtgraph.

        Uses the cached ``_display_params`` (built once per stack/mode
        swap) so per-frame scrubs don't rebuild axes. ``auto_range`` is
        True only on the initial render after a stack/mode change;
        every subsequent frame change passes False to preserve the
        user's zoom/pan.
        """
        if self._display_params is None:
            return
        p = self._display_params
        # Fetch the actual 2D frame for ``idx``. Wrapped in
        # try/except because the FrameSource can be in a briefly-
        # released state during the silx detach/reattach dance —
        # if a play-tick or scrub fires in that window, get_polar
        # / get_cartesian raise ``RuntimeError("FrameSource not
        # acquired")``. We swallow that quietly; the reattach path
        # re-renders the active frame once acquire completes.
        try:
            if self._mode == MODE_RAW:
                if self._raw_image_stack is None:
                    return
                frame = np.asarray(self._raw_image_stack[idx]).T
            elif self._mode == MODE_POLAR:
                if self._frame_source is None or not self._frame_source.is_open:
                    return
                frame = self._frame_source.get_polar(idx)
            else:  # MODE_CARTESIAN
                if self._frame_source is None or not self._frame_source.is_open:
                    return
                frame = self._frame_source.get_cartesian(idx).T
        except (RuntimeError, ValueError, OSError, KeyError):
            return

        self._plot.setLabel("bottom", p.x_label[0], units=p.x_label[1])
        self._plot.setLabel("left", p.y_label[0], units=p.y_label[1])
        image, levels = self._maybe_apply_log(frame, p.levels)
        self._view.setImage(
            image,
            autoRange=auto_range,
            autoLevels=False,
            levels=levels,
            pos=p.pos,
            scale=p.scale,
        )
        # pyqtgraph's setImage internally calls roiClicked() which
        # force-shows the bottom timeline strip whenever the image has
        # a time axis. With single 2D frames there is no time axis so
        # the strip stays hidden naturally — no extra action needed
        # since the Timeline toggle was removed in the lazy-loading
        # milestone.
        self._hide_pyqtgraph_timeline()

    def _maybe_apply_log(
        self, image: np.ndarray, levels: tuple[float, float]
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """If log-scale is on, return (log10(clip(image, floor)), levels')
        with levels recomputed on the transformed first frame.

        Floor is the 1st percentile of strictly-positive finite values
        (or 1e-6 fallback) so the log transform is well-defined for
        zero / negative pixels (background, masked regions). The
        original ``image`` array is not modified.
        """
        if not self._log_scale:
            return image, levels
        finite = image[np.isfinite(image)]
        pos = finite[finite > 0]
        if pos.size > 0:
            floor = float(np.percentile(pos, 1.0))
        else:
            floor = 1e-6
        if floor <= 0:
            floor = 1e-6
        transformed = np.log10(np.clip(image, floor, None))
        ref = transformed[0] if transformed.ndim == 3 else transformed
        return transformed, _robust_levels(ref)

    def _on_log_toggled(self, checked: bool) -> None:
        """Re-render in the active mode with log/linear contrast.

        Saves and restores the viewbox range so toggling contrast
        doesn't reset the user's zoom or pan. Frame index is preserved
        too — pyqtgraph's setImage keeps the time-axis position when
        the stack shape is unchanged.
        """
        self._log_scale = bool(checked)
        saved: tuple[tuple[float, float], tuple[float, float]] | None = None
        try:
            xr, yr = self._plot.getViewBox().viewRange()
            saved = ((float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1])))
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer._on_log_toggled", exc_info=True)
            saved = None
        self._render_active_mode()
        if saved is not None:
            try:
                self._plot.getViewBox().setRange(
                    xRange=saved[0], yRange=saved[1], padding=0
                )
            except Exception:
                logger.debug("suppressed exception in GIWAXSImageViewer._on_log_toggled", exc_info=True)
                pass

    def _hide_pyqtgraph_timeline(self) -> None:
        """Keep pyqtgraph's bottom timeline strip hidden.

        pyqtgraph's ``setImage`` internally calls ``roiClicked()``
        which re-shows the strip whenever the image has a time axis.
        After the lazy-loading milestone we only ever pass 2D frames
        (no time axis), so the strip stays hidden naturally — but
        we re-apply hidden state explicitly to defend against future
        pyqtgraph versions that might force it back on. The splitter
        handle is set to 0-width in ``__init__`` so there's no
        residual line clipping the image's x-axis label.
        """
        ui = self._view.ui
        ui.roiPlot.setVisible(False)
        ui.splitter.setSizes([1, 0])

    # NB: _on_time_changed (the old pyqtgraph sigTimeChanged slot) is
    # removed. Frame changes flow through set_frame; emissions of
    # frameChanged + matchedStructuresChanged happen there. Left as a
    # comment so future code reviews see the deliberate removal.

    def _render_overlays(self, frame: int) -> None:
        # Raw mode has no peak data to draw — return before touching any
        # overlay path. The four _PeakShapeItems were already cleared by
        # show_raw_stack via clear(), so they have no leftover geometry.
        if self._mode == MODE_RAW:
            return
        peaks = self._frame_peaks.get(frame, {})
        det = peaks.get("detected")
        fit = peaks.get("fitted")

        # Apply the Display-dock min-score filter to the detected
        # overlay. Epsilon is half the slider's natural resolution
        # (0.01 per step → 0.005 tolerance) so a peak whose score
        # the slider was just seeded from is guaranteed to pass,
        # regardless of FP roundoff in the underlying value.
        if det is not None and self._detected_score_cutoff > 0.0 and len(det) > 0:
            try:
                cutoff = self._detected_score_cutoff - 0.005
                mask = np.asarray(det.score) >= cutoff
                if mask.sum() < len(det):
                    keep_ids = [int(det.ids[i]) for i in np.where(mask)[0]]
                    det = _peaks_subset(det, keep_ids)
            except Exception:
                # If the table doesn't have a usable score column
                # (older files), silently skip the filter rather
                # than break the overlay.
                logger.debug("suppressed exception in GIWAXSImageViewer._render_overlays", exc_info=True)
                pass

        manual_list = list(self._manual_peaks.get(frame, []))
        # Manual boxes are scratch labels — keep them invisible
        # unless they are the active selection. The ManualPeak
        # instances stay in ``_manual_peaks`` so things like the
        # post-Add-to-fitted Ctrl+Z (FittedRowAction.undo) can
        # reselect the source manual and bring its yellow ROI
        # back. Hit-testing in ``_on_select_at`` still operates on
        # ``_manual_peaks`` directly, so the user can click the
        # invisible region to re-select / reveal the peak.
        selected_is_manual = (
            self._selected is not None and self._selected.kind == "manual"
        )
        if not selected_is_manual:
            manual_list = []
        # When an ROI is active the selected peak is shown via the ROI handles —
        # exclude the manual peak from the manual-overlay path so it doesn't
        # render twice. (Detected/fitted overlays still draw the underlying
        # row; the white SELECTION_STYLE highlight is what's suppressed.)
        roi_active = self._roi_item is not None and self._selected is not None
        if roi_active and self._selected.kind == "manual":
            manual_list = [
                m for m in manual_list if m is not self._selected.manual_ref
            ]
        manual_table = _peaks_from_manual(manual_list)
        # Selection highlight: a one-row PeakTable from whatever's selected,
        # except suppressed when the ROI is doing the highlighting itself.
        # Matched-structure selections (multi_peak_ids set) expand to
        # every peak in the structure — pull the geometry from the frame's
        # fitted table so the boxes track ROI edits / re-fits.
        sel_table: PeakTable | None = None
        if self._selected is not None and not roi_active:
            if self._selected.multi_peak_ids and fit is not None:
                sel_table = _peaks_subset(fit, self._selected.multi_peak_ids)
                # Fallback to the representative peak if every id has
                # gone stale (e.g. user re-ran fitting since the
                # selection was set).
                if len(sel_table) == 0:
                    sel_table = _peaks_from_manual([
                        ManualPeak(
                            radius=self._selected.radius,
                            angle=self._selected.angle,
                            radius_width=self._selected.radius_width,
                            angle_width=self._selected.angle_width,
                            is_ring=self._selected.is_ring,
                            temp_id=self._selected.peak_id,
                        )
                    ])
            else:
                sel_table = _peaks_from_manual([
                    ManualPeak(
                        radius=self._selected.radius,
                        angle=self._selected.angle,
                        radius_width=self._selected.radius_width,
                        angle_width=self._selected.angle_width,
                        is_ring=self._selected.is_ring,
                        temp_id=self._selected.peak_id,
                    )
                ])

        # Fitted-preview overlay: only meaningful while a candidate peak
        # (manual or detected) is the active selection — both are subject to
        # an Add-to-fitted commit and the cyan box previews where the saved
        # row would land. Hide for fitted/matched selections (already on
        # file with their own box).
        preview_table: PeakTable | None = None
        if (
            self._fitted_preview_geom is not None
            and self._selected is not None
            and self._selected.kind in ("manual", "detected")
        ):
            cr, wr, ca, wa = self._fitted_preview_geom
            if self._fitted_preview_is_ring:
                # Ring preview: angular dimensions ignored, full sweep at
                # the canonical (45°, ∞) ring convention. ``wr`` is the
                # literal radial box width the caller wants drawn — see
                # ``set_fitted_preview`` (dumb painter, no convention
                # math here).
                preview_table = _peaks_from_manual([
                    ManualPeak(
                        radius=cr, angle=45.0,
                        radius_width=wr,
                        angle_width=float("inf"),
                        is_ring=True, temp_id=0,
                    )
                ])
            else:
                # Both ``wr`` and ``wa`` are the literal box widths the
                # caller wants drawn (``MainWindow._update_fitted_preview``
                # has already done the convention math: both modes use
                # ``2σ_r × 2σ_a``). Do NOT double / scale anything here
                # — that produced a preview twice as wide as the saved
                # row when the caller's contract changed under us.
                preview_table = _peaks_from_manual([
                    ManualPeak(
                        radius=cr, angle=ca,
                        radius_width=wr,
                        angle_width=wa,
                        is_ring=False, temp_id=0,
                    )
                ])

        # Live angular extent of the displayed polar stack so ring
        # overlays (and any segment whose stored angle_width spills
        # past the image bounds) clip to the data instead of the
        # global ±180° fallback.
        extent = self.angular_extent()
        if self._mode == MODE_POLAR:
            self._detected.set_polar(det, extent=extent)
            self._fitted.set_polar(fit, extent=extent)
            self._manual.set_polar(manual_table, extent=extent)
            if sel_table is not None:
                self._selection.set_polar(sel_table, extent=extent)
            else:
                self._selection.clear_path()
            if preview_table is not None:
                self._fitted_preview.set_polar(preview_table, extent=extent)
            else:
                self._fitted_preview.clear_path()
        else:
            self._detected.set_cartesian(det, extent=extent)
            self._fitted.set_cartesian(fit, extent=extent)
            self._manual.set_cartesian(manual_table, extent=extent)
            if sel_table is not None:
                self._selection.set_cartesian(sel_table, extent=extent)
            else:
                self._selection.clear_path()
            if preview_table is not None:
                self._fitted_preview.set_cartesian(preview_table, extent=extent)
            else:
                self._fitted_preview.clear_path()

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
        extent = self.angular_extent()
        vb = self._plot.getViewBox()
        for i, s in enumerate(structures):
            item = _PeakShapeItem(**matched_pen_for(i))
            if self._mode == MODE_POLAR:
                item.set_polar(s.peaks, extent=extent)
            else:
                item.set_cartesian(s.peaks, extent=extent)
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
        # The Cartesian / Polar radios are meaningless in RAW mode (raw
        # frames carry no q-axes), so swallow the toggle. Step 7 hides
        # the radios entirely in raw sessions; this guard is the
        # belt-and-braces backup if the radios are still reachable.
        if self._mode == MODE_RAW:
            return
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
                logger.debug("suppressed exception in GIWAXSImageViewer._apply_cmap", exc_info=True)
                cmap = None
            if cmap is not None:
                break
        if cmap is not None:
            self._view.setColorMap(cmap)

    # -- Cursor readout (status bar) --

    def _on_cursor_pos(self, pt: QPointF) -> None:
        info = self._compute_cursor_info(pt)
        self.cursorMoved.emit(info)

    def _on_cursor_left(self) -> None:
        self.cursorMoved.emit(None)

    def _compute_cursor_info(self, pt: QPointF) -> dict | None:
        """Translate a data-space cursor point into a status-bar payload.

        Returns one of three shapes, distinguished by the ``mode`` key:

        - ``"pixel"`` — raw mode: ``row, col, intensity``.
        - ``"cartesian"`` — q-cartesian view: ``q_xy, q_z, intensity``.
        - ``"polar"`` — q-polar view: ``r, theta, intensity``.

        Polar-view axes are **x = radius, y = angle** (matches what
        ``_polar_params`` puts on the plot, NOT the math convention).
        Intensity is looked up against the polar cache when the polar
        view is active; if that returns NaN (uncovered region of the
        polar transform), we fall back to the cartesian grid via the
        derived ``(q_xy, q_z)`` so users see something useful at the
        rim of the polar image.
        """
        x, y = pt.x(), pt.y()
        frame = self.current_frame
        if self._mode == MODE_RAW and self._raw_image_stack is not None:
            stack = self._raw_image_stack
            n_fr, n_rows, n_cols = stack.shape
            col = int(round(x))
            row = int(round(y))
            if (
                0 <= frame < n_fr
                and 0 <= row < n_rows
                and 0 <= col < n_cols
            ):
                intensity = float(stack[frame, row, col])
            else:
                intensity = float("nan")
            return {
                "mode": "pixel",
                "row": row,
                "col": col,
                "intensity": intensity,
            }
        if self._stack is None:
            return None
        if self._mode == MODE_CARTESIAN:
            q_xy_val = float(x)
            q_z_val = float(y)
            intensity = self._lookup_cartesian_intensity(
                frame, q_xy_val, q_z_val
            )
            return {
                "mode": "cartesian",
                "q_xy": q_xy_val,
                "q_z": q_z_val,
                "intensity": intensity,
            }
        # MODE_POLAR — viewer's polar image is laid out with radius on
        # the x axis and angle on the y axis (see _polar_params).
        r_val = float(x)
        theta_deg = float(y)
        intensity = float("nan")
        if self._polar_cache is not None:
            polar_stack, radius_axis, angle_axis = self._polar_cache
            if (
                0 <= frame < polar_stack.shape[0]
                and len(radius_axis) > 0
                and len(angle_axis) > 0
            ):
                r_idx = _bin_index(radius_axis, r_val)
                a_idx = _bin_index(angle_axis, theta_deg)
                intensity = float(polar_stack[frame, r_idx, a_idx])
        # Polar transform leaves NaN in uncovered regions; fall back
        # to the cartesian grid so the readout still shows a real
        # intensity near the edge.
        if intensity != intensity:  # NaN
            q_xy_val, q_z_val = polar_to_qxyz(r_val, theta_deg)
            intensity = self._lookup_cartesian_intensity(
                frame, q_xy_val, q_z_val
            )
        return {
            "mode": "polar",
            "r": r_val,
            "theta": theta_deg,
            "intensity": intensity,
        }

    def _lookup_cartesian_intensity(
        self, frame: int, q_xy_val: float, q_z_val: float
    ) -> float:
        """Pixel-bin intensity lookup against the cartesian stack.

        Uses floor-based binning (not nearest-neighbour) so the
        returned intensity stays constant while the cursor is inside
        the same displayed pixel — matches pyqtgraph's ``pos=axis[0]``
        / ``scale=step`` image transform exactly.

        Returns NaN while the FrameSource is released by the silx
        detach/reattach dance (pipeline runs, Add-to-fitted, clear-
        peaks, save-as). Without this guard, ``stack3d[...]`` delegates
        into ``FrameSource.get_cartesian`` which raises
        ``RuntimeError("FrameSource not acquired")`` every time the
        cursor moves over the cartesian plot during a write. Polar
        mode uses the ``self._polar_cache is None`` guard for the
        same reason; this is the cartesian-side mirror.
        """
        if self._stack is None:
            return float("nan")
        if self._frame_source is None or not self._frame_source.is_open:
            return float("nan")
        stack3d = self._stack.image_stack
        qxy_axis = self._stack.q_xy
        qz_axis = self._stack.q_z
        if not (
            0 <= frame < stack3d.shape[0]
            and len(qxy_axis) > 0
            and len(qz_axis) > 0
        ):
            return float("nan")
        qxy_idx = _bin_index(qxy_axis, q_xy_val)
        qz_idx = _bin_index(qz_axis, q_z_val)
        return float(stack3d[frame, qz_idx, qxy_idx])

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
        if self._mode != MODE_POLAR or self._busy:
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
        # Single-manual-box policy: any pre-existing manual peak on this
        # frame is replaced atomically. Modelled as one undo entry so
        # Ctrl+Z rewinds the whole swap rather than two staged steps.
        frame = self.current_frame
        existing = self._manual_peaks.get(frame, [])
        old_peak = existing[0] if existing else None
        if old_peak is not None:
            self._undoable_remove_manual(frame, old_peak)
        # _undoable_add_manual auto-selects on the current frame, so no
        # explicit selection call is needed here.
        self._undoable_add_manual(frame, peak)
        self._push_undo(
            ManualReplaceAction(frame=frame, old_peak=old_peak, new_peak=peak)
        )

    def _on_select_at(self, pos: QPointF) -> None:
        # Raw mode has no q-space overlays to hit-test. Polar and
        # Cartesian both run the same hit-test pipeline; the click
        # coordinates arrive in whatever space the viewbox is currently
        # showing (polar = (r, a), cartesian = (q_xy, q_z)), and the
        # mode-specific helpers below normalise to polar before
        # checking containment.
        if self._mode not in (MODE_POLAR, MODE_CARTESIAN) or self._busy:
            return
        x, y = float(pos.x()), float(pos.y())
        cart = self._mode == MODE_CARTESIAN
        frame = self.current_frame
        peaks_for_frame = self._frame_peaks.get(frame) or {}

        def hit_manual(peak: ManualPeak) -> bool:
            return _cart_box_contains(peak, x, y) if cart else _polar_box_contains(peak, x, y)

        def hit_table(tbl: PeakTable, i: int) -> bool:
            return (
                _cart_table_row_contains(tbl, i, x, y) if cart
                else _polar_table_row_contains(tbl, i, x, y)
            )

        # Priority order: manual > fitted > detected > matched. Matched is
        # last because it's a subset of fitted; the rare case where the user
        # wants the matched-context selection is still reachable by hiding
        # the fitted overlay.
        # 1) manual
        if self._visibility.get("manual", True):
            for peak in reversed(self._manual_peaks.get(frame, [])):
                if hit_manual(peak):
                    self._set_selected(SelectedPeak.from_manual(peak, frame))
                    return

        # 2) fitted, 3) detected — same hit-test against the PeakTable rows.
        for kind in ("fitted", "detected"):
            if not self._visibility.get(kind, True):
                continue
            table = peaks_for_frame.get(kind)
            if table is None or len(table) == 0:
                continue
            for i in reversed(range(len(table))):
                if hit_table(table, i):
                    self._set_selected(SelectedPeak(
                        kind=kind,
                        frame=frame,
                        peak_id=int(table.ids[i]),
                        radius=float(table.radius[i]),
                        angle=float(table.angle[i]),
                        radius_width=float(table.radius_width[i]),
                        angle_width=float(table.angle_width[i]),
                        is_ring=bool(table.is_ring[i]),
                        score=float(table.score[i]),
                        amplitude=float(table.amplitude[i]),
                    ))
                    return

        # 4) matched — only when the master toggle is on. The hit's peak_id
        # is the underlying fitted id (which is what delete_peak consumes).
        if self._matched_master_visible:
            structures = self._matched_per_frame.get(frame, [])
            for s_idx, s in reversed(list(enumerate(structures))):
                if not self._is_matched_item_visible(s.unique_id):
                    continue
                tbl = s.peaks
                color = matched_pen_for(s_idx)["color"]
                for i in reversed(range(len(tbl))):
                    if hit_table(tbl, i):
                        self._set_selected(SelectedPeak(
                            kind="matched",
                            frame=frame,
                            peak_id=int(tbl.ids[i]),
                            radius=float(tbl.radius[i]),
                            angle=float(tbl.angle[i]),
                            radius_width=float(tbl.radius_width[i]),
                            angle_width=float(tbl.angle_width[i]),
                            is_ring=bool(tbl.is_ring[i]),
                            structure_uid=s.unique_id,
                            structure_label=s.label,
                            structure_color=color,
                            score=float(tbl.score[i]),
                            amplitude=float(tbl.amplitude[i]),
                            # Clicking any peak of the structure
                            # promotes the whole structure into the
                            # selection — overlay highlights every
                            # peak in it, table syncs the structure
                            # row.
                            multi_peak_ids=[int(x) for x in tbl.ids],
                        ))
                        return

        # Click on empty space → deselect
        if self._selected is not None:
            self._set_selected(None)

    def _set_selected(
        self, sel: SelectedPeak | None, *, preserve_manual: bool = False,
    ) -> None:
        """Update the selection and sync the ROI + emit selectionChanged once.

        Side effect: if we're transitioning **away** from a
        manual-peak selection to anything else (a different peak, or
        nothing), drop the previous manual peak via
        ``remove_manual_peak``. This makes manual boxes truly
        transient — clicking off them abandons the draw. A
        ``ManualRemoveAction`` is pushed so ``Ctrl+Z`` brings the box
        back. The new-box draw path already removes the old peak via
        ``ManualReplaceAction`` *before* this method runs, so no
        double-remove happens there.

        Programmatic deselects from ``clear_selection`` bypass this
        method (they set ``self._selected = None`` directly), which
        preserves the manual peak across pipeline resets and other
        non-user-driven selection clears.

        ``preserve_manual``: when True, the manual-removal side
        effect is suppressed. Used by host flows that intentionally
        keep the manual peak around across a programmatic
        switch — currently ``MainWindow._on_add_to_fitted`` (so the
        manual source survives the auto-switch to the new fitted
        peak; without this, the silent ``ManualRemoveAction`` push
        would force the user to press Ctrl+Z twice to fully revert
        an Add-to-fitted commit) and the matching redo closure in
        ``_push_fitted_add_undo``.
        """
        if sel is None and self._selected is None:
            return
        if (
            sel is not None
            and self._selected is not None
            and sel.kind == self._selected.kind
            and sel.frame == self._selected.frame
            and sel.peak_id == self._selected.peak_id
            and sel.structure_uid == self._selected.structure_uid
        ):
            return
        prev = self._selected
        transitioning_away_from_manual = (
            not preserve_manual
            and prev is not None
            and prev.kind == "manual"
            and prev.manual_ref is not None
            and (sel is None or sel.manual_ref is not prev.manual_ref)
        )
        if transitioning_away_from_manual:
            bucket = self._manual_peaks.get(prev.frame, [])
            if prev.manual_ref in bucket:
                # ``remove_manual_peak`` clears ``self._selected`` to
                # None internally and emits ``selectionChanged(None)``
                # since the removed peak was the active selection. We
                # then re-apply the actual new ``sel`` below, which
                # emits ``selectionChanged(sel)`` a second time. The
                # transient None emit is harmless — listeners that
                # store state will end up with the right value.
                self.remove_manual_peak(prev.frame, prev.manual_ref)
        self._selected = sel
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(sel)

    def keyPressEvent(self, ev) -> None:  # type: ignore[override]
        if (
            ev.key() == Qt.Key.Key_Delete
            and self._selected is not None
            and not self._busy
        ):
            sel = self._selected
            if sel.kind == "manual" and sel.manual_ref is not None:
                self.remove_manual_peak(self.current_frame, sel.manual_ref)
            else:
                # File-resident peaks go through MainWindow → mlgidbase
                # delete_peak (cascading + with confirmation).
                self.deletePeakRequested.emit(sel)
            ev.accept()
            return
        # Esc on a selected manual peak removes it. Manual boxes are an
        # in-memory scratchpad — no file write, no confirmation.
        # File-resident selections fall through (Esc is meaningless for
        # them; Delete is the documented binding).
        if (
            ev.key() == Qt.Key.Key_Escape
            and self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is not None
            and not self._busy
        ):
            self.remove_manual_peak(self.current_frame, self._selected.manual_ref)
            ev.accept()
            return
        super().keyPressEvent(ev)

    # -- Resizable ROI on the selected peak --

    def _sync_roi(self) -> None:
        """Create / update / destroy the resize ROI to match the selection.

        Polar mode only, and only for editable kinds (manual / detected).
        Fitted and matched selections show the box but no ROI — fitted
        boxes encode the ``2σ`` convention so dragging their bounds
        would misrepresent the underlying Gaussian; matched is a
        derived view of fitted_peaks. Both edit through Add-to-fitted
        / delete instead.

        Handles ring peaks (``is_ring`` true or non-finite ``angle_width``)
        and peaks whose box edges fall outside the visible polar range:
        the ROI is clamped to the data axes so the handles are reachable,
        and ring peaks get only the radial (left/right) handles since their
        angular extent is the whole quadrant by definition.
        """
        self._teardown_roi()

        if (
            self._selected is None
            or self._mode != MODE_POLAR
            or self._busy
            or self._selected.kind in ("fitted", "matched")
        ):
            return

        # Need the polar axes to clamp against; bail if not yet computed.
        if self._polar_cache is None:
            return
        _, radius_axis, angle_axis = self._polar_cache
        if radius_axis.size == 0 or angle_axis.size == 0:
            return
        r_min, r_max = float(radius_axis[0]), float(radius_axis[-1])
        a_min, a_max = float(angle_axis[0]), float(angle_axis[-1])

        sel = self._selected
        is_ring_box = sel.is_ring or not np.isfinite(sel.angle_width)

        if is_ring_box:
            a_lo, a_hi = a_min, a_max
        else:
            a_lo = max(sel.angle - sel.angle_width / 2.0, a_min)
            a_hi = min(sel.angle + sel.angle_width / 2.0, a_max)
            if a_hi <= a_lo:
                return  # peak entirely outside visible angular range
        r_lo = max(sel.radius - sel.radius_width / 2.0, r_min)
        r_hi = min(sel.radius + sel.radius_width / 2.0, r_max)
        if r_hi <= r_lo:
            return

        pos = (r_lo, a_lo)
        size = (r_hi - r_lo, a_hi - a_lo)

        # ROI pen colored by the source overlay's hue so the user keeps a
        # visual link to which list the peak came from.
        roi_color = OVERLAY_STYLE.get(sel.kind, OVERLAY_STYLE["manual"])["color"]
        pen = pg.mkPen(QColor(roi_color), width=2.0)
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        hover_pen = pg.mkPen(
            QColor(SELECTION_STYLE["color"]), width=SELECTION_STYLE["width"]
        )
        hover_pen.setCosmetic(True)

        roi = pg.ROI(pos=pos, size=size, pen=pen, hoverPen=hover_pen, movable=True)
        # Edge-only handles (no corners): each handle drags one edge while the
        # opposite edge stays anchored. Rings have only radial handles — the
        # angular bounds are the whole quadrant by construction.
        roi.addScaleHandle([1.0, 0.5], [0.0, 0.5])  # right
        roi.addScaleHandle([0.0, 0.5], [1.0, 0.5])  # left
        if not is_ring_box:
            roi.addScaleHandle([0.5, 1.0], [0.5, 0.0])  # top
            roi.addScaleHandle([0.5, 0.0], [0.5, 1.0])  # bottom
        roi.setZValue(60)
        # Track which dimensions are user-editable for _on_roi_changed.
        roi._mlgid_ring_box = is_ring_box  # type: ignore[attr-defined]
        roi.sigRegionChangeStarted.connect(self._on_roi_drag_started)
        roi.sigRegionChanged.connect(self._on_roi_changed)
        roi.sigRegionChangeFinished.connect(self._on_roi_drag_finished)

        self._plot.getViewBox().addItem(roi, ignoreBounds=True)
        self._roi_item = roi

    def _teardown_roi(self) -> None:
        if self._roi_item is None:
            return
        roi = self._roi_item
        for sig_name in (
            "sigRegionChangeStarted", "sigRegionChanged", "sigRegionChangeFinished",
        ):
            try:
                getattr(roi, sig_name).disconnect()
            except (RuntimeError, TypeError):
                pass
        self._plot.getViewBox().removeItem(roi)
        self._roi_item = None

    def _sync_roi_geometry(self) -> None:
        """Adjust the existing ROI to match the SelectedPeak without rebuilding.

        Used by undo/redo and external geometry updates — blocks signals so we
        don't recursively re-enter ``_on_roi_changed``.
        """
        if self._roi_item is None or self._selected is None:
            return
        roi = self._roi_item
        roi.blockSignals(True)
        try:
            roi.setPos(
                [
                    self._selected.radius - self._selected.radius_width / 2.0,
                    self._selected.angle - self._selected.angle_width / 2.0,
                ],
                update=False,
            )
            roi.setSize([self._selected.radius_width, self._selected.angle_width])
        finally:
            roi.blockSignals(False)

    def _on_roi_drag_started(self) -> None:
        if self._selected is None:
            return
        self._roi_drag_before = self._selected.polar_tuple()

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
        new_r = x0 + w / 2.0
        new_a = y0 + h / 2.0

        sel = self._selected
        # Ring peaks have only radial handles — the angular extent is fixed
        # at the whole visible quadrant (and the underlying angle_width is
        # often inf). Don't propagate the ROI's angular pos/size into the
        # peak's geometry, or we'd corrupt the ring on every drag.
        is_ring_box = bool(getattr(roi, "_mlgid_ring_box", False))

        sel.radius = new_r
        sel.radius_width = w
        if not is_ring_box:
            sel.angle = new_a
            sel.angle_width = h

        if sel.kind == "manual" and sel.manual_ref is not None:
            sel.manual_ref.radius = new_r
            sel.manual_ref.radius_width = w
            if not is_ring_box:
                sel.manual_ref.angle = new_a
                sel.manual_ref.angle_width = h
        else:
            # Mutate the in-memory PeakTable so the colored detected/fitted
            # outline tracks the drag live. Disk + matched-resync happen on
            # drag-end via _on_roi_drag_finished. For ring boxes we don't
            # touch angle / angle_width since those handles aren't shown.
            peaks_for_frame = self._frame_peaks.get(sel.frame) or {}
            table = peaks_for_frame.get(sel.kind)
            if table is not None and len(table) > 0:
                matches = np.where(table.ids == sel.peak_id)[0]
                if matches.size > 0:
                    idx = int(matches[0])
                    table.radius[idx] = new_r
                    table.radius_width[idx] = w
                    if not is_ring_box:
                        table.angle[idx] = new_a
                        table.angle_width[idx] = h
                    cur_a = float(table.angle[idx])
                    table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(new_r, cur_a)

        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(sel)

    def _on_roi_drag_finished(self) -> None:
        if self._selected is None or self._roi_drag_before is None:
            return
        before = self._roi_drag_before
        after = self._selected.polar_tuple()
        self._roi_drag_before = None
        if before == after:
            return  # idle release — nothing to record

        sel = self._selected
        if sel.kind == "manual" and sel.manual_ref is not None:
            self._push_undo(ManualGeomAction(
                frame=sel.frame, peak=sel.manual_ref,
                before=before, after=after,
            ))
        elif sel.kind in ("detected", "fitted"):
            self._push_undo(FileGeomAction(
                frame=sel.frame, kind=sel.kind, peak_id=sel.peak_id,
                before=before, after=after,
            ))
            # Re-derive matched overlays if a fitted edit changed an
            # underlying row used by any matched solution.
            if sel.kind == "fitted":
                self._refresh_matched_for(sel.frame)
                self._render_overlays(self.current_frame)
            # Persist to the file via MainWindow (see peakRowWriteRequested).
            self.peakRowWriteRequested.emit(
                sel.frame, sel.kind, int(sel.peak_id),
                {"radius": after[0], "angle": after[1],
                 "radius_width": after[2], "angle_width": after[3]},
            )
        # Drag-end notification (all kinds) — subscribers that want the
        # final geometry without per-tick refits hook here.
        self.peakGeometryDragFinished.emit(sel)

    def update_peak_geometry_external(self, peak: ManualPeak) -> None:
        """Sync the ROI to a peak whose geometry was changed elsewhere
        (e.g. by dragging a profile region). Suppresses ROI signals so this
        doesn't loop back into ``_on_roi_changed``.
        """
        if (
            self._selected is None
            or self._selected.kind != "manual"
            or self._selected.manual_ref is not peak
            or self._roi_item is None
        ):
            return
        # Mirror the new geometry onto the SelectedPeak snapshot.
        self._selected.radius = peak.radius
        self._selected.angle = peak.angle
        self._selected.radius_width = peak.radius_width
        self._selected.angle_width = peak.angle_width
        self._sync_roi_geometry()
        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(self._selected)

    def update_detected_geometry_external(self, sel: "SelectedPeak") -> None:
        """Sync the detected-peak overlay to a SelectedPeak whose geometry
        was changed elsewhere — currently the profile region drag (see
        ``profile_viewer.detectedPeakGeometryChanged``).

        Mirrors the in-memory mutation that ``_on_roi_changed`` does
        for detected/fitted peaks during an image-side ROI drag:
        updates the cached ``_frame_peaks`` PeakTable so the colored
        overlay tracks the drag live; recomputes the row's Cartesian
        ``q_xy`` / ``q_z`` from the new polar coordinates; re-renders
        and re-syncs the image-space ROI. Disk persistence happens
        separately on drag-end via the host's
        ``_on_peak_row_write_requested`` slot.
        """
        if (
            self._selected is None
            or self._selected.kind != "detected"
            or sel is not self._selected
            or self._roi_item is None
        ):
            return
        peaks_for_frame = self._frame_peaks.get(sel.frame) or {}
        table = peaks_for_frame.get("detected")
        if table is None or len(table) == 0:
            return
        matches = np.where(table.ids == sel.peak_id)[0]
        if matches.size == 0:
            return
        idx = int(matches[0])
        table.radius[idx] = sel.radius
        table.radius_width[idx] = sel.radius_width
        table.angle[idx] = sel.angle
        table.angle_width[idx] = sel.angle_width
        table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(sel.radius, sel.angle)
        self._sync_roi_geometry()
        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(self._selected)
