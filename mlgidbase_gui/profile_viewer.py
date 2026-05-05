from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QWidget

from mlgidbase_gui.fit import fit_gaussian_on_axis
from mlgidbase_gui.image_viewer import OVERLAY_STYLE, ManualPeak

PROFILE_PEN_COLOR = "#e8e8e8"
PROFILE_PEN_WIDTH = 1.2

REGION_COLOR = OVERLAY_STYLE["manual"]["color"]
REGION_BRUSH_ALPHA = 30  # 0-255

# Distinct from the white data curve and the yellow region markers.
FIT_PEN_COLOR = "#ff7eb6"
FIT_PEN_WIDTH = 1.6

# Multiple of the box width added on each side when auto-zooming to a peak.
# View window = (1 + 2 * ZOOM_PAD_FACTOR) * box_width.
ZOOM_PAD_FACTOR = 1.0


class ProfileViewer(QWidget):
    """Two stacked plots showing the radial and angular intensity profiles
    of the polar image.

    Radial profile is mean intensity across all angles, indexed by radius.
    Angular profile is mean intensity across all radii, indexed by angle.
    When a manual peak is selected, draggable ``LinearRegionItem``s appear in
    each profile representing the peak's bounds. Dragging a region mutates the
    peak in place and emits ``peakGeometryChanged`` so the 2D ROI can sync.
    """

    peakGeometryChanged = Signal(object)  # ManualPeak whose geometry changed

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        pen = pg.mkPen(PROFILE_PEN_COLOR, width=PROFILE_PEN_WIDTH)
        region_pen = pg.mkPen(REGION_COLOR, width=1.5)
        region_pen.setCosmetic(True)
        region_brush = pg.mkBrush(QColor(*_hex_to_rgb(REGION_COLOR), REGION_BRUSH_ALPHA))
        fit_pen = pg.mkPen(FIT_PEN_COLOR, width=FIT_PEN_WIDTH)
        fit_pen.setCosmetic(True)

        self._radial_plot = pg.PlotWidget()
        self._radial_plot.setLabel("bottom", "radius", units="Å⁻¹")
        self._radial_plot.setLabel("left", "intensity")
        self._radial_plot.setTitle("Radial profile")
        self._radial_plot.showGrid(x=True, y=True, alpha=0.2)
        self._radial_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self._radial_curve = self._radial_plot.plot([], [], pen=pen)
        self._radial_fit_curve = self._radial_plot.plot([], [], pen=fit_pen)
        self._radial_region = pg.LinearRegionItem(
            values=(0.0, 0.0), brush=region_brush, pen=region_pen
        )
        self._radial_region.setZValue(50)
        self._radial_region.setVisible(False)
        self._radial_region.sigRegionChanged.connect(self._on_radial_changed)
        self._radial_plot.addItem(self._radial_region)
        layout.addWidget(self._radial_plot)

        self._angular_plot = pg.PlotWidget()
        self._angular_plot.setLabel("bottom", "angle", units="deg")
        self._angular_plot.setLabel("left", "intensity")
        self._angular_plot.setTitle("Angular profile")
        self._angular_plot.showGrid(x=True, y=True, alpha=0.2)
        self._angular_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self._angular_curve = self._angular_plot.plot([], [], pen=pen)
        self._angular_fit_curve = self._angular_plot.plot([], [], pen=fit_pen)
        self._angular_region = pg.LinearRegionItem(
            values=(0.0, 0.0), brush=region_brush, pen=region_pen
        )
        self._angular_region.setZValue(50)
        self._angular_region.setVisible(False)
        self._angular_region.sigRegionChanged.connect(self._on_angular_changed)
        self._angular_plot.addItem(self._angular_region)
        layout.addWidget(self._angular_plot)

        self._polar_stack: np.ndarray | None = None
        self._radius: np.ndarray | None = None
        self._angle: np.ndarray | None = None
        self._selected: ManualPeak | None = None
        self._current_frame = 0

    # -- Public API --

    def set_polar_stack(
        self, polar_stack: np.ndarray, radius: np.ndarray, angle: np.ndarray
    ) -> None:
        """Load the polar-stack data; renders frame 0 immediately."""
        self._polar_stack = polar_stack
        self._radius = radius
        self._angle = angle
        self._current_frame = 0
        self._recompute_curves()

    def set_frame(self, frame: int) -> None:
        if self._polar_stack is None:
            return
        if not 0 <= frame < self._polar_stack.shape[0]:
            return
        self._current_frame = frame
        self._recompute_curves()

    def clear(self) -> None:
        self._radial_curve.setData([], [])
        self._angular_curve.setData([], [])
        self._radial_fit_curve.setData([], [])
        self._angular_fit_curve.setData([], [])
        self._polar_stack = None
        self._radius = None
        self._angle = None
        self.set_selected_peak(None)

    # -- Selected-peak edge handles --

    def set_selected_peak(self, peak: ManualPeak | None) -> None:
        """Show / hide / sync the edge regions for the given peak.

        Profiles re-integrate over the *complementary* axis of the box (radial
        profile uses the box's angular range, angular profile uses the radial
        range). Deselecting restores full integration over the entire image.
        Also auto-zooms each profile to a window slightly wider than the box.
        """
        self._selected = peak
        visible = peak is not None
        self._radial_region.setVisible(visible)
        self._angular_region.setVisible(visible)
        if peak is not None:
            self.sync_regions_from_peak(peak)
            self._zoom_to_peak(peak)
        else:
            self._radial_plot.enableAutoRange()
            self._angular_plot.enableAutoRange()
        self._recompute_curves()

    def sync_regions_from_peak(self, peak: ManualPeak) -> None:
        """Programmatically update region bounds without re-emitting changes."""
        if peak is not self._selected:
            return
        r_lo = peak.radius - peak.radius_width / 2.0
        r_hi = peak.radius + peak.radius_width / 2.0
        a_lo = peak.angle - peak.angle_width / 2.0
        a_hi = peak.angle + peak.angle_width / 2.0
        for region, lo, hi in (
            (self._radial_region, r_lo, r_hi),
            (self._angular_region, a_lo, a_hi),
        ):
            region.blockSignals(True)
            try:
                region.setRegion((float(lo), float(hi)))
            finally:
                region.blockSignals(False)
        # The integration window changed → recompute the slice profiles.
        self._recompute_curves()

    # -- Internals --

    def _recompute_curves(self) -> None:
        """Re-render both profile curves and any fit overlays.

        Selected: radial profile averages columns within the box's angular
        range; angular profile averages rows within the box's radial range.
        Fit a 1D Gaussian + linear background near the box centre on each
        axis and render that on top so users can drag the box edges to match.
        Unselected: full-image averages on both axes, fit curves cleared.
        """
        if (
            self._polar_stack is None
            or self._radius is None
            or self._angle is None
        ):
            return
        if not 0 <= self._current_frame < self._polar_stack.shape[0]:
            return
        img = self._polar_stack[self._current_frame]

        if self._selected is not None:
            peak = self._selected
            a_slice = _bounds_to_slice(
                self._angle,
                peak.angle - peak.angle_width / 2.0,
                peak.angle + peak.angle_width / 2.0,
            )
            r_slice = _bounds_to_slice(
                self._radius,
                peak.radius - peak.radius_width / 2.0,
                peak.radius + peak.radius_width / 2.0,
            )
            radial_src = img[:, a_slice] if a_slice.stop > a_slice.start else img
            angular_src = img[r_slice, :] if r_slice.stop > r_slice.start else img
            radial = np.nanmean(radial_src, axis=1)
            angular = np.nanmean(angular_src, axis=0)
        else:
            radial = np.nanmean(img, axis=1)
            angular = np.nanmean(img, axis=0)

        self._radial_curve.setData(self._radius, radial)
        self._angular_curve.setData(self._angle, angular)

        if self._selected is not None:
            self._update_fit_curves(self._selected, radial, angular)
        else:
            self._radial_fit_curve.setData([], [])
            self._angular_fit_curve.setData([], [])

    def _update_fit_curves(
        self, peak: ManualPeak, radial: np.ndarray, angular: np.ndarray
    ) -> None:
        if self._radius is not None:
            rfit = fit_gaussian_on_axis(
                self._radius, radial, peak.radius, peak.radius_width
            )
            if rfit is not None:
                self._radial_fit_curve.setData(rfit.x, rfit.y)
            else:
                self._radial_fit_curve.setData([], [])
        if self._angle is not None:
            afit = fit_gaussian_on_axis(
                self._angle, angular, peak.angle, peak.angle_width
            )
            if afit is not None:
                self._angular_fit_curve.setData(afit.x, afit.y)
            else:
                self._angular_fit_curve.setData([], [])

    def _zoom_to_peak(self, peak: ManualPeak) -> None:
        """Set each profile's X range to a window slightly wider than the box,
        and let Y auto-scale to whatever data is currently visible in X.
        """
        for plot, center, width in (
            (self._radial_plot, peak.radius, peak.radius_width),
            (self._angular_plot, peak.angle, peak.angle_width),
        ):
            if not (np.isfinite(center) and np.isfinite(width)) or width <= 0:
                plot.enableAutoRange()
                continue
            half = width / 2.0 + ZOOM_PAD_FACTOR * width
            plot.setXRange(center - half, center + half, padding=0)
            vb = plot.getViewBox()
            vb.setAutoVisible(y=True)
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)

    def _on_radial_changed(self) -> None:
        if self._selected is None:
            return
        lo, hi = self._radial_region.getRegion()
        self._selected.radius_width = abs(float(hi) - float(lo))
        self._selected.radius = (float(hi) + float(lo)) / 2.0
        # Angular profile slices over the radial range — needs refresh.
        self._recompute_curves()
        self.peakGeometryChanged.emit(self._selected)

    def _on_angular_changed(self) -> None:
        if self._selected is None:
            return
        lo, hi = self._angular_region.getRegion()
        self._selected.angle_width = abs(float(hi) - float(lo))
        self._selected.angle = (float(hi) + float(lo)) / 2.0
        # Radial profile slices over the angular range — needs refresh.
        self._recompute_curves()
        self.peakGeometryChanged.emit(self._selected)


def _hex_to_rgb(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _bounds_to_slice(axis: np.ndarray, lo: float, hi: float) -> slice:
    """Convert a (lo, hi) range in physical units to an index slice into a
    monotonic 1D axis. Tolerates non-finite bounds (treated as full range).
    """
    if not np.isfinite(lo):
        lo_idx = 0
    else:
        lo_idx = int(np.searchsorted(axis, lo, side="left"))
    if not np.isfinite(hi):
        hi_idx = len(axis)
    else:
        hi_idx = int(np.searchsorted(axis, hi, side="right"))
    lo_idx = max(0, min(lo_idx, len(axis)))
    hi_idx = max(lo_idx, min(hi_idx, len(axis)))
    return slice(lo_idx, hi_idx)
