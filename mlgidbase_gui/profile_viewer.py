from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QWidget

from mlgidbase_gui.fit import GaussianFit, fit_gaussian_on_axis, gaussian_from_box
from mlgidbase_gui.image_viewer import OVERLAY_STYLE, ManualPeak, SelectedPeak

# Multiple of box width on each side over which the rendered Gaussian curve
# extends past the box bounds. Big enough that the tails fade into the
# baseline; small enough that very tight peaks don't render as a spike on a
# nearly empty axis.
FIT_RENDER_PAD_FACTOR = 3.0

PROFILE_PEN_COLOR = "#e8e8e8"
PROFILE_PEN_WIDTH = 1.2

REGION_COLOR = OVERLAY_STYLE["manual"]["color"]
REGION_BRUSH_ALPHA = 30  # 0-255

# Distinct from the white data curve and the yellow region markers.
FIT_PEN_COLOR = "#ff7eb6"
FIT_PEN_WIDTH = 1.6

# Multiple of the box width added on each side when auto-zooming to a peak.
# View window = (1 + 2 * ZOOM_PAD_FACTOR) * box_width. Matched to
# FIT_RENDER_PAD_FACTOR below so the rendered Gaussian's tails are visible
# without manual panning.
ZOOM_PAD_FACTOR = 3.0


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
    # Emitted whenever the cached fit pair changes (computed, cleared, or both
    # axes failed). Carries (radial_fit, angular_fit) — either may be None.
    fitParamsChanged = Signal(object, object)

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
        # SelectedPeak (any kind) so profiles/fits can render for detected /
        # fitted / matched / manual alike. Region drags only mutate when the
        # kind is "manual" — file-resident peaks edit through the 2D ROI.
        self._selected: SelectedPeak | None = None
        self._current_frame = 0
        # Last successfully-computed fits — exposed via last_fit_params() so
        # the parameter panel's "Add to fitted" button can write them to the
        # fitted_peaks dataset. Cleared whenever the curves are cleared.
        self._last_radial_fit: GaussianFit | None = None
        self._last_angular_fit: GaussianFit | None = None

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
        self._set_fit_cache(None, None)
        self.set_selected_peak(None)

    def last_fit_params(self) -> dict[str, GaussianFit | None]:
        """Most recent radial / angular Gaussian fits, or None if not fitted.

        Used by MainWindow's "Add to fitted" handler to read amplitude /
        center / sigma without re-running the fit. Stays valid until the
        next ``_recompute_curves`` call (frame change, deselect, etc.).
        """
        return {
            "radial": self._last_radial_fit,
            "angular": self._last_angular_fit,
        }

    # -- Selected-peak edge handles --

    def set_selected_peak(self, peak: SelectedPeak | None) -> None:
        """Show / hide / sync the edge regions for the given selection.

        Profiles re-integrate over the *complementary* axis of the box (radial
        profile uses the box's angular range, angular profile uses the radial
        range). Deselecting restores full integration over the entire image.
        Also auto-zooms each profile to a window slightly wider than the box.

        Region drags edit the underlying peak only for ``kind == "manual"``;
        for other kinds the regions are visible but read-only because file
        resident peaks edit through the 2D ROI (which has its own undo /
        write hooks).
        """
        self._selected = peak
        visible = peak is not None
        is_manual = peak is not None and peak.kind == "manual"
        # For ring peaks (or any peak whose angle_width is non-finite) the
        # angular region would span the entire axis or fail to render — hide
        # it so the profile stays usable. The radial region still tracks.
        is_ring_box = (
            peak is not None
            and (peak.is_ring or not np.isfinite(peak.angle_width))
        )

        self._radial_region.setVisible(visible)
        self._radial_region.setMovable(is_manual)
        self._angular_region.setVisible(visible and not is_ring_box)
        self._angular_region.setMovable(is_manual)

        if peak is not None:
            self.sync_regions_from_peak(peak)
            self._zoom_to_peak(peak)
        else:
            self._radial_plot.enableAutoRange()
            self._angular_plot.enableAutoRange()
        self._recompute_curves()

    def sync_regions_from_peak(self, peak: SelectedPeak | ManualPeak) -> None:
        """Programmatically update region bounds without re-emitting changes.

        Accepts ``SelectedPeak`` (the new wire) or ``ManualPeak`` (legacy
        callers). Returns silently when the argument doesn't match the
        currently-shown selection.
        """
        if isinstance(peak, ManualPeak):
            if (
                self._selected is None
                or self._selected.kind != "manual"
                or self._selected.manual_ref is not peak
            ):
                return
            r, a = peak.radius, peak.angle
            dr, da = peak.radius_width, peak.angle_width
            is_ring_box = peak.is_ring or not np.isfinite(da)
        else:
            if peak is not self._selected:
                return
            r, a = peak.radius, peak.angle
            dr, da = peak.radius_width, peak.angle_width
            is_ring_box = peak.is_ring or not np.isfinite(da)
        r_lo = r - dr / 2.0
        r_hi = r + dr / 2.0
        self._radial_region.blockSignals(True)
        try:
            self._radial_region.setRegion((float(r_lo), float(r_hi)))
        finally:
            self._radial_region.blockSignals(False)
        # Skip the angular region for ring boxes — its bounds would be
        # ±inf / span the entire axis. setVisible already hid it.
        if not is_ring_box:
            a_lo = a - da / 2.0
            a_hi = a + da / 2.0
            self._angular_region.blockSignals(True)
            try:
                self._angular_region.setRegion((float(a_lo), float(a_hi)))
            finally:
                self._angular_region.blockSignals(False)
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
            self._set_fit_cache(None, None)

    def _update_fit_curves(
        self, peak: SelectedPeak, radial: np.ndarray, angular: np.ndarray
    ) -> None:
        # Manual + detected peaks are candidate boxes — refit the data
        # inside the box on every drag so the user can adjust the bounds
        # against the live curve, and so "Add to fitted" picks up an
        # actual FWHM (detected boxes don't carry the FWHM convention).
        # Fitted + matched peaks are stored under the convention
        #   radial border = FWHM   →   FWHM_r = radius_width
        #   azimuthal border = 2 × FWHM   →   FWHM_a = angle_width / 2
        # so we draw the Gaussian implied by that convention exactly. This
        # keeps the displayed curve in sync with the saved box parameters
        # — no refit drift on re-select.
        do_real_fit = peak.kind in ("manual", "detected")
        rfit: GaussianFit | None = None
        afit: GaussianFit | None = None
        if self._radius is not None:
            r_lo = peak.radius - peak.radius_width / 2.0
            r_hi = peak.radius + peak.radius_width / 2.0
            r_pad = FIT_RENDER_PAD_FACTOR * peak.radius_width
            render_r = (r_lo - r_pad, r_hi + r_pad)
            if do_real_fit:
                rfit = fit_gaussian_on_axis(
                    self._radius, radial, peak.radius, peak.radius_width,
                    fit_range=(r_lo, r_hi),
                    render_range=render_r,
                )
            else:
                # Stored convention: radial box width == FWHM.
                rfit = gaussian_from_box(
                    self._radius, radial, peak.radius, peak.radius_width,
                    render_range=render_r,
                )
            if rfit is not None:
                self._radial_fit_curve.setData(rfit.x, rfit.y)
            else:
                self._radial_fit_curve.setData([], [])
        if self._angle is not None:
            if np.isfinite(peak.angle_width) and peak.angle_width > 0:
                a_lo = peak.angle - peak.angle_width / 2.0
                a_hi = peak.angle + peak.angle_width / 2.0
                a_pad = FIT_RENDER_PAD_FACTOR * peak.angle_width
                render_a = (a_lo - a_pad, a_hi + a_pad)
                if do_real_fit:
                    afit = fit_gaussian_on_axis(
                        self._angle, angular, peak.angle, peak.angle_width,
                        fit_range=(a_lo, a_hi),
                        render_range=render_a,
                    )
                else:
                    # Stored convention: azimuthal box width == 2 × FWHM.
                    afit = gaussian_from_box(
                        self._angle, angular, peak.angle,
                        peak.angle_width / 2.0,
                        render_range=render_a,
                    )
            if afit is not None:
                self._angular_fit_curve.setData(afit.x, afit.y)
            else:
                self._angular_fit_curve.setData([], [])
        self._set_fit_cache(rfit, afit)

    def _set_fit_cache(
        self, rfit: GaussianFit | None, afit: GaussianFit | None,
    ) -> None:
        if rfit is self._last_radial_fit and afit is self._last_angular_fit:
            return
        self._last_radial_fit = rfit
        self._last_angular_fit = afit
        self.fitParamsChanged.emit(rfit, afit)

    def _zoom_to_peak(self, peak: SelectedPeak) -> None:
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
        # Region drags only mutate manual peaks. For non-manual selections
        # the region is set non-movable so this slot should never fire, but
        # guard defensively.
        if (
            self._selected is None
            or self._selected.kind != "manual"
            or self._selected.manual_ref is None
        ):
            return
        lo, hi = self._radial_region.getRegion()
        new_w = abs(float(hi) - float(lo))
        new_r = (float(hi) + float(lo)) / 2.0
        self._selected.radius_width = new_w
        self._selected.radius = new_r
        self._selected.manual_ref.radius_width = new_w
        self._selected.manual_ref.radius = new_r
        # Angular profile slices over the radial range — needs refresh.
        self._recompute_curves()
        self.peakGeometryChanged.emit(self._selected.manual_ref)

    def _on_angular_changed(self) -> None:
        if (
            self._selected is None
            or self._selected.kind != "manual"
            or self._selected.manual_ref is None
        ):
            return
        lo, hi = self._angular_region.getRegion()
        new_h = abs(float(hi) - float(lo))
        new_a = (float(hi) + float(lo)) / 2.0
        self._selected.angle_width = new_h
        self._selected.angle = new_a
        self._selected.manual_ref.angle_width = new_h
        self._selected.manual_ref.angle = new_a
        # Radial profile slices over the angular range — needs refresh.
        self._recompute_curves()
        self.peakGeometryChanged.emit(self._selected.manual_ref)


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
