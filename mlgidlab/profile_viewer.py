from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QWidget

from mlgidlab.fit import GaussianFit, fit_gaussian_on_axis
from mlgidlab.image_viewer import (
    OVERLAY_STYLE,
    ManualPeak,
    SelectedPeak,
    _disable_viewport_scroll,
)

# Multiple of box width on each side over which the rendered Gaussian curve
# extends past the box bounds. Big enough that the tails fade into the
# baseline; small enough that very tight peaks don't render as a spike on a
# nearly empty axis.
FIT_RENDER_PAD_FACTOR = 3.0

# Fit interval for fitted / matched peaks, expressed as a multiple of the
# stored FWHM around the peak center. The stored FWHM (radial: radius_width;
# azimuthal: angle_width / 2) is generally tighter than the region we want
# to fit over — the user expects to *see* the same FWHM on screen as is
# saved in the file, but they want the *fit* to be computed over a wider
# window so the Gaussian's tails get to settle into the baseline.
FITTED_FIT_REGION_FACTOR = 3.0

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
        # Bottom margin keeps the "radius" axis label off the viewport's
        # lower edge — pyqtgraph's auto-sized bottom-axis cell can leave
        # too little room and the clipped label opens a small scrollable
        # region in the dock.
        self._radial_plot.plotItem.layout.setContentsMargins(0, 0, 0, 12)
        # PlotWidget is a QGraphicsView under the hood — pin the
        # scrollbars off and drop the frame border so the plot
        # contents (axes + labels) sit flush against the dock and
        # the user can't accidentally pan/scroll the scene.
        self._radial_plot.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._radial_plot.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._radial_plot.setFrameStyle(QFrame.Shape.NoFrame)
        _disable_viewport_scroll(self._radial_plot)
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
        self._angular_plot.plotItem.layout.setContentsMargins(0, 0, 0, 12)
        self._angular_plot.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._angular_plot.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._angular_plot.setFrameStyle(QFrame.Shape.NoFrame)
        _disable_viewport_scroll(self._angular_plot)
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
        # When True, the angular Gaussian fit is skipped + cleared. Driven
        # by the host (MainWindow) from the parameter panel's ring toggle:
        # if Add-to-fitted will commit a ring there's no angular dimension
        # to save, so showing the angular fit would be misleading.
        self._skip_angular_fit: bool = False
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

    def set_skip_angular_fit(self, skip: bool) -> None:
        """Tell the viewer not to compute or render the angular Gaussian.

        Used while the parameter panel's ring toggle is active — saving
        as ring drops the angular dimension entirely, so showing a fit
        the user can't actually save would be misleading.
        """
        if self._skip_angular_fit == skip:
            return
        self._skip_angular_fit = skip
        # Re-render so the curve appears / disappears immediately.
        self._recompute_curves()

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

        Region markers (yellow LinearRegionItems) are shown only for kinds
        the user actively shapes — ``manual`` (draggable) and ``detected``
        (read-only but tracks the box). For ``fitted`` / ``matched`` the
        box is fixed by the storage convention and the region markers
        would just clutter the view; the live Gaussian curve still renders.
        """
        self._selected = peak
        visible = peak is not None
        is_manual = peak is not None and peak.kind == "manual"
        # Only manual + detected are user-shaped — show the region marker
        # only for those. Fitted / matched still get the Gaussian curve
        # (handled in _update_fit_curves) but no region overlay.
        show_regions = peak is not None and peak.kind in ("manual", "detected")
        # For ring peaks (or any peak whose angle_width is non-finite) the
        # angular region would span the entire axis or fail to render — hide
        # it so the profile stays usable. The radial region still tracks.
        is_ring_box = (
            peak is not None
            and (peak.is_ring or not np.isfinite(peak.angle_width))
        )

        self._radial_region.setVisible(show_regions)
        self._radial_region.setMovable(is_manual)
        self._angular_region.setVisible(show_regions and not is_ring_box)
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
            sel = self._selected  # SelectedPeak with kind == "manual"
        else:
            if peak is not self._selected:
                return
            sel = peak
        r_range = _radial_fit_range(sel)
        a_range = _angular_fit_range(sel)
        # Geometry changes can flip ``is_ring`` (via the Save-fitted-as-
        # ring toggle) or the angular_width (via the 2D ROI), so the
        # angular region's visibility has to be re-evaluated here.
        # ``set_selected_peak`` only fires on selection *change*, which
        # would otherwise leave the angular region painting the pre-
        # ring borders even after the box is expanded to a full sweep.
        # Radial visibility / movability depend only on selection kind,
        # which doesn't change here.
        is_ring_box = sel.is_ring or not np.isfinite(sel.angle_width)
        show_regions = sel.kind in ("manual", "detected")
        self._angular_region.setVisible(show_regions and not is_ring_box)

        self._radial_region.blockSignals(True)
        try:
            self._radial_region.setRegion(
                (float(r_range[0]), float(r_range[1]))
            )
        finally:
            self._radial_region.blockSignals(False)
        # Skip the angular region for ring boxes — its bounds would be
        # ±inf / span the entire axis. setVisible already hid it.
        if a_range is not None:
            self._angular_region.blockSignals(True)
            try:
                self._angular_region.setRegion(
                    (float(a_range[0]), float(a_range[1]))
                )
            finally:
                self._angular_region.blockSignals(False)
        # The integration window changed → recompute the slice profiles.
        self._recompute_curves()
        # When the box moved via the 2D ROI (manual/detected drag), the
        # region may now sit outside the profile's current X range.
        # Pan/expand each plot so the borders stay visible — without
        # this the user dragging the ROI past the visible profile
        # window loses sight of the interval.
        if self._radial_region.isVisible():
            self._ensure_region_in_view(self._radial_plot, self._radial_region)
        if a_range is not None and self._angular_region.isVisible():
            self._ensure_region_in_view(self._angular_plot, self._angular_region)

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
        # All peak kinds now drive a real Gaussian fit; only the *interval*
        # we fit over changes:
        #   manual / detected  → box bounds (the user-controlled region)
        #   fitted / matched   → ``FITTED_FIT_REGION_FACTOR × FWHM`` around
        #                        the center, where the FWHM is derived from
        #                        the storage convention
        #                          radial:   FWHM_r = radius_width
        #                          azimuth:  FWHM_a = angle_width / 2
        # The wider fitted/matched window lets the Gaussian's tails settle
        # into the baseline so the fit isn't biased by the FWHM boundary.
        # The image-space box continues to render the stored FWHM extents.
        rfit: GaussianFit | None = None
        afit: GaussianFit | None = None
        r_range = _radial_fit_range(peak)
        if self._radius is not None and r_range is not None:
            r_lo, r_hi = r_range
            r_pad = FIT_RENDER_PAD_FACTOR * (r_hi - r_lo)
            render_r = (r_lo - r_pad, r_hi + r_pad)
            rfit = fit_gaussian_on_axis(
                self._radius, radial, peak.radius, peak.radius_width,
                fit_range=(r_lo, r_hi),
                render_range=render_r,
            )
            if rfit is not None:
                self._radial_fit_curve.setData(rfit.x, rfit.y)
            else:
                self._radial_fit_curve.setData([], [])
        # Skip angular fit when the host has flagged ring-save mode — the
        # angular dimension won't be saved, so showing a fit would lie
        # about what Add-to-fitted will write.
        a_range = None if self._skip_angular_fit else _angular_fit_range(peak)
        if self._angle is not None and a_range is not None:
            a_lo, a_hi = a_range
            a_pad = FIT_RENDER_PAD_FACTOR * (a_hi - a_lo)
            render_a = (a_lo - a_pad, a_hi + a_pad)
            afit = fit_gaussian_on_axis(
                self._angle, angular, peak.angle, peak.angle_width,
                fit_range=(a_lo, a_hi),
                render_range=render_a,
            )
            if afit is not None:
                self._angular_fit_curve.setData(afit.x, afit.y)
            else:
                self._angular_fit_curve.setData([], [])
        elif self._angle is not None:
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
        # Keep the dragged region inside the visible plot area so the
        # user can always see what they're editing.
        self._ensure_region_in_view(self._radial_plot, self._radial_region)
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
        self._ensure_region_in_view(self._angular_plot, self._angular_region)
        self.peakGeometryChanged.emit(self._selected.manual_ref)

    def _ensure_region_in_view(
        self, plot: pg.PlotWidget, region_item: pg.LinearRegionItem,
    ) -> None:
        """Keep the selection region's borders inside the plot's X
        range. Pans the view to follow the region when it has moved
        outside the current window, expands when the region is wider
        than the current view. Padding leaves a small gap so the
        edges aren't flush with the plot boundary.

        Called both on direct profile-region drags (manual peak edge
        edits) and on programmatic syncs from the 2D ROI (manual /
        detected peak moves) — the latter is what was previously
        leaving the box stranded off-screen.
        """
        lo, hi = region_item.getRegion()
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return
        span = max(abs(hi - lo), 1e-9)
        pad = 0.5 * span
        target_lo, target_hi = lo - pad, hi + pad
        vb = plot.getViewBox()
        cur_lo, cur_hi = vb.viewRange()[0]
        cur_lo, cur_hi = float(cur_lo), float(cur_hi)
        cur_width = cur_hi - cur_lo
        target_width = target_hi - target_lo

        if target_width > cur_width:
            # Region is wider than the visible window — expand to fit
            # without losing the existing centre point.
            new_lo = min(cur_lo, target_lo)
            new_hi = max(cur_hi, target_hi)
        elif target_lo < cur_lo:
            # Region's left edge fell off-screen → slide the view left
            # by exactly the overshoot, keeping the original width.
            shift = cur_lo - target_lo
            new_lo, new_hi = cur_lo - shift, cur_hi - shift
        elif target_hi > cur_hi:
            # Region's right edge fell off-screen → slide the view right.
            shift = target_hi - cur_hi
            new_lo, new_hi = cur_lo + shift, cur_hi + shift
        else:
            return  # already fully visible
        plot.setXRange(new_lo, new_hi, padding=0)


def _radial_fit_range(peak: SelectedPeak) -> tuple[float, float] | None:
    """Radial Gaussian fit interval for ``peak``.

    For manual / detected peaks the interval is the box itself (the user
    controls the box bounds, so we fit over what they drew). For
    fitted / matched peaks the box bounds *are* the FWHM — the fit interval
    expands to ``FITTED_FIT_REGION_FACTOR × FWHM`` around the center so
    the Gaussian fit has room outside the FWHM to settle into the baseline.
    Returns ``None`` only when widths are non-finite or zero (caller hides
    the curve in that case).
    """
    r = peak.radius
    dr = peak.radius_width
    if not (np.isfinite(r) and np.isfinite(dr) and dr > 0):
        return None
    if peak.kind in ("manual", "detected"):
        half = dr / 2.0
    else:  # fitted / matched
        # radial box width == FWHM_r by storage convention.
        half = FITTED_FIT_REGION_FACTOR * dr / 2.0
    return (r - half, r + half)


def _angular_fit_range(peak: SelectedPeak) -> tuple[float, float] | None:
    """Angular Gaussian fit interval for ``peak``. None for ring peaks
    (angle_width inf or non-finite — angular slice is the whole axis).
    """
    a = peak.angle
    da = peak.angle_width
    if peak.is_ring or not (np.isfinite(a) and np.isfinite(da) and da > 0):
        return None
    if peak.kind in ("manual", "detected"):
        half = da / 2.0
    else:
        # azimuthal box width == 2 × FWHM_a by storage convention, so
        # FWHM_a = da / 2 — same factor expansion as radial.
        fwhm_a = da / 2.0
        half = FITTED_FIT_REGION_FACTOR * fwhm_a / 2.0
    return (a - half, a + half)


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
