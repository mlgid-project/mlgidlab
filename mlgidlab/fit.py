"""1D Gaussian fitting helpers for the radial / angular profile overlays.

**Display-only, never persisted.** This module's output is used to
paint live preview curves on the profile viewer for the active
manual / detected box. It is deliberately not pygidFIT and must
not be written into ``fitted_peaks`` — the persistence path goes
through ``mlgidlab.manual_fit.fit_one_peak`` → pygidfit's 2D fit.
Physics-audit finding F-06 is closed by keeping these two
responsibilities cleanly separated:

  * ``fit.py``        — fast 1D scipy fit, live preview overlay only
  * ``manual_fit.py`` — slow 2D pygidfit fit, persisted truth

If a future caller ever wants to persist 1D-fit output, the audit
verdict must be re-opened: pygidFIT's 2D cluster model is the
upstream contract for fitted peaks, and the 1D Gaussian + linear
background here is not it.

Pure scipy/numpy — no Qt. Returns the fit curve sampled on a fine grid plus
the fitted parameters; callers decide where to draw it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

import logging
logger = logging.getLogger(__name__)

# Default window for the fit, expressed as a multiplier of the box width on
# each side of the box centre. Wider window gives the fit more context for
# background estimation but costs more CPU.
DEFAULT_FIT_WINDOW_FACTOR = 2.0
# Number of samples on the rendered fit curve. Cheap; smoother is better.
FIT_RENDER_SAMPLES = 250


@dataclass
class GaussianFit:
    x: np.ndarray            # rendered x grid
    y: np.ndarray            # fitted curve at x
    amplitude: float
    center: float
    sigma: float
    slope: float
    intercept: float

    @property
    def fwhm(self) -> float:
        return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(self.sigma))


def gaussian_with_linear_bg(
    x: np.ndarray, amplitude: float, center: float, sigma: float,
    slope: float, intercept: float,
) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2)) + slope * x + intercept


def gaussian_with_constant_bg(
    x: np.ndarray, amplitude: float, center: float, sigma: float,
    intercept: float,
) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2)) + intercept


def gaussian_from_stored_params(
    axis: np.ndarray,
    center: float,
    fwhm: float,
    amplitude: float,
    *,
    data: np.ndarray | None = None,
    render_range: tuple[float, float] | None = None,
) -> GaussianFit | None:
    """Render a 1D Gaussian curve **from persisted parameters** — no
    fitting, no data lookup for the Gaussian shape.

    For a peak that already lives in ``fitted_peaks`` or
    ``matched_*``, the profile overlay shows the projection of the
    persisted 2D Gaussian onto the radial / angular axis. The
    projection of an axis-aligned 2D Gaussian along one axis is just
    the 1D Gaussian with the same centre + FWHM in that direction, so
    we sample ``amplitude * exp(-(x-center)^2 / (2*sigma^2))`` over
    ``render_range`` and return it shaped like ``GaussianFit`` so the
    profile viewer can plot it uniformly with the live preview path.

    Baseline. When ``data`` is provided alongside ``axis`` (same
    length), the curve is offset by the local minimum of the data
    inside the rendered window — so the Gaussian sits on top of the
    profile's actual baseline instead of falling to zero in the
    tails. The persisted ``amplitude`` is the 2D fit's peak height
    above local background, so this baseline reconstruction makes
    the overlay visually agree with the data. Without ``data`` we
    fall back to ``intercept=0`` (legacy behaviour).

    Closes the "profile must reflect 2D fit" half of physics-audit
    finding F-06.
    """
    if not (np.isfinite(center) and np.isfinite(fwhm)) or fwhm <= 0:
        return None
    if axis is None or len(axis) == 0:
        return None
    if not np.isfinite(amplitude):
        return None
    sigma = abs(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if render_range is not None:
        rlo = max(float(render_range[0]), float(axis[0]))
        rhi = min(float(render_range[1]), float(axis[-1]))
        if not np.isfinite(rlo) or not np.isfinite(rhi) or rhi <= rlo:
            rlo, rhi = float(axis[0]), float(axis[-1])
    else:
        rlo, rhi = float(axis[0]), float(axis[-1])
    x_fine = np.linspace(rlo, rhi, FIT_RENDER_SAMPLES)

    # Reconstruct the local baseline from the data, when provided.
    # Take the *minimum* of the data inside the render window — the
    # persisted amplitude is peak-above-background, so the baseline
    # is what sits under the peak. Robust to noisy tails because
    # we're aggregating across the full rendered window.
    intercept = 0.0
    if data is not None and len(data) == len(axis):
        axis_arr = np.asarray(axis)
        in_window = (axis_arr >= rlo) & (axis_arr <= rhi)
        if in_window.any():
            window_y = np.asarray(data)[in_window]
            finite = np.isfinite(window_y)
            if finite.any():
                intercept = float(np.nanmin(window_y[finite]))

    y_fine = float(amplitude) * np.exp(
        -((x_fine - float(center)) ** 2) / (2.0 * sigma ** 2)
    ) + intercept
    return GaussianFit(
        x=x_fine, y=y_fine,
        amplitude=float(amplitude), center=float(center), sigma=sigma,
        slope=0.0, intercept=intercept,
    )


def gaussian_from_box(
    axis: np.ndarray,
    data: np.ndarray,
    center: float,
    fwhm: float,
    *,
    render_range: tuple[float, float] | None = None,
) -> GaussianFit | None:
    """Synthesize a Gaussian whose FWHM equals ``fwhm`` and is centred at
    ``center``. Amplitude/baseline are read off the data within the FWHM.

    Used by the profile viewer for non-manual selections so the displayed
    Gaussian matches the box's stored convention exactly (no refit drift).
    Pass ``render_range=(lo, hi)`` to extend the rendered curve past the
    FWHM window — the curve still represents the same Gaussian, just
    sampled over a wider x range.
    """
    if not (np.isfinite(center) and np.isfinite(fwhm)) or fwhm <= 0:
        return None
    if axis is None or len(axis) == 0:
        return None
    lo = center - fwhm / 2.0
    hi = center + fwhm / 2.0
    lo_idx = max(0, int(np.searchsorted(axis, lo, side="left")))
    hi_idx = min(len(axis), int(np.searchsorted(axis, hi, side="right")))
    if hi_idx - lo_idx < 2:
        return None
    y = np.asarray(data[lo_idx:hi_idx], dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return None
    y_finite = y[finite]
    intercept = float(np.nanmin(y_finite))
    amplitude = float(np.nanmax(y_finite) - intercept)
    if amplitude <= 0:
        amplitude = float(np.nanmax(y_finite) - np.nanmin(y_finite))
    if amplitude <= 0:
        amplitude = 1.0
    sigma = abs(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    if render_range is not None:
        rlo = max(float(render_range[0]), float(axis[0]))
        rhi = min(float(render_range[1]), float(axis[-1]))
        if not np.isfinite(rlo) or not np.isfinite(rhi) or rhi <= rlo:
            rlo, rhi = float(axis[lo_idx]), float(axis[hi_idx - 1])
    else:
        rlo, rhi = float(axis[lo_idx]), float(axis[hi_idx - 1])
    x_fine = np.linspace(rlo, rhi, FIT_RENDER_SAMPLES)
    y_fine = gaussian_with_constant_bg(x_fine, amplitude, center, sigma, intercept)
    return GaussianFit(
        x=x_fine, y=y_fine,
        amplitude=amplitude, center=center, sigma=sigma,
        slope=0.0, intercept=intercept,
    )


def fit_gaussian_on_axis(
    axis: np.ndarray,
    data: np.ndarray,
    center_init: float,
    width_init: float,
    *,
    window_factor: float = DEFAULT_FIT_WINDOW_FACTOR,
    fit_range: tuple[float, float] | None = None,
    render_range: tuple[float, float] | None = None,
) -> GaussianFit | None:
    """Fit a 1D Gaussian + linear background to ``data`` along ``axis``.

    By default the fit consumes a window centred on ``center_init`` whose
    half-width is ``(0.5 + window_factor) * width_init`` (so several times
    wider than the user's box, giving the linear-bg term context). Pass
    ``fit_range=(lo, hi)`` to use exactly that window instead — typically
    the bounds of a user-drawn box, so the fit reflects only what's inside.

    Pass ``render_range=(lo, hi)`` to render the resulting Gaussian curve
    on a *different* x range than the fit window — useful when callers
    want to display the Gaussian's tails extending beyond a tight fit box.
    The fit itself is unaffected; only the returned ``x``/``y`` arrays
    sample the rendered curve over ``render_range`` instead of the fit
    window. ``render_range`` is clipped to ``[axis[0], axis[-1]]``.

    Returns None if the inputs aren't fittable or scipy can't converge.
    """
    if len(axis) < 8:
        return None
    if not (np.isfinite(center_init) and np.isfinite(width_init)) or width_init <= 0:
        return None

    if fit_range is not None:
        lo, hi = float(fit_range[0]), float(fit_range[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return None
    else:
        half = max(width_init * (0.5 + window_factor), (axis[-1] - axis[0]) * 0.05)
        lo, hi = center_init - half, center_init + half

    lo_idx = int(np.searchsorted(axis, lo, side="left"))
    hi_idx = int(np.searchsorted(axis, hi, side="right"))
    lo_idx = max(0, lo_idx)
    hi_idx = min(len(axis), hi_idx)
    if hi_idx - lo_idx < 6:
        return None

    x = axis[lo_idx:hi_idx]
    y = data[lo_idx:hi_idx]
    finite = np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 6:
        return None

    sigma_init = max(width_init / 2.355, (x[-1] - x[0]) / 50.0)

    # Linear background term (slope*x + intercept) is fitted in both the
    # wide-window and box-bounded paths. Wide-window fits use it freely;
    # box-bounded fits constrain amp/sigma so the slope can't be absorbed
    # into a "very wide gaussian + parabola" local minimum.
    n_edge = max(len(x) // 8, 2)
    bg_x = np.concatenate([x[:n_edge], x[-n_edge:]])
    bg_y = np.concatenate([y[:n_edge], y[-n_edge:]])
    try:
        slope_init, intercept_init = np.polyfit(bg_x, bg_y, 1)
    except Exception:
        logger.debug("suppressed exception in fit_gaussian_on_axis", exc_info=True)
        slope_init, intercept_init = 0.0, float(np.nanmean(y))

    bg_at_center = float(slope_init) * center_init + float(intercept_init)
    amp_init = float(np.nanmax(y) - bg_at_center)
    if amp_init <= 0:
        amp_init = float(np.nanmax(y) - np.nanmin(y))
    if amp_init <= 0:
        amp_init = 1.0

    try:
        if fit_range is None:
            popt, _ = curve_fit(
                gaussian_with_linear_bg,
                x, y,
                p0=[amp_init, center_init, sigma_init, slope_init, intercept_init],
                maxfev=2000,
            )
        else:
            # Box-bounded fits can fall into a "very wide Gaussian + large
            # negative offset ≈ parabola" local minimum because there's no
            # flat baseline to anchor amp/sigma. Bound sigma to the window
            # width and keep amp ≥ 0 so the optimizer must converge on an
            # actual peak shape inside the box. Slope/intercept are left
            # unbounded so the linear-bg term can absorb sloped baselines.
            window_w = float(x[-1] - x[0])
            sample_step = window_w / max(len(x) - 1, 1)
            sigma_lo = max(sample_step * 0.5, 1e-6)
            sigma_hi = max(window_w, sigma_lo * 2.0)
            popt, _ = curve_fit(
                gaussian_with_linear_bg,
                x, y,
                p0=[amp_init, center_init, sigma_init, slope_init, intercept_init],
                bounds=(
                    [0.0,    float(x[0]),  sigma_lo, -np.inf, -np.inf],
                    [np.inf, float(x[-1]), sigma_hi,  np.inf,  np.inf],
                ),
                maxfev=5000,
            )
        amplitude, center, sigma, slope, intercept = (float(v) for v in popt)
    except Exception:
        logger.debug("suppressed exception in fit_gaussian_on_axis", exc_info=True)
        return None

    # Reject obviously-degenerate fits.
    if not np.isfinite(sigma) or abs(sigma) > (x[-1] - x[0]) * 5.0:
        return None
    if amplitude < 0:
        # Allow inverted peaks but only if they're shallow; reject pathological cases.
        if abs(amplitude) > 100.0 * (np.nanmax(y) - np.nanmin(y) + 1):
            return None

    if render_range is not None:
        rlo = max(float(render_range[0]), float(axis[0]))
        rhi = min(float(render_range[1]), float(axis[-1]))
        if not np.isfinite(rlo) or not np.isfinite(rhi) or rhi <= rlo:
            rlo, rhi = float(x[0]), float(x[-1])
    else:
        rlo, rhi = float(x[0]), float(x[-1])
    x_fine = np.linspace(rlo, rhi, FIT_RENDER_SAMPLES)
    y_fine = gaussian_with_linear_bg(
        x_fine, amplitude, center, sigma, slope, intercept,
    )
    return GaussianFit(
        x=x_fine, y=y_fine,
        amplitude=amplitude, center=center, sigma=abs(sigma),
        slope=slope, intercept=intercept,
    )


def fit_gaussian_anchored(
    axis: np.ndarray,
    data: np.ndarray,
    *,
    center: float,
    sigma: float,
    center_drift: float = 0.0,
    sigma_factor: float = 1.0,
    fit_range: tuple[float, float] | None = None,
    render_range: tuple[float, float] | None = None,
) -> GaussianFit | None:
    """Fit a Gaussian + linear bg with centre / sigma loosely anchored.

    When ``center_drift == 0`` and ``sigma_factor == 1.0`` this
    behaves as a strict fix — only ``(amplitude, slope, intercept)``
    are free. Otherwise scipy is bounded to:

    * centre ∈ ``[center - center_drift, center + center_drift]``
    * sigma  ∈ ``[sigma / sigma_factor, sigma * sigma_factor]``

    Used by ``MainWindow._refresh_2d_preview`` so the pink projected
    curve in 2D mode sits cleanly on the 1D-integrated grey trace
    even when pygidfit's 2D centroid differs slightly from the
    1D-projected centroid (theta ≠ 0, asymmetric peaks, polar-grid
    interpolation mismatch between mlgidlab and pygidfit). The
    bounded drift lets the curve realign on the actual data peak
    without straying so far from pygidfit's geometry that the cyan
    image-side preview box would feel disconnected.

    Returns ``None`` when there aren't enough finite samples or
    scipy doesn't converge — same precedent as
    ``fit_gaussian_on_axis``; the caller (host) blanks the curve in
    that case.
    """
    if len(axis) < 8:
        return None
    if not (np.isfinite(center) and np.isfinite(sigma)) or sigma <= 0:
        return None
    if center_drift < 0 or sigma_factor < 1.0:
        return None

    if fit_range is not None:
        lo, hi = float(fit_range[0]), float(fit_range[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return None
    else:
        half = max(3.0 * sigma, (axis[-1] - axis[0]) * 0.05)
        lo, hi = center - half, center + half

    lo_idx = int(np.searchsorted(axis, lo, side="left"))
    hi_idx = int(np.searchsorted(axis, hi, side="right"))
    lo_idx = max(0, lo_idx)
    hi_idx = min(len(axis), hi_idx)
    if hi_idx - lo_idx < 6:
        return None

    x = axis[lo_idx:hi_idx]
    y = data[lo_idx:hi_idx]
    finite = np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 6:
        return None

    # Estimate the linear background from the edge samples — same
    # heuristic as ``fit_gaussian_on_axis`` for a stable initial
    # guess.
    n_edge = max(len(x) // 8, 2)
    bg_x = np.concatenate([x[:n_edge], x[-n_edge:]])
    bg_y = np.concatenate([y[:n_edge], y[-n_edge:]])
    try:
        slope_init, intercept_init = np.polyfit(bg_x, bg_y, 1)
    except Exception:
        logger.debug(
            "suppressed exception in fit_gaussian_anchored", exc_info=True
        )
        slope_init, intercept_init = 0.0, float(np.nanmean(y))

    bg_at_center = float(slope_init) * center + float(intercept_init)
    amp_init = float(np.nanmax(y) - bg_at_center)
    if amp_init <= 0:
        amp_init = float(np.nanmax(y) - np.nanmin(y))
    if amp_init <= 0:
        amp_init = 1.0

    target_center = float(center)
    target_sigma = float(sigma)

    strict = center_drift == 0.0 and sigma_factor == 1.0
    if strict:
        # Centre + sigma frozen — only 3 parameters free. Reuses the
        # same curve_fit call shape as the bounded path so a future
        # consolidation is easy.
        def model(xs, amp, slope, intercept):
            return gaussian_with_linear_bg(
                xs, amp, target_center, target_sigma, slope, intercept,
            )
        try:
            popt, _ = curve_fit(
                model, x, y,
                p0=[amp_init, slope_init, intercept_init],
                maxfev=2000,
            )
            amplitude, slope, intercept = (float(v) for v in popt)
        except Exception:
            logger.debug(
                "suppressed exception in fit_gaussian_anchored", exc_info=True
            )
            return None
        fitted_center = target_center
        fitted_sigma = target_sigma
    else:
        # Bounded 5-param fit. Centre and sigma allowed to drift
        # within the supplied tolerances so the curve realigns on
        # the data peak when pygidfit's 2D centroid doesn't match
        # the 1D-projected centroid.
        c_lo = target_center - center_drift
        c_hi = target_center + center_drift
        s_lo = target_sigma / sigma_factor
        s_hi = target_sigma * sigma_factor
        try:
            popt, _ = curve_fit(
                gaussian_with_linear_bg,
                x, y,
                p0=[
                    amp_init, target_center, target_sigma,
                    slope_init, intercept_init,
                ],
                bounds=(
                    [0.0, c_lo, s_lo, -np.inf, -np.inf],
                    [np.inf, c_hi, s_hi, np.inf, np.inf],
                ),
                maxfev=5000,
            )
            (
                amplitude, fitted_center, fitted_sigma,
                slope, intercept,
            ) = (float(v) for v in popt)
        except Exception:
            logger.debug(
                "suppressed exception in fit_gaussian_anchored", exc_info=True
            )
            return None

    if not np.isfinite(amplitude):
        return None

    if render_range is not None:
        rlo = max(float(render_range[0]), float(axis[0]))
        rhi = min(float(render_range[1]), float(axis[-1]))
        if not np.isfinite(rlo) or not np.isfinite(rhi) or rhi <= rlo:
            rlo, rhi = float(x[0]), float(x[-1])
    else:
        rlo, rhi = float(x[0]), float(x[-1])
    x_fine = np.linspace(rlo, rhi, FIT_RENDER_SAMPLES)
    y_fine = gaussian_with_linear_bg(
        x_fine, amplitude, fitted_center, fitted_sigma, slope, intercept,
    )
    return GaussianFit(
        x=x_fine, y=y_fine,
        amplitude=amplitude,
        center=fitted_center, sigma=abs(fitted_sigma),
        slope=slope, intercept=intercept,
    )
