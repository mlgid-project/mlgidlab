"""1D Gaussian fitting helpers used to overlay an expected peak shape on the
radial / angular profile plots.

Pure scipy/numpy — no Qt. Returns the fit curve sampled on a fine grid plus
the fitted parameters; callers decide where to draw it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

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
