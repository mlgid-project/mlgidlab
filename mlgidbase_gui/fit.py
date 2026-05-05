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


def fit_gaussian_on_axis(
    axis: np.ndarray,
    data: np.ndarray,
    center_init: float,
    width_init: float,
    *,
    window_factor: float = DEFAULT_FIT_WINDOW_FACTOR,
) -> GaussianFit | None:
    """Fit a 1D Gaussian + linear background to a window centred on `center_init`.

    Returns None if the inputs aren't fittable or scipy can't converge.
    """
    if len(axis) < 8:
        return None
    if not (np.isfinite(center_init) and np.isfinite(width_init)) or width_init <= 0:
        return None

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
        popt, _ = curve_fit(
            gaussian_with_linear_bg,
            x, y,
            p0=[amp_init, center_init, sigma_init, slope_init, intercept_init],
            maxfev=2000,
        )
    except Exception:
        return None

    amplitude, center, sigma, slope, intercept = (float(v) for v in popt)
    # Reject obviously-degenerate fits.
    if not np.isfinite(sigma) or abs(sigma) > (x[-1] - x[0]) * 5.0:
        return None
    if amplitude < 0:
        # Allow inverted peaks but only if they're shallow; reject pathological cases.
        if abs(amplitude) > 100.0 * (np.nanmax(y) - np.nanmin(y) + 1):
            return None

    x_fine = np.linspace(x[0], x[-1], FIT_RENDER_SAMPLES)
    y_fine = gaussian_with_linear_bg(x_fine, *popt)
    return GaussianFit(
        x=x_fine, y=y_fine,
        amplitude=amplitude, center=center, sigma=abs(sigma),
        slope=slope, intercept=intercept,
    )
