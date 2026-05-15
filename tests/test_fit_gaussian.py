"""Coverage of ``fit.fit_gaussian_on_axis`` — pure scipy/numpy, no Qt.

Recovery is asserted on a synthetic Gaussian + linear background with
fixed-seed noise; only the *deterministic* None guards are asserted
(scipy non-convergence is matrix-flaky and never asserted). Source:
fit.py:110-244.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidlab.fit import fit_gaussian_on_axis

_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _synthetic(center, sigma, amplitude, slope, intercept, *, n=200, noise=0.0):
    axis = np.linspace(-10.0, 10.0, n)
    clean = (
        amplitude * np.exp(-((axis - center) ** 2) / (2.0 * sigma**2))
        + slope * axis
        + intercept
    )
    if noise:
        clean = clean + np.random.default_rng(42).normal(0.0, noise, size=n)
    return axis, clean


@pytest.mark.parametrize("center", [-3.0, 0.0, 2.5])
def test_recovers_parameters(center):
    sigma, amplitude, slope, intercept = 1.2, 50.0, 0.7, 3.0
    axis, data = _synthetic(
        center, sigma, amplitude, slope, intercept, noise=0.5
    )
    fit = fit_gaussian_on_axis(axis, data, center_init=center, width_init=2.0)
    assert fit is not None
    assert fit.center == pytest.approx(center, abs=0.2)
    assert fit.sigma == pytest.approx(sigma, rel=0.15)
    assert fit.amplitude == pytest.approx(amplitude, rel=0.15)
    assert fit.slope == pytest.approx(slope, abs=0.3)
    assert fit.intercept == pytest.approx(intercept, abs=1.5)


def test_fwhm_property():
    axis, data = _synthetic(0.0, 1.5, 40.0, 0.0, 2.0)
    fit = fit_gaussian_on_axis(axis, data, center_init=0.0, width_init=2.5)
    assert fit is not None
    assert fit.fwhm == pytest.approx(_FWHM_K * fit.sigma)


def test_none_short_axis():
    axis = np.linspace(0.0, 1.0, 5)  # len < 8
    assert fit_gaussian_on_axis(axis, axis, 0.5, 1.0) is None


def test_none_nonpositive_width():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert fit_gaussian_on_axis(axis, data, 0.0, 0.0) is None
    assert fit_gaussian_on_axis(axis, data, 0.0, -1.0) is None


def test_none_nonfinite_init():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert fit_gaussian_on_axis(axis, data, np.nan, 1.0) is None
    assert fit_gaussian_on_axis(axis, data, 0.0, np.inf) is None


def test_none_inverted_fit_range():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert (
        fit_gaussian_on_axis(axis, data, 0.0, 1.0, fit_range=(5.0, -5.0))
        is None
    )


def test_none_too_few_samples_in_window():
    # A 10-point axis spanning a huge range with a tiny fit window:
    # fewer than 6 samples fall inside → None.
    axis = np.linspace(0.0, 1000.0, 10)
    data = np.zeros(10)
    assert (
        fit_gaussian_on_axis(
            axis, data, center_init=500.0, width_init=0.01,
            fit_range=(499.9, 500.1),
        )
        is None
    )
