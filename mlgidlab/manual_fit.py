"""Single-peak 2D fit wrapper around ``pygidfit.fit_data``.

The GUI's "Add to fitted" action used to commit a 1D-Gaussian-derived
row to ``fitted_peaks`` with the 2D shape coefficients (A, B, C,
theta) zero-filled — physics-audit finding F-06's concrete failure
mode (the persisted manual fit was already partly fictitious).

This module replaces that path. ``fit_one_peak`` accepts a single
user-drawn box plus the active **cartesian** frame + q axes +
experimental geometry, replicates pygidfit's pipeline polar
conversion internally (``_get_polar_grid`` + ``polar_conversion``
+ ``img_preprocessing``), and runs ``pygidfit.fit_data`` on a
length-1 input. Using pygidfit's own conversion is what makes the
manual fit byte-identical to what the pipeline ``run_fitting``
writes for the same detected box — an earlier revision passed
mlgidlab's polar image (different resolution + scipy interpolation +
no ``img_preprocessing``) and produced visible drift between the
two paths.

On any pygidfit failure (raises, returns empty, NaN amplitude) we
raise ``ManualFitError`` so callers can fall back to the legacy
1D + zero-fill behaviour or surface a clear error to the user.
"""
from __future__ import annotations

from dataclasses import dataclass

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ManualFitError(RuntimeError):
    """Raised when ``pygidfit.fit_data`` cannot produce a fit for a
    user-drawn box (clustering returned no boxes, the fit didn't
    converge, etc.). Caller is expected to fall back to the legacy
    1D + zero-fill path so the user is never blocked from committing
    a peak."""


@dataclass(frozen=True)
class ManualFitResult:
    """Output of a single-peak 2D fit, in the same field convention as
    the persisted ``fitted_peaks`` row (polar geometry + 2D shape
    coefficients + amplitude). All scalars."""

    radius: float
    radius_width: float
    angle: float
    angle_width: float
    amplitude: float
    A: float
    B: float
    C: float
    theta: float


# Hardcoded polar grid shape pygidfit's ``ProcessDataFromFile.process_single_frame``
# uses (see ``mlgidbase/pygidfit_functions.py::_run_pygidfit_from_file``
# at line ~51: ``polar_shape=np.array([512, 1024])``). Matching this
# resolution + interpolation method makes the manual fit byte-identical
# to the pipeline ``run_fitting`` result. Changing it without changing
# pygidfit would re-introduce the drift the wrapper exists to remove.
_PYGIDFIT_POLAR_SHAPE = (512, 1024)


def fit_one_peak(
    cartesian_image: np.ndarray,
    q_xy: np.ndarray,
    q_z: np.ndarray,
    *,
    radius: float,
    radius_width: float,
    angle: float,
    angle_width: float,
    wavelength_angstrom: float,
    q_xy_max: float,
    q_z_max: float,
    ai_deg: float = 0.0,
    crit_angle: float = 0.0,
    theta_fixed: bool = True,
    clustering_distance_peaks: float = 10.0,
    clustering_distance_rings: float = 10.0,
    clustering_extend: int = 2,
) -> ManualFitResult:
    """Run ``pygidfit.fit_data`` on a single user-drawn box.

    Inputs match pygidfit's pipeline convention:

    * ``cartesian_image``: the entry's ``data/img_gid_q`` frame in
      reciprocal space. Shape ``(n_qz, n_qxy)`` per pygid's array
      layout — this is what ``FrameSource.get_cartesian(frame)``
      returns.
    * ``q_xy`` / ``q_z``: 1-D coordinate axes in Å⁻¹.
    * ``radius`` / ``radius_width`` / ``angle`` / ``angle_width``:
      the user's box in polar q-space (Å⁻¹ / degrees).
    * geometry kwargs (``wavelength_angstrom``, ``q_*_max``,
      ``ai_deg``, ``crit_angle``): read from the entry's instrument
      metadata.
    * fit kwargs (``theta_fixed``, ``clustering_distance_*``,
      ``clustering_extend``): caller should pass the same values
      the pipeline panel uses so manual Add-to-fitted runs with
      the SAME config as the next ``run_fitting`` would.

    Internally the wrapper replicates
    ``pygidfit.process_scans.ProcessDataFromFile.process_single_frame``:
    apply ``img_preprocessing`` (masks rows below the sample
    horizon to NaN using ``crit_angle``), build a polar grid via
    ``_get_polar_grid`` at the hardcoded ``(512, 1024)`` resolution,
    resample with ``polar_conversion`` (cv2 linear interp), then
    call ``fit_data``. Doing the conversion here — rather than
    accepting mlgidlab's polar resample — is what makes the result
    byte-identical to the pipeline output.

    Returns a ``ManualFitResult`` with the 2D-fit values pygidfit
    produces for the box. Raises ``ManualFitError`` on any pygidfit
    failure so the caller can fall back to the legacy 1D path or
    surface a clear error.
    """
    # Lazy import — pygidfit pulls torch / scipy / scikit-learn and
    # warming those at module import would slow GUI startup. The
    # first ``Add to fitted`` click pays the cost; subsequent clicks
    # are fast.
    try:
        import cv2  # noqa: F401
        from pygidfit.process_scans import (
            _get_polar_grid,
            fit_data,
            img_preprocessing,
            polar_conversion,
        )
    except Exception as exc:  # pragma: no cover — env-dependent
        raise ManualFitError(
            f"pygidfit / cv2 not available ({exc!r}); cannot run 2D fit"
        ) from exc

    # 1) Preprocess: mask rows below the sample horizon (q_z <
    # q_z_critical from ``calc_smpl_hor(ai, crit_angle, wavelength)``)
    # and any non-positive pixels to NaN. This matches what
    # ProcessDataFromFile.process_single_frame does and is the
    # difference that gave the most drift in the manual-vs-pipeline
    # A/B without it.
    try:
        img_pre = img_preprocessing(
            np.asarray(cartesian_image),
            float(ai_deg),
            float(crit_angle),
            float(wavelength_angstrom),
            np.asarray(q_z),
        )
    except Exception as exc:
        raise ManualFitError(
            f"pygidfit.img_preprocessing raised: {exc!r}"
        ) from exc

    # 2) Build polar grid at pygidfit's hardcoded shape. The
    # ``beam_center=[0, 0]`` is what mlgidbase passes too — pygidfit
    # treats the image as a quadrant whose ``(0,0)`` index lies at
    # the q-space origin. (See pygidfit's _get_polar_grid for the
    # exact construction; the wrapper just mirrors the pipeline.)
    try:
        yy, zz, ang_deg_max = _get_polar_grid(
            img_pre.shape, _PYGIDFIT_POLAR_SHAPE, [0, 0],
        )
        polar_img = polar_conversion(img_pre, yy, zz, cv2.INTER_LINEAR)
    except Exception as exc:
        raise ManualFitError(
            f"pygidfit polar conversion raised: {exc!r}"
        ) from exc

    # 3) Compute ``q_abs_max`` the same way ProcessDataFromFile does:
    # ``np.sqrt(np.nanmax(q_z)**2 + np.nanmax(q_xy)**2)``. NOT
    # ``max(polar_radius_axis)`` (which is what an earlier revision
    # of this wrapper used and was a second source of drift).
    q_abs_max = float(np.sqrt(
        np.nanmax(np.asarray(q_z)) ** 2 + np.nanmax(np.asarray(q_xy)) ** 2
    ))

    # 4) Run pygidfit's fit on the single box. Each box parameter is
    # an array of length 1.
    radius_arr = np.array([float(radius)], dtype=float)
    radius_width_arr = np.array([float(radius_width)], dtype=float)
    angle_arr = np.array([float(angle)], dtype=float)
    angle_width_arr = np.array([float(angle_width)], dtype=float)
    try:
        container, _peaks_pool = fit_data(
            polar_img,
            radius=radius_arr,
            radius_width=radius_width_arr,
            angle=angle_arr,
            angle_width=angle_width_arr,
            wavelength=float(wavelength_angstrom),
            q_xy_max=float(q_xy_max),
            q_z_max=float(q_z_max),
            q_abs_max=q_abs_max,
            ang_deg_max=ang_deg_max,
            clustering_distance_peaks=float(clustering_distance_peaks),
            clustering_distance_rings=float(clustering_distance_rings),
            clustering_extend=int(clustering_extend),
            theta_fixed=bool(theta_fixed),
            debug=False,
            multiprocessing=False,
            peaks_pool=None,
            ai=float(ai_deg),
        )
    except Exception as exc:
        raise ManualFitError(
            f"pygidfit.fit_data raised: {exc!r}"
        ) from exc

    # pygidfit may cluster the single box away or produce a degenerate
    # output — both surface as a container with empty arrays.
    amp = np.asarray(container.amplitude).ravel()
    if amp.size == 0:
        raise ManualFitError(
            "pygidfit produced no output for the user-drawn box "
            "(clustering may have dropped the input or the fit did "
            "not converge)"
        )
    if not np.isfinite(amp[0]):
        raise ManualFitError(
            "pygidfit returned NaN amplitude — the 2D fit did not "
            "converge on the user-drawn box"
        )

    # Return pygidfit's container values verbatim, no width
    # conversion. mlgidbase's pipeline ``run_fitting`` path stores
    # the same container directly, so manually-added peaks and
    # pipeline-fitted peaks share the same width convention
    # (pygidfit's ``2σ`` in both ``radius_width`` and ``angle_width``,
    # per ``_data2container``'s ``*2`` scaling).
    return ManualFitResult(
        radius=float(np.asarray(container.radius).ravel()[0]),
        radius_width=float(np.asarray(container.radius_width).ravel()[0]),
        angle=float(np.asarray(container.angle).ravel()[0]),
        angle_width=float(np.asarray(container.angle_width).ravel()[0]),
        amplitude=float(amp[0]),
        A=float(np.asarray(container.A).ravel()[0]),
        B=float(np.asarray(container.B).ravel()[0]),
        C=float(np.asarray(container.C).ravel()[0]),
        theta=float(np.asarray(container.theta).ravel()[0]),
    )
