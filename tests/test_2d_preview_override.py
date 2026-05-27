"""Profile-viewer 2D-preview integration-window override.

In 2D fit-mode, ``MainWindow._refresh_2d_preview`` runs pygidfit and
pushes ``(box, rfit, afit)`` through
``ProfileViewer.set_2d_preview``. The ``box`` field overrides the
integration window used by ``_recompute_curves`` so the grey
integrated trace lines up with the pink projected-Gaussian — both
reference pygidfit's refined box, not the user-drawn ROI.

This test exercises ``set_2d_preview`` directly with a synthetic
polar stack engineered so the user box and the override box pick
out distinct slices: the radial trace mean differs between the two
windows. The test also asserts the profile-side edit regions hide
while the override is active (the back-feed loop is disabled).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlgidlab.image_viewer import SelectedPeak
from mlgidlab.profile_viewer import ProfileViewer


def _polar_stack_with_lobe(
    n_frames: int = 1, n_radius: int = 32, n_angle: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a polar stack where the radial profile depends on
    the angular slice chosen.

    Image is uniform 1.0 except for a narrow bright lobe at
    angle ≈ 60° (high values across all radii). A radial-profile
    integration that includes the lobe column averages to ~5; one
    that excludes it averages to ~1. So integrating over distinct
    angular boxes produces visibly different traces — exactly the
    discriminator the override test needs.
    """
    radius = np.linspace(1.0, 4.0, n_radius, dtype=float)
    angle = np.linspace(0.0, 90.0, n_angle, dtype=float)
    img = np.ones((n_radius, n_angle), dtype=float)
    # Bright lobe at angle ≈ 60° — exactly one column wide so the
    # slice-window dependence is sharp.
    lobe_col = int(np.argmin(np.abs(angle - 60.0)))
    img[:, lobe_col] = 5.0
    stack = np.stack([img for _ in range(n_frames)], axis=0)
    return stack, radius, angle


def test_override_slices_image_over_pygidfit_box(qtbot):
    """When ``set_2d_preview`` carries a box, ``_recompute_curves``
    integrates over that box instead of the SelectedPeak's box."""
    pv = ProfileViewer()
    qtbot.addWidget(pv)
    stack, radius, angle = _polar_stack_with_lobe()
    pv.set_polar_stack(stack, radius, angle)

    # User-drawn box at angle 20° — well clear of the bright lobe.
    user_peak = SelectedPeak(
        kind="manual", frame=0, peak_id=0,
        radius=2.5, radius_width=1.0,
        angle=20.0, angle_width=10.0,
    )
    pv.set_selected_peak(user_peak)

    # Without an override, the radial trace averages over the user
    # box's angular slice (15°-25°), which excludes the lobe.
    baseline = pv.last_radial_profile()
    assert baseline is not None
    assert baseline.mean() == pytest.approx(1.0)

    # Push an override box that DOES include the lobe (centred 60°).
    pv.set_2d_preview(
        box=(2.5, 1.0, 60.0, 10.0),
        rfit=None,
        afit=None,
    )
    overridden = pv.last_radial_profile()
    assert overridden is not None
    # Lobe is one of ~7 columns in a 10° angular slice over a 90°/64
    # axis. Trace mean is well above 1.0 (uniform) and below 5.0
    # (pure-lobe). Bracket assertion is sufficient — exact value
    # depends on the angular grid spacing and isn't the point.
    assert 1.0 < overridden.mean() < 5.0
    assert overridden.mean() != pytest.approx(baseline.mean())


def test_override_hides_profile_edit_regions(qtbot):
    """While the override is active, the yellow draggable region
    overlays on the profile plots are hidden — they'd encode a
    window the user can't drag without pygidfit clobbering it."""
    pv = ProfileViewer()
    qtbot.addWidget(pv)
    stack, radius, angle = _polar_stack_with_lobe()
    pv.set_polar_stack(stack, radius, angle)

    peak = SelectedPeak(
        kind="manual", frame=0, peak_id=0,
        radius=2.5, radius_width=1.0,
        angle=20.0, angle_width=10.0,
    )
    pv.set_selected_peak(peak)
    # Manual peak → regions visible by default.
    assert pv._radial_region.isVisible()
    assert pv._angular_region.isVisible()

    pv.set_2d_preview(box=(2.5, 1.0, 60.0, 10.0), rfit=None, afit=None)
    assert not pv._radial_region.isVisible()
    assert not pv._angular_region.isVisible()

    # Clearing the override restores them.
    pv.set_2d_preview(box=None, rfit=None, afit=None)
    assert pv._radial_region.isVisible()
    assert pv._angular_region.isVisible()


def test_clearing_override_restores_user_box_integration(qtbot):
    """``set_2d_preview(None, None, None)`` reverts to user-box
    integration. Symmetric with the failure-fallback path in
    ``MainWindow._refresh_2d_preview``."""
    pv = ProfileViewer()
    qtbot.addWidget(pv)
    stack, radius, angle = _polar_stack_with_lobe()
    pv.set_polar_stack(stack, radius, angle)

    peak = SelectedPeak(
        kind="manual", frame=0, peak_id=0,
        radius=2.5, radius_width=1.0,
        angle=20.0, angle_width=10.0,
    )
    pv.set_selected_peak(peak)
    user_box_mean = pv.last_radial_profile().mean()

    pv.set_2d_preview(box=(2.5, 1.0, 60.0, 10.0), rfit=None, afit=None)
    assert pv.last_radial_profile().mean() != pytest.approx(user_box_mean)

    pv.set_2d_preview(box=None, rfit=None, afit=None)
    assert pv.last_radial_profile().mean() == pytest.approx(user_box_mean)
