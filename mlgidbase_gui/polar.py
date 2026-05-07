"""Cartesian ↔ polar resampling for q-space GIWAXS images.

Pure numpy/scipy — no Qt, no pygid. Convention: angle in degrees from the q_xy
(sample-plane) axis toward q_z (out-of-plane), so 0° = along q_xy, 90° = along q_z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class PolarImage:
    image: np.ndarray  # shape (n_radius, n_angle)
    radius: np.ndarray  # (n_radius,) Å⁻¹
    angle: np.ndarray   # (n_angle,) degrees


def _polar_extent(
    q_xy: np.ndarray, q_z: np.ndarray
) -> tuple[float, float, float]:
    """Compute the (radius_max, angle_min_deg, angle_max_deg) covered.

    The polar grid must span the actual data box, not just the
    upper-right quadrant: pygid emits converted images whose
    (q_xy, q_z) box can sit in any quadrant or straddle multiple
    quadrants depending on the ``vert_positive`` / ``hor_positive``
    conversion flags. Sampling outside that box drops data the user
    expects to see (e.g. negative-q_xy peaks vanish from the polar
    view entirely).

    The angular convention matches pygid: ``angle = atan2(q_z, q_xy)``
    in degrees, with 0° along +q_xy and 90° along +q_z. Atan2 returns
    a value in ``(-180, 180]``; for the data ranges produced by
    pygid's ``vert_positive`` / ``hor_positive`` toggles, the four
    corners always lie in a contiguous arc, so simple ``min`` / ``max``
    of corner angles gives the correct grid extent.

    Pygid's own conventions on those toggles
    (``project_repos/pygid/pygid/coordmaps.py:309-315``):

    - ``vert_positive=True``  →  ang in [0°,  180°]
    - ``hor_positive=True``   →  ang in [-90°, 90°]
    - both                    →  ang in [0°,  90°]   (upper-right)
    - neither                 →  full angular range from corners

    The corner-based detection here yields each of those automatically.
    """
    qxy = np.asarray(q_xy, dtype=float)
    qz = np.asarray(q_z, dtype=float)
    qxy_lo, qxy_hi = float(qxy.min()), float(qxy.max())
    qz_lo, qz_hi = float(qz.min()), float(qz.max())
    corners = (
        (qxy_lo, qz_lo),
        (qxy_lo, qz_hi),
        (qxy_hi, qz_lo),
        (qxy_hi, qz_hi),
    )
    radius_max = max(float(np.hypot(xy, z)) for xy, z in corners)
    angles_deg = [
        float(np.rad2deg(np.arctan2(z, xy)))
        for xy, z in corners
        if not (xy == 0.0 and z == 0.0)
    ]
    if not angles_deg:
        # Degenerate: the data box collapses to the origin. Fall
        # back to the upper-right quadrant so callers still get a
        # sensible (empty) polar grid.
        return radius_max, 0.0, 90.0
    return radius_max, min(angles_deg), max(angles_deg)


def cartesian_to_polar(
    image: np.ndarray,
    q_xy: np.ndarray,
    q_z: np.ndarray,
    n_radius: int = 1000,
    n_angle: int = 900,
) -> PolarImage:
    """Resample a Cartesian (q_z, q_xy)-indexed image onto a polar (radius, angle) grid.

    The polar grid spans the actual angular extent of the input data
    box (not just the upper-right quadrant), so converted images that
    sit in any combination of quadrants — including those produced
    by pygid with ``vert_positive`` / ``hor_positive`` left off —
    render in full. The Cartesian-to-polar mapping uses signed
    ``Q_xy`` / ``Q_z``, so the q_xy / q_z arrays may be increasing,
    decreasing, positive, negative, or mixed without any pre-flipping.
    """
    if image.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {image.shape}")
    if q_xy.size < 2 or q_z.size < 2:
        raise ValueError(
            f"q_xy and q_z must each have at least 2 samples; got "
            f"q_xy.size={q_xy.size}, q_z.size={q_z.size}"
        )

    radius_max, angle_min, angle_max = _polar_extent(q_xy, q_z)
    radius = np.linspace(0.0, radius_max, n_radius)
    angle = np.linspace(angle_min, angle_max, n_angle)

    R, A = np.meshgrid(radius, np.deg2rad(angle), indexing="ij")
    Q_xy = R * np.cos(A)
    Q_z = R * np.sin(A)

    # ``dx`` may be negative when q_xy is decreasing; the formula
    # ``(target - origin) / step`` recovers the correct fractional
    # column index either way (the negative step cancels).
    dx = float(q_xy[1] - q_xy[0])
    dy = float(q_z[1] - q_z[0])
    col = (Q_xy - float(q_xy[0])) / dx
    row = (Q_z - float(q_z[0])) / dy

    polar = map_coordinates(
        image.astype(np.float32, copy=False),
        [row, col],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return PolarImage(image=polar, radius=radius, angle=angle)


def stack_to_polar(
    stack: np.ndarray, q_xy: np.ndarray, q_z: np.ndarray, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply cartesian_to_polar to each frame of a (frames, q_z, q_xy) stack."""
    first = cartesian_to_polar(stack[0], q_xy, q_z, **kwargs)
    out = np.empty(
        (stack.shape[0], first.image.shape[0], first.image.shape[1]),
        dtype=first.image.dtype,
    )
    out[0] = first.image
    for i in range(1, stack.shape[0]):
        out[i] = cartesian_to_polar(stack[i], q_xy, q_z, **kwargs).image
    return out, first.radius, first.angle
