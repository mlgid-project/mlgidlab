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


def cartesian_to_polar(
    image: np.ndarray,
    q_xy: np.ndarray,
    q_z: np.ndarray,
    n_radius: int = 1000,
    n_angle: int = 900,
) -> PolarImage:
    """Resample a Cartesian (q_z, q_xy)-indexed image onto a polar (radius, angle) grid."""
    if image.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {image.shape}")

    radius_max = float(np.hypot(q_xy.max(), q_z.max()))
    radius = np.linspace(0.0, radius_max, n_radius)
    angle = np.linspace(0.0, 90.0, n_angle)

    R, A = np.meshgrid(radius, np.deg2rad(angle), indexing="ij")
    Q_xy = R * np.cos(A)
    Q_z = R * np.sin(A)

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
