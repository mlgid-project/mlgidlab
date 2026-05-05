"""Lazy wrappers around mlgidbase pipeline stages.

Imports of mlgidbase are deferred so the GUI can run without it installed.
No Qt imports — keep this module independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Loggers we attach a Qt sink to during a run.
PIPELINE_LOGGERS: tuple[str, ...] = (
    "mlgidbase",
    "pygid",
    "mlgiddetect",
    "pygidfit",
    "mlgidmatch",
)


@dataclass
class PipelineCommand:
    """A single mlgidbase method call."""

    op_name: str  # "run_detection" | "run_fitting" | "run_matching"
    kwargs: dict[str, Any] = field(default_factory=dict)


def is_mlgidbase_available() -> bool:
    try:
        import mlgidbase  # noqa: F401
        return True
    except ImportError:
        return False


def add_peak_kwargs_for(peak) -> dict:
    """Build the per-mode kwargs dict for ``mlgidBASE.add_peak``.

    Tutorial 7 takes polar args for rings (angle / angle_width / radius /
    radius_width) and Cartesian for segments (q_xy / dq_xy / q_z / dq_z).
    For segments we return the Cartesian bounding box of the polar
    rectangle — accurate for narrow segments, conservative for wider ones.
    """
    import math

    if peak.is_ring:
        return {
            "angle": float(peak.angle),
            "angle_width": float(peak.angle_width),
            "radius": float(peak.radius),
            "radius_width": float(peak.radius_width),
        }

    a = math.radians(peak.angle)
    da = math.radians(peak.angle_width)
    rs = (
        max(peak.radius - peak.radius_width / 2.0, 0.0),
        peak.radius + peak.radius_width / 2.0,
    )
    n_sub = 8
    angs = [a - da / 2.0 + (a + da / 2.0 - (a - da / 2.0)) * i / (n_sub - 1)
            for i in range(n_sub)]
    qxys = [r * math.cos(ang) for r in rs for ang in angs]
    qzs = [r * math.sin(ang) for r in rs for ang in angs]
    return {
        "q_xy": float((min(qxys) + max(qxys)) / 2.0),
        "dq_xy": float(max(qxys) - min(qxys)),
        "q_z": float((min(qzs) + max(qzs)) / 2.0),
        "dq_z": float(max(qzs) - min(qzs)),
    }


def execute(file_path: Path, command: PipelineCommand) -> Any:
    """Run one pipeline command on a NeXus file. Lazily imports mlgidbase."""
    from mlgidbase import mlgidBASE  # noqa: N814

    analysis = mlgidBASE(filename=str(file_path))
    method = getattr(analysis, command.op_name)
    return method(**command.kwargs)
