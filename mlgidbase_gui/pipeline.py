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
    """Build the kwargs dict for ``mlgidBASE.add_peak``.

    Always pass polar (angle / angle_width / radius / radius_width) for both
    rings and segments. mlgidBASE.add_peak (see ``peak_operations._calc_new_peak``)
    accepts either polar or cartesian: if all four polar values are non-None
    it uses them verbatim and recomputes ``q_xy / q_z`` from them, otherwise
    it back-computes polar widths from cartesian widths. The cartesian
    bounding box of a polar wedge is strictly wider than the polar widths,
    so the back-converted polar widths come out inflated — which previously
    made saved segments look much bigger than the user-drawn box.
    """
    return {
        "angle": float(peak.angle),
        "angle_width": float(peak.angle_width),
        "radius": float(peak.radius),
        "radius_width": float(peak.radius_width),
    }


def execute(file_path: Path, command: PipelineCommand) -> Any:
    """Run one pipeline command on a NeXus file. Lazily imports mlgidbase."""
    from mlgidbase import mlgidBASE  # noqa: N814

    analysis = mlgidBASE(filename=str(file_path))
    method = getattr(analysis, command.op_name)
    return method(**command.kwargs)
