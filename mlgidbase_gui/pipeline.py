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
    kwargs = dict(command.kwargs)
    # Run-matching takes a ``cif_prepr`` value that mlgidBASE accepts as
    # either a path-to-pickle string or a CifPattern instance. The GUI lets
    # the user point at raw .cif files / a folder of CIFs too — translate
    # those into a CifPattern here, on the worker thread, since pattern
    # construction simulates each CIF and can take seconds.
    if command.op_name == "run_matching":
        cif_in = kwargs.get("cif_prepr")
        if isinstance(cif_in, str) and cif_in:
            wrapped = _maybe_build_cif_pattern_from_raw(cif_in, file_path)
            if wrapped is not None:
                kwargs["cif_prepr"] = wrapped
    method = getattr(analysis, command.op_name)
    return method(**kwargs)


def parse_cif_input(cif_input: str, nexus_file: Path) -> Any:
    """Pre-load a CIF input into a reusable cache value.

    Returns:
      • ``CifPattern`` for raw .cif paths or a CIF folder (slow — simulates
        every CIF; the GUI runs this in a worker thread so the parsing
        cost is paid once, then reused for every Run-Matching).
      • ``CifPattern`` for a pickle (loaded via ``pickle.load``).

    Either result can be passed verbatim to ``mlgidBASE.run_matching`` as
    ``cif_prepr`` — ``load_cif_prepr`` accepts a ``CifPattern`` and a
    string path, so the cache is uniformly a ``CifPattern``.
    """
    import os
    import pickle

    paths = [p.strip() for p in (cif_input or "").split(";") if p.strip()]
    if not paths:
        raise ValueError("Empty CIF input")

    # Single pickle → load it once into memory. mlgidBASE would otherwise
    # re-open it on every run; not expensive but free to avoid.
    if len(paths) == 1 and paths[0].lower().endswith((".pickle", ".pkl")):
        with open(paths[0], "rb") as fh:
            return pickle.load(fh)

    # Raw .cif input goes through CifPattern construction.
    return _build_cif_pattern_from_raw(paths, nexus_file)


def _maybe_build_cif_pattern_from_raw(
    cif_in: str, nexus_file: Path
) -> Any:
    """Return a CifPattern when ``cif_in`` points at raw CIFs, else None.

    Used by ``execute`` to translate string-form ``cif_prepr`` kwargs to
    ``CifPattern`` instances when the user hasn't pre-parsed the input via
    the panel's "Parse CIFs" button. Returns ``None`` for pickle input so
    mlgidBASE's own loader sees the path. ``parse_cif_input`` is the
    panel-facing equivalent that always returns a usable cache value.
    """
    paths = [p.strip() for p in (cif_in or "").split(";") if p.strip()]
    if not paths:
        return None
    # Single pickle → forward as-is. mlgidBASE.load_cif_prepr handles it.
    if len(paths) == 1 and paths[0].lower().endswith((".pickle", ".pkl")):
        return None
    return _build_cif_pattern_from_raw(paths, nexus_file)


def _build_cif_pattern_from_raw(
    paths: list[str], nexus_file: Path
) -> Any:
    """Construct a ``CifPattern`` from a list of raw .cif paths or a folder.

    Internal helper shared by ``parse_cif_input`` (panel preload path) and
    ``_maybe_build_cif_pattern_from_raw`` (worker fallback path).
    Validates that all .cif paths share a folder (CifPattern indexes into
    a single folder) and falls back to scanning a directory's .cifs when
    that's the only path given.
    """
    import os

    # Directory → use every .cif inside.
    if len(paths) == 1 and os.path.isdir(paths[0]):
        folder_path = paths[0]
        cifs = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(".cif")
        )
        if not cifs:
            raise FileNotFoundError(
                f"No .cif files found in folder: {folder_path}"
            )
    else:
        # One-or-more raw .cif paths. They must share a folder so CifPattern
        # can locate them through ``folder_path``.
        if not all(p.lower().endswith(".cif") for p in paths):
            raise ValueError(
                "Mixed input — pass either a single pickle, a folder, or "
                "one-or-more .cif files (semicolon-separated)."
            )
        folders = {os.path.dirname(os.path.abspath(p)) for p in paths}
        if len(folders) != 1:
            raise ValueError(
                "When passing multiple .cif files, they must live in the "
                "same folder. Got: " + ", ".join(sorted(folders))
            )
        folder_path = folders.pop()
        cifs = [os.path.basename(p) for p in paths]

    from mlgidmatch.preprocess.cif_preprocess import CifPattern  # noqa: E501

    params = _exp_params_from_nexus(nexus_file)
    # ``create_all=True`` populates the 1D ring patterns (``all_patterns_q1d``
    # / ``all_patterns_int1d``) in addition to the default 2D segment data.
    # mlgidmatch's ``test_rings`` reads the 1D fields directly and crashes
    # with ``'NoneType' object is not subscriptable`` when they're None,
    # so we always populate both — the cache is built once per Parse click
    # anyway and the 1D pass is cheap relative to the 3D pattern simulation
    # that ``create_elementary`` already does.
    return CifPattern(
        params=params,
        folder_path=folder_path,
        cifs=cifs,
        create_all=True,
    )


def _exp_params_from_nexus(nexus_file: Path) -> Any:
    """Derive ExpParameters from a NeXus file's instrument metadata.

    Energy is recovered from the wavelength stored in
    ``instrument/monochromator/wavelength`` (meters); the angle of
    incidence from ``instrument/angle_of_incidence``. q-axis extents come
    from the first entry's ``data/q_xy`` / ``data/q_z`` arrays. Falls back
    to ExpParameters() defaults when any of these are missing.
    """
    from pygidsim.experiment import ExpParameters

    defaults = dict(q_xy_max=2.7, q_z_max=2.7, ai=0.3, en=18000.0)
    try:
        import h5py
        import numpy as np

        from mlgidbase_gui.file_model import is_entry_group_name

        with h5py.File(nexus_file, "r") as f:
            entry_names = sorted(
                k for k in f.keys() if is_entry_group_name(k)
            )
            if not entry_names:
                return ExpParameters(**defaults)
            entry = f[entry_names[0]]
            q_xy_max = float(np.max(np.abs(entry["data/q_xy"][()])))
            q_z_max = float(np.max(np.abs(entry["data/q_z"][()])))
            wl_m = float(np.asarray(entry["instrument/monochromator/wavelength"]).ravel()[0])
            ai = float(np.asarray(entry["instrument/angle_of_incidence"]).ravel()[0])
        # h*c / λ in eV — physical constants are exact in SI 2019.
        h = 6.62607015e-34
        c = 299792458.0
        eV = 1.602176634e-19
        en = (h * c / wl_m) / eV if wl_m > 0 else defaults["en"]
        return ExpParameters(
            q_xy_max=q_xy_max,
            q_z_max=q_z_max,
            ai=ai,
            en=en,
        )
    except Exception:
        return ExpParameters(**defaults)
