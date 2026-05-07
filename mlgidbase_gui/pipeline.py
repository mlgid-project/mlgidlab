"""Lazy wrappers around mlgidbase pipeline stages.

Imports of mlgidbase are deferred so the GUI can run without it installed.
No Qt imports — keep this module independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    import logging

    from mlgidbase import mlgidBASE  # noqa: N814

    # Some labeled training files (e.g. ``organic_labeled.h5``) carry
    # fitted_peaks rows that only have polar coordinates — Cartesian
    # ``q_xy`` / ``q_z`` and ``amplitude`` are stored as zeros. mlgidmatch
    # reads ``q_xy`` / ``q_z`` directly and filters by ``amplitude >
    # intensity_threshold``, so without back-filling every peak collapses
    # to the origin with zero intensity and matching silently returns
    # "no solutions". Normalise *before* opening the file via mlgidBASE
    # so the rebuilt analysis sees the patched dataset.
    if command.op_name == "run_matching":
        try:
            _backfill_fitted_peaks_polar_to_cartesian(
                file_path, command.kwargs.get("entry"),
                command.kwargs.get("frame_num"),
                logging.getLogger("mlgidBASE"),
            )
        except Exception as exc:
            logging.getLogger("mlgidBASE").warning(
                "Could not normalise fitted_peaks in %s: %s",
                file_path, exc,
            )

    analysis = mlgidBASE(filename=str(file_path))
    kwargs = dict(command.kwargs)
    # Run-matching takes a ``cif_prepr`` value that mlgidBASE accepts as
    # either a path-to-pickle string or a CifPattern instance. The GUI lets
    # the user point at raw .cif files / a folder of CIFs too — translate
    # those into a CifPattern here, on the worker thread, since pattern
    # construction simulates each CIF and can take seconds.
    if command.op_name == "run_matching":
        cif_in = kwargs.get("cif_prepr")
        active_entry = kwargs.get("entry")
        if isinstance(cif_in, str) and cif_in:
            # Build the CifPattern against the *active entry's*
            # ExpParameters so multi-energy datasets get correct
            # per-entry simulations. Otherwise we'd silently match
            # against entry_0's wavelength for every entry and yield
            # zero solutions whenever entry_0 differs from the one
            # being processed.
            wrapped = _maybe_build_cif_pattern_from_raw(
                cif_in, file_path, active_entry
            )
            if wrapped is not None:
                kwargs["cif_prepr"] = wrapped
        elif _looks_like_cif_pattern(cif_in) and active_entry is not None:
            # Cached / pre-parsed CifPattern carries fixed ExpParameters
            # from whichever entry the panel preloaded against. Surface
            # a warning when those params don't match the entry being
            # matched right now — common on per-material datasets like
            # ``organic_labeled.h5`` where each entry has its own beam
            # energy and incidence angle. The match still runs, but the
            # log line tells the user why no solutions appeared.
            try:
                _warn_if_cif_params_mismatch(
                    cif_in, file_path, active_entry,
                    logging.getLogger("mlgidBASE"),
                )
            except Exception:
                pass
    method = getattr(analysis, command.op_name)
    result = method(**kwargs)

    # Consolidate matched_<type>_NNNN groups produced by mlgidmatch.
    # mlgidmatch returns one "solution" per consistent combination of
    # structures, so the same (CIF, h, k, l, peak_list) tuple often
    # appears across many groups when one structure is shared across
    # multiple combinations — visually noisy in the GUI's matched
    # overlay. Replace the per-solution groups with a single group
    # holding one row per unique identification (highest-probability
    # instance kept).
    if command.op_name == "run_matching":
        try:
            _dedupe_matched_groups(
                file_path,
                command.kwargs.get("entry"),
                command.kwargs.get("frame_num"),
                command.kwargs.get("peaks_type", "segments"),
                logging.getLogger("mlgidBASE"),
            )
        except Exception as exc:
            logging.getLogger("mlgidBASE").warning(
                "Could not dedupe matched groups in %s: %s",
                file_path, exc,
            )

    return result


def _backfill_fitted_peaks_polar_to_cartesian(
    file_path: Path,
    entry: str | None,
    frame_num: Any,
    logger: Any,
) -> None:
    """Back-fill missing q_xy / q_z / amplitude in fitted_peaks rows.

    Some files store fitted_peaks with only the polar coordinates
    populated (``radius`` / ``angle``) and leave the Cartesian fields
    plus ``amplitude`` at zero. mlgidmatch reads q_xy / q_z directly
    and filters by ``amplitude > intensity_threshold``, so without
    this normalisation every peak ends up at (0, 0) with zero
    intensity and is dropped before any structure matching happens.

    Walks every ``data/analysis/frameNNNNN/fitted_peaks`` dataset under
    each affected entry and, *only* on rows where (q_xy == 0 AND
    q_z == 0 AND radius > 0), writes:

      - ``q_xy = radius * cos(angle_deg)``
      - ``q_z  = radius * sin(angle_deg)``

    On rows where ``amplitude == 0`` we substitute ``1.0`` so the
    default ``intensity_threshold=0`` filter (`amplitude > 0`) lets
    the row through. Rows whose Cartesian or amplitude fields are
    already non-zero are left untouched — this routine never
    overwrites real data.

    No-op when the file is already populated correctly.
    """
    import h5py
    import numpy as np

    from mlgidbase_gui.file_model import is_entry_group_name

    if entry is not None:
        entries = [entry]
    else:
        with h5py.File(file_path, "r") as f:
            entries = sorted(k for k in f.keys() if is_entry_group_name(k))

    patched_total = 0
    with h5py.File(file_path, "r+") as f:
        for ent in entries:
            analysis_grp = f.get(f"{ent}/data/analysis")
            if analysis_grp is None:
                continue
            if frame_num is None:
                frame_keys = list(analysis_grp.keys())
            elif isinstance(frame_num, (list, tuple)):
                frame_keys = [f"frame{int(n):05d}" for n in frame_num]
            else:
                frame_keys = [f"frame{int(frame_num):05d}"]

            for fk in frame_keys:
                ds_path = f"{ent}/data/analysis/{fk}/fitted_peaks"
                if ds_path not in f:
                    continue
                ds = f[ds_path]
                fp = ds[()]
                # Identify rows that need back-fill. q_xy/q_z exactly 0
                # is suspicious only when radius is meaningfully > 0;
                # leave genuine origin peaks alone.
                need_xy = (
                    (fp["q_xy"] == 0.0)
                    & (fp["q_z"] == 0.0)
                    & (fp["radius"] > 0.0)
                )
                need_amp = fp["amplitude"] == 0.0
                if not need_xy.any() and not need_amp.any():
                    continue
                if need_xy.any():
                    ang_rad = np.deg2rad(fp["angle"][need_xy])
                    fp["q_xy"][need_xy] = fp["radius"][need_xy] * np.cos(ang_rad)
                    fp["q_z"][need_xy] = fp["radius"][need_xy] * np.sin(ang_rad)
                if need_amp.any():
                    # Placeholder unit amplitude so intensity-filtered
                    # matching still sees the peak. Real amplitudes
                    # require re-running run_fitting against the image.
                    fp["amplitude"][need_amp] = 1.0
                ds[...] = fp
                patched_total += int(max(need_xy.sum(), need_amp.sum()))
                logger.warning(
                    "fitted_peaks @ %s/%s: back-filled %d row(s) where "
                    "Cartesian q_xy/q_z were zero (computed from polar) "
                    "and %d row(s) with zero amplitude (set to 1.0 as a "
                    "placeholder so the intensity filter does not drop "
                    "them). Re-run fitting for real amplitudes.",
                    ent, fk, int(need_xy.sum()), int(need_amp.sum()),
                )
    if patched_total:
        logger.info(
            "Normalised fitted_peaks in %s: %d row(s) patched across "
            "%d entry(ies). Matching can now proceed.",
            file_path, patched_total, len(entries),
        )


def _dedupe_matched_groups(
    file_path: Path,
    entry: str | None,
    frame_num: Any,
    peaks_type: str,
    logger: Any,
) -> None:
    """Collapse repeated structure identifications across matched groups.

    mlgidmatch emits one ``matched_<type>_NNNN`` group per "solution"
    (a self-consistent combination of structures). When a single
    structure participates in many combinations, its
    (CIF, h, k, l, peak_list) row gets written into every one of those
    groups, cluttering the saved data and the GUI's matched overlay.

    This helper, run after each ``run_matching`` call, walks the
    affected ``data/analysis/frameNNNNN`` groups, gathers every row
    across all ``matched_<peaks_type>_*`` datasets, deduplicates by
    the (CIF, h, k, l, sorted-peak-list) key (keeping the highest
    probability), and rewrites the result as a single
    ``matched_<peaks_type>_0000`` dataset. The original groups are
    deleted in the same write pass.

    No-op when at most one group exists already.
    """
    import h5py
    import numpy as np

    from mlgidbase_gui.file_model import is_entry_group_name

    prefix = f"matched_{peaks_type}_"

    if entry is not None:
        entries = [entry]
    else:
        with h5py.File(file_path, "r") as f:
            entries = sorted(k for k in f.keys() if is_entry_group_name(k))

    consolidated_total = 0
    with h5py.File(file_path, "r+") as f:
        for ent in entries:
            analysis_grp = f.get(f"{ent}/data/analysis")
            if analysis_grp is None:
                continue
            if frame_num is None:
                frame_keys = list(analysis_grp.keys())
            elif isinstance(frame_num, (list, tuple)):
                frame_keys = [f"frame{int(n):05d}" for n in frame_num]
            else:
                frame_keys = [f"frame{int(frame_num):05d}"]

            for fk in frame_keys:
                frame_grp = analysis_grp.get(fk)
                if frame_grp is None:
                    continue
                matched_keys = sorted(
                    k for k in frame_grp.keys() if k.startswith(prefix)
                )
                if len(matched_keys) <= 1:
                    continue

                # Pick a reference dtype so the consolidated dataset
                # round-trips through pygid's reader unchanged.
                ref_ds = frame_grp[matched_keys[0]]
                ref_dtype = ref_ds.dtype

                # Map (cif, h, k, l, peak_list_tuple) -> best row.
                best: dict[tuple, Any] = {}
                row_count_in = 0
                for mk in matched_keys:
                    rows = frame_grp[mk][()]
                    row_count_in += int(rows.shape[0])
                    for row in rows:
                        peak_arr = np.asarray(row["peak_list"]).astype(np.int32)
                        key = (
                            bytes(row["CIF"]),
                            int(row["h"]), int(row["k"]), int(row["l"]),
                            tuple(np.sort(peak_arr).tolist()),
                        )
                        prev = best.get(key)
                        if prev is None or float(row["probability"]) > float(prev["probability"]):
                            # Copy the row so freeing the source array
                            # doesn't take the data with it.
                            best[key] = np.asarray(row, dtype=ref_dtype).copy()

                if not best:
                    continue

                # Sort consolidated rows by probability descending so
                # the highest-confidence identifications come first.
                ordered = sorted(
                    best.values(),
                    key=lambda r: float(r["probability"]),
                    reverse=True,
                )
                consolidated = np.empty(len(ordered), dtype=ref_dtype)
                for i, row in enumerate(ordered):
                    consolidated[i] = row

                # Delete every existing matched_<type>_NNNN dataset
                # before writing the consolidated one — pygid's
                # _save_matched_data uses the same delete-then-write
                # pattern so this stays consistent with the rest of
                # the codebase.
                for mk in matched_keys:
                    del frame_grp[mk]
                frame_grp.create_dataset(f"{prefix}0000", data=consolidated)
                consolidated_total += row_count_in - len(ordered)
                logger.info(
                    "Consolidated %d matched %s rows from %d groups into "
                    "1 group with %d unique identifications at %s/%s.",
                    row_count_in, peaks_type, len(matched_keys),
                    len(ordered), ent, fk,
                )
    if consolidated_total:
        logger.info(
            "Removed %d duplicate matched-row(s) total in %s.",
            consolidated_total, file_path,
        )


def _looks_like_cif_pattern(obj: Any) -> bool:
    """True if ``obj`` quacks like a ``CifPattern`` — has a ``params``
    attribute exposing ``en`` / ``ai``. Avoids importing mlgidmatch
    just for an isinstance check."""
    if obj is None:
        return False
    params = getattr(obj, "params", None)
    return params is not None and hasattr(params, "en") and hasattr(params, "ai")


def _warn_if_cif_params_mismatch(
    cif_pattern: Any, nexus_file: Path, entry: str, logger: Any,
) -> None:
    """Emit a warning when the cached CifPattern's params disagree with
    the active entry's params. Threshold: >0.5% relative on either
    energy or angle of incidence (well above HDF5 round-trip noise)."""
    cur = _exp_params_from_nexus(nexus_file, entry)
    cached_en = float(getattr(cif_pattern.params, "en", 0.0))
    cached_ai = float(getattr(cif_pattern.params, "ai", 0.0))
    cur_en = float(getattr(cur, "en", 0.0))
    cur_ai = float(getattr(cur, "ai", 0.0))
    en_ok = cached_en > 0 and abs(cached_en - cur_en) / cached_en < 0.005
    ai_ok = (
        abs(cached_ai - cur_ai) < 1e-3
        if abs(cached_ai) < 1e-3
        else abs(cached_ai - cur_ai) / abs(cached_ai) < 0.05
    )
    if not (en_ok and ai_ok):
        logger.warning(
            "Pre-parsed CifPattern was built against ExpParameters "
            "(en=%.0f eV, ai=%.4f) that don't match entry %r "
            "(en=%.0f eV, ai=%.4f). Pattern simulation is using the "
            "wrong wavelength/incidence — re-parse CIFs with this "
            "entry active, or run matching from the raw CIF folder so "
            "each entry gets its own simulation.",
            cached_en, cached_ai, entry, cur_en, cur_ai,
        )


def parse_cif_input(
    cif_input: str, nexus_file: Path, entry: str | None = None
) -> Any:
    """Pre-load a CIF input into a reusable cache value.

    Returns:
      • ``CifPattern`` for raw .cif paths or a CIF folder (slow — simulates
        every CIF; the GUI runs this in a worker thread so the parsing
        cost is paid once, then reused for every Run-Matching).
      • ``CifPattern`` for a pickle (loaded via ``pickle.load``).

    Either result can be passed verbatim to ``mlgidBASE.run_matching`` as
    ``cif_prepr`` — ``load_cif_prepr`` accepts a ``CifPattern`` and a
    string path, so the cache is uniformly a ``CifPattern``.

    For raw .cif input, pass ``entry`` to bind the simulation to one
    specific entry's ExpParameters — the cached pattern is then only
    valid for that entry. The panel uses this so caches stay in sync
    with the active entry on multi-energy datasets.
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
    return _build_cif_pattern_from_raw(paths, nexus_file, entry)


def _maybe_build_cif_pattern_from_raw(
    cif_in: str, nexus_file: Path, entry: str | None = None
) -> Any:
    """Return a CifPattern when ``cif_in`` points at raw CIFs, else None.

    Used by ``execute`` to translate string-form ``cif_prepr`` kwargs to
    ``CifPattern`` instances when the user hasn't pre-parsed the input via
    the panel's "Parse CIFs" button. Returns ``None`` for pickle input so
    mlgidBASE's own loader sees the path. ``parse_cif_input`` is the
    panel-facing equivalent that always returns a usable cache value.

    ``entry`` is forwarded to ``_build_cif_pattern_from_raw`` so the
    simulation uses the per-entry ExpParameters; vital for files with
    mixed-energy entries.
    """
    paths = [p.strip() for p in (cif_in or "").split(";") if p.strip()]
    if not paths:
        return None
    # Single pickle → forward as-is. mlgidBASE.load_cif_prepr handles it.
    if len(paths) == 1 and paths[0].lower().endswith((".pickle", ".pkl")):
        return None
    return _build_cif_pattern_from_raw(paths, nexus_file, entry)


def _build_cif_pattern_from_raw(
    paths: list[str], nexus_file: Path, entry: str | None = None
) -> Any:
    """Construct a ``CifPattern`` from a list of raw .cif paths or a folder.

    Internal helper shared by ``parse_cif_input`` (panel preload path) and
    ``_maybe_build_cif_pattern_from_raw`` (worker fallback path).
    Validates that all .cif paths share a folder (CifPattern indexes into
    a single folder) and falls back to scanning a directory's .cifs when
    that's the only path given.

    ``entry`` selects which NeXus entry's instrument metadata is used
    when computing ExpParameters — required for multi-entry files
    where energies differ across entries.
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

    params = _exp_params_from_nexus(nexus_file, entry)
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


def _exp_params_from_nexus(
    nexus_file: Path, entry: str | None = None
) -> Any:
    """Derive ExpParameters from a NeXus file's instrument metadata.

    Energy is recovered from the wavelength stored in
    ``instrument/monochromator/wavelength`` (meters); the angle of
    incidence from ``instrument/angle_of_incidence``. q-axis extents
    come from ``data/q_xy`` / ``data/q_z``.

    Pass ``entry`` to read from a specific entry — multi-entry files
    where each entry was collected at a different beam energy / angle
    of incidence (e.g. ``organic_labeled.h5`` with per-material
    entries) need per-entry params, otherwise CIF pattern simulation
    runs against the wrong wavelength and matching silently returns
    "no solutions". When ``entry`` is None, the alphabetically-first
    entry is used as a fallback (preserves previous behaviour for
    callers that don't yet plumb through an entry name). Falls back
    to ExpParameters() defaults when any value is missing.
    """
    from pygidsim.experiment import ExpParameters

    defaults = dict(q_xy_max=2.7, q_z_max=2.7, ai=0.3, en=18000.0)
    try:
        import h5py
        import numpy as np

        from mlgidbase_gui.file_model import is_entry_group_name

        with h5py.File(nexus_file, "r") as f:
            if entry is not None and entry in f:
                entry_name = entry
            else:
                entry_names = sorted(
                    k for k in f.keys() if is_entry_group_name(k)
                )
                if not entry_names:
                    return ExpParameters(**defaults)
                entry_name = entry_names[0]
            grp = f[entry_name]
            q_xy_max = float(np.max(np.abs(grp["data/q_xy"][()])))
            q_z_max = float(np.max(np.abs(grp["data/q_z"][()])))
            wl_m = float(np.asarray(grp["instrument/monochromator/wavelength"]).ravel()[0])
            ai = float(np.asarray(grp["instrument/angle_of_incidence"]).ravel()[0])
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
