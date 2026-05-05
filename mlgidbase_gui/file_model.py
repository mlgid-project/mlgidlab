"""Pure-Python readers for the mlgidBASE NeXus file format.

Schema reference: project_repos/mlgidBASE/docs/tutorials/output_file_format.md
No Qt imports — keep this module independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

ENTRY_PREFIX = "entry_"
FRAME_KEY_FMT = "frame{:05d}"
ANALYSIS_REL = "data/analysis"
IMG_REL = "data/img_gid_q"
QXY_REL = "data/q_xy"
QZ_REL = "data/q_z"

PeakKind = str  # "detected" | "fitted"


@dataclass
class PeakTable:
    """Centers, widths, and ids for a set of peaks at one frame.

    Both Cartesian (q_xy, q_z) and polar (angle, radius) coordinates are kept
    because the peak operations API exposes both.
    """

    q_xy: np.ndarray
    q_z: np.ndarray
    angle: np.ndarray
    radius: np.ndarray
    angle_width: np.ndarray
    radius_width: np.ndarray
    is_ring: np.ndarray
    ids: np.ndarray

    @classmethod
    def from_dataset(cls, ds: h5py.Dataset) -> PeakTable:
        arr = ds[()]
        return cls(
            q_xy=np.asarray(arr["q_xy"], dtype=float),
            q_z=np.asarray(arr["q_z"], dtype=float),
            angle=np.asarray(arr["angle"], dtype=float),
            radius=np.asarray(arr["radius"], dtype=float),
            angle_width=np.asarray(arr["angle_width"], dtype=float),
            radius_width=np.asarray(arr["radius_width"], dtype=float),
            is_ring=np.asarray(arr["is_ring"], dtype=bool),
            ids=np.asarray(arr["id"], dtype=int),
        )

    def __len__(self) -> int:
        return int(self.ids.size)


@dataclass
class EntryStack:
    """Image stack + reciprocal-space axes for one entry."""

    image_stack: np.ndarray  # shape (n_frames, n_qz, n_qxy)
    q_xy: np.ndarray         # shape (n_qxy,)
    q_z: np.ndarray          # shape (n_qz,)

    @property
    def n_frames(self) -> int:
        return int(self.image_stack.shape[0])


def list_entries(file_path: Path) -> list[str]:
    """Return entry group names like ['entry_0000', 'entry_0001'] in sorted order."""
    with h5py.File(file_path, "r") as f:
        return sorted(name for name in f if name.startswith(ENTRY_PREFIX))


def load_entry(file_path: Path, entry: str) -> EntryStack:
    """Load the full image stack and axes for one entry."""
    with h5py.File(file_path, "r") as f:
        img = np.asarray(f[f"{entry}/{IMG_REL}"][()])
        q_xy = np.asarray(f[f"{entry}/{QXY_REL}"][()], dtype=float)
        q_z = np.asarray(f[f"{entry}/{QZ_REL}"][()], dtype=float)
    return EntryStack(image_stack=img, q_xy=q_xy, q_z=q_z)


def load_peaks(
    file_path: Path, entry: str, frame: int
) -> dict[PeakKind, PeakTable | None]:
    """Read detected/fitted peak tables for one frame; missing tables → None."""
    frame_key = FRAME_KEY_FMT.format(frame)
    out: dict[PeakKind, PeakTable | None] = {"detected": None, "fitted": None}
    with h5py.File(file_path, "r") as f:
        group_path = f"{entry}/{ANALYSIS_REL}/{frame_key}"
        if group_path not in f:
            return out
        group = f[group_path]
        if "detected_peaks" in group:
            out["detected"] = PeakTable.from_dataset(group["detected_peaks"])
        if "fitted_peaks" in group:
            out["fitted"] = PeakTable.from_dataset(group["fitted_peaks"])
    return out


@dataclass
class MatchedStructure:
    """One matched crystal structure within a frame.

    Each ``matched_*`` NeXus dataset can contain several rows — multi-phase
    solutions — and each row corresponds to one ``MatchedStructure``. The
    ``peaks`` table is a subset of the frame's ``fitted_peaks`` taken at the
    indices listed in ``peak_list``, so the existing peak-rendering helpers
    can draw matched peaks without any special-casing.
    """

    solution_field: str  # e.g. "matched_segments_0000"
    local_idx: int       # row within ``solution_field``
    cif: str             # CIF basename (no extension)
    h: int
    k: int
    l: int
    probability: float
    peaks: PeakTable
    # Indices into the frame's ``fitted_peaks`` table that produced ``peaks``.
    # Kept so matched overlays can be re-derived after an in-memory edit of a
    # fitted peak without a second file read.
    peak_list: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def unique_id(self) -> str:
        """Stable handle within one frame."""
        return f"{self.solution_field}/{self.local_idx}"

    @property
    def label(self) -> str:
        ori = "rand" if (self.h, self.k, self.l) == (0, 0, 0) else (
            f"({self.h}{self.k}{self.l})"
        )
        return f"{self.cif} {ori}  p={self.probability:.2f}"


def load_matched_peaks(
    file_path: Path,
    entry: str,
    frame: int,
    fitted_peaks: PeakTable | None,
) -> list[MatchedStructure]:
    """Read all ``matched_*`` solutions for a frame.

    Each row of each ``matched_*`` dataset becomes one ``MatchedStructure``.
    Returns an empty list when the file has no matches for this frame, or when
    ``fitted_peaks`` is None — peak_list indices reference the frame's
    fitted_peaks, so without them the matched peaks have no geometry.
    """
    if fitted_peaks is None or len(fitted_peaks) == 0:
        return []
    frame_key = FRAME_KEY_FMT.format(frame)
    n_fit = len(fitted_peaks)
    out: list[MatchedStructure] = []
    with h5py.File(file_path, "r") as f:
        group_path = f"{entry}/{ANALYSIS_REL}/{frame_key}"
        if group_path not in f:
            return out
        group = f[group_path]
        for name in sorted(group.keys()):
            if not name.startswith("matched_"):
                continue
            ds = group[name]
            arr = ds[()]
            for i in range(len(arr)):
                cif_raw = arr["CIF"][i]
                cif_str = (
                    cif_raw.decode("utf-8", errors="replace")
                    if isinstance(cif_raw, bytes)
                    else str(cif_raw)
                )
                # Strip the .cif extension for display.
                if cif_str.lower().endswith(".cif"):
                    cif_str = cif_str[:-4]
                idx = np.asarray(arr["peak_list"][i], dtype=int)
                # Tolerate stale matches: drop indices that no longer exist
                # in fitted_peaks (e.g. fitting was re-run after matching).
                idx = idx[(idx >= 0) & (idx < n_fit)]
                if idx.size == 0:
                    continue
                subset = PeakTable(
                    q_xy=fitted_peaks.q_xy[idx],
                    q_z=fitted_peaks.q_z[idx],
                    angle=fitted_peaks.angle[idx],
                    radius=fitted_peaks.radius[idx],
                    angle_width=fitted_peaks.angle_width[idx],
                    radius_width=fitted_peaks.radius_width[idx],
                    is_ring=fitted_peaks.is_ring[idx],
                    ids=fitted_peaks.ids[idx],
                )
                out.append(
                    MatchedStructure(
                        solution_field=name,
                        local_idx=i,
                        cif=cif_str,
                        h=int(arr["h"][i]),
                        k=int(arr["k"][i]),
                        l=int(arr["l"][i]),
                        probability=float(arr["probability"][i]),
                        peaks=subset,
                        peak_list=idx,
                    )
                )
    return out


def add_fitted_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    *,
    angle: float,
    angle_width: float,
    radius: float,
    radius_width: float,
    amplitude: float,
    is_ring: bool = False,
    theta: float = 0.0,
    score: float = 0.0,
    A: float = 0.0,
    B: float = 0.0,
    C: float = 0.0,
    is_cut_qz: bool = False,
    is_cut_qxy: bool = False,
    visibility: int = 0,
) -> int:
    """Append a new row to the frame's ``fitted_peaks`` dataset.

    Mirrors mlgidbase's row layout (see pygid.datasaver pygid_results_dtype):
    polar geometry + amplitude + 2D-Gaussian shape params (A, B, C, theta)
    + flags + an auto-assigned id. q_xy/q_z are recomputed from polar.

    Caller-supplied fields drive only the meaningful values; the 2D-shape
    params (A/B/C/theta) and score default to zero because a 1D-fit-derived
    row carries no 2D context. Returns the new peak's ``id``.
    """
    frame_key = FRAME_KEY_FMT.format(frame)
    with h5py.File(file_path, "r+") as f:
        ds_path = f"{entry}/{ANALYSIS_REL}/{frame_key}/fitted_peaks"
        if ds_path not in f:
            raise KeyError(
                f"fitted_peaks dataset missing at {ds_path} — run fitting "
                "at least once on this frame before adding manual fitted rows."
            )
        ds = f[ds_path]
        arr = ds[()]
        new_id = int(arr["id"].max() + 1) if len(arr) > 0 else 0
        # Build the new row by name to stay schema-resilient if the dtype
        # ever gains/loses a field.
        new_row = np.zeros(1, dtype=arr.dtype)
        fields = {
            "amplitude": amplitude,
            "angle": angle,
            "angle_width": angle_width,
            "radius": radius,
            "radius_width": radius_width,
            "q_z": radius * np.sin(np.deg2rad(angle)),
            "q_xy": radius * np.cos(np.deg2rad(angle)),
            "theta": theta,
            "score": score,
            "A": A,
            "B": B,
            "C": C,
            "is_ring": is_ring,
            "is_cut_qz": is_cut_qz,
            "is_cut_qxy": is_cut_qxy,
            "visibility": visibility,
            "id": new_id,
        }
        for name, value in fields.items():
            if name in arr.dtype.names:
                new_row[name] = value
        new_arr = np.concatenate([arr, new_row])
        # pygid creates fixed-shape datasets (maxshape == shape), so resize
        # is unavailable — delete and recreate, preserving any attrs.
        attrs = dict(ds.attrs)
        del f[ds_path]
        new_ds = f.create_dataset(ds_path, data=new_arr)
        for k, v in attrs.items():
            new_ds.attrs[k] = v
        return new_id


def update_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    kind: str,
    peak_id: int,
    *,
    angle: float,
    angle_width: float,
    radius: float,
    radius_width: float,
) -> None:
    """Mutate one row of detected_peaks/fitted_peaks in place, keyed by `id`.

    The id field is unique within a frame (mlgidbase reindexes on delete), so
    finding a row by id is unambiguous. Polar fields are written verbatim;
    q_xy/q_z are recomputed from the new (radius, angle).

    Raises KeyError when no row with ``peak_id`` exists — the caller should
    treat this as "the file moved on under us" and clear the undo stack.
    """
    if kind not in ("detected", "fitted"):
        raise ValueError(f"update_peak_row only handles detected/fitted, not {kind!r}")
    ds_name = f"{kind}_peaks"
    frame_key = FRAME_KEY_FMT.format(frame)
    with h5py.File(file_path, "r+") as f:
        ds = f[f"{entry}/{ANALYSIS_REL}/{frame_key}/{ds_name}"]
        arr = ds[()]
        matches = np.where(arr["id"] == peak_id)[0]
        if matches.size == 0:
            raise KeyError(
                f"peak_id={peak_id} not found in {entry}/{frame_key}/{ds_name}"
            )
        idx = int(matches[0])
        arr["radius"][idx] = radius
        arr["radius_width"][idx] = radius_width
        arr["angle"][idx] = angle
        arr["angle_width"][idx] = angle_width
        arr["q_xy"][idx] = radius * np.cos(np.deg2rad(angle))
        arr["q_z"][idx] = radius * np.sin(np.deg2rad(angle))
        ds[...] = arr
