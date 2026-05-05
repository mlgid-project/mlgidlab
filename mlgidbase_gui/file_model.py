"""Pure-Python readers for the mlgidBASE NeXus file format.

Schema reference: project_repos/mlgidBASE/docs/tutorials/output_file_format.md
No Qt imports — keep this module independently testable.
"""
from __future__ import annotations

from dataclasses import dataclass
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
                    )
                )
    return out
