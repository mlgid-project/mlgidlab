"""Deleting a fitted peak must keep matched structures, not wipe them.

Before: ``_delete_peaks_scoped`` cleared every ``matched_*`` row on the
frame whenever any fitted peak was deleted, because the matched
``peak_list`` stores *positions* into ``fitted_peaks`` and deleting a row
shifts them. So removing one fitted prediction made every matched
structure on the frame disappear.

Now ``file_model.remap_matched_peak_lists`` reindexes instead:
- a deleted peak is dropped from any structure that referenced it,
- the surviving indices shift down to keep pointing at the same physical
  peaks,
- structures that referenced none of the deleted peaks keep their full
  membership, and
- no structure row is removed, even if its ``peak_list`` becomes empty.

These tests drive ``file_model`` directly (no Qt): they build a NeXus
file whose ``fitted_peaks`` ids are deliberately *not* equal to their
positions (mlgidLAB keeps ids stable across deletes), and a
``matched_segments_*`` dataset in the real on-disk format
(``peak_list`` = vlen int32 positions, see
``mlgidbase.mlgidmatch_functions``).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mlgidlab import file_model

# fitted_peaks dtype as written by the pipeline (mirrors conftest's
# PYGID_PEAK_DTYPE; only the fields the readers touch matter here).
_FIT_DTYPE = np.dtype(
    [
        ("amplitude", "f4"), ("angle", "f4"), ("angle_width", "f4"),
        ("radius", "f4"), ("radius_width", "f4"), ("q_z", "f4"),
        ("q_xy", "f4"), ("theta", "f4"), ("score", "f4"),
        ("A", "f4"), ("B", "f4"), ("C", "f4"),
        ("is_ring", "bool"), ("is_cut_qz", "bool"), ("is_cut_qxy", "bool"),
        ("visibility", "i4"), ("id", "i4"),
    ]
)
_VLEN_INT = h5py.vlen_dtype(np.int32)
_MATCHED_DTYPE = np.dtype(
    [
        ("CIF", "S64"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("probability", "f4"), ("peak_list", _VLEN_INT),
    ]
)
ENTRY = "entry_0000"


def _build(
    tmp_path: Path,
    fitted_ids: list[int],
    solutions: list[tuple[str, list[int]]],
) -> Path:
    """A NeXus file with one frame: ``fitted_peaks`` carrying ``fitted_ids``
    (radius encodes the original position so peaks stay identifiable by id),
    and one ``matched_segments_0000`` dataset whose rows are ``solutions``
    as ``(cif, peak_list_positions)``."""
    path = tmp_path / "matched.h5"
    n = len(fitted_ids)
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group(f"{ENTRY}/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset("img_gid_q", data=np.zeros((1, 8, 8), np.float32))
        data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
        data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))

        fit = np.zeros(n, dtype=_FIT_DTYPE)
        fit["id"] = fitted_ids
        fit["radius"] = [float(i + 1) for i in range(n)]  # position marker
        fit["angle"] = 45.0
        fit["angle_width"] = 5.0
        fit["radius_width"] = 0.2
        g = data.create_group("analysis/frame00000", track_order=True)
        g.create_dataset("fitted_peaks", data=fit)

        rows = np.empty(len(solutions), dtype=_MATCHED_DTYPE)
        for i, (cif, plist) in enumerate(solutions):
            rows["CIF"][i] = cif.encode()
            rows["h"][i] = i
            rows["k"][i] = 0
            rows["l"][i] = 0
            rows["probability"][i] = 0.5
            rows["peak_list"][i] = np.asarray(plist, dtype=np.int32)
        g.create_dataset("matched_segments_0000", data=rows)
    return path


def _ids_per_solution(path: Path) -> list[set[int]]:
    """Fitted ids each rendered matched structure references, in file order."""
    fitted = file_model.load_peaks(path, ENTRY, 0)["fitted"]
    structs = file_model.load_matched_peaks(path, ENTRY, 0, fitted)
    return [set(int(x) for x in s.peaks.ids) for s in structs]


def _raw_peak_lists(path: Path) -> list[list[int]]:
    with h5py.File(path, "r") as f:
        ds = f[f"{ENTRY}/data/analysis/frame00000/matched_segments_0000"]
        arr = ds[()]
        return [list(np.asarray(arr["peak_list"][i], dtype=int))
                for i in range(len(arr))]


def test_positions_are_by_position_not_id(tmp_path):
    # ids deliberately != positions; id 2 sits at position 1, id 7 at 3.
    path = _build(tmp_path, [5, 2, 9, 7], [("A", [0, 1, 3])])
    assert file_model.fitted_positions_for_ids(path, ENTRY, 0, [2]) == [1]
    assert file_model.fitted_positions_for_ids(path, ENTRY, 0, [7, 5]) == [0, 3]
    assert file_model.fitted_positions_for_ids(path, ENTRY, 0, [999]) == []


def test_delete_member_drops_only_that_peak_keeps_structure(tmp_path):
    # Structures over fitted ids [5,2,9,7] at positions [0,1,2,3]:
    #   A -> positions 0,1,3  (ids 5,2,7)
    #   B -> position  1      (id 2)            -> becomes empty
    #   C -> positions 2,3    (ids 9,7)
    path = _build(
        tmp_path, [5, 2, 9, 7],
        [("A", [0, 1, 3]), ("B", [1]), ("C", [2, 3])],
    )
    # Delete fitted id 2 (position 1), mirroring _delete_peaks_scoped.
    positions = file_model.fitted_positions_for_ids(path, ENTRY, 0, [2])
    assert positions == [1]
    assert file_model.delete_peak_row(
        path, ENTRY, frame=0, kind="fitted", peak_id=2
    ) == 1
    changed = file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    # All three rows change: A drops a peak + shifts, B empties, C shifts.
    assert changed == 3

    # Raw peak_list after remap: A drops 1 and shifts 3->2; B empties; C shifts.
    assert _raw_peak_lists(path) == [[0, 2], [], [1, 2]]

    # All three rows survive (B kept though empty).
    assert len(_raw_peak_lists(path)) == 3

    ids = _ids_per_solution(path)
    # B (empty) is skipped by load_matched_peaks, so two render:
    assert ids == [{5, 7}, {9, 7}]  # A lost id 2; C unchanged membership


def test_delete_non_member_leaves_membership_but_reindexes(tmp_path):
    # Delete a peak no structure references, sitting BELOW the referenced
    # ones: membership is unchanged, but indices must shift so the same
    # physical peaks stay referenced.
    path = _build(
        tmp_path, [5, 2, 9, 7],
        [("A", [2, 3])],  # ids 9,7 — does NOT include position 0 (id 5)
    )
    positions = file_model.fitted_positions_for_ids(path, ENTRY, 0, [5])
    assert positions == [0]
    file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=5)
    file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    # 2,3 -> 1,2 (shifted down by one); still ids 9,7.
    assert _raw_peak_lists(path) == [[1, 2]]
    assert _ids_per_solution(path) == [{9, 7}]


def test_delete_below_all_no_op_for_higher_only_shifts(tmp_path):
    # A structure entirely ABOVE the deleted position shifts; one entirely
    # BELOW is byte-identical.
    path = _build(
        tmp_path, [10, 11, 12, 13, 14],
        [("low", [0, 1]), ("high", [3, 4])],
    )
    positions = file_model.fitted_positions_for_ids(path, ENTRY, 0, [12])  # pos 2
    file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=12)
    file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    assert _raw_peak_lists(path) == [[0, 1], [2, 3]]  # low unchanged; high -1
    assert _ids_per_solution(path) == [{10, 11}, {13, 14}]


def test_bulk_delete_two_positions(tmp_path):
    # Delete two fitted peaks at once (positions 1 and 3).
    path = _build(
        tmp_path, [5, 2, 9, 7, 8],
        [("A", [0, 1, 2, 3, 4])],  # all five
    )
    positions = file_model.fitted_positions_for_ids(path, ENTRY, 0, [2, 7])
    assert positions == [1, 3]
    for pid in (2, 7):
        file_model.delete_peak_row(path, ENTRY, frame=0, kind="fitted", peak_id=pid)
    file_model.remap_matched_peak_lists(path, ENTRY, 0, positions)
    # 0->0, 1 dropped, 2->1, 3 dropped, 4->2  =>  [0,1,2]
    assert _raw_peak_lists(path) == [[0, 1, 2]]
    assert _ids_per_solution(path) == [{5, 9, 8}]


def test_remap_no_op_without_matches(tmp_path):
    # No matched datasets at all: remap is a clean 0-change no-op.
    path = _build(tmp_path, [5, 2, 9], [])
    with h5py.File(path, "r+") as f:
        del f[f"{ENTRY}/data/analysis/frame00000/matched_segments_0000"]
    assert file_model.remap_matched_peak_lists(path, ENTRY, 0, [1]) == 0


def test_remap_empty_positions_is_no_op(tmp_path):
    path = _build(tmp_path, [5, 2, 9], [("A", [0, 2])])
    before = _raw_peak_lists(path)
    assert file_model.remap_matched_peak_lists(path, ENTRY, 0, []) == 0
    assert _raw_peak_lists(path) == before
