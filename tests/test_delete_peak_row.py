"""``file_model.delete_peak_row`` — kind-scoped peak removal that does
NOT cascade across detected / fitted / matched (the way
``mlgidbase.delete_peak`` does). Used by the GUI's "Delete peak"
flow so removing a fitted row leaves the detected row intact, and
vice versa.

Plus a coverage check on the fitted-delete + matched-clear cascade
the host wires up via ``MainWindow._delete_file_peak_scoped`` —
fitted row deletion invalidates ``matched_*`` peak_list integer
indices and the host clears ``matched_*`` on the same frame to
match the F-04 invalidate-on-refit pattern.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from mlgidlab import file_model

# Minimal peak dtype — only the fields the row-delete helper needs
# (``id`` for matching, plus a couple of geometry fields so the
# remaining rows are visibly identifiable in assertions). Mirrors
# the structured dtype layout pygid / mlgidbase write to detected /
# fitted datasets; full per-field listing lives in
# ``tests/conftest.py::PYGID_PEAK_DTYPE``.
_TEST_PEAK_DTYPE = [
    ("id", "i4"),
    ("radius", "f4"),
    ("angle", "f4"),
    ("radius_width", "f4"),
    ("angle_width", "f4"),
    ("amplitude", "f4"),
]


def _seed_file_with_three_kinds(path, *, with_matched: bool = True):
    """A frame with 3 detected, 3 fitted, and 2 matched solution
    rows. Ids are 0/1/2 in both detected and fitted (mlgidbase's
    natural convention)."""
    rng = np.random.default_rng(0)
    dt = np.dtype(_TEST_PEAK_DTYPE)
    detected = np.zeros(3, dtype=dt)
    detected["id"] = [0, 1, 2]
    detected["radius"] = [1.0, 2.0, 3.0]
    detected["angle"] = [10.0, 30.0, 80.0]
    fitted = np.zeros(3, dtype=dt)
    fitted["id"] = [0, 1, 2]
    fitted["radius"] = [1.1, 2.1, 3.1]
    fitted["angle"] = [11.0, 31.0, 81.0]
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q",
            data=rng.random((1, 8, 8), dtype=np.float32),
        )
        data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
        data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
        g = data.create_group("analysis/frame00000", track_order=True)
        g.create_dataset("detected_peaks", data=detected)
        g.create_dataset("fitted_peaks", data=fitted)
        if with_matched:
            mdtype = np.dtype(
                [("CIF", "S32"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
                 ("peak_list", "S16"), ("probability", "f4")]
            )
            mrows = np.zeros(2, dtype=mdtype)
            mrows["CIF"] = [b"a.cif", b"b.cif"]
            mrows["peak_list"] = [b"[0, 1]", b"[1, 2]"]
            mrows["probability"] = [0.9, 0.7]
            g.create_dataset("matched_segments_0000", data=mrows)
    return path


def test_delete_fitted_leaves_detected_intact(tmp_path):
    """The user's specific complaint: deleting a fitted peak via the
    GUI was wiping the detected peak with the same id (mlgidbase
    cascade). With the kind-scoped helper, only fitted_peaks is
    touched."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped.h5")
    removed = file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=1,
    )
    assert removed == 1
    with h5py.File(path, "r") as f:
        det = f["entry_0000/data/analysis/frame00000/detected_peaks"][()]
        fit = f["entry_0000/data/analysis/frame00000/fitted_peaks"][()]
    # detected is untouched — all three rows still there.
    assert sorted(det["id"].tolist()) == [0, 1, 2]
    # fitted lost row id=1 only.
    assert sorted(fit["id"].tolist()) == [0, 2]


def test_delete_detected_leaves_fitted_intact(tmp_path):
    """Symmetric: deleting a detected peak doesn't wipe the fitted."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped2.h5")
    removed = file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="detected", peak_id=2,
    )
    assert removed == 1
    with h5py.File(path, "r") as f:
        det = f["entry_0000/data/analysis/frame00000/detected_peaks"][()]
        fit = f["entry_0000/data/analysis/frame00000/fitted_peaks"][()]
    assert sorted(det["id"].tolist()) == [0, 1]
    assert sorted(fit["id"].tolist()) == [0, 1, 2]


def test_delete_preserves_unrelated_row_ids(tmp_path):
    """Unlike mlgidbase's _delete_*, this helper does NOT re-index
    ids on the surviving rows. A row with id=2 stays id=2 even when
    the row with id=1 was deleted in front of it. Keeps matched_*
    peak_list references stable (their indices into the kept rows
    are unchanged, modulo positional shifts which the host's
    matched_* remap handles via remap_matched_peak_lists)."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped3.h5", with_matched=False)
    file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=0,
    )
    with h5py.File(path, "r") as f:
        fit = f["entry_0000/data/analysis/frame00000/fitted_peaks"][()]
    # id=0 gone; the remaining rows kept their original ids 1, 2 —
    # NOT reindexed to 0, 1 (that's mlgidbase's convention, not ours).
    assert sorted(fit["id"].tolist()) == [1, 2]


def test_delete_missing_id_is_noop(tmp_path):
    """Removing an id that isn't present returns 0 and leaves the
    file untouched (no row count change, no error)."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped4.h5", with_matched=False)
    removed = file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="detected", peak_id=99,
    )
    assert removed == 0
    with h5py.File(path, "r") as f:
        det = f["entry_0000/data/analysis/frame00000/detected_peaks"][()]
    assert sorted(det["id"].tolist()) == [0, 1, 2]


def test_delete_rejects_unsupported_kind(tmp_path):
    """Matched live in per-solution datasets; the row-delete helper
    only handles detected / fitted. ``"matched"`` raises so the
    host knows to take a different path."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped5.h5")
    with pytest.raises(ValueError, match="detected or fitted"):
        file_model.delete_peak_row(
            path, "entry_0000", frame=0, kind="matched", peak_id=0,
        )


def test_delete_preserves_dataset_attrs(tmp_path):
    """Dataset is recreated under the same name with the same dtype
    and attrs; the in-place delete + recreate pattern (same as
    ``clear_peaks`` and ``add_fitted_peak_row``) must carry attrs
    forward so a downstream reader doesn't see a half-stripped
    dataset."""
    path = _seed_file_with_three_kinds(tmp_path / "scoped6.h5")
    # Stamp a marker attr.
    with h5py.File(path, "r+") as f:
        ds = f["entry_0000/data/analysis/frame00000/fitted_peaks"]
        ds.attrs["mlgid_marker"] = "kept"
    file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=1,
    )
    with h5py.File(path, "r") as f:
        ds = f["entry_0000/data/analysis/frame00000/fitted_peaks"]
        assert ds.attrs.get("mlgid_marker") == "kept"


def test_delete_nonexistent_dataset_is_noop(tmp_path):
    """When the frame's analysis group doesn't carry a fitted_peaks
    dataset at all (e.g. detection never ran), the helper returns
    0 without raising."""
    path = tmp_path / "empty_analysis.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        d = f.create_group("entry_0000/data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset(
            "img_gid_q", data=rng.random((1, 8, 8), dtype=np.float32)
        )
        d.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
        d.create_group("analysis/frame00000")  # no peaks datasets
    removed = file_model.delete_peak_row(
        path, "entry_0000", frame=0, kind="fitted", peak_id=0,
    )
    assert removed == 0
