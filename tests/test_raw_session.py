"""Raw-file discovery, RawSession lifecycle, and path classification.

The ``list_raw_entries`` / ``RawSession`` parts are pure logic (no
QApplication); ``_classify_h5_path`` is a ``MainWindow`` method so its
test carries the ``gui`` marker. Source: file_model.py:595-651,
session.py:111-162, main_window.py:2349-2371.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlgidlab import file_model
from mlgidlab.session import RawSession


def test_list_raw_entries_applies_size_and_ndim_filter(synthetic_raw):
    entries = file_model.list_raw_entries(synthetic_raw)
    assert len(entries) == 1
    e = entries[0]
    assert e.dataset_path == "raw/data0/image"
    assert e.shape == (4, 64, 64)


def test_raw_session_open_empty_list_raises_valueerror():
    with pytest.raises(ValueError):
        RawSession.open([])


def test_raw_session_open_missing_path_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        RawSession.open([tmp_path / "does_not_exist.h5"])


def test_raw_session_open_happy(synthetic_raw):
    session = RawSession.open([synthetic_raw])
    assert session.kind == "raw"
    assert session.raw_paths == [synthetic_raw.resolve()]
    assert session.temp_path == synthetic_raw.resolve()


@pytest.mark.gui
def test_classify_h5_path(main_window, synthetic_nexus, synthetic_raw, tmp_path):
    assert main_window._classify_h5_path(synthetic_nexus) == "nexus"
    assert main_window._classify_h5_path(synthetic_raw) == "raw"

    not_h5 = tmp_path / "garbage.txt"
    not_h5.write_text("not an hdf5 file")
    assert main_window._classify_h5_path(Path(not_h5)) is None
