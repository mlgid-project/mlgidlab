"""Shared fixtures for the mlgidLAB smoke harness.

The environment is locked down *before* any Qt or h5py import so the
suite is headless and hermetic:

* ``QT_QPA_PLATFORM=offscreen`` — render with no display, no xvfb.
* ``XDG_CONFIG_HOME`` redirected to a temp dir — QSettings (recent
  files, theme, playback) cannot read or clobber the real user config,
  so construction starts from a clean slate every run.
* ``HDF5_USE_FILE_LOCKING=FALSE`` — mirrors ``mlgidlab.__init__.main``;
  matters once the file-open increments land, set here for parity.

These run at conftest import, which pytest executes before collecting
tests and before the pytest-qt ``qapp`` fixture constructs the
QApplication, so the offscreen platform is in place in time.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --- environment lockdown (must precede Qt / h5py import) -------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# A per-process temp config root. Kept for the whole session; the OS
# cleans /tmp, and we never want test runs sharing QSettings state.
_CONFIG_ROOT = tempfile.mkdtemp(prefix="mlgidlab-test-config-")
os.environ["XDG_CONFIG_HOME"] = _CONFIG_ROOT

import pytest  # noqa: E402


@pytest.fixture
def main_window(qtbot):
    """Construct a fresh ``MainWindow`` and guarantee teardown.

    Imported lazily inside the fixture so the environment lockdown
    above is fully applied before ``mlgidlab`` (and its Qt/h5py
    imports) is touched. ``qtbot`` ensures a QApplication exists.

    Teardown runs even if a test fails: ``close()`` exercises the real
    ``closeEvent`` shutdown path (silx detach, worker quit). With no
    session loaded it must not raise or prompt.
    """
    from mlgidlab.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    try:
        yield window
    finally:
        window.close()


@pytest.fixture
def synthetic_nexus(tmp_path):
    """A minimal valid NeXus file: 1 entry, 3 frames, no peaks.

    Matches exactly what the read path requires, verified against
    source:

    * ``file_model`` constants ``IMG_REL='data/img_gid_q'``,
      ``QXY_REL='data/q_xy'``, ``QZ_REL='data/q_z'``
      (``file_model.py:42-44``).
    * ``list_entries`` keeps an entry only if its ``data`` group has a
      ``signal`` attr equal to ``"img_gid_q"`` (``file_model.py:132``)
      and the group name passes ``is_entry_group_name``
      (``entry``/``entry_0000``/``entry_horiz`` all valid).

    The ``analysis`` group is intentionally omitted: it is written by
    ``normalize_for_pygid`` on the file-open *worker* path, which the
    tests bypass by calling ``_set_active_session`` directly. Peaks
    are optional and their absence is handled gracefully, so the
    viewer just shows empty overlays.

    Imports are local so the conftest environment lockdown is fully
    applied before h5py touches HDF5.
    """
    import h5py
    import numpy as np

    path = tmp_path / "synthetic.h5"
    n_frames, n_qz, n_qxy = 3, 16, 24
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data")
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset(
            "img_gid_q",
            data=rng.random((n_frames, n_qz, n_qxy), dtype=np.float32),
        )
        data.create_dataset(
            "q_xy", data=np.linspace(-1.0, 3.0, n_qxy, dtype=np.float32)
        )
        data.create_dataset(
            "q_z", data=np.linspace(0.0, 4.0, n_qz, dtype=np.float32)
        )
    return path
