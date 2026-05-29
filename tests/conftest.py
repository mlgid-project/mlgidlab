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

import sys  # noqa: E402

import pytest  # noqa: E402

import pytestqt.qtbot as _qtbot_mod  # noqa: E402
_PIN=[]
_oaw=_qtbot_mod._add_widget
def _paw(item,widget,**kw):
    _PIN.append(widget); _oaw(item,widget,**kw)
_qtbot_mod._add_widget=_paw

# Real pytest exit status, captured at session end and used by
# pytest_unconfigure to hard-exit before the crashy native teardown.
_PYTEST_EXIT_STATUS = 0


def pytest_sessionfinish(session, exitstatus):
    global _PYTEST_EXIT_STATUS
    _PYTEST_EXIT_STATUS = int(exitstatus)


def pytest_unconfigure(config):
    """Skip the crashy native interpreter teardown.

    On headless CI the PySide6 + silx(OpenGL) + h5py C++ stack tears
    down its static singletons in an order that SIGSEGVs at interpreter
    exit *after* a clean run: pytest prints ``N passed`` and returns 0,
    then the process dies with exit code 139. It reproduces on every
    Python (3.11-3.14) and never locally (a real GL/display masks it),
    so it is not a test failure and faulthandler cannot catch it (it is
    gone by then).

    ``pytest_unconfigure`` is the final hook, run *after* the terminal
    reporter has printed the summary, so no output is lost. We flush and
    ``os._exit`` with pytest's real status (captured in
    ``pytest_sessionfinish``): a genuine failure still exits non-zero
    (CI stays honest), only the post-success native teardown is
    bypassed. ``os._exit`` skips Python atexit/cleanup, which is safe
    for a short-lived test process (no coverage plugin in the dev
    deps).
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_PYTEST_EXIT_STATUS)


# The peak structured dtype, kept as a plain list of (name, fmt) tuples
# so numpy stays out of module scope (the env lockdown above must apply
# before any numpy/h5py import). Field set + order is the ground truth
# from mlgidLAB itself, NOT pygid (absent from this checkout): it mirrors
# the ``fields`` dict written by ``add_fitted_peak_row``
# (file_model.py:840-858) and the names read by
# ``PeakTable.from_dataset`` (file_model.py:93-104).
PYGID_PEAK_DTYPE = [
    ("amplitude", "f4"),
    ("angle", "f4"),
    ("angle_width", "f4"),
    ("radius", "f4"),
    ("radius_width", "f4"),
    ("q_z", "f4"),
    ("q_xy", "f4"),
    ("theta", "f4"),
    ("score", "f4"),
    ("A", "f4"),
    ("B", "f4"),
    ("C", "f4"),
    ("is_ring", "bool"),
    ("is_cut_qz", "bool"),
    ("is_cut_qxy", "bool"),
    ("visibility", "i4"),
    ("id", "i4"),
]


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


@pytest.fixture
def synthetic_nexus_with_peaks(tmp_path):
    """A valid NeXus file (1 entry, 3 frames) plus a populated analysis
    tree on ``frame00000`` only.

    Self-contained — does not chain off ``synthetic_nexus`` — so the
    analysis groups can be written in the same ``h5py.File`` open. The
    layout matches what the read path expects, verified against source:

    * group path ``entry_0000/data/analysis/frame00000/<kind>_peaks``
      (``ANALYSIS_REL`` / ``FRAME_KEY_FMT``, file_model.py:21,44).
    * the structured dtype is ``PYGID_PEAK_DTYPE`` (see its comment).
    * frames 1 & 2 get *no* analysis group, so ``load_peaks`` returns
      ``{detected:None, fitted:None}`` (file_model.py:677-678) and
      ``add_fitted_peak_row`` on them raises ``KeyError``
      (file_model.py:829-833).

    Imports are local so the conftest environment lockdown is fully
    applied before h5py touches HDF5.
    """
    import h5py
    import numpy as np

    path = tmp_path / "synthetic_peaks.h5"
    n_frames, n_qz, n_qxy = 3, 16, 24
    dt = np.dtype(PYGID_PEAK_DTYPE)
    rng = np.random.default_rng(0)

    detected = np.zeros(3, dtype=dt)
    detected["id"] = [0, 1, 2]
    detected["score"] = [0.40, 0.75, 1.00]
    detected["amplitude"] = [10.0, 20.0, 30.0]
    detected["angle"] = [10.0, 45.0, 80.0]
    detected["radius"] = [1.0, 2.0, 3.0]
    detected["angle_width"] = [5.0, 5.0, 5.0]
    detected["radius_width"] = [0.2, 0.2, 0.2]
    detected["q_xy"] = detected["radius"] * np.cos(np.deg2rad(detected["angle"]))
    detected["q_z"] = detected["radius"] * np.sin(np.deg2rad(detected["angle"]))

    fitted = np.zeros(2, dtype=dt)
    fitted["id"] = [0, 1]
    fitted["score"] = [0.50, 0.90]
    fitted["amplitude"] = [12.0, 22.0]
    fitted["angle"] = [20.0, 60.0]
    fitted["radius"] = [1.5, 2.5]
    fitted["angle_width"] = [4.0, 4.0]
    fitted["radius_width"] = [0.3, 0.3]
    fitted["q_xy"] = fitted["radius"] * np.cos(np.deg2rad(fitted["angle"]))
    fitted["q_z"] = fitted["radius"] * np.sin(np.deg2rad(fitted["angle"]))

    fitted_errors = np.zeros(0, dtype=dt)

    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group("entry_0000/data", track_order=True)
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
        g = data.create_group("analysis/frame00000", track_order=True)
        g.create_dataset("detected_peaks", data=detected)
        g.create_dataset("fitted_peaks", data=fitted)
        g.create_dataset("fitted_peaks_errors", data=fitted_errors)
    return path


@pytest.fixture
def synthetic_raw(tmp_path):
    """A raw HDF5 file with one qualifying 3-D detector dataset.

    ``list_raw_entries`` keeps a dataset only if it is 3-D with both
    spatial dims ≥ ``RAW_MIN_DETECTOR_HW`` (==32) and a numeric dtype
    (file_model.py:595,634-640). This file deliberately includes two
    datasets that must be *filtered out* so the size / ndim guards are
    exercised, and carries no ``signal`` attr so ``_classify_h5_path``
    reads it as ``raw`` (not ``nexus``).

    Imports are local so the conftest environment lockdown is fully
    applied before h5py touches HDF5.
    """
    import h5py
    import numpy as np

    path = tmp_path / "synthetic_raw.h5"
    rng = np.random.default_rng(1)
    with h5py.File(path, "w", track_order=True) as f:
        f.create_dataset(
            "raw/data0/image",
            data=rng.integers(0, 1000, size=(4, 64, 64), dtype=np.uint32),
        )
        # Filtered: spatial dims 16 < RAW_MIN_DETECTOR_HW (32).
        f.create_dataset(
            "raw/small",
            data=rng.integers(0, 100, size=(2, 16, 16), dtype=np.uint16),
        )
        # Filtered: ndim != 3.
        f.create_dataset(
            "raw/flat",
            data=rng.integers(0, 100, size=(64, 64), dtype=np.uint16),
        )
    return path
