from __future__ import annotations

import os
import sys

# Disable HDF5's SWMR file locking. silx's Hdf5TreeModel opens each
# loaded file as ``r`` and pygid (via mlgidbase) reopens the same
# path as ``r+`` for the Figure Export window's renderer; with
# default locking on, the second open fails with
# "file is already open for read-only". Setting this before any
# ``h5py`` import means neither side acquires the OS-level lock that
# would block the other. Must be set before the first ``h5py``
# import — keep it at the very top of the package init.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from PySide6.QtWidgets import QApplication

__version__ = "0.0.1"


def main() -> int:
    from mlgidlab.main_window import MainWindow
    from mlgidlab.theme import apply_dark_theme

    app = QApplication(sys.argv)
    # Set both org + app names so QSettings has a stable key path on
    # every platform (used by the Recent files menu, may grow other
    # persisted preferences over time).
    app.setOrganizationName("mlgidLAB")
    app.setApplicationName("mlgidLAB")
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()
