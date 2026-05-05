from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

__version__ = "0.0.1"


def main() -> int:
    from mlgidbase_gui.main_window import MainWindow
    from mlgidbase_gui.theme import apply_dark_theme

    app = QApplication(sys.argv)
    app.setApplicationName("mlgidBASE GUI")
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()
