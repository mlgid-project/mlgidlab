"""Application-wide dark theme.

Uses qdarkstyle for the Qt widget chrome and aligns pyqtgraph's defaults so
plots blend with the rest of the UI.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
import qdarkstyle
from PySide6.QtWidgets import QApplication

# Background / foreground tuned to match qdarkstyle's panel colour so plot
# widgets sit flush with their surrounding docks.
PG_BACKGROUND = "#19232d"
PG_FOREGROUND = "#dfe1e2"


def apply_dark_theme(app: QApplication) -> None:
    pg.setConfigOption("background", PG_BACKGROUND)
    pg.setConfigOption("foreground", PG_FOREGROUND)
    pg.setConfigOption("antialias", True)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6") + _OVERRIDES)


# qdarkstyle reserves a left column on QComboBox dropdown items for the check
# indicator, which leaves a visible gap before the text on uncheckable combos.
# Pin the indicator width to zero and shrink the row padding to recover it.
_OVERRIDES = """
QComboBox QAbstractItemView {
    padding: 0px;
}
QComboBox QAbstractItemView::item {
    padding-left: 6px;
    padding-right: 6px;
    min-height: 20px;
}
QComboBox QAbstractItemView::indicator {
    width: 0px;
    height: 0px;
}
"""
