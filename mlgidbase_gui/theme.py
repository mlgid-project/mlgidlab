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


# Qt's default QComboBox reserves an icon column (PM_SmallIconSize, ~16 px)
# on every dropdown item. Combos in this app don't use icons, so the column
# shows up as a visible empty box before each item's text. Setting
# qproperty-iconSize collapses that column for every QComboBox application-
# wide. The other rules tighten item padding and zero out the (already
# unused) check indicator that qdarkstyle reserves space for.
_OVERRIDES = """
QComboBox {
    qproperty-iconSize: 0px 0px;
}
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
    image: none;
}
"""
