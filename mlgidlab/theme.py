"""Application-wide theming.

``apply_dark_theme`` is the default — uses qdarkstyle for the Qt widget
chrome and aligns pyqtgraph's defaults so plots blend with the rest of
the UI. ``apply_light_theme`` is the inverse: strips the stylesheet and
uses white/black pyqtgraph defaults so the GUI reads as a standard
light-mode Qt app.

Switched at runtime by ``MainWindow._set_theme``.
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


# Light-mode pyqtgraph defaults — white background, black axes / text.
# Matches what pyqtgraph ships out of the box; restated here so the
# light-theme switcher can revert from the dark configuration.
PG_LIGHT_BACKGROUND = "w"
PG_LIGHT_FOREGROUND = "k"


def apply_light_theme(app: QApplication) -> None:
    """Strip the qdarkstyle stylesheet + flip pyqtgraph to white.

    Existing widgets reread the global QApplication stylesheet, so
    a runtime switch is mostly seamless. pyqtgraph plots use
    ``pg.setConfigOption`` defaults at construction time — already-
    constructed plots keep their dark palette until the next
    ``setImage`` / replot, which most workflows trigger naturally
    on the next frame change or file open.
    """
    pg.setConfigOption("background", PG_LIGHT_BACKGROUND)
    pg.setConfigOption("foreground", PG_LIGHT_FOREGROUND)
    pg.setConfigOption("antialias", True)
    # Empty stylesheet → fall back to Qt's native (light) palette.
    # Keep ``_OVERRIDES`` since the QComboBox icon-column fix is
    # theme-agnostic.
    app.setStyleSheet(_OVERRIDES)


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
