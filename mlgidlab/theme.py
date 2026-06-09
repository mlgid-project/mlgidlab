"""Application-wide theming.

Both themes apply a full qdarkstyle stylesheet — ``DarkPalette`` for
dark, ``LightPalette`` for light — so the look is **independent of the
host's Qt palette**. The previous light theme set an empty stylesheet
and fell back to the OS palette, which on a dark-desktop machine left
"light mode" looking dark. Each also pushes matching pyqtgraph
background / foreground defaults so plots blend with the chrome.

Switched at runtime by ``MainWindow._set_theme`` (which also forces a
live re-polish and refreshes existing plot colours — config options
below only affect *newly* created pg items).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
import qdarkstyle
from qdarkstyle.dark.palette import DarkPalette
from qdarkstyle.light.palette import LightPalette
from PySide6.QtWidgets import QApplication

# pyqtgraph background / foreground per theme. Dark matches qdarkstyle's
# panel colour (#19232d); light matches its panel colour (#fafafa) so the
# plots sit flush with the surrounding docks in either theme.
PG_DARK_BACKGROUND = "#19232d"
PG_DARK_FOREGROUND = "#dfe1e2"
PG_LIGHT_BACKGROUND = "#fafafa"
PG_LIGHT_FOREGROUND = "#000000"

# Look-up used by the runtime switcher to recolour already-built plots.
PG_COLORS = {
    "dark": (PG_DARK_BACKGROUND, PG_DARK_FOREGROUND),
    "light": (PG_LIGHT_BACKGROUND, PG_LIGHT_FOREGROUND),
}


def pg_colors(theme: str) -> tuple[str, str]:
    """``(background, foreground)`` for ``"dark"`` / ``"light"``."""
    return PG_COLORS.get(theme, PG_COLORS["dark"])


def _apply(app: QApplication, *, palette, background: str, foreground: str) -> None:
    pg.setConfigOption("background", background)
    pg.setConfigOption("foreground", foreground)
    pg.setConfigOption("antialias", True)
    app.setStyleSheet(
        qdarkstyle.load_stylesheet(qt_api="pyside6", palette=palette) + _OVERRIDES
    )


def apply_dark_theme(app: QApplication) -> None:
    _apply(
        app,
        palette=DarkPalette,
        background=PG_DARK_BACKGROUND,
        foreground=PG_DARK_FOREGROUND,
    )


def apply_light_theme(app: QApplication) -> None:
    """Apply qdarkstyle's **LightPalette** stylesheet + light pyqtgraph
    defaults. A real light theme, not a strip-to-OS-default fallback, so
    it reads as light on every desktop."""
    _apply(
        app,
        palette=LightPalette,
        background=PG_LIGHT_BACKGROUND,
        foreground=PG_LIGHT_FOREGROUND,
    )


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
