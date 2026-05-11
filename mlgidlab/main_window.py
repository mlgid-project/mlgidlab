from __future__ import annotations

import json
import math
from pathlib import Path

from PySide6.QtCore import (
    QCoreApplication,
    QSettings,
    QSignalBlocker,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QFrame,
    QPlainTextEdit,
    QProgressDialog,
    QRadioButton,
    QDoubleSpinBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.hdf5 import Hdf5TreeModel, Hdf5TreeView
from silx.gui.hdf5.NexusSortFilterProxyModel import NexusSortFilterProxyModel

from mlgidlab import file_model
from mlgidlab.image_viewer import (
    GIWAXSImageViewer,
    MATCHED_STYLE,
    ManualPeak,
    OVERLAY_KINDS,
    OVERLAY_STYLE,
    SelectedPeak,
    matched_pen_for,
)
from mlgidlab.parameter_panel import ParameterPanel
from mlgidlab.pipeline import (
    PipelineCommand,
    add_peak_kwargs_for,
    is_mlgidbase_available,
)
from mlgidlab.pipeline_panel import PipelinePanel
from mlgidlab.profile_viewer import ProfileViewer
from mlgidlab.conversion_panel import ConversionPanel
from mlgidlab.session import BaseSession, NexusSession, RawSession, Session
from mlgidlab.workers import (
    CifParseWorker,
    ConversionWorker,
    CopyWorker,
    PipelineWorker,
    PrefetchWorker,
)

APP_NAME = "mlgidLAB"
NEXUS_FILTER = "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;All files (*)"
# Open dialog now auto-classifies NeXus vs raw; one filter does for both.
OPEN_FILTER = "HDF5 (*.h5 *.hdf5 *.nxs);;All files (*)"


def _make_pen_swatch(style: dict, width: int = 26, height: int = 12) -> QPixmap:
    """Render a small line preview matching an overlay's pen color/style."""
    pix = QPixmap(width, height)
    pix.fill(Qt.GlobalColor.transparent)
    pen = QPen(QColor(style["color"]), 2)
    pen.setStyle(style["style"])
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(pen)
    painter.drawLine(2, height // 2, width - 2, height // 2)
    painter.end()
    return pix


def _make_color_swatch(color: str, width: int = 26, height: int = 12) -> QPixmap:
    """Solid-line swatch in the given color — used for the matched-peaks
    master row where only the colour matters and there is no per-row
    line style to mirror.
    """
    return _make_pen_swatch(
        {"color": color, "style": MATCHED_STYLE["style"]}, width, height
    )


class _MlgidHdf5TreeModel(Hdf5TreeModel):
    """Silx tree model that swaps the file-root icon for raw sessions.

    The default ``Hdf5TreeModel`` uses ``SP_FileIcon`` for every loaded
    HDF5 file. Distinguishing converted-NeXus files (the pipeline runs
    on these) from raw detector files (they need conversion first)
    helps the user spot which is which when both are open in the file
    browser dock at the same time.

    The set of "raw" filesystem paths is owned by ``MainWindow`` and
    pushed in via ``set_raw_paths``; the model emits ``dataChanged``
    so existing rows refresh without a full rebuild.

    All read-only model overrides (``data`` / ``flags`` / ``rowCount`` /
    ``columnCount`` / ``hasChildren`` / ``index``) are wrapped in
    defensive try/except blocks that swallow ``ValueError`` /
    ``RecursionError`` / ``KeyError`` / ``OSError`` / ``RuntimeError``
    and return safe defaults. silx's ``Hdf5Item`` keeps an
    ``h5py.Group`` reference that can outlive the file handle
    (post-pipeline-run detach/reattach, session swap, file close);
    silx's own model methods don't defend against that and raise
    ``ValueError: Invalid group (or file) id`` from inside
    ``len(self.obj)``. Qt's QSortFilterProxyModel then re-fires the
    failing call through every proxy layer, producing a stack-busting
    recursion (40+ frames of ``QSortFilterProxyModel::data`` →
    ``QSortFilterProxyModel::rowCount`` → ``mapToSource``). Swallowing
    the error at our layer stops the storm at its source; the view
    paints a blank row for that frame, which the next natural repaint
    (after the proxy/source rebuild that follows the silx-dance
    completes) overwrites with the correct content.
    """

    # Exceptions raised when an Hdf5Item holds a stale h5py reference
    # or when Qt re-fires a failed call recursively.
    _STALE_EXC = (ValueError, KeyError, OSError, RuntimeError, RecursionError)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._raw_paths: set[str] = set()
        from PySide6.QtWidgets import QApplication, QStyle
        style = QApplication.style()
        self._raw_icon = style.standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon)
        self._nexus_icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)

    def set_raw_paths(self, paths) -> None:
        # No dataChanged.emit here on purpose. Forcing silx's tree to
        # repaint while h5py items may still be in lazy-init has
        # produced reentrancy storms in QSortFilterProxyModel under
        # PySide6. Icons just take effect on the next natural paint
        # (resize / scroll / new insert), which is good enough.
        self._raw_paths = {str(p) for p in paths}

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.DecorationRole
            and index.column() == self.NAME_COLUMN
            and not index.parent().isValid()
        ):
            try:
                node = self.nodeFromIndex(index)
                obj = getattr(node, "obj", None)
                if obj is not None:
                    filename = getattr(obj, "filename", None)
                    if filename:
                        if str(filename) in self._raw_paths:
                            return self._raw_icon
                        return self._nexus_icon
            except self._STALE_EXC:
                # If silx / h5py is in a transient bad state, fall
                # through to super().data() rather than propagating
                # an exception that Qt would re-fire endlessly.
                pass
        try:
            return super().data(index, role)
        except self._STALE_EXC:
            # silx's Hdf5Item.dataDescription walks `len(self.obj)` on
            # a possibly-stale h5py group and propagates a ValueError
            # ("Invalid group (or file) id") through every proxy layer.
            # Returning None lets the view paint a blank cell; the next
            # repaint after the silx-dance completes shows real data.
            return None

    def flags(self, index):
        try:
            return super().flags(index)
        except self._STALE_EXC:
            return Qt.ItemFlag.NoItemFlags

    def rowCount(self, parent=None):
        try:
            if parent is None:
                return super().rowCount()
            return super().rowCount(parent)
        except self._STALE_EXC:
            return 0

    def columnCount(self, parent=None):
        try:
            if parent is None:
                return super().columnCount()
            return super().columnCount(parent)
        except self._STALE_EXC:
            return 0

    def hasChildren(self, parent=None):
        try:
            if parent is None:
                return super().hasChildren()
            return super().hasChildren(parent)
        except self._STALE_EXC:
            return False

    def index(self, row, column, parent=None):
        try:
            if parent is None:
                return super().index(row, column)
            return super().index(row, column, parent)
        except self._STALE_EXC:
            from PySide6.QtCore import QModelIndex
            return QModelIndex()


class _MlgidHdf5TreeView(Hdf5TreeView):
    """Hdf5TreeView that builds its default model from our subclass.

    Also disables silx's built-in file-drop handler so all drag-and-
    drop events fall through to ``MainWindow.dropEvent``. Silx's
    default behaviour is to accept any URL drop on the tree and call
    ``insertFileAsync`` directly — that creates an orphan tree node
    with no matching ``Session`` in our session list, and later
    queries (selection changes, pipeline detach/reattach) blow up
    against the orphan's stale h5py handle.
    """

    def createDefaultModel(self):
        model = _MlgidHdf5TreeModel(self)
        model.setFileDropEnabled(False)
        proxy = NexusSortFilterProxyModel(self)
        proxy.setSourceModel(model)
        return proxy


class _ExportPeaksDialog(QDialog):
    """Modal kind/scope picker for Tools → Export peaks as CSV.

    Two QButtonGroups hold the kind (Detected/Fitted/Matched) and the
    scope (Active frame / Active entry / All entries) respectively;
    Active-frame is greyed when the active stack has only one frame
    so the option doesn't masquerade as different from Active-entry.
    """

    def __init__(self, parent: QWidget, *, has_multiple_frames: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export peaks as CSV")
        layout = QVBoxLayout(self)

        kind_box = QGroupBox("Peak kind")
        kind_layout = QVBoxLayout(kind_box)
        self._rb_detected = QRadioButton("Detected")
        self._rb_fitted = QRadioButton("Fitted (with fit errors)")
        self._rb_matched = QRadioButton("Matched (flattened: one row per peak)")
        self._rb_fitted.setChecked(True)
        for rb in (self._rb_detected, self._rb_fitted, self._rb_matched):
            kind_layout.addWidget(rb)
        layout.addWidget(kind_box)

        scope_box = QGroupBox("Scope")
        scope_layout = QVBoxLayout(scope_box)
        self._rb_frame = QRadioButton("Active frame")
        self._rb_entry = QRadioButton("Active entry (all frames)")
        self._rb_all = QRadioButton("All entries (one combined CSV)")
        self._rb_entry.setChecked(True)
        if not has_multiple_frames:
            self._rb_frame.setEnabled(False)
            self._rb_frame.setToolTip(
                "Available only when the active entry has more than one frame."
            )
        for rb in (self._rb_frame, self._rb_entry, self._rb_all):
            scope_layout.addWidget(rb)
        layout.addWidget(scope_box)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_kind(self) -> str:
        if self._rb_detected.isChecked():
            return "detected"
        if self._rb_matched.isChecked():
            return "matched"
        return "fitted"

    def selected_scope(self) -> str:
        if self._rb_frame.isChecked():
            return "frame"
        if self._rb_all.isChecked():
            return "all"
        return "entry"


# Playback settings persisted via QSettings under the keys below. The
# defaults give a 2× speed-up over the previous fixed 100 ms interval
# while still leaving headroom for cold-cache disk reads (~70-100 ms
# per fresh frame on local SSD). Users who want true frame-by-frame
# stepping can dial Frame interval up; users who want a fixed total
# duration (e.g. 5 s overview regardless of frame count) can flip to
# Total play time.
PLAYBACK_MODE_FRAME = "frame_interval_ms"
PLAYBACK_MODE_TOTAL = "total_time_s"
DEFAULT_PLAYBACK_FRAME_MS = 50          # 20 fps — was 100 ms / 10 fps
DEFAULT_PLAYBACK_TOTAL_S = 3.0
PLAYBACK_FRAME_MS_MIN = 10              # 100 fps requested ceiling
PLAYBACK_FRAME_MS_MAX = 2000            # 0.5 fps floor
PLAYBACK_TOTAL_S_MIN = 0.5
PLAYBACK_TOTAL_S_MAX = 600.0            # 10 minutes max

# Real tick cap. The eye stops perceiving extra frames much above
# ~20 fps, and large frames cannot be painted faster than ~50 ms
# regardless. When the user requests a faster per-frame rate (e.g.
# 3 s total over 300 frames = 10 ms / frame) we keep the timer at
# 50 ms and skip frames instead — see ``_compute_play_schedule``.
PLAYBACK_TICK_FLOOR_MS = 50


class _SettingsDialog(QDialog):
    """Application-wide settings dialog.

    Currently only carries the frame-playback section, but its
    layout reserves room for future settings groups (rendering,
    pipeline defaults, etc.) so adding a new section is just
    appending another ``QGroupBox`` to the outer layout.

    On accept, every changed value is written back to QSettings and
    the host MainWindow is told to re-apply (so an in-flight
    playback timer picks up the new interval immediately).
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)

        outer = QVBoxLayout(self)

        # --- Playback section -------------------------------------------------
        playback_box = QGroupBox("Frame playback")
        playback_layout = QVBoxLayout(playback_box)

        hint = QLabel(
            "<i>Controls the speed of the Display-dock Play button.</i>"
        )
        hint.setWordWrap(True)
        playback_layout.addWidget(hint)

        # Two mutually-exclusive modes. The active radio's spinbox is
        # the one that takes effect; the other stays editable so the
        # user can flip between modes without losing their values.
        mode_box = QButtonGroup(self)
        mode_box.setExclusive(True)
        self._rb_frame = QRadioButton("Time per frame")
        self._rb_total = QRadioButton("Total play time")
        mode_box.addButton(self._rb_frame)
        mode_box.addButton(self._rb_total)
        self._rb_frame.toggled.connect(self._refresh_enabled)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._spin_frame_ms = QSpinBox()
        self._spin_frame_ms.setRange(
            PLAYBACK_FRAME_MS_MIN, PLAYBACK_FRAME_MS_MAX
        )
        self._spin_frame_ms.setSingleStep(10)
        self._spin_frame_ms.setSuffix(" ms")
        self._spin_frame_ms.setToolTip(
            "Time spent on each frame. Lower = faster playback. The 10 ms "
            "lower bound caps playback at 100 fps; cold-cache disk reads "
            "may stutter below ~50 ms on large files."
        )

        self._spin_total_s = QDoubleSpinBox()
        self._spin_total_s.setRange(
            PLAYBACK_TOTAL_S_MIN, PLAYBACK_TOTAL_S_MAX
        )
        self._spin_total_s.setSingleStep(0.5)
        self._spin_total_s.setDecimals(2)
        self._spin_total_s.setSuffix(" s")
        self._spin_total_s.setToolTip(
            "Total time to traverse the whole stack (first frame → last "
            "frame). The per-frame interval is computed at play-start "
            "from the active entry's frame count, so swapping entries "
            "automatically adjusts the speed."
        )

        form.addRow(self._rb_frame, self._spin_frame_ms)
        form.addRow(self._rb_total, self._spin_total_s)
        playback_layout.addLayout(form)
        outer.addWidget(playback_box)

        # --- Buttons + outer wiring -------------------------------------------
        outer.addStretch(1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        # Load current values from QSettings (with defaults).
        settings = QSettings()
        mode = settings.value(
            MainWindow._PLAYBACK_MODE_KEY, PLAYBACK_MODE_FRAME
        )
        # QSettings returns strings on Linux but raw types on macOS;
        # coerce defensively.
        try:
            frame_ms = int(settings.value(
                MainWindow._PLAYBACK_FRAME_MS_KEY, DEFAULT_PLAYBACK_FRAME_MS
            ))
        except (TypeError, ValueError):
            frame_ms = DEFAULT_PLAYBACK_FRAME_MS
        try:
            total_s = float(settings.value(
                MainWindow._PLAYBACK_TOTAL_S_KEY, DEFAULT_PLAYBACK_TOTAL_S
            ))
        except (TypeError, ValueError):
            total_s = DEFAULT_PLAYBACK_TOTAL_S
        # Clamp into the spinbox range so out-of-bounds stored values
        # don't silently revert to the spinbox minimum.
        frame_ms = max(PLAYBACK_FRAME_MS_MIN,
                       min(PLAYBACK_FRAME_MS_MAX, frame_ms))
        total_s = max(PLAYBACK_TOTAL_S_MIN,
                      min(PLAYBACK_TOTAL_S_MAX, total_s))
        self._spin_frame_ms.setValue(frame_ms)
        self._spin_total_s.setValue(total_s)
        if mode == PLAYBACK_MODE_TOTAL:
            self._rb_total.setChecked(True)
        else:
            self._rb_frame.setChecked(True)
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        frame_active = self._rb_frame.isChecked()
        self._spin_frame_ms.setEnabled(frame_active)
        self._spin_total_s.setEnabled(not frame_active)

    def save_to_qsettings(self) -> None:
        """Write the dialog's current values to QSettings.

        Called by the host on accept. Stores both spinbox values so a
        later mode-flip preserves the user's last value in each mode.
        """
        settings = QSettings()
        mode = PLAYBACK_MODE_FRAME if self._rb_frame.isChecked() else PLAYBACK_MODE_TOTAL
        settings.setValue(MainWindow._PLAYBACK_MODE_KEY, mode)
        settings.setValue(
            MainWindow._PLAYBACK_FRAME_MS_KEY, int(self._spin_frame_ms.value())
        )
        settings.setValue(
            MainWindow._PLAYBACK_TOTAL_S_KEY, float(self._spin_total_s.value())
        )


class MainWindow(QMainWindow):
    # Cross-thread invocation signals for the prefetch worker (queued
    # auto-connection to slots on the worker's own QThread). Emitting
    # is the safe cross-thread equivalent of calling the worker's
    # methods directly; the queued delivery serialises with the
    # worker's other queued slots and its internal QTimer ticks.
    _prefetchConfigure = Signal(str, str, int, int)
    _prefetchUpdate = Signal(int, bool, int)
    _prefetchRelease = Signal()

    def __init__(self) -> None:
        super().__init__()
        # Multiple files can be open at once — each as its own Session in the
        # file browser. The "active" one drives entry_combo, the image viewer,
        # and per-file actions (save, save-as, close, pipeline). Switching is
        # automatic when the user clicks a node from a different file.
        self._sessions: list[BaseSession] = []
        self._active_session: BaseSession | None = None
        # Opens run serially through the existing single-thread CopyWorker
        # plumbing; extra paths from a multi-select dialog wait here.
        self._open_queue: list[Path] = []
        self._thread: QThread | None = None
        self._worker: CopyWorker | None = None
        self._progress: QProgressDialog | None = None
        self._pipe_thread: QThread | None = None
        self._pipe_worker: PipelineWorker | None = None
        # Queue of PipelineCommands waiting to run sequentially. The
        # "All entries" option in the pipeline panel expands to one
        # command per entry; each finished run dequeues the next so the
        # user gets per-entry log lines and per-entry error recovery.
        self._pipeline_queue: list[PipelineCommand] = []
        # CIF-parse worker thread. CifPattern construction is slow for
        # raw CIFs so we run it off the GUI thread; only one parse runs
        # at a time (the panel's button stays disabled while it's in
        # flight).
        self._cif_parse_thread: QThread | None = None
        self._cif_parse_worker: CifParseWorker | None = None
        # Conversion worker thread (raw → NeXus). Kept separate from
        # the pipeline worker because conversion runs on raw inputs,
        # while pipeline runs on converted NeXus files; the two never
        # need to share a worker.
        self._conv_thread: QThread | None = None
        self._conv_worker: ConversionWorker | None = None
        self._conv_progress: QProgressDialog | None = None

        # Background prefetch worker. Lives on its own QThread so it
        # can read frames + compute polar resamples without ever
        # blocking the GUI. Spawned lazily on first entry load (no
        # cost on cold startup) and survives across entry switches —
        # the worker is reconfigured per-entry rather than rebuilt.
        # See ``_ensure_prefetch_worker`` and the ``_prefetch*``
        # signals on the class for the cross-thread wiring.
        self._prefetch_thread: QThread | None = None
        self._prefetch_worker: PrefetchWorker | None = None

        # Frame step per play-tick. Stays at 1 unless the requested
        # per-frame interval drops below ``PLAYBACK_TICK_FLOOR_MS``,
        # in which case ``_compute_play_schedule`` bumps it so the
        # play-head jumps multiple frames per tick to honour the
        # total-time target without overrunning the 20 fps practical
        # ceiling. Refreshed on every Play press + every settings
        # change while playing.
        self._play_step: int = 1

        # Stash of the manual peak's geometry captured the moment the
        # "Save fitted as ring" toggle goes ON. Set to a tuple of
        # (peak_ref, radius, angle, radius_width, angle_width, is_ring)
        # while ring is active; cleared on the toggle's OFF transition
        # after the box has been restored. Allows the auto-uncheck
        # that follows a successful Add-to-fitted to revert the box
        # to its pre-ring shape without the host needing to track a
        # commit/cancel distinction.
        self._ring_pre_geom: tuple[
            ManualPeak, float, float, float, float, bool
        ] | None = None

        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        self._build_menu()
        self._build_central()
        self._build_docks()
        # View menu is built after docks because it pulls
        # toggleViewAction()s from them. Settings is built last so it
        # sits at the right end of the menu bar — the conventional
        # rightmost-menu placement that matches user expectations.
        self._build_view_menu()
        self._build_settings_menu()
        # Status bar depends on the viewer + entry combo existing; build
        # after central + docks.
        self._build_status_bar()
        self._update_title()
        self._update_actions()
        # Accept dropped files anywhere on the main window so the user
        # can drag NeXus / raw paths in from a file manager. The drop
        # handler classifies each file by content and dispatches.
        self.setAcceptDrops(True)

    @property
    def session(self) -> Session | None:
        """The currently-active session — drives viewer + entry_combo + save.

        Most callers were written before multi-file support and reach for
        ``self.session``; making it a property of the active session keeps
        those call sites working without per-call refactors.
        """
        return self._active_session

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self._build_file_menu(file_menu)
        self._build_edit_menu(bar)
        self._build_tools_menu(bar)

    def _build_edit_menu(self, bar) -> None:
        edit_menu = bar.addMenu("&Edit")
        self.action_undo = QAction("&Undo", self)
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.action_undo.triggered.connect(self._action_undo)
        edit_menu.addAction(self.action_undo)

        self.action_redo = QAction("&Redo", self)
        # Bind both Ctrl+Y (Win/Linux default) and Ctrl+Shift+Z so muscle
        # memory from either platform works.
        self.action_redo.setShortcuts([
            QKeySequence(QKeySequence.StandardKey.Redo),
            QKeySequence("Ctrl+Shift+Z"),
        ])
        self.action_redo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.action_redo.triggered.connect(self._action_redo)
        edit_menu.addAction(self.action_redo)

    def _build_tools_menu(self, bar) -> None:
        """Bulk-edit operations that don't fit the per-peak ROI workflow.

        Currently scoped to "clear all of one kind for the active entry".
        Future additions (export, copy peaks across frames, statistics,
        symmetry ops, etc.) will land here too — see the README for the
        full roadmap.
        """
        tools_menu = bar.addMenu("&Tools")
        # The four clear-* actions all do the same kind of thing (wipe one
        # peak family) so they live under a single hover-expanding
        # "Clear peaks" submenu rather than cluttering the Tools root.
        clear_menu = tools_menu.addMenu("&Clear peaks")

        self.action_clear_detected = QAction("Detected", self)
        self.action_clear_detected.triggered.connect(
            lambda: self._action_clear_file_peaks("detected")
        )
        clear_menu.addAction(self.action_clear_detected)

        self.action_clear_fitted = QAction("Fitted", self)
        self.action_clear_fitted.triggered.connect(
            lambda: self._action_clear_file_peaks("fitted")
        )
        clear_menu.addAction(self.action_clear_fitted)

        # "Matched and fitted" rather than just "Matched" because the
        # action cascades — clearing matched here also wipes fitted on
        # the same scope, since matched solutions reference fitted ids
        # and an orphaned matched_* group would render against missing
        # peak rows.
        self.action_clear_matched = QAction("Matched and fitted", self)
        self.action_clear_matched.triggered.connect(
            lambda: self._action_clear_file_peaks("matched")
        )
        clear_menu.addAction(self.action_clear_matched)

        # Reset submenu — full wipe of det + fit + match (and manual,
        # in-memory) at three scopes. "Active frame" is greyed out
        # when fewer than two frames are loaded since on a single-
        # frame file it would just duplicate "Active entry".
        clear_menu.addSeparator()
        reset_menu = clear_menu.addMenu("&Reset all peaks")
        reset_menu.setToolTipsVisible(True)
        self._reset_menu = reset_menu

        self.action_reset_entry = QAction("Active entry (all frames)", self)
        self.action_reset_entry.setToolTip(
            "Clear detected, fitted, matched, and manual peaks on the "
            "currently displayed entry."
        )
        self.action_reset_entry.triggered.connect(
            lambda: self._action_reset_analysis("entry")
        )
        reset_menu.addAction(self.action_reset_entry)

        self.action_reset_all = QAction("All entries", self)
        self.action_reset_all.setToolTip(
            "Clear detected, fitted, matched, and manual peaks on every "
            "entry in the active file."
        )
        self.action_reset_all.triggered.connect(
            lambda: self._action_reset_analysis("all")
        )
        reset_menu.addAction(self.action_reset_all)

        self.action_reset_frame = QAction("Active frame", self)
        self.action_reset_frame.setToolTip(
            "Clear detected, fitted, and matched peaks on just the "
            "currently displayed frame of the active entry. Manual "
            "peaks are wiped (they live in memory across frames)."
        )
        self.action_reset_frame.triggered.connect(
            lambda: self._action_reset_analysis("frame")
        )
        reset_menu.addAction(self.action_reset_frame)
        # Re-evaluate the per-scope enabled states right before the
        # submenu is shown — n_frames / session state can change between
        # menu opens, and aboutToShow keeps the gate cheap (no signal
        # plumbing on every viewer event).
        reset_menu.aboutToShow.connect(self._refresh_reset_menu_state)

        # Export the current frame to PNG. Works for either NeXus or
        # raw mode — pyqtgraph's ImageExporter operates on the active
        # plot item regardless of which stack supplied the data.
        tools_menu.addSeparator()
        self.action_export_png = QAction("Export current frame as PNG…", self)
        self.action_export_png.triggered.connect(self._action_export_png)
        tools_menu.addAction(self.action_export_png)

        # CSV export of detected/fitted/matched peaks. NeXus-only.
        self.action_export_csv = QAction("Export peaks as CSV…", self)
        self.action_export_csv.triggered.connect(self._action_export_csv)
        tools_menu.addAction(self.action_export_csv)

    def _action_clear_file_peaks(self, kind: str) -> None:
        """Empty every ``<kind>_peaks`` dataset for the active entry.

        Cascade rules:
        - clearing ``fitted`` also clears ``matched`` (matched rows
          reference fitted ids; orphaned matched_* groups can't render).
        - the menu's ``matched`` action is labelled "Matched and fitted"
          and likewise wipes both — see Tools-menu wiring above for the
          rename rationale.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        if not self._confirm_clear(kind):
            return
        kinds_to_clear = [kind]
        if kind == "fitted":
            kinds_to_clear.append("matched")
        elif kind == "matched":
            kinds_to_clear.append("fitted")

        self._detach_silx_tree()
        try:
            removed_total = 0
            for k in kinds_to_clear:
                removed_total += file_model.clear_peaks(
                    self.session.temp_path, entry, k
                )
        except Exception as exc:
            QMessageBox.critical(self, "Clear failed", str(exc))
            self._reattach_silx_tree()
            return
        self._reattach_silx_tree()

        self.session.mark_dirty()
        self._update_title()
        # Bulk wipe invalidates every FileGeomAction and the selection.
        self.viewer.clear_history()
        self.viewer.clear_selection()
        self._load_entry_into_viewer(entry, preserve_view=True)
        self.pipeline_panel.append_log(
            f"Cleared {kind} peaks ({removed_total} rows total) on {entry}"
        )

    def _refresh_reset_menu_state(self) -> None:
        """Gate Reset submenu actions on session + frame availability.

        Active-frame is greyed out with a single frame loaded since the
        clear would be identical to Active-entry. Active-entry / All
        entries need only an open session.
        """
        has_session = self.session is not None and self._pipe_thread is None
        n_frames = getattr(self.viewer, "n_frames", 0) if has_session else 0
        self.action_reset_entry.setEnabled(has_session)
        self.action_reset_all.setEnabled(has_session)
        self.action_reset_frame.setEnabled(has_session and n_frames > 1)

    def _action_reset_analysis(self, scope: str) -> None:
        """Wipe det + fit + match (and manual peaks) at the requested scope.

        ``scope`` is one of:
        - ``"entry"`` — active entry, every frame in it.
        - ``"all"``   — every entry in the active file, every frame.
        - ``"frame"`` — active entry, just the active frame.

        Manual peaks are session-wide and live in memory only; every
        scope clears them outright since the user asked for a true reset.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        active_entry = self.entry_combo.currentText()
        if scope in ("entry", "frame") and not active_entry:
            return
        if scope == "frame" and getattr(self.viewer, "n_frames", 0) <= 1:
            return

        # Build the scope-specific list of (entry, frame|None) tuples
        # the inner h5 wipe loop iterates over.
        if scope == "all":
            try:
                targets = [(e, None) for e in file_model.list_entries(self.session.temp_path)]
            except Exception as exc:
                QMessageBox.critical(self, "Reset failed", f"Could not list entries: {exc}")
                return
            scope_label = f"all {len(targets)} entries"
        elif scope == "entry":
            targets = [(active_entry, None)]
            scope_label = f"entry {active_entry}"
        else:  # frame
            frame_idx = int(self.viewer.current_frame)
            targets = [(active_entry, frame_idx)]
            scope_label = f"frame {frame_idx} of {active_entry}"

        reply = QMessageBox.question(
            self,
            "Reset analysis",
            f"Remove every detected, fitted, matched, and manual peak "
            f"on {scope_label}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Manual peaks are global session state — drop them once,
        # regardless of scope.
        self.viewer.clear_all_manual_peaks()

        self._detach_silx_tree()
        try:
            removed_total = 0
            for entry, frame in targets:
                for kind in ("detected", "fitted", "matched"):
                    removed_total += file_model.clear_peaks(
                        self.session.temp_path, entry, kind, frame=frame
                    )
        except Exception as exc:
            QMessageBox.critical(self, "Reset failed", str(exc))
            self._reattach_silx_tree()
            return
        self._reattach_silx_tree()

        self.session.mark_dirty()
        self._update_title()
        self.viewer.clear_history()
        self.viewer.clear_selection()
        # Refresh the displayed entry — the cleared one if the user
        # was looking at it, otherwise the currently-active one.
        if active_entry:
            self._load_entry_into_viewer(active_entry, preserve_view=True)
        self.pipeline_panel.append_log(
            f"Reset analysis: cleared {removed_total} peak rows on {scope_label} "
            f"(plus all manual peaks)"
        )

    def _action_export_png(self) -> None:
        """Export the currently-displayed image (with overlays) to PNG.

        Uses pyqtgraph's ImageExporter on the viewer's PlotItem so the
        output mirrors what the user sees — colormap, levels, axes,
        and any visible peak overlays. Available in both NeXus and
        raw modes.
        """
        if self.viewer.n_frames == 0:
            QMessageBox.information(
                self, "Nothing to export",
                "Open a file and load an entry before exporting.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export current frame as PNG", "frame.png",
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter
            exporter = ImageExporter(self.viewer._plot)
            exporter.export(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Export failed",
                f"Could not write PNG: {exc}",
            )
            return
        # Confirm visually so the user knows where the file landed.
        self.statusBar().showMessage(f"Wrote {path}", 5000)

    def _action_export_csv(self) -> None:
        """Pop the kind/scope dialog and write peaks to a CSV.

        NeXus-only — raw sessions don't have peak datasets. The actual
        flatten + write lives in ``file_model.export_peaks_csv`` /
        ``export_matched_csv``; the GUI's job here is dialog wiring,
        scope resolution, and the silx detach/reattach that frees the
        file's HDF5 handle for r-mode reads.
        """
        if self.session is None or self.session.kind != "nexus":
            QMessageBox.information(
                self, "Export peaks",
                "Open a NeXus file first — raw files have no peak datasets.",
            )
            return
        active_entry = self.entry_combo.currentText()
        if not active_entry:
            QMessageBox.information(
                self, "Export peaks", "No active entry to export from."
            )
            return
        n_frames = getattr(self.viewer, "n_frames", 0)
        dlg = _ExportPeaksDialog(self, has_multiple_frames=n_frames > 1)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        kind = dlg.selected_kind()
        scope = dlg.selected_scope()

        # Suggest a filename rooted at the original-file basename so
        # batched exports from multiple opens don't collide on disk.
        base = self.session.original_path.stem
        suggest = f"{base}_{kind}_{scope}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export peaks as CSV", suggest,
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return

        # Resolve the scope into an (entry, frame|None) target list
        # consumed by the file_model exporters.
        if scope == "all":
            try:
                entries = file_model.list_entries(self.session.temp_path)
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", f"Could not list entries: {exc}")
                return
            targets: list[tuple[str, int | None]] = [(e, None) for e in entries]
        elif scope == "entry":
            targets = [(active_entry, None)]
        else:
            targets = [(active_entry, int(self.viewer.current_frame))]

        # silx may hold a read handle on the temp file; detach so h5py
        # can open it without contention.
        self._detach_silx_tree()
        try:
            if kind == "matched":
                n = file_model.export_matched_csv(
                    self.session.temp_path, targets, Path(path)
                )
            else:
                n = file_model.export_peaks_csv(
                    self.session.temp_path, targets, kind, Path(path)
                )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            self._reattach_silx_tree()
            return
        self._reattach_silx_tree()

        self.statusBar().showMessage(
            f"Wrote {n} {kind} peak rows ({scope}) to {path}", 6000
        )
        self.pipeline_panel.append_log(
            f"Exported {n} {kind} peak rows ({scope}) to {path}"
        )

    def _confirm_clear(self, kind: str) -> bool:
        descriptions = {
            "detected": ("detected peaks",
                         "every row of detected_peaks for the active entry"),
            "fitted":   ("fitted + matched peaks",
                         "every row of fitted_peaks AND every matched_* "
                         "solution for the active entry "
                         "(matched references fitted, so it has to go too)"),
            "matched":  ("matched and fitted peaks",
                         "every matched_* solution AND every row of "
                         "fitted_peaks for the active entry "
                         "(matched references fitted, so the cascade goes "
                         "both ways here — see the Tools menu)"),
        }
        title, body = descriptions.get(kind, (kind, kind))
        reply = QMessageBox.question(
            self,
            f"Clear {title}",
            f"Remove {body}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _build_view_menu(self) -> None:
        """Expose dock visibility toggles in a top-level View menu.

        Each dock already has a built-in ``toggleViewAction()`` whose label
        and check state stay in sync with the dock — reusing them keeps the
        menu correct without manual bookkeeping.
        """
        view_menu = self.menuBar().addMenu("&View")
        for dock in (
            self._tree_dock,
            self._display_dock,
            self._pipeline_dock,
            self._conversion_dock,
            self._logs_dock,
            self._profile_dock,
        ):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        # Toggle for the cursor-readout segment of the status bar — some
        # users find the per-pixel readout distracting; on by default.
        self.action_toggle_cursor_readout = QAction(
            "Show cursor readout", self
        )
        self.action_toggle_cursor_readout.setCheckable(True)
        self.action_toggle_cursor_readout.setChecked(True)
        self.action_toggle_cursor_readout.toggled.connect(
            self._set_cursor_readout_visible
        )
        view_menu.addAction(self.action_toggle_cursor_readout)

    def _build_settings_menu(self) -> None:
        """Build the top-level Settings menu.

        Houses application-wide preferences that don't justify a
        dedicated dock or main-toolbar slot. Currently exposes the
        frame-playback settings; future entries (e.g. default
        colormap, default render quality, log-verbosity toggle) hang
        off the same menu.

        The menu is built after View so it sits at the rightmost
        position, which is where users instinctively reach for
        Settings in cross-platform apps.
        """
        settings_menu = self.menuBar().addMenu("&Settings")
        self.action_playback_settings = QAction(
            "&Playback settings…", self
        )
        self.action_playback_settings.setToolTip(
            "Configure how the Display-dock Play button drives frame "
            "advance — either fixed time per frame or fixed total "
            "duration regardless of frame count."
        )
        self.action_playback_settings.triggered.connect(
            self._action_playback_settings
        )
        settings_menu.addAction(self.action_playback_settings)

    def _action_playback_settings(self) -> None:
        """Open the playback-settings dialog.

        On accept, persist the dialog's values via QSettings and, if
        the play timer is currently running, re-apply the new
        interval mid-flight so the change is felt immediately. The
        next press of Play also re-reads via ``_compute_play_schedule``
        so a setting change applied while paused still takes effect.
        """
        dlg = _SettingsDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        dlg.save_to_qsettings()
        # If playback is currently running, push the new schedule onto
        # the timer right away. The next tick will use it.
        if self._play_timer.isActive():
            interval, step = self._compute_play_schedule()
            self._play_timer.setInterval(interval)
            self._play_step = step
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, True, step,
                )

    def _action_undo(self) -> None:
        # Covers manual add/remove, manual geom edits, and detected/fitted
        # geom edits. File-level deletes (delete_peak) are not undoable —
        # see the confirmation dialog in _on_delete_peak_requested.
        if hasattr(self, "viewer"):
            self.viewer.undo_last_action()

    def _action_redo(self) -> None:
        if hasattr(self, "viewer"):
            self.viewer.redo_last_action()

    def _build_file_menu(self, file_menu) -> None:

        # Single Open action — file content is auto-classified as
        # NeXus or raw inside ``_action_open`` so users don't have to
        # pick the right entry point. Raw files are bundled into one
        # ``RawSession`` in the same way the old "Open raw" action did.
        self.action_open = QAction("&Open…", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._action_open)
        file_menu.addAction(self.action_open)

        # Recent-files submenu — populated lazily on aboutToShow so the
        # missing-file filter stays accurate across sessions.
        self._recent_menu = file_menu.addMenu("Open &recent")
        self._recent_menu.setToolTipsVisible(True)
        self._recent_menu.aboutToShow.connect(self._refresh_recent_files_menu)
        # Build once now so the menu shows real entries on first open
        # (aboutToShow only fires when the user actually opens the
        # submenu — but the parent File menu's expansion looks better
        # if the count is right from the start).
        self._refresh_recent_files_menu()

        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._action_save)
        file_menu.addAction(self.action_save)

        self.action_save_as = QAction("Save &As…", self)
        self.action_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.action_save_as.triggered.connect(self._action_save_as)
        file_menu.addAction(self.action_save_as)

        self.action_close_file = QAction("&Close", self)
        self.action_close_file.setShortcut(QKeySequence.StandardKey.Close)
        self.action_close_file.triggered.connect(self._action_close_file)
        file_menu.addAction(self.action_close_file)

        file_menu.addSeparator()

        action_exit = QAction("E&xit", self)
        action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    # -- Recent files (QSettings-backed) --

    _RECENT_FILES_KEY = "recentFiles"
    _MAX_RECENT_FILES = 10

    # Playback settings (persisted via QSettings). See the module-level
    # PLAYBACK_* constants for defaults and bounds.
    _PLAYBACK_MODE_KEY = "playbackMode"
    _PLAYBACK_FRAME_MS_KEY = "playbackFrameIntervalMs"
    _PLAYBACK_TOTAL_S_KEY = "playbackTotalTimeS"

    def _load_recent_files(self) -> list[dict]:
        """Return the persisted recent-files list as a list of dicts.

        Each entry is ``{"type": "nexus"|"raw", "path": str}``. The
        list is stored as a JSON string in QSettings to keep the
        serialization explicit and robust across PySide/Qt platforms
        (raw QStringList round-tripping has bitten us before).
        """
        settings = QSettings()
        blob = settings.value(self._RECENT_FILES_KEY, "[]")
        if not isinstance(blob, str):
            return []
        try:
            data = json.loads(blob)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [
            d for d in data
            if isinstance(d, dict)
            and d.get("type") in ("nexus", "raw")
            and isinstance(d.get("path"), str)
        ]

    def _save_recent_files(self, items: list[dict]) -> None:
        QSettings().setValue(self._RECENT_FILES_KEY, json.dumps(items))

    def _add_recent_file(self, path: str | Path, kind: str) -> None:
        """Push ``path`` onto the front of the recent list.

        Move-to-front semantics: if ``path`` is already in the list it
        gets bubbled up to the top instead of duplicated. The list is
        capped at ``_MAX_RECENT_FILES``.
        """
        if kind not in ("nexus", "raw"):
            return
        path_str = str(path)
        items = self._load_recent_files()
        items = [i for i in items if i.get("path") != path_str]
        items.insert(0, {"type": kind, "path": path_str})
        items = items[: self._MAX_RECENT_FILES]
        self._save_recent_files(items)
        self._refresh_recent_files_menu()

    def _refresh_recent_files_menu(self) -> None:
        """Rebuild the submenu, dropping entries whose files have moved."""
        self._recent_menu.clear()
        items = self._load_recent_files()
        # Filter to existing files and rewrite the persisted list
        # if any are missing — keeps the user from being surprised
        # by stale entries reappearing the next session.
        present = [i for i in items if Path(i["path"]).exists()]
        if len(present) != len(items):
            self._save_recent_files(present)
        if not present:
            empty = QAction("(no recent files)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for entry in present:
            path = entry["path"]
            kind = entry["type"]
            basename = Path(path).name
            # NeXus rows show plain basename; raw rows get a "[raw] "
            # prefix so the two are distinguishable without an icon.
            label = basename if kind == "nexus" else f"[raw]  {basename}"
            action = QAction(label, self)
            # Tooltip shows the full path so the user can disambiguate
            # files with the same basename living in different folders.
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, p=path, k=kind: self._open_recent(p, k)
            )
            self._recent_menu.addAction(action)
        self._recent_menu.addSeparator()
        clear_action = QAction("Clear recent files", self)
        clear_action.triggered.connect(self._clear_recent_files)
        self._recent_menu.addAction(clear_action)

    def _clear_recent_files(self) -> None:
        self._save_recent_files([])
        self._refresh_recent_files_menu()

    def _open_recent(self, path: str, kind: str) -> None:
        """Open a file from the Recent-files submenu.

        Routes by recorded ``kind`` (``"nexus"`` or ``"raw"``) so we
        don't depend on extension sniffing. Drops the entry from the
        list if the file has gone missing since it was recorded.
        """
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(
                self,
                "Recent files",
                f"File no longer exists:\n{path}\n\n"
                "Removing from the recent list.",
            )
            items = [i for i in self._load_recent_files() if i.get("path") != path]
            self._save_recent_files(items)
            self._refresh_recent_files_menu()
            return
        if kind == "nexus":
            self._open_queue.append(p)
            self._process_open_queue()
            return
        # Raw mode — synchronous open through RawSession; same shape
        # as the unified _open_paths raw branch but without the
        # classification step (kind was recorded with the recent
        # entry, so we already know).
        try:
            session = RawSession.open([p])
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        model = self.tree.findHdf5TreeModel()
        for raw_path in session.raw_paths:
            model.insertFile(str(raw_path))
        self._sessions.append(session)
        self._set_active_session(session)
        self._refresh_tree_raw_paths()

    def _build_central(self) -> None:
        self.viewer = GIWAXSImageViewer(self)
        self.data_viewer = DataViewerFrame(self)

        self.tabs = QTabWidget(self)
        # documentMode flattens the tab-pane border so the image fills
        # the full tab area without the small inset that lets pyqtgraph
        # show a few pixels of scrollable margin.
        self.tabs.setDocumentMode(True)
        # documentMode is partial under qdarkstyle — the pane keeps a
        # small border + padding from the dark stylesheet which traps
        # ~2 px of overflow from the central widget. Override via an
        # explicit zero-pad stylesheet so the viewer fills flush.
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 0px; padding: 0px; margin: 0px; }"
        )
        self.tabs.addTab(self.viewer, "Image")
        self.tabs.addTab(self.data_viewer, "Data")
        self.setCentralWidget(self.tabs)

    def _build_docks(self) -> None:
        # Make the side docks own the bottom corners so the bottom Profile
        # dock stays aligned with the central image/data tabs.
        self.setCorner(
            Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Left: HDF5 tree (silx) — subclass swaps the root icon for raw
        # sessions so NeXus and raw files are distinguishable at a glance.
        self.tree = _MlgidHdf5TreeView(self)
        self.tree.setSortingEnabled(True)
        # Single-click silently updates Data tab; double-click jumps to it.
        self.tree.selectionModel().selectionChanged.connect(
            self._on_tree_selection_changed
        )
        self.tree.activated.connect(self._on_tree_activated)
        self._tree_dock = QDockWidget("File browser", self)
        self._tree_dock.setWidget(self.tree)
        self._tree_dock.setObjectName("FileBrowserDock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._tree_dock)

        # Right: entry selector + overlay toggles
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self.entry_combo = QComboBox()
        self.entry_combo.currentTextChanged.connect(self._on_entry_changed)
        form.addRow("Entry:", self.entry_combo)

        # Frame slider — drives viewer.set_frame; Hidden for single-
        # frame stacks where it would just take vertical space without
        # any function.
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_label = QLabel("Frame —")
        self.frame_label.setMinimumWidth(80)
        self.frame_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        # Play / pause toggle. Drives a QTimer that calls
        # viewer.set_frame(current + 1) on every tick; stops when the
        # last frame is reached. Standard-icon based so qdarkstyle
        # picks up the right colour automatically.
        self.play_button = QToolButton()
        self.play_button.setCheckable(True)
        self._icon_play = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPlay
        )
        self._icon_pause = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPause
        )
        self.play_button.setIcon(self._icon_play)
        self.play_button.setToolTip(
            "Play frames from the current position to the end.\n"
            "Stops at the last frame; click again to pause."
        )
        self.play_button.toggled.connect(self._on_play_toggled)
        # Driver for playback. Interval + step are resolved from
        # QSettings (see ``_compute_play_schedule``) on every Play
        # start, so a setting change picks up on the next press
        # without restarting the timer. Default mode is "time per
        # frame" at 50 ms = 20 fps. Requested rates below 50 ms /
        # frame don't speed up the timer — instead, the play-head
        # advances by ``self._play_step`` frames per tick so the
        # target total time is honoured while the timer stays at the
        # 20 fps practical ceiling.
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(DEFAULT_PLAYBACK_FRAME_MS)
        self._play_timer.timeout.connect(self._on_play_tick)
        frame_row = QWidget()
        frame_h = QHBoxLayout(frame_row)
        frame_h.setContentsMargins(0, 0, 0, 0)
        frame_h.setSpacing(6)
        frame_h.addWidget(self.play_button)
        frame_h.addWidget(self.frame_slider, 1)
        frame_h.addWidget(self.frame_label)
        # Stash the row's parent label so we can hide both in unison.
        self._frame_row_widget = frame_row
        form.addRow("Frame:", frame_row)
        layout.addLayout(form)
        # Both the slider and its "Frame:" label start hidden — they're
        # only useful once a multi-frame stack is loaded.
        self._set_frame_slider_visible(False)

        layout.addWidget(QLabel("Overlays"))
        # Manual peaks intentionally omitted: the GUI now keeps at most
        # one manual box per frame (drawn → replaced → committed via
        # Add-to-fitted/detected, removed via Esc / Delete), so a
        # visibility toggle for "all manual peaks" no longer has work
        # to do. The viewer's internal _visibility["manual"] stays True
        # by default — see GIWAXSImageViewer.__init__.
        self._overlay_checks: dict[str, QCheckBox] = {}
        for kind, label in (
            ("detected", "Detected peaks"),
            ("fitted", "Fitted peaks"),
        ):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            swatch = QLabel()
            swatch.setPixmap(_make_pen_swatch(OVERLAY_STYLE[kind]))
            row.addWidget(swatch)
            chk = QCheckBox(label)
            chk.setChecked(True)
            chk.toggled.connect(
                lambda v, k=kind: self.viewer.set_overlay_visible(k, v)
            )
            row.addWidget(chk)
            row.addStretch(1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            layout.addWidget(row_widget)
            self._overlay_checks[kind] = chk

        # Matched peaks: master toggle + per-structure rows. The per-structure
        # rows are rebuilt on every frame change because different frames can
        # have different matching solutions.
        matched_master_row = QHBoxLayout()
        matched_master_row.setContentsMargins(0, 0, 0, 0)
        matched_master_row.setSpacing(6)
        # Empty spacer where the swatch would go — colors live on each row.
        matched_master_row.addSpacing(_make_pen_swatch(OVERLAY_STYLE["detected"]).width() + 4)
        self._matched_master_check = QCheckBox("Matched peaks")
        self._matched_master_check.setChecked(True)
        self._matched_master_check.toggled.connect(self._on_matched_master_toggled)
        # Per-structure checkboxes are rebuilt in _refresh_matched_panel
        # but kept indexed here so the master-toggle cascade and the
        # "single structure on while master off" promotion path can
        # reach them by uid.
        self._matched_struct_checkboxes: dict[str, QCheckBox] = {}
        matched_master_row.addWidget(self._matched_master_check)
        matched_master_row.addStretch(1)
        matched_master_widget = QWidget()
        matched_master_widget.setLayout(matched_master_row)
        layout.addWidget(matched_master_widget)

        # Container for the dynamic per-structure rows. Indented so it reads
        # as a sub-list of the master toggle.
        self._matched_struct_container = QWidget()
        self._matched_struct_layout = QVBoxLayout(self._matched_struct_container)
        self._matched_struct_layout.setContentsMargins(20, 0, 0, 0)
        self._matched_struct_layout.setSpacing(2)
        layout.addWidget(self._matched_struct_container)
        # Lives in its own field so we can find/remove the placeholder row.
        self._matched_empty_label: QLabel | None = None
        self._refresh_matched_panel(0, [])
        self.viewer.matchedStructuresChanged.connect(self._refresh_matched_panel)

        layout.addSpacing(6)

        self.parameter_panel = ParameterPanel(self)
        layout.addWidget(self.parameter_panel)

        layout.addSpacing(6)
        hint = QLabel(
            "<i>Polar mode: <b>Ctrl+Alt-drag</b> to label, click any peak "
            "(detected / fitted / matched / manual) to select, drag edges "
            "to resize manual / detected, <b>Delete</b> to remove. "
            "Add-to-fitted accepts manual or detected selections — the "
            "cyan box previews the saved FWHM. "
            "<b>LMB double-click</b> resets zoom. "
            "<b>Ctrl+Z</b> / <b>Ctrl+Shift+Z</b> undo / redo.</i>"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch(1)

        # Wrap the dock content in a QScrollArea so files with many
        # matched structures (one row per (CIF, hkl) match) don't push
        # the parameter panel and shortcut hint off the bottom of the
        # screen. Vertical scrolling kicks in on demand; horizontal is
        # locked off so narrow docks wrap their form rows instead of
        # introducing an x-axis scrollbar.
        display_scroll = QScrollArea(self)
        display_scroll.setWidgetResizable(True)
        display_scroll.setFrameShape(QFrame.Shape.NoFrame)
        display_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        display_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        display_scroll.setWidget(panel)
        self._display_dock = QDockWidget("Display", self)
        self._display_dock.setWidget(display_scroll)
        self._display_dock.setObjectName("DisplayDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._display_dock)

        # Pipeline dock — tabified with Display on the right.
        self.pipeline_panel = PipelinePanel(self)
        # Let the panel resolve "Active entry" / "Active frame" at click time
        # without pulling MainWindow into its imports. Returning None for
        # either falls through to mlgidBASE's all-entries / all-frames default.
        self.pipeline_panel.set_active_entry_resolver(
            lambda: self.entry_combo.currentText() or None
        )
        self.pipeline_panel.set_active_frame_resolver(
            lambda: self.viewer.current_frame if self.session is not None else None
        )
        self.pipeline_panel.runRequested.connect(self._on_run_requested)
        self.pipeline_panel.parseCifsRequested.connect(self._on_parse_cifs_requested)
        self._pipeline_dock = QDockWidget("Pipeline", self)
        self._pipeline_dock.setWidget(self.pipeline_panel)
        self._pipeline_dock.setObjectName("PipelineDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._pipeline_dock)
        self.tabifyDockWidget(self._display_dock, self._pipeline_dock)

        # Conversion dock — mode-exclusive sibling of the Pipeline dock.
        # Visible only when the active session is a RawSession; switching
        # between Nexus and Raw sessions hides one and shows the other.
        # Both share the same dock slot (tabified with Display) so the
        # right side never grows beyond two visible tabs.
        self.conversion_panel = ConversionPanel(self)
        self.conversion_panel.conversionRunRequested.connect(
            self._on_conversion_run
        )
        self._conversion_dock = QDockWidget("Conversion", self)
        self._conversion_dock.setWidget(self.conversion_panel)
        self._conversion_dock.setObjectName("ConversionDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._conversion_dock)
        self.tabifyDockWidget(self._display_dock, self._conversion_dock)
        # Default state matches the default session (none): pipeline dock
        # shown so the user can see what would be available once they
        # open a converted file. ``_apply_session_mode`` handles toggles
        # from then on.
        self._conversion_dock.setVisible(False)

        # Shared Logs dock — tabified next to Display / Pipeline / Conversion.
        # Both panels emit ``logMessage`` / ``logCleared``; we route them
        # through this single widget so the log history is visible in
        # either mode (and a switch from Conversion to NeXus doesn't hide
        # the running log).
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("monospace"))
        self._log_view.setMaximumBlockCount(4000)
        self._log_view.setPlaceholderText(
            "Pipeline and conversion logs land here."
        )
        self._logs_dock = QDockWidget("Logs", self)
        self._logs_dock.setWidget(self._log_view)
        self._logs_dock.setObjectName("LogsDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._logs_dock)
        self.tabifyDockWidget(self._display_dock, self._logs_dock)

        # Route both panels' log messages into the shared widget. Both
        # panels' ``append_log`` / ``clear_log`` already emit these
        # signals — every existing call site keeps working.
        self.pipeline_panel.logMessage.connect(self._log_view.appendPlainText)
        self.pipeline_panel.logCleared.connect(self._log_view.clear)
        self.conversion_panel.logMessage.connect(self._log_view.appendPlainText)
        self.conversion_panel.logCleared.connect(self._log_view.clear)

        self._display_dock.raise_()

        # Bottom: profile viewer. Default to ~30% of window height so the
        # central image stays the main focus.
        self.profile_viewer = ProfileViewer(self)
        self._profile_dock = QDockWidget("Profiles", self)
        self._profile_dock.setWidget(self.profile_viewer)
        self._profile_dock.setObjectName("ProfileDock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._profile_dock)
        self.resizeDocks(
            [self._profile_dock], [max(self.height() // 3, 280)], Qt.Orientation.Vertical
        )
        self.viewer.frameChanged.connect(self.profile_viewer.set_frame)
        # Bidirectional Display-dock slider sync: viewer pushes frame
        # changes into the slider (e.g. user scrubs the pyqtgraph
        # timeline below the image), and the slider's valueChanged
        # already pushes back into the viewer via _on_frame_slider_changed.
        self.viewer.frameChanged.connect(self._on_viewer_frame_changed)
        # Bidirectional sync between 2D ROI and profile-edge regions. The
        # profile viewer only handles ManualPeak, so we filter the
        # SelectedPeak-typed signals down to the manual case before forwarding.
        self.viewer.selectionChanged.connect(self._forward_selection_to_profile)
        self.viewer.peakGeometryChanged.connect(self._forward_geom_to_profile)
        self.profile_viewer.peakGeometryChanged.connect(self.viewer.update_peak_geometry_external)
        # The faint fitted-preview box for the selected manual peak follows
        # the profile viewer's 1D Gaussian fits. It also has to drop when
        # the selection changes away from a manual peak.
        self.profile_viewer.fitParamsChanged.connect(self._update_fitted_preview)
        self.viewer.selectionChanged.connect(self._on_selection_for_preview)
        # Live readout of the same 1D fits in the parameter panel so the
        # user can see the fitted-peak parameters next to the detected ones.
        self.profile_viewer.fitParamsChanged.connect(self.parameter_panel.set_fits)

        # Parameter readout — both selection and geometry changes feed the same slot.
        self.viewer.selectionChanged.connect(self.parameter_panel.set_peak)
        self.viewer.peakGeometryChanged.connect(self.parameter_panel.set_peak)

        # Commit / delete actions on the parameter panel. Add-to-detected and
        # delete reuse the existing PipelineWorker path.
        self.parameter_panel.addToDetectedRequested.connect(self._on_add_to_detected)
        self.parameter_panel.addToFittedRequested.connect(self._on_add_to_fitted)
        # Refresh the cyan preview overlay immediately when the user
        # toggles ring/segment — otherwise the preview would lag until
        # the next fit recompute.
        self.parameter_panel.saveAsRingChanged.connect(self._on_save_as_ring_changed)
        self.parameter_panel.deletePeakRequested.connect(
            lambda: self._on_delete_peak_requested(self.viewer.selected_peak)
        )

        # Direct-h5py geometry writes for detected/fitted ROI edits.
        self.viewer.peakRowWriteRequested.connect(self._on_peak_row_write_requested)
        # Delete keypress on file-resident peaks.
        self.viewer.deletePeakRequested.connect(self._on_delete_peak_requested)

        # Keep _ring_pre_geom in sync with the manual peak it points at.
        # When the user replaces the box (single-box policy) while ring
        # is active, the new box also needs ring expansion; when the
        # box is removed (Esc / Delete / Add-to-detected), the stash
        # goes stale and must be invalidated.
        self.viewer.manualPeakRemoved.connect(self._on_manual_peak_removed)
        self.viewer.manualPeakAdded.connect(self._on_manual_peak_added)

    # -- Actions --

    def _action_open(self) -> None:
        """Unified open: pick HDF5 files and auto-classify each as NeXus or raw.

        Multi-select is supported. Each picked file is classified by
        content (not extension) inside ``_open_paths`` — NeXus files
        stream through the per-file copy worker queue, raw files are
        bundled into a single shared ``RawSession`` matching the old
        Open-raw bulk behaviour. Files that match neither classifier
        are reported in the log and skipped.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open file(s)", "", OPEN_FILTER
        )
        if not paths:
            return
        self._open_paths([Path(p) for p in paths])

    def _classify_h5_path(self, path: Path) -> str | None:
        """Return ``"nexus"``, ``"raw"``, or ``None`` from file content.

        NeXus is detected by the presence of at least one entry whose
        ``data`` group has ``signal == "img_gid_q"`` (the same filter
        used everywhere else in the GUI). Raw detection falls back to
        any 3D detector-shaped dataset. Both readers swallow their
        exceptions so a non-HDF5 file or a permissions error returns
        ``None`` rather than crashing the open.
        """
        try:
            if file_model.list_entries(path):
                return "nexus"
        except Exception:
            pass
        try:
            if file_model.list_raw_entries(path):
                return "raw"
        except Exception:
            pass
        return None

    def _open_paths(self, paths: list[Path]) -> None:
        """Open a mixed batch of NeXus + raw files, auto-classifying each.

        Used by both the unified File → Open action and the drag-and-drop
        handler. NeXus paths queue through the existing copy worker;
        raw paths are bundled into one ``RawSession`` so the Conversion
        panel can apply one config to the whole batch.
        """
        nexus_paths: list[Path] = []
        raw_paths: list[Path] = []
        rejected: list[Path] = []
        for p in paths:
            if not p.is_file():
                rejected.append(p)
                continue
            kind = self._classify_h5_path(p)
            if kind == "nexus":
                nexus_paths.append(p)
            elif kind == "raw":
                raw_paths.append(p)
            else:
                rejected.append(p)
        if rejected:
            self.pipeline_panel.append_log(
                "Could not classify (no q-signal entries, no raw 3D "
                "detector datasets): "
                + ", ".join(str(p) for p in rejected)
            )
        if nexus_paths:
            self._open_queue.extend(nexus_paths)
            self._process_open_queue()
        if raw_paths:
            try:
                session = RawSession.open(raw_paths)
            except Exception as exc:
                QMessageBox.critical(self, "Open failed", str(exc))
                return
            model = self.tree.findHdf5TreeModel()
            for raw_path in session.raw_paths:
                model.insertFile(str(raw_path))
            self._sessions.append(session)
            self._set_active_session(session)
            for raw_path in session.raw_paths:
                self._add_recent_file(raw_path, "raw")
            self._refresh_tree_raw_paths()

    def _refresh_tree_raw_paths(self) -> None:
        """Push the active set of raw filesystem paths into the tree model.

        Called whenever the session list changes so the file browser's
        custom raw-icon stays accurate. NeXus sessions don't need to be
        listed — anything the model hasn't been told about as raw
        renders with the default NeXus icon.
        """
        model = self.tree.findHdf5TreeModel()
        if not isinstance(model, _MlgidHdf5TreeModel):
            return
        raw_paths: set[str] = set()
        for s in self._sessions:
            if s.kind == "raw" and isinstance(s, RawSession):
                for p in s.raw_paths:
                    raw_paths.add(str(p))
        model.set_raw_paths(raw_paths)

    def _process_open_queue(self) -> None:
        """Kick off the next queued open if no copy is in flight.

        When the queue is exhausted the shared progress dialog is finally
        closed and destroyed — see the comment in ``_open_path`` for why
        we keep one dialog spanning the batch instead of creating a fresh
        one per file.
        """
        if self._thread is not None:
            return
        if not self._open_queue:
            self._dismiss_open_progress()
            return
        self._open_path(self._open_queue.pop(0))

    def _dismiss_open_progress(self) -> None:
        """Hide + destroy the shared open-progress dialog, if any."""
        if self._progress is None:
            return
        self._progress.close()
        # ``close()`` only hides the dialog and keeps it parented to the
        # MainWindow as a hidden child; the WindowModal overlay state on
        # the parent isn't fully released until the dialog is destroyed.
        # ``deleteLater`` schedules destruction on the next event-loop
        # turn, which is what un-dims the window.
        self._progress.deleteLater()
        self._progress = None

    def _action_save(self) -> None:
        self._save(confirm=True)

    def _save(self, confirm: bool, session: BaseSession | None = None) -> bool:
        """Overwrite the original from the temp. Returns True on success.

        Raw sessions have no writable temp copy — Save and Save As are
        no-ops for them. The action is also disabled in the menu, but
        guard here too in case a shortcut fires.
        """
        target = session if session is not None else self._active_session
        if target is None or target.kind != "nexus":
            return False
        assert isinstance(target, NexusSession)
        if confirm:
            reply = QMessageBox.question(
                self,
                "Save",
                f"Overwrite the original file?\n\n{target.original_path}",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Save:
                return False
        try:
            target.save()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self._update_title()
        return True

    def _action_save_as(self) -> None:
        if self.session is None or self.session.kind != "nexus":
            return
        assert isinstance(self.session, NexusSession)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save As",
            str(self.session.original_path),
            NEXUS_FILTER,
        )
        if not path:
            return
        # Release the viewer's FrameSource handle *before* the rename
        # so the file isn't open when shutil.copy2 + rename runs (matters
        # on Windows; harmless on Linux). silx is also detached so the
        # tree can be rebuilt at the new basename.
        self._detach_silx_tree()
        try:
            self.session.save_as(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Save As failed", str(exc))
            # Re-attach so the viewer can keep reading; otherwise the
            # user is left looking at a frozen image with the file
            # detached but nothing written.
            self._reattach_silx_tree()
            return
        # Save As renamed the temp file to match the new basename. The
        # FrameSource was created against the old basename and would
        # otherwise fail to reopen — point it at the new path before
        # the reattach reacquires.
        if (
            self.viewer._frame_source is not None
            and isinstance(self.session, NexusSession)
        ):
            self.viewer._frame_source.relocate(self.session.temp_path)
        # Rebuild the silx tree from the updated session paths.
        self._reattach_silx_tree()
        self._update_title()
        # The user just wrote a new file at ``path``; surface it in the
        # recent menu so it's reopenable from the next session.
        self._add_recent_file(path, "nexus")

    def _action_close_file(self) -> None:
        """Close just the currently-active file. Other files stay open."""
        active = self._active_session
        if active is None:
            return
        if not self._confirm_discard_changes(active):
            return
        self._close_session(active)

    # -- Session lifecycle --

    def _open_path(self, path: Path) -> None:
        # Additive: never tear down existing sessions. The new file simply
        # gets appended to the file browser when the copy completes.
        self._thread = QThread(self)
        self._worker = CopyWorker(path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_open_finished)

        # One shared progress dialog spans the whole open queue. Creating
        # a new WindowModal QProgressDialog per file (and only ``close()``-
        # ing the previous one) used to leave the parent visibly dimmed
        # across the entire batch — Qt re-applied the modal overlay
        # before the previous dialog's hide had finished painting, and
        # the parented hidden dialogs accumulated as zombie children.
        if self._progress is None:
            self._progress = QProgressDialog("Opening file…", "", 0, 0, self)
            self._progress.setWindowTitle(APP_NAME)
            self._progress.setWindowModality(Qt.WindowModality.WindowModal)
            self._progress.setCancelButton(None)
            self._progress.setMinimumDuration(0)
            self._progress.show()
        # Per-file label so the user can see which file is being copied.
        self._progress.setLabelText(f"Opening {path.name}…")

        self._thread.start()

    def _on_open_finished(
        self, session: Session | None, error: Exception | None
    ) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        # Progress dialog is *not* closed per file — it spans the whole
        # queue and is dismissed by ``_process_open_queue`` once the
        # queue is empty. Closing here would re-trigger the modal flicker
        # that this consolidation was meant to avoid.

        if error is not None:
            QMessageBox.critical(self, "Open failed", str(error))
        elif session is not None:
            # Patch any pygid-incompatible metadata in the temp copy
            # (e.g. 0-D angle_of_incidence) before silx + the pipeline
            # see it. Failure here is non-fatal — the file might still
            # work for normal viewing even if a pipeline run later
            # complains.
            try:
                patched = file_model.normalize_for_pygid(session.temp_path)
            except Exception:
                patched = {"angle": [], "frames": []}
            if patched["angle"]:
                self.pipeline_panel.append_log(
                    "Normalized 0-D angle_of_incidence in: "
                    + ", ".join(patched["angle"])
                )
            if patched["frames"]:
                self.pipeline_panel.append_log(
                    "Created missing per-frame analysis groups in: "
                    + ", ".join(patched["frames"])
                )
            self._sessions.append(session)
            self.tree.findHdf5TreeModel().insertFile(str(session.temp_path))
            # Newly-opened file becomes the active one — the user almost
            # always wants to inspect what they just opened.
            self._set_active_session(session)
            # Remember the original (not the temp) so the recent menu
            # reopens the file at its real location next session.
            self._add_recent_file(session.original_path, "nexus")
            self._refresh_tree_raw_paths()

        # Keep draining the queue regardless of this open's outcome so a
        # single bad file in a batch doesn't strand the rest.
        self._process_open_queue()

    def _close_session(self, session: BaseSession) -> None:
        """Remove ``session`` from the window: tear down its tree entry,
        delete its temp dir, and pick a new active if it was the active one.
        """
        was_active = session is self._active_session
        if session not in self._sessions:
            return
        # Stop playback before pulling the file out from under the
        # viewer — the timer's next tick would otherwise read from a
        # released FrameSource.
        if was_active:
            self._pause_playback()
        self._sessions.remove(session)
        # silx exposes no "remove single file" API on Hdf5TreeModel, so we
        # rebuild the tree from the remaining sessions. Cheap — sessions
        # are typically <5 and the model just re-opens HDF5 files.
        self._detach_silx_tree()
        if was_active:
            # Active state is tied to viewer/entry_combo content — drop it
            # before swapping so we don't leak the old session's overlays.
            self.viewer.clear()
            self.viewer.clear_history()
            self.profile_viewer.clear()
            self.entry_combo.blockSignals(True)
            self.entry_combo.clear()
            self.entry_combo.blockSignals(False)
            self._active_session = None
        session.close()
        self._reattach_silx_tree()
        # Closed session may have been raw — rebuild the tree's raw set.
        self._refresh_tree_raw_paths()
        if was_active:
            new_active = self._sessions[-1] if self._sessions else None
            if new_active is not None:
                self._set_active_session(new_active)
            else:
                self._update_title()
                self._update_actions()

    def _set_active_session(self, session: BaseSession | None) -> None:
        """Make ``session`` the active one and reload viewer-side state.

        No-op when ``session`` is already active. Blocked while a pipeline
        run is in flight — the worker captured the active temp_path at run
        time and ``_on_pipeline_finished`` reaches for ``self.session``,
        so swapping mid-flight would corrupt that path.
        """
        if session is self._active_session:
            return
        if self._pipe_thread is not None:
            return
        # Stop playback before swapping so the timer doesn't tick into
        # the new session's viewer state mid-construction.
        self._pause_playback()
        # Tear down viewer-side state belonging to the prior active session
        # before swapping — the new session's overlays must replace, not
        # accumulate on top of, whatever was previously shown.
        self.viewer.clear()
        self.viewer.clear_history()
        self.profile_viewer.clear()
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.blockSignals(False)
        self._active_session = session
        # ExpParameters are derived from the active NeXus metadata, so a
        # CIF cache built against the prior session may be misleading
        # for the new one. Forget it; the user can re-Parse when ready.
        if hasattr(self, "pipeline_panel"):
            self.pipeline_panel.clear_cif_cache()
        if session is not None:
            self._populate_entries()
        else:
            # No active session → wipe the per-entry options on the
            # pipeline panel so they don't reference a closed file.
            self.pipeline_panel.set_available_entries([])
        self._apply_session_mode(session)
        self._update_title()
        self._update_actions()
        # Status bar reflects the active session's entry / frame; the
        # entry change handler will fire shortly when the entry combo
        # repopulates, but pushing the file label now keeps the bar
        # consistent with the title bar even before that fires.
        self._update_status_entry()
        self._update_status_frame()

    def _apply_session_mode(self, session: BaseSession | None) -> None:
        """Toggle dock visibility + viewer affordances for the session kind.

        Pipeline and Conversion docks are mode-exclusive: only one is ever
        visible at a time. Switching between a NeXus and a Raw session
        flips them in lockstep. With no active session, default to the
        Pipeline dock visible — that matches the cold-start UI.

        Raw mode also hides everything that doesn't apply to a raw
        detector frame: peak overlays / matched structures / profile
        viewer / parameter panel / Cartesian-Polar radios / Tools >
        Clear-peaks submenu. The user gets a clean canvas focused on
        the conversion workflow.
        """
        is_raw = session is not None and session.kind == "raw"
        self._pipeline_dock.setVisible(not is_raw)
        self._conversion_dock.setVisible(is_raw)
        if is_raw:
            self._conversion_dock.raise_()
        else:
            # Keep Display in front by default for NeXus sessions; users
            # who prefer Pipeline up-front can click its tab.
            self._display_dock.raise_()
        # Hide NeXus-mode-only widgets in raw mode.
        self._profile_dock.setVisible(not is_raw)
        if hasattr(self, "parameter_panel"):
            self.parameter_panel.setVisible(not is_raw)
        # Cartesian / Polar radios — meaningless before conversion.
        self.viewer.set_mode_radios_visible(not is_raw)
        # Tools > Clear peaks submenu has nothing to clear in raw mode.
        for action in (
            getattr(self, "action_clear_detected", None),
            getattr(self, "action_clear_fitted", None),
            getattr(self, "action_clear_matched", None),
        ):
            if action is not None:
                action.setEnabled(not is_raw)

    def _confirm_discard_changes(self, session: BaseSession | None = None) -> bool:
        target = session if session is not None else self._active_session
        if target is None or not target.dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"{target.original_path.name} has unsaved changes. "
            f"Save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save(confirm=False, session=target)
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False

    # -- silx tree helpers --

    def _detach_silx_tree(self) -> None:
        """Release silx's read handles + the viewer's FrameSource handle.

        Required before any code path opens an HDF5 file ``r+`` (pipeline
        runs, direct h5py edits) since open read handles would otherwise
        block the writer. After the lazy-loading milestone the viewer
        also holds a long-lived h5py handle through its FrameSource —
        that handle must be released here in addition to silx's.

        Both calls are wrapped in try/except: silx's ``clear()`` walks
        every Hdf5Item to close its owned file via ``obj.filename``,
        and on a stale ``obj`` that raises ``ValueError: Not a file or
        file object``. We swallow such errors so a partial-clear
        doesn't strand the detach half-done — the next reattach
        rebuilds the model from scratch anyway.
        """
        try:
            self.tree.findHdf5TreeModel().clear()
        except Exception:
            # silx's clear() can blow up when an Hdf5Item references a
            # closed h5py file. Swallow it; the reattach rebuilds the
            # whole tree from the live session list.
            pass
        try:
            self.data_viewer.setData(None)
        except Exception:
            pass
        self.viewer.release_frame_source()
        # Tell the background prefetch worker to drop its own h5py
        # handle too. mlgidbase opens the same file r+ in the worker
        # we're about to spawn; an outstanding read handle from the
        # prefetcher would either contend (Windows) or silently
        # serve pre-write data into the LRU mid-pipeline (Linux).
        if self._prefetch_worker is not None:
            self._prefetchRelease.emit()

    def _reattach_silx_tree(self) -> None:
        """Re-insert every session's files + reopen the viewer's
        FrameSource handle.

        NeXus sessions contribute one file (the temp working copy); raw
        sessions contribute every selected raw input so the user can keep
        browsing all of them while configuring conversion. The custom
        per-file icon set is repushed afterwards because clear() emptied
        the model. The viewer's FrameSource is reopened so subsequent
        frame reads can stream from disk again.

        Each ``insertFile`` is independent — one bad path doesn't
        strand the rest. silx returns a node reference on success and
        raises ``OSError`` on a missing/corrupt file; either way we
        continue with the next session.
        """
        model = self.tree.findHdf5TreeModel()
        for s in self._sessions:
            try:
                if isinstance(s, RawSession):
                    for raw_path in s.raw_paths:
                        model.insertFile(str(raw_path))
                else:
                    model.insertFile(str(s.temp_path))
            except Exception:
                # One bad session shouldn't strand the rest. The user
                # will see the missing entry; rebuild at next detach.
                pass
        self._refresh_tree_raw_paths()
        self.viewer.acquire_frame_source()

    # -- Entry / viewer wiring --

    def _populate_entries(self) -> None:
        if self.session is None:
            return
        if self.session.kind == "raw":
            self._populate_raw_entries()
            return
        try:
            entries = file_model.list_entries(self.session.temp_path)
        except Exception as exc:
            QMessageBox.warning(self, "Read failed", f"Could not list entries: {exc}")
            return
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.addItems(entries)
        self.entry_combo.blockSignals(False)
        # Push the same entry list into the pipeline panel's per-section
        # scope dropdowns so the user can pick a specific entry instead
        # of being limited to ACTIVE / ALL.
        self.pipeline_panel.set_available_entries(entries)
        if entries:
            self._load_entry_into_viewer(entries[0])
        else:
            # Empty entry combo isn't always "empty file" — it's much more
            # often "file has entries but none are 2D q-images". Tell the
            # user exactly what's in the file so they don't think the GUI
            # silently dropped a working file.
            self._warn_no_q_entries()

    def _populate_raw_entries(self) -> None:
        """Walk every raw file in the active session and populate the
        entry combo with its 3D detector-image candidates.

        Combo items are labeled ``filename::dataset/path`` so the user
        can disambiguate when the batch contains multiple files. Pipeline
        panel's per-entry scope dropdown is cleared — pipeline ops aren't
        meaningful in raw mode. The Conversion panel also receives the
        same set of (file, entries) tuples for its selection tree.
        """
        assert isinstance(self.session, RawSession)
        # Maintain a mapping from combo label → RawEntry so the change
        # handler can resolve a click without re-walking the HDF5 file.
        self._raw_entries: dict[str, file_model.RawEntry] = {}
        labels: list[str] = []
        panel_inputs: list[tuple[Path, list[file_model.RawEntry]]] = []
        for raw_path in self.session.raw_paths:
            try:
                entries = file_model.list_raw_entries(raw_path)
            except Exception as exc:
                self.conversion_panel.append_log(
                    f"Could not read {raw_path.name}: {exc}"
                )
                panel_inputs.append((raw_path, []))
                continue
            panel_inputs.append((raw_path, entries))
            for re in entries:
                self._raw_entries[re.label] = re
                labels.append(re.label)
        # Push the same data into the Conversion panel for its selection
        # tree. Done before populating the combo so the panel paint
        # happens once on activation.
        self.conversion_panel.set_raw_inputs(panel_inputs)
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.addItems(labels)
        self.entry_combo.blockSignals(False)
        self.pipeline_panel.set_available_entries([])
        if labels:
            # Auto-load the first candidate so the user sees something
            # immediately. The change handler handles further picks.
            self._load_raw_entry_into_viewer(labels[0])
        else:
            QMessageBox.information(
                self,
                "No raw datasets found",
                "None of the selected raw files contain a 3D detector "
                "dataset (shape (N, H, W) with H, W ≥ 32). Check the "
                "files in the tree on the left to see their structure.",
            )

    def _load_raw_entry_into_viewer(self, label: str) -> None:
        """Load the picked raw entry into the viewer in pixel coords.

        ``label`` is the combo's display string (file::dataset/path).
        Resolved through ``self._raw_entries`` to a ``RawEntry`` so
        the loader can pull the right dataset.
        """
        raw_entry = getattr(self, "_raw_entries", {}).get(label)
        if raw_entry is None:
            return
        try:
            arr = file_model.load_raw_dataset(raw_entry)
        except Exception as exc:
            QMessageBox.warning(
                self, "Load failed",
                f"Could not load {raw_entry.label}: {exc}",
            )
            return
        self.viewer.show_raw_stack(arr)
        self._refresh_frame_slider()

    def _warn_no_q_entries(self) -> None:
        """Diagnose why the entry combo ended up empty.

        The viewer + pipeline only handle ``img_gid_q`` entries. Files
        with reduced-data entries (``horiz_cut_gid``, ``rad_cut_gid``,
        polar-only ``img_gid_pol``, etc.) load fine but produce no
        viewable entry, which previously looked like the GUI broke.
        """
        try:
            signals = file_model.list_entry_signals(self.session.temp_path)
        except Exception:
            signals = {}
        if not signals:
            # Truly empty — file genuinely has no entry_* groups.
            QMessageBox.information(
                self,
                "Nothing to show",
                f"{self.session.original_path.name} has no entry_* groups "
                "to display.",
            )
            return
        rows = "\n".join(
            f"  • {name} — signal = {signal!r}"
            for name, signal in signals.items()
        )
        QMessageBox.information(
            self,
            "No 2D q-image entries",
            f"{self.session.original_path.name} loaded successfully but "
            "contains no entries with the 2D q-image data the viewer + "
            "pipeline operate on (signal = 'img_gid_q').\n\n"
            f"Entries found:\n{rows}\n\n"
            "These are likely reduced data (1D cuts, polar grids, or "
            "post-processed outputs). To use the GUI's detection / "
            "fitting / matching tools, open a NeXus file produced by "
            "the pygid → mlgidDETECT pipeline that still carries the "
            "raw q-image stack.",
        )

    def _on_entry_changed(self, entry: str) -> None:
        if not entry or self.session is None:
            return
        if self.session.kind == "raw":
            self._load_raw_entry_into_viewer(entry)
        else:
            self._load_entry_into_viewer(entry)
        self._update_status_entry()
        self._update_status_frame()

    def _on_frame_slider_changed(self, value: int) -> None:
        """User dragged the Display-dock slider — push to the viewer.

        ``viewer.set_frame`` is a no-op when ``value`` already matches
        the current frame, so the bidirectional sync (slider→viewer→
        slider via _on_viewer_frame_changed) doesn't recurse.
        """
        self.viewer.set_frame(value)

    def _on_viewer_frame_changed(self, frame: int) -> None:
        """Viewer changed frame (timeline scrub, programmatic seek, etc.)
        — keep the Display-dock slider + label in sync without
        re-emitting valueChanged back into the viewer. Also pushes
        the new play-head into the background prefetch worker so
        its sliding window slides with the user.
        """
        self.frame_label.setText(self._frame_label_text(frame))
        if self.frame_slider.value() != frame:
            self.frame_slider.blockSignals(True)
            try:
                self.frame_slider.setValue(int(frame))
            finally:
                self.frame_slider.blockSignals(False)
        self._update_status_frame()
        # Tell the prefetch worker where the play-head is now. The
        # ``active`` flag tracks the play-button state so a manual
        # scrub doesn't accidentally wake the worker. Pass the live
        # step so the worker walks the same stride as the player.
        if self._prefetch_worker is not None:
            active = self.play_button.isChecked()
            step = self._play_step if active else 1
            self._prefetchUpdate.emit(int(frame), active, step)

    def _on_play_toggled(self, checked: bool) -> None:
        """Start / stop the frame-playback timer.

        Press → if the current frame is already at the end, restart
        from frame 0; otherwise advance from the current frame. Press
        again to pause. The icon flips between Play and Pause.

        Refuses to start during a pipeline run (the viewer is gated
        ``busy`` during those, so frame edits would block anyway).

        Reads the current playback settings from QSettings on every
        press so a setting change applies on the next play without
        any restart machinery.
        """
        if checked:
            if self._pipe_thread is not None or self.viewer.n_frames <= 1:
                # Bail: the play button toggle was either an erroneous
                # programmatic click or fired while a pipeline run owns
                # the viewer. Unchecking re-fires this slot with
                # checked=False, which is a no-op.
                self.play_button.setChecked(False)
                return
            # Wrap around when at the end so the second click of Play
            # always plays the full sequence.
            if self.viewer.current_frame >= self.viewer.n_frames - 1:
                self.viewer.set_frame(0)
            interval, step = self._compute_play_schedule()
            self._play_timer.setInterval(interval)
            self._play_step = step
            self.play_button.setIcon(self._icon_pause)
            self.play_button.setToolTip("Pause playback")
            self._play_timer.start()
            # Activate the background prefetch worker — it'll start
            # warming frames just ahead of the play-head, stepping
            # the same way the player does so prefetched frames
            # actually match the ones we'll display.
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, True, step,
                )
        else:
            self._play_timer.stop()
            self._play_step = 1
            self.play_button.setIcon(self._icon_play)
            self.play_button.setToolTip(
                "Play frames from the current position to the end.\n"
                "Stops at the last frame; click again to pause."
            )
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, False, 1,
                )

    def _compute_play_schedule(self) -> tuple[int, int]:
        """Resolve ``(timer_interval_ms, frame_step)`` from QSettings.

        The user expresses a desired *per-frame* duration — either
        directly (Time-per-frame mode) or implicitly (Total-time mode
        ÷ n_frames). If that desired duration is at or above
        ``PLAYBACK_TICK_FLOOR_MS`` (≈ 20 fps), playback uses it
        directly with ``step=1``. If it's *below* that floor, the
        timer is held at the floor and ``step`` is bumped so the
        play-head jumps multiple frames per tick — i.e. we honour the
        target total time by skipping frames instead of asking Qt to
        fire faster than the display + disk can keep up. 20 fps is
        more than enough to perceive the time-series motion; the
        skipped frames are still reachable via the slider.

        Out-of-bounds / unparseable stored values fall back to the
        defaults so a corrupted QSettings entry can't soft-lock the
        Play button.
        """
        settings = QSettings()
        mode = settings.value(self._PLAYBACK_MODE_KEY, PLAYBACK_MODE_FRAME)
        if mode == PLAYBACK_MODE_TOTAL:
            try:
                total_s = float(settings.value(
                    self._PLAYBACK_TOTAL_S_KEY, DEFAULT_PLAYBACK_TOTAL_S
                ))
            except (TypeError, ValueError):
                total_s = DEFAULT_PLAYBACK_TOTAL_S
            total_s = max(PLAYBACK_TOTAL_S_MIN,
                          min(PLAYBACK_TOTAL_S_MAX, total_s))
            steps = max(self.viewer.n_frames - 1, 1)
            desired_ms = total_s * 1000.0 / steps
        else:
            try:
                frame_ms = int(settings.value(
                    self._PLAYBACK_FRAME_MS_KEY, DEFAULT_PLAYBACK_FRAME_MS
                ))
            except (TypeError, ValueError):
                frame_ms = DEFAULT_PLAYBACK_FRAME_MS
            desired_ms = float(max(PLAYBACK_FRAME_MS_MIN,
                                   min(PLAYBACK_FRAME_MS_MAX, frame_ms)))

        if desired_ms < PLAYBACK_TICK_FLOOR_MS:
            # Below the 20 fps ceiling — bunch frames together per tick.
            # step = ceil(floor / desired) so the per-frame time stays
            # ≤ desired (= we never play slower than asked). Interval
            # then = desired * step, which lands at or just above the
            # floor.
            step = max(1, int(math.ceil(PLAYBACK_TICK_FLOOR_MS / desired_ms)))
            interval_ms = max(PLAYBACK_TICK_FLOOR_MS,
                              int(round(desired_ms * step)))
        else:
            step = 1
            interval_ms = int(round(desired_ms))
        return interval_ms, step

    def _on_play_tick(self) -> None:
        """One step of frame playback.

        Stops at end-of-stack. Auto-pauses if the viewer becomes busy
        (pipeline run kicked off mid-playback) or the user closed the
        file. The slider's ``valueChanged`` connection routes the
        frame change through ``viewer.set_frame`` so the existing
        sync paths fire exactly once per step.
        """
        if (
            self.viewer.n_frames <= 1
            or self._pipe_thread is not None
            or self.session is None
        ):
            self.play_button.setChecked(False)
            return
        step = max(1, self._play_step)
        next_frame = self.viewer.current_frame + step
        if next_frame >= self.viewer.n_frames:
            # Snap to the last frame so the user always sees the end
            # of the sequence even when ``step`` would overshoot — then
            # pause. Click Play again to wrap to frame 0 (the toggle
            # handler handles the wrap).
            last = self.viewer.n_frames - 1
            if self.viewer.current_frame < last:
                self.viewer.set_frame(last)
            self.play_button.setChecked(False)
            return
        self.viewer.set_frame(next_frame)

    def _pause_playback(self) -> None:
        """Stop the playback timer if it's running.

        Called from session-swap / file-close / pipeline-start paths
        so playback doesn't tick into a torn-down viewer or contend
        with a pipeline write. Safe to call when playback is already
        stopped.
        """
        if self.play_button.isChecked():
            self.play_button.setChecked(False)

    # -- Background prefetch worker ---------------------------------------

    def _ensure_prefetch_worker(self) -> None:
        """Spawn the prefetch worker + thread on first use. Idempotent.

        Lazy spawn keeps startup fast for users who only ever view
        single-frame files (no playback, no prefetch worth running).
        Once spawned, the worker survives across entry switches —
        each new entry triggers ``configure()`` rather than a
        rebuild.
        """
        if self._prefetch_worker is not None:
            return
        self._prefetch_thread = QThread(self)
        self._prefetch_worker = PrefetchWorker()
        self._prefetch_worker.moveToThread(self._prefetch_thread)
        # Cross-thread wiring. configure / update_state / release run
        # on the worker's thread via queued connections; prefetched
        # signal delivers back to the GUI thread.
        self._prefetchConfigure.connect(
            self._prefetch_worker.configure, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetchUpdate.connect(
            self._prefetch_worker.update_state, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetchRelease.connect(
            self._prefetch_worker.release, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetch_worker.prefetched.connect(
            self._on_prefetched, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetch_thread.start()

    def _configure_prefetch_for_active_entry(self) -> None:
        """Tell the worker about the active entry's shape + LRU size.

        Called after every successful entry load (in
        ``_load_entry_into_viewer``) and after the silx-reattach
        path completes a pipeline run. No-op for single-frame
        stacks (nothing to prefetch) and for raw sessions
        (FrameSource isn't used).
        """
        if (
            self.session is None
            or self.session.kind != "nexus"
            or self.viewer._frame_source is None
            or self.viewer.n_frames <= 1
        ):
            # Release the worker if we have one — no work on idle.
            if self._prefetch_worker is not None:
                self._prefetchRelease.emit()
            return
        self._ensure_prefetch_worker()
        fs = self.viewer._frame_source
        # Sliding-window size = LRU - 1 so the prefetcher can never
        # evict frames the play-head still needs to reach.
        window = max(1, fs.cart_lru_size - 1)
        entry = self.entry_combo.currentText()
        if not entry:
            return
        self._prefetchConfigure.emit(
            str(self.session.temp_path), entry, fs.n_frames, window,
        )
        # Start in paused state — the worker only ticks during
        # active playback. The play-button toggle (and any frame
        # change while playing) will flip ``active=True`` via
        # _prefetchUpdate.
        self._prefetchUpdate.emit(self.viewer.current_frame, False, 1)

    @Slot(int, object, object, object, object)
    def _on_prefetched(
        self,
        idx: int,
        cart: object,
        polar: object,
        radius: object,
        angle: object,
    ) -> None:
        """Deposit a prefetched frame into the active FrameSource's LRU.

        Runs on the GUI thread (queued from the worker). The
        FrameSource's LRUs are touched only here and from the
        synchronous ``get_cartesian`` / ``get_polar`` paths, both
        of which live on the GUI thread — so no locking is needed.

        Drops the result silently if the FrameSource has been
        released (post-pipeline detach), since a stale signal in
        flight should not warm a closed cache.
        """
        fs = self.viewer._frame_source
        if fs is None or not fs.is_open:
            return
        try:
            fs.warm_cartesian(int(idx), cart)        # type: ignore[arg-type]
            fs.warm_polar(int(idx), polar, radius, angle)  # type: ignore[arg-type]
        except Exception:
            # Defensive — a stale signal during teardown shouldn't
            # propagate.
            pass

    def _refresh_frame_slider(self) -> None:
        """Match the slider's range + value to the active stack's
        frame count. Called after every show_stack — covers entry
        switches, file opens, and pipeline-finished reloads.
        Single-frame stacks hide the row entirely.
        """
        n = self.viewer.n_frames
        cur = self.viewer.current_frame
        self.frame_slider.blockSignals(True)
        try:
            if n <= 1:
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(0)
                self.frame_slider.setValue(0)
            else:
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(n - 1)
                self.frame_slider.setValue(int(cur))
        finally:
            self.frame_slider.blockSignals(False)
        self.frame_label.setText(self._frame_label_text(cur))
        self._set_frame_slider_visible(n > 1)

    def _set_frame_slider_visible(self, visible: bool) -> None:
        """Show or hide the slider row + its left-column label.

        QFormLayout doesn't have a single "hide row" call in older
        PySide6 versions, so we pull the label widget out of the form
        directly and toggle it alongside the slider widget.
        """
        self._frame_row_widget.setVisible(visible)
        # The left-column "Frame:" label was added by addRow; reach it
        # via labelForField so it hides in lockstep.
        form = self._frame_row_widget.parentWidget().layout()
        try:
            label = form.labelForField(self._frame_row_widget)
        except Exception:
            label = None
        if label is not None:
            label.setVisible(visible)

    def _frame_label_text(self, idx: int) -> str:
        n = self.viewer.n_frames
        if n <= 1:
            return "Frame —"
        return f"Frame {int(idx)} / {n - 1}"

    def _safe_selected_h5_nodes(self) -> list:
        """Return ``selectedH5Nodes`` results, swallowing silx model errors.

        Under certain races (mid-pipeline detach/reattach, freshly
        inserted file with not-yet-resolved h5py state), silx's tree
        model can raise on attribute lookup deep inside the proxy
        chain. Qt then re-fires the call, producing a stack-busting
        recursion that brings down the click handler. We catch
        anything from that path here so a single bad click can't
        wedge the GUI.
        """
        try:
            return list(self.tree.selectedH5Nodes())
        except (RecursionError, RuntimeError, KeyError, OSError) as exc:
            self.pipeline_panel.append_log(
                f"WARN — silx tree query failed ({type(exc).__name__}); "
                "rebuilding the file browser"
            )
            # Drastic but reliable: tear the tree down and rebuild it
            # from the live session list. Any orphan / half-loaded
            # silx items get dropped in the process.
            self._detach_silx_tree()
            self._reattach_silx_tree()
            return []

    def _on_tree_selection_changed(self, *_: object) -> None:
        nodes = self._safe_selected_h5_nodes()
        if not nodes:
            return
        node = nodes[0]
        self.data_viewer.setData(node)
        # Multiple files may be loaded — clicking into a different file's
        # subtree promotes that file to the active session so the entry
        # combo, image viewer, and per-file actions follow the user's
        # focus without an extra click.
        self._activate_session_for_node(node)
        # Click into entry_X anywhere → switch the image tab to that
        # entry. The entry-combo signal already triggers the viewer
        # reload, so we just push the new value here.
        self._activate_entry_for_node(node)

    def _on_tree_activated(self, *_: object) -> None:
        nodes = self._safe_selected_h5_nodes()
        if not nodes:
            return
        node = nodes[0]
        self.data_viewer.setData(node)
        self.tabs.setCurrentWidget(self.data_viewer)
        self._activate_session_for_node(node)
        self._activate_entry_for_node(node)

    def _activate_entry_for_node(self, node) -> None:
        """If the clicked node is inside an ``entry_*`` group, switch the
        entry combo (and therefore the image viewer) to that entry.

        No-ops for clicks on the file root or on nodes outside any entry
        group (e.g. top-level metadata). Also no-ops if the entry isn't
        in the combo — that would mean it's a non-q entry filtered out
        by ``list_entries``, where the viewer can't render anything
        useful anyway.
        """
        entry = self._node_entry_name(node)
        if entry is None:
            return
        if self.entry_combo.findText(entry) < 0:
            return
        if self.entry_combo.currentText() == entry:
            return
        # Triggers _on_entry_changed → _load_entry_into_viewer.
        self.entry_combo.setCurrentText(entry)

    @staticmethod
    def _node_entry_name(node) -> str | None:
        """Extract the ``entry_*`` group name from a node's HDF5 path.

        silx exposes the absolute path as ``local_name`` (e.g.
        ``/entry_0000/data/img_gid_q``); we take the first component if
        it begins with ``entry_``. Returns None for nodes outside any
        entry group.
        """
        for getter in (
            lambda n: getattr(n, "local_name", None),
            lambda n: n.h5py_object.name,
        ):
            try:
                p = getter(node)
            except Exception:
                continue
            if p:
                parts = str(p).lstrip("/").split("/")
                if parts and file_model.is_entry_group_name(parts[0]):
                    return parts[0]
                return None
        return None

    def _activate_session_for_node(self, node) -> None:
        """If ``node`` lives in a non-active session's file, swap active.

        silx normalizes paths through the OS, so a literal ``Path`` equality
        with ``session.temp_path`` can fail when one side has a symlink,
        dotfile component, or trailing slash that the other doesn't —
        previously this silently left the wrong session active and the
        pipeline ran on the most recently opened file regardless of which
        tree the user clicked. ``Path.resolve()`` collapses both sides to
        a canonical absolute form before the comparison.
        """
        fname = self._node_filename(node)
        if fname is None:
            return
        try:
            target = fname.resolve()
        except OSError:
            target = fname
        for s in self._sessions:
            # Raw sessions own multiple files in the tree; any of them
            # should activate the same RawSession. NeXus sessions own
            # exactly one file (the temp working copy).
            if isinstance(s, RawSession):
                candidate_paths = list(s.raw_paths)
            else:
                candidate_paths = [s.temp_path]
            for candidate in candidate_paths:
                try:
                    candidate_resolved = candidate.resolve()
                except OSError:
                    candidate_resolved = candidate
                if candidate_resolved == target:
                    if s is not self._active_session:
                        self._set_active_session(s)
                    return

    @staticmethod
    def _node_filename(node) -> Path | None:
        """Resolve the filesystem path of the file ``node`` was loaded from.

        silx exposes this differently across versions — fall through the
        known accessors and give up silently if nothing answers.
        """
        for getter in (
            lambda n: getattr(n, "local_filename", None),
            lambda n: n.h5py_object.file.filename,
        ):
            try:
                p = getter(node)
            except Exception:
                continue
            if p:
                return Path(p)
        return None

    # -- Profile viewer adapters --

    def _forward_selection_to_profile(self, sel: SelectedPeak | None) -> None:
        # Profiles render for any kind of selection; the profile viewer
        # internally makes regions non-movable for non-manual peaks since
        # those are edited through the 2D ROI.
        self.profile_viewer.set_selected_peak(sel)

    def _forward_geom_to_profile(self, sel: SelectedPeak | None) -> None:
        if sel is None:
            return
        self.profile_viewer.sync_regions_from_peak(sel)

    def _on_selection_for_preview(self, sel: SelectedPeak | None) -> None:
        """Drop the fitted-preview overlay when the active selection isn't
        a candidate-for-fitted peak. Manual + detected are both candidates
        — Add-to-fitted is enabled for either — so the preview is shown
        for both kinds. Fitted / matched already have a stored box, so a
        cyan refit overlay there would be visual noise.
        """
        if sel is None or sel.kind not in ("manual", "detected"):
            self.viewer.set_fitted_preview(None, None, None, None)

    def _update_fitted_preview(self, rfit, afit) -> None:
        """Sync the viewer's fitted-preview box to the latest 1D fits.

        Relevant for manual + detected selections — both feed Add-to-fitted.
        File-resident fitted / matched peaks already carry their stored box
        and aren't previewed here. ``rfit`` / ``afit`` may be ``None`` (no
        convergence) → clear the preview unless we're previewing a ring,
        in which case only ``rfit`` matters.
        """
        sel = self.viewer.selected_peak
        if sel is None or sel.kind not in ("manual", "detected"):
            self.viewer.set_fitted_preview(None, None, None, None)
            return
        save_as_ring = self.parameter_panel.save_as_ring()
        if rfit is None:
            self.viewer.set_fitted_preview(None, None, None, None)
            return
        if save_as_ring:
            # Angular fit isn't required for rings — pass placeholders.
            self.viewer.set_fitted_preview(
                float(rfit.center), float(rfit.fwhm),
                None, None,
                is_ring=True,
            )
            return
        if afit is None:
            self.viewer.set_fitted_preview(None, None, None, None)
            return
        self.viewer.set_fitted_preview(
            float(rfit.center), float(rfit.fwhm),
            float(afit.center), float(afit.fwhm),
            is_ring=False,
        )

    def _on_save_as_ring_changed(self, is_ring: bool) -> None:
        """Toggle between segment / ring preview.

        Three coordinated effects:

        1. The profile viewer skips the angular Gaussian fit while ring
           is active — that fit wouldn't be saved by Add-to-fitted.
        2. If a manual peak is selected, its angular sweep is widened
           to span the full polar plot height (so the radial profile
           integrates over the entire angular axis, matching what the
           ring fit will eventually represent). The pre-ring geometry
           is stashed so unticking the box — including the auto-uncheck
           that fires after Add-to-fitted commits — restores the box.
        3. The fitted-preview is recomputed against the new fit cache.
        """
        sel = self.viewer.selected_peak
        manual_ref = (
            sel.manual_ref if sel is not None and sel.kind == "manual" else None
        )

        if is_ring and manual_ref is not None:
            # Stash pre-ring geometry once. If the user ticks → unticks
            # → re-ticks without committing, we keep the original stash
            # so the eventual restore returns to the very first state,
            # not the intermediate ring state.
            if self._ring_pre_geom is None:
                self._ring_pre_geom = (
                    manual_ref,
                    manual_ref.radius,
                    manual_ref.angle,
                    manual_ref.radius_width,
                    manual_ref.angle_width,
                    manual_ref.is_ring,
                )
            extent = self.viewer.angular_extent()
            if extent is not None:
                a_lo, a_hi = extent
                ring_angle = 0.5 * (a_lo + a_hi)
                ring_width = abs(a_hi - a_lo)
                self.viewer.set_manual_geometry(
                    manual_ref,
                    radius=manual_ref.radius,
                    angle=ring_angle,
                    radius_width=manual_ref.radius_width,
                    angle_width=ring_width,
                    is_ring=True,
                )
        elif not is_ring and self._ring_pre_geom is not None:
            (
                stashed_peak,
                pre_r,
                pre_a,
                pre_dr,
                pre_da,
                pre_is_ring,
            ) = self._ring_pre_geom
            self._ring_pre_geom = None
            # Only restore if the stashed peak still exists — the user
            # may have drawn a replacement (which removes the original
            # via the single-box policy) while ring was active. In that
            # case the new peak inherited the ring geometry but has no
            # captured pre-state, so leave it alone.
            for peaks in self.viewer._manual_peaks.values():
                if stashed_peak in peaks:
                    self.viewer.set_manual_geometry(
                        stashed_peak,
                        radius=pre_r,
                        angle=pre_a,
                        radius_width=pre_dr,
                        angle_width=pre_da,
                        is_ring=pre_is_ring,
                    )
                    break

        # Drop the angular fit *before* recomputing the preview so the
        # cached afit is None when _update_fitted_preview reads it.
        self.profile_viewer.set_skip_angular_fit(is_ring)
        fits = self.profile_viewer.last_fit_params()
        self._update_fitted_preview(fits.get("radial"), fits.get("angular"))

    def _on_manual_peak_added(self, _frame: int, peak: ManualPeak) -> None:
        """Apply the active ring expansion to a freshly added manual peak.

        When the user draws a new manual box while the ring checkbox is
        on, the single-box-replace removes the old (with its ring stash)
        and adds the new one. Without this slot, the new box would stay
        as drawn — confusing because the checkbox is still ticked. We
        mirror what ``_on_save_as_ring_changed(True)`` would do for the
        new peak: stash its pre-ring shape, then expand to the full
        angular sweep.
        """
        if not self.parameter_panel.save_as_ring():
            return
        # Stash pre-ring state for the new peak. Any earlier stash
        # already pointed at a peak that's been removed (which our
        # manualPeakRemoved slot has already cleared).
        self._ring_pre_geom = (
            peak,
            peak.radius,
            peak.angle,
            peak.radius_width,
            peak.angle_width,
            peak.is_ring,
        )
        extent = self.viewer.angular_extent()
        if extent is None:
            return
        a_lo, a_hi = extent
        self.viewer.set_manual_geometry(
            peak,
            radius=peak.radius,
            angle=0.5 * (a_lo + a_hi),
            radius_width=peak.radius_width,
            angle_width=abs(a_hi - a_lo),
            is_ring=True,
        )

    def _on_manual_peak_removed(self, _frame: int, peak: ManualPeak) -> None:
        """Invalidate ``_ring_pre_geom`` when the peak it references goes away.

        Without this, an Esc / Delete / Add-to-detected on a ring-
        expanded peak would leave a dangling stash; later unticking
        the ring checkbox would walk the manual list looking for that
        ghost and find nothing, but the stash stays set and could
        mis-fire on a later toggle cycle.
        """
        if (
            self._ring_pre_geom is not None
            and self._ring_pre_geom[0] is peak
        ):
            self._ring_pre_geom = None

    # -- Pipeline --

    def _on_parse_cifs_requested(self, cif_input: str) -> None:
        """Run CIF parsing on a worker thread + post the result back.

        CIF preprocessing simulates every CIF and can take several
        seconds; the worker keeps the GUI responsive. Only one parse
        runs at a time — the panel's button stays disabled until we
        post the result back via ``set_cif_pattern``.
        """
        if self._cif_parse_thread is not None:
            return
        if self.session is None:
            self.pipeline_panel.set_cif_pattern(
                None, RuntimeError("Open a NeXus file first.")
            )
            return
        nexus_file = self.session.temp_path
        # Pass the active entry through so CifPattern is simulated
        # against that entry's energy / angle of incidence — multi-
        # energy datasets need this to match correctly.
        active_entry = self.entry_combo.currentText() or None
        self._cif_parse_thread = QThread(self)
        self._cif_parse_worker = CifParseWorker(
            cif_input, nexus_file, active_entry
        )
        self._cif_parse_worker.moveToThread(self._cif_parse_thread)
        self._cif_parse_thread.started.connect(self._cif_parse_worker.run)
        self._cif_parse_worker.finished.connect(self._on_parse_cifs_finished)
        self._cif_parse_thread.start()

    def _on_parse_cifs_finished(
        self, result: object | None, error: Exception | None
    ) -> None:
        if self._cif_parse_thread is not None:
            self._cif_parse_thread.quit()
            self._cif_parse_thread.wait()
            self._cif_parse_thread.deleteLater()
            self._cif_parse_thread = None
        if self._cif_parse_worker is not None:
            self._cif_parse_worker.deleteLater()
            self._cif_parse_worker = None
        self.pipeline_panel.set_cif_pattern(result, error)
        if error is not None:
            self.pipeline_panel.append_log(f"CIF parse failed: {error}")
        elif result is not None:
            n = len(getattr(result, "cifs", []) or [])
            self.pipeline_panel.append_log(
                f"CIF cache loaded ({n} CIFs) — reused across matching runs"
            )

    def _on_run_requested(self, command: PipelineCommand) -> None:
        """Dispatch a runRequested command from the pipeline panel.

        "All entries" runs are expanded into one ``PipelineCommand`` per
        q-entry and queued sequentially — the user gets per-entry log
        lines and a single bad entry doesn't strand the others. A command
        that already names an explicit ``entry`` (or runs on a file with
        a single entry) goes straight through unchanged.

        ``add_peak`` and ``delete_peak`` always carry a specific
        ``entry`` already; only the run_* ops are subject to expansion.
        """
        if self.session is None:
            return
        if (
            command.op_name in ("run_detection", "run_fitting", "run_matching")
            and "entry" not in command.kwargs
        ):
            try:
                entries = file_model.list_entries(self.session.temp_path)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Pipeline", f"Could not list entries: {exc}"
                )
                return
            if not entries:
                # No q-entries to run on — fall through and let mlgidbase
                # raise its usual "no entries" message in the log.
                self._enqueue_pipeline(command)
                return
            for entry in entries:
                self._enqueue_pipeline(
                    PipelineCommand(
                        command.op_name,
                        {**command.kwargs, "entry": entry},
                    )
                )
        else:
            self._enqueue_pipeline(command)

    def _enqueue_pipeline(self, command: PipelineCommand) -> None:
        """Queue ``command`` and start it if no run is in flight."""
        self._pipeline_queue.append(command)
        if self._pipe_thread is None:
            self._run_next_pipeline_command()

    def _run_next_pipeline_command(self) -> None:
        """Pop the next queued command and start it, if any."""
        if self._pipe_thread is not None or not self._pipeline_queue:
            return
        command = self._pipeline_queue.pop(0)
        self._on_pipeline_run(command)

    def _on_pipeline_run(self, command: PipelineCommand) -> None:
        if self.session is None or self._pipe_thread is not None:
            return

        # Stop frame playback if it's running — the pipeline owns the
        # file r+ for the duration of the run and ticking would either
        # contend on the silx detach or read post-write data
        # mid-render.
        self._pause_playback()
        self.pipeline_panel.set_running(True)
        self.parameter_panel.set_busy(True)
        self.viewer.set_busy(True)
        self._update_status_pipeline(command, running=True)
        # Any pipeline op that reshuffles peak ids (everything except
        # add_peak, which only appends) invalidates pending FileGeomActions.
        # add_peak is handled by commit_manual_peak's targeted scrub.
        if command.op_name != "add_peak":
            self.viewer.clear_history()
            self.viewer.clear_selection()
        # Per-run header: include the entry scope when present so the
        # user can see which entry is being processed in a multi-entry
        # batch.
        entry_tag = command.kwargs.get("entry")
        if entry_tag:
            self.pipeline_panel.append_log(
                f"--- {command.op_name} on {entry_tag} ---"
            )
        else:
            self.pipeline_panel.append_log(
                f"--- {command.op_name} (all entries) ---"
            )

        # Release silx's read handles on every loaded temp file so mlgidbase
        # can open the active one r+. Sibling files are reattached on finish.
        self._detach_silx_tree()

        self._pipe_thread = QThread(self)
        self._pipe_worker = PipelineWorker(self.session.temp_path, command)
        self._pipe_worker.moveToThread(self._pipe_thread)
        self._pipe_worker.log.connect(self.pipeline_panel.append_log)
        self._pipe_worker.finished.connect(self._on_pipeline_finished)
        self._pipe_thread.started.connect(self._pipe_worker.run)
        self._pipe_thread.start()

    # -- Conversion (raw → NeXus) --

    def _on_conversion_run(self, cfg, scans: list) -> None:
        """Spawn the ConversionWorker for a fresh run.

        ``cfg`` is a ``ConversionConfig``; ``scans`` is a list of
        ``RawScan``. We don't refuse on overlapping output paths here
        — pygid handles overwrite-or-append per scan via ``cfg``'s
        flags.
        """
        if self._conv_thread is not None:
            QMessageBox.information(
                self, "Conversion in progress",
                "A conversion run is already in flight; please wait for it "
                "to finish before starting another.",
            )
            return
        # Modal progress dialog — a long batch can run for minutes; the
        # user needs a way to see it's progressing without watching the
        # log pane scroll.
        self._conv_progress = QProgressDialog(
            "Converting…", "", 0, max(len(scans), 1), self
        )
        self._conv_progress.setWindowTitle(APP_NAME)
        self._conv_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._conv_progress.setCancelButton(None)
        self._conv_progress.setMinimumDuration(0)
        self._conv_progress.setLabelText(
            f"Running {len(scans)} scan(s)…"
        )
        self._conv_progress.show()

        self.conversion_panel.set_running(True)
        self.conversion_panel.clear_log()
        self.conversion_panel.append_log(
            f"Starting conversion: {len(scans)} scan(s) → {cfg.output_dir}"
        )

        self._conv_thread = QThread(self)
        self._conv_worker = ConversionWorker(scans, cfg)
        self._conv_worker.moveToThread(self._conv_thread)
        self._conv_thread.started.connect(self._conv_worker.run)
        self._conv_worker.log.connect(self.conversion_panel.append_log)
        self._conv_worker.progress.connect(self._on_conversion_progress)
        self._conv_worker.finished.connect(self._on_conversion_finished)
        self._conv_thread.start()

    def _on_conversion_progress(self, done: int, total: int) -> None:
        if self._conv_progress is None:
            return
        self._conv_progress.setMaximum(max(total, 1))
        self._conv_progress.setValue(done)

    def _on_conversion_finished(
        self, output_paths: list | None, error: Exception | None
    ) -> None:
        if self._conv_thread is not None:
            self._conv_thread.quit()
            self._conv_thread.wait()
            self._conv_thread.deleteLater()
            self._conv_thread = None
        if self._conv_worker is not None:
            self._conv_worker.deleteLater()
            self._conv_worker = None
        if self._conv_progress is not None:
            self._conv_progress.close()
            self._conv_progress.deleteLater()
            self._conv_progress = None

        self.conversion_panel.set_running(False)

        if error is not None:
            self.conversion_panel.append_log(f"ERROR - {error}")
            QMessageBox.critical(self, "Conversion failed", str(error))
            return

        outputs = list(output_paths or [])
        if not outputs:
            self.conversion_panel.append_log(
                "Conversion completed but produced no output paths."
            )
            return

        self.conversion_panel.append_log(
            "Conversion DONE. Output files:\n  " + "\n  ".join(str(p) for p in outputs)
        )

        # Auto-open: queue every produced file as a NeXus session. The
        # existing CopyWorker path normalizes pygid metadata and handles
        # silx-tree insertion; ``_set_active_session`` swaps focus once
        # the first file lands.
        for out_path in outputs:
            self._open_queue.append(Path(out_path))
        self._process_open_queue()

    def _on_pipeline_finished(self, _result: object, error: Exception | None) -> None:
        if self._pipe_thread is not None:
            self._pipe_thread.quit()
            self._pipe_thread.wait()
            self._pipe_thread.deleteLater()
            self._pipe_thread = None
        if self._pipe_worker is not None:
            self._pipe_worker.deleteLater()
            self._pipe_worker = None

        if error is not None:
            self.pipeline_panel.append_log(f"ERROR - {error}")
            # In a queued multi-entry batch, surface the error in the log
            # but only show the modal once at the *end* (otherwise the user
            # gets a dialog per entry and the run halts in front of every
            # one). For single-command runs (queue empty) keep the modal.
            if not self._pipeline_queue:
                QMessageBox.critical(self, "Pipeline error", str(error))
        else:
            self.pipeline_panel.append_log("DONE")

        if self.session is not None and error is None:
            self.session.mark_dirty()

        # If more commands are queued, run the next one without
        # tearing down the silx tree / viewer state for the user — keep
        # the busy gating active and chain straight into the next run.
        if self._pipeline_queue:
            self._run_next_pipeline_command()
            return

        # Queue drained — final cleanup. Reattach silx, refresh the
        # viewer for the active entry, lift busy gating.
        self.pipeline_panel.set_running(False)
        self.parameter_panel.set_busy(False)
        self.viewer.set_busy(False)
        self._update_status_pipeline(running=False)
        self._reattach_silx_tree()
        if self.session is not None:
            entry = self.entry_combo.currentText()
            if entry:
                # Same entry, same axes — preserve the user's zoom and
                # frame across the overlay refresh.
                self._load_entry_into_viewer(entry, preserve_view=True)
            self._update_title()

    def _on_add_to_detected(self) -> None:
        if self.session is None or self._pipe_thread is not None:
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind != "manual" or sel.manual_ref is None or not entry:
            return
        manual_peak = sel.manual_ref
        frame = self.viewer.current_frame
        kwargs = {
            "entry": entry,
            "frame_num": frame,
            **add_peak_kwargs_for(manual_peak),
        }
        # Manual peak is left in place after the run so it can also be
        # committed to fitted_peaks or further tweaked. See the comment in
        # _on_pipeline_finished for the rationale.
        self._on_pipeline_run(PipelineCommand("add_peak", kwargs))

    def _on_add_to_fitted(self) -> None:
        """Append a row to fitted_peaks using the profile viewer's 1D fits.

        Accepts the active selection when ``kind`` is manual or detected —
        both are candidate boxes whose 1D fit is the natural input for
        fitted_peaks. Geometry comes from the radial / angular Gaussian
        centers (and FWHMs); amplitude from the radial fit. Fields not
        measurable from 1D fits (theta, A, B, C, score) get zeroed —
        downstream code that needs them should re-run the proper 2D fit.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind not in ("manual", "detected") or not entry:
            return
        # The ring/segment toggle in the parameter panel decides which
        # storage convention to use — defaults to the source's is_ring on
        # selection but the user can flip it before clicking. For rings,
        # the angular fit is irrelevant (and frequently undefined) so we
        # only require the radial fit.
        save_as_ring = self.parameter_panel.save_as_ring()
        frame = self.viewer.current_frame

        fits = self.profile_viewer.last_fit_params()
        rfit = fits.get("radial")
        afit = fits.get("angular")
        if rfit is None:
            QMessageBox.warning(
                self, "Add to fitted",
                "No radial Gaussian fit is available for this peak. Drag "
                "the box until the pink fit curve appears in the radial "
                "profile, then try again.",
            )
            return
        if not save_as_ring and afit is None:
            QMessageBox.warning(
                self, "Add to fitted",
                "No angular Gaussian fit is available — required for "
                "segment peaks. Drag the box until the pink fit curve "
                "appears in the angular profile, or check 'Save fitted "
                "as ring' if this is a ring.",
            )
            return

        if save_as_ring:
            # Canonical ring convention used in already-labelled fitted_peaks
            # rows (angle = 45°, angle_width = ∞). q_xy / q_z are recomputed
            # from this in add_fitted_peak_row.
            angle_to_save = 45.0
            angle_width_to_save = float("inf")
        else:
            angle_to_save = float(afit.center)
            # Box convention: radial border = FWHM, azimuthal border =
            # 2 × FWHM. The wider azimuthal box gives the next refit
            # enough context to converge on the same Gaussian; the radial
            # box hugs the FWHM tightly because that's where peak position
            # matters most for downstream matching.
            angle_width_to_save = float(2.0 * afit.fwhm)

        self._detach_silx_tree()
        try:
            new_id = file_model.add_fitted_peak_row(
                self.session.temp_path, entry, frame,
                radius=float(rfit.center),
                radius_width=float(rfit.fwhm),
                angle=angle_to_save,
                angle_width=angle_width_to_save,
                amplitude=float(rfit.amplitude),
                is_ring=save_as_ring,
            )
        except KeyError as exc:
            QMessageBox.warning(self, "Add to fitted", str(exc))
            self._reattach_silx_tree()
            return
        except Exception as exc:
            QMessageBox.critical(self, "Add to fitted", str(exc))
            self._reattach_silx_tree()
            return
        self._reattach_silx_tree()

        # Selection is left alone so the user can keep editing or commit
        # again; the cyan fitted overlay simply appears alongside the
        # original box.
        self.session.mark_dirty()
        self._update_title()
        # File-level mutation invalidates pending FileGeomActions whose ids
        # were ordered before the new row.
        self.viewer.clear_history()
        # Pull the fresh fitted_peaks (and matched, which references it) back
        # into the viewer — same entry, so preserve the user's zoom + frame.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Ring toggle is sticky across selections but reset on commit so
        # the next Add-to-fitted defaults back to segment unless the user
        # explicitly opts in again.
        self.parameter_panel.reset_save_as_ring()
        self.pipeline_panel.append_log(
            f"Added fitted peak id={new_id} "
            f"({'ring' if save_as_ring else 'segment'} from {sel.kind}) on "
            f"{entry}/frame{frame:05d}"
        )

    def _on_delete_peak_requested(self, sel: SelectedPeak | None) -> None:
        """Confirm + cascade-delete a non-manual peak via mlgidbase."""
        if (
            sel is None or sel.kind == "manual"
            or self.session is None or self._pipe_thread is not None
        ):
            return
        if not is_mlgidbase_available():
            QMessageBox.information(
                self, "Delete peak",
                "mlgidbase is not installed; cannot delete peaks.",
            )
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        reply = QMessageBox.question(
            self,
            "Delete peak",
            (
                f"Delete {sel.kind} peak id={sel.peak_id} on frame {sel.frame}?\n\n"
                "It will be removed from detected, fitted, and all matched "
                "solutions. This cannot be undone."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cmd = PipelineCommand(
            "delete_peak",
            {
                "entry": entry,
                "frame_num": int(sel.frame),
                "peak_id": int(sel.peak_id),
            },
        )
        self._on_pipeline_run(cmd)

    def _on_peak_row_write_requested(
        self, frame: int, kind: str, peak_id: int, polar: dict
    ) -> None:
        """Persist a detected/fitted box edit straight to the NeXus file.

        Drops silx's read handle for the duration of the write (matching the
        pipeline-run dance in ``_on_pipeline_run``) so h5py can open r+, then
        re-attaches. On KeyError (peak vanished), the undo/redo stacks are
        cleared since they're keyed on stale ids.
        """
        if self.session is None:
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        self._detach_silx_tree()
        try:
            file_model.update_peak_row(
                self.session.temp_path, entry, frame, kind, peak_id, **polar
            )
        except KeyError:
            QMessageBox.warning(
                self, "Edit failed",
                f"Peak id={peak_id} no longer exists in the file. "
                "Undo history has been cleared.",
            )
            self.viewer.clear_history()
        except Exception as exc:
            QMessageBox.critical(self, "Edit failed", str(exc))
            self.viewer.clear_history()
        else:
            self.session.mark_dirty()
            self._update_title()
        finally:
            self._reattach_silx_tree()

    def _load_entry_into_viewer(
        self, entry: str, *, preserve_view: bool = False
    ) -> None:
        """Load ``entry`` into the image viewer.

        ``preserve_view``: when True, the viewer keeps its current zoom and
        frame index across the reload. Used after pipeline ops and direct
        h5py edits (the underlying stack is unchanged — only peak overlays
        are different). Switching to a different entry passes False so the
        viewer auto-ranges to the new axes.
        """
        assert self.session is not None
        try:
            stack = file_model.load_entry(self.session.temp_path, entry)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"Could not load {entry}: {exc}")
            return
        self.viewer.show_stack(stack, preserve_view=preserve_view)
        # Match the slider to the new stack's frame range. preserve_view
        # already restores the prior frame index inside show_stack; we
        # only need to repopulate the slider's bounds + label here.
        self._refresh_frame_slider()
        for frame in range(stack.n_frames):
            try:
                peaks = file_model.load_peaks(self.session.temp_path, entry, frame)
            except Exception:
                peaks = {kind: None for kind in OVERLAY_KINDS}
            self.viewer.set_peaks(frame, peaks)
            # Matched solutions reference fitted_peaks indices — pass them in
            # so the loader can resolve geometry without a second file read.
            try:
                matched = file_model.load_matched_peaks(
                    self.session.temp_path, entry, frame, peaks.get("fitted")
                )
            except Exception:
                matched = []
            self.viewer.set_matched_structures(frame, matched)
        # Initial panel state for whichever frame the viewer is showing now.
        self._refresh_matched_panel(
            self.viewer.current_frame,
            self.viewer.matched_structures(self.viewer.current_frame),
        )
        # Hand the polar transform to the profile viewer. After the
        # lazy-loading milestone ``viewer.polar_data()`` returns a
        # ``(_LazyPolarStack, radius, angle)`` tuple — frames are
        # resampled on demand inside the FrameSource so this no longer
        # forces an eager precompute of the full polar stack. The
        # profile viewer indexes the wrapper by frame; the cursor
        # readout uses tuple indexing for single-pixel lookups.
        polar = self.viewer.polar_data()
        if polar is not None:
            self.profile_viewer.set_polar_stack(*polar)
        # Configure the prefetch worker for the new entry. Single-
        # frame stacks short-circuit inside the helper; multi-frame
        # stacks spawn the worker (if not already) and reset its
        # _done set so the next Play press starts filling from
        # frame current+1 onward.
        self._configure_prefetch_for_active_entry()

    # -- UI state --

    def _refresh_matched_panel(self, _frame: int, structures: list) -> None:
        """Rebuild the per-structure rows under the Matched-peaks master.

        Called on every frame change and after a fresh entry load. We blow
        away the old QCheckBox widgets and create new ones — the structure
        list is small (1-N rows) so this is cheap.
        """
        # Clear children of the dynamic container.
        while self._matched_struct_layout.count():
            item = self._matched_struct_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._matched_empty_label = None
        self._matched_struct_checkboxes.clear()

        if not structures:
            self._matched_empty_label = QLabel("<i>No matched solutions for this frame.</i>")
            self._matched_empty_label.setWordWrap(True)
            self._matched_struct_layout.addWidget(self._matched_empty_label)
            return

        for i, s in enumerate(structures):
            pen = matched_pen_for(i)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            swatch = QLabel()
            # Mirror the *exact* pen used to render the structure on the
            # image so the user can map a row to its overlay shape even
            # when colour repeats — the dashed/dotted swatch flags it.
            swatch.setPixmap(_make_pen_swatch(pen))
            row.addWidget(swatch)
            chk = QCheckBox(s.label)
            chk.setChecked(self.viewer.matched_visibility(_frame, s.unique_id))
            chk.toggled.connect(
                lambda v, uid=s.unique_id: self._on_matched_structure_toggled(uid, v)
            )
            row.addWidget(chk)
            row.addStretch(1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            self._matched_struct_layout.addWidget(row_widget)
            self._matched_struct_checkboxes[s.unique_id] = chk

    def _on_matched_master_toggled(self, checked: bool) -> None:
        """Master toggles cascade to every per-structure row.

        Unchecking the master now also unchecks every structure
        checkbox; checking it back rechecks them all. The viewer's
        own master flag is updated either way so its hit-test gating
        stays in sync. Per-checkbox ``setChecked`` calls are blocked
        from re-emitting ``toggled`` so the structure-toggled slot
        doesn't interpret the cascade as a user-driven single-show.
        """
        self.viewer.set_matched_master_visible(checked)
        for uid, chk in self._matched_struct_checkboxes.items():
            with QSignalBlocker(chk):
                chk.setChecked(checked)
            self.viewer.set_matched_structure_visible(uid, checked)

    def _on_matched_structure_toggled(self, uid: str, checked: bool) -> None:
        """Per-structure toggle. Promotes a ``check while master is off``
        click into a "show only this one" view: every other structure is
        unchecked, the master is auto-ticked (without re-cascading), and
        only the freshly-checked structure ends up visible.
        """
        self.viewer.set_matched_structure_visible(uid, checked)
        if not checked:
            return
        if self._matched_master_check.isChecked():
            return
        # Master was off → user wants to see this single structure.
        # Force the others off (both UI + viewer state) before flipping
        # the master ON, since the master toggle would otherwise
        # cascade and re-show every structure.
        for other_uid, chk in self._matched_struct_checkboxes.items():
            if other_uid == uid:
                continue
            with QSignalBlocker(chk):
                chk.setChecked(False)
            self.viewer.set_matched_structure_visible(other_uid, False)
        with QSignalBlocker(self._matched_master_check):
            self._matched_master_check.setChecked(True)
        # blockSignals suppressed _on_matched_master_toggled, so call
        # the viewer's master flag directly.
        self.viewer.set_matched_master_visible(True)

    def _update_title(self) -> None:
        if self.session is None:
            self.setWindowTitle(APP_NAME)
            self._update_status_file()
            return
        marker = "*" if self.session.dirty else ""
        self.setWindowTitle(
            f"{self.session.original_path.name}{marker} — {APP_NAME}"
        )
        self._update_status_file()

    def _build_status_bar(self) -> None:
        """Permanent status-bar widgets: file / entry / frame / pipeline + cursor.

        Each label lives in the status bar's permanent-widget slot so
        Qt's transient ``showMessage`` calls (PNG / CSV export confirmations)
        still render correctly alongside them — Qt clears the transient
        message after its timeout but leaves the permanent labels alone.
        """
        sb = self.statusBar()
        self._sb_file = QLabel("no file")
        self._sb_entry = QLabel("")
        self._sb_frame = QLabel("")
        self._sb_pipeline = QLabel("idle")
        self._sb_cursor = QLabel("")
        for w in (self._sb_file, self._sb_entry, self._sb_frame,
                  self._sb_pipeline, self._sb_cursor):
            # Light separation so the eye can scan the row.
            w.setStyleSheet("padding: 0 8px; border-left: 1px solid #444;")
            sb.addPermanentWidget(w)
        # The cursor readout is the chattiest widget; let it stretch
        # so values don't truncate, others stay tight.
        self._sb_cursor.setMinimumWidth(360)
        self.viewer.cursorMoved.connect(self._on_status_cursor_moved)
        self._status_cursor_visible = True

    def _update_status_file(self) -> None:
        if self.session is None:
            self._sb_file.setText("no file")
            return
        marker = "*" if self.session.dirty else ""
        self._sb_file.setText(
            f"{self.session.original_path.name}{marker}"
        )

    def _update_status_entry(self) -> None:
        entry = self.entry_combo.currentText() if hasattr(self, "entry_combo") else ""
        self._sb_entry.setText(entry or "")

    def _update_status_frame(self) -> None:
        n = getattr(self.viewer, "n_frames", 0)
        if n <= 0:
            self._sb_frame.setText("")
            return
        cur = int(getattr(self.viewer, "current_frame", 0))
        # Frames are 0-indexed everywhere else in the GUI (peak rows,
        # NeXus group keys), so the denominator is the max index
        # ``n - 1``, not the count. 17-frame stack → "frame 16 / 16"
        # at the end. Single-frame entries elide the "/ total" since
        # there's no navigation possible.
        if n == 1:
            self._sb_frame.setText(f"frame {cur}")
        else:
            self._sb_frame.setText(f"frame {cur} / {n - 1}")

    def _update_status_pipeline(self, command=None, *, running: bool) -> None:
        if not running:
            self._sb_pipeline.setText("idle")
            return
        if command is None:
            self._sb_pipeline.setText("running…")
            return
        op = command.op_name if hasattr(command, "op_name") else str(command)
        entry = command.kwargs.get("entry") if hasattr(command, "kwargs") else None
        self._sb_pipeline.setText(
            f"running: {op} on {entry}" if entry else f"running: {op}"
        )

    def _on_status_cursor_moved(self, info) -> None:
        if not self._status_cursor_visible:
            self._sb_cursor.setText("")
            return
        if not info:
            self._sb_cursor.setText("")
            return
        mode = info.get("mode")
        inten = info.get("intensity", float("nan"))
        inten_str = "—" if inten != inten else f"{inten:.3g}"  # NaN check
        if mode == "pixel":
            self._sb_cursor.setText(
                f"row={info['row']}, col={info['col']}, I={inten_str}"
            )
        elif mode == "cartesian":
            self._sb_cursor.setText(
                f"q_xy={info['q_xy']:.3f}, q_z={info['q_z']:.3f}, I={inten_str}"
            )
        elif mode == "polar":
            self._sb_cursor.setText(
                f"r={info['r']:.3f}, θ={info['theta']:.1f}°, I={inten_str}"
            )
        else:
            self._sb_cursor.setText("")

    def _set_cursor_readout_visible(self, visible: bool) -> None:
        self._status_cursor_visible = bool(visible)
        self._sb_cursor.setVisible(self._status_cursor_visible)

    def _update_actions(self) -> None:
        has_session = self.session is not None
        # Save/Save As only apply to NeXus sessions — raw sessions have no
        # writable temp copy. Close still works either way.
        is_nexus = has_session and self.session.kind == "nexus"
        self.action_save.setEnabled(is_nexus)
        self.action_save_as.setEnabled(is_nexus)
        self.action_close_file.setEnabled(has_session)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drops carrying local file URLs.

        Acceptance is loose at enter time — content classification
        happens in ``dropEvent`` so the cursor reflects "yes you can
        drop" while the user drags over the window. Non-file payloads
        (text, internal Qt drags) are ignored.
        """
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:
        # Same gate as dragEnter — Qt fires move events repeatedly while
        # the drag is in flight and the proposed-action state has to
        # stay accepted across them or the drop won't fire.
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Open every dropped file via the unified _open_paths classifier."""
        urls = event.mimeData().urls()
        paths: list[Path] = []
        for u in urls:
            if not u.isLocalFile():
                continue
            local = u.toLocalFile()
            if not local:
                continue
            paths.append(Path(local))
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._open_paths(paths)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Each loaded file may have unsaved changes — prompt per dirty
        # session in load order so the user gets the same per-file save
        # dialog they would on _action_close_file.
        for s in list(self._sessions):
            if not self._confirm_discard_changes(s):
                event.ignore()
                return
        # Stop frame playback so the timer doesn't fire one last tick
        # against a torn-down viewer during shutdown.
        self._pause_playback()
        # All clear — tear everything down. silx must release its handles
        # before we delete the temp files.
        self._detach_silx_tree()
        self.viewer.clear()
        self.profile_viewer.clear()
        # Drop the shared open-progress dialog if it's still up (rare —
        # the modal would normally have blocked the close). Otherwise its
        # leftover modal overlay would dim the next opened window too.
        self._dismiss_open_progress()
        # Stop a CIF parse if one is running — closing the window while
        # CifPattern construction is in flight otherwise drops the worker
        # thread on the floor and Qt complains at exit.
        if self._cif_parse_thread is not None:
            self._cif_parse_thread.quit()
            self._cif_parse_thread.wait()
        # Stop a conversion run if one is in flight — pygid + h5py do
        # their own cleanup on a clean thread exit.
        if self._conv_thread is not None:
            self._conv_thread.quit()
            self._conv_thread.wait()
        # Shut the background prefetch worker down cleanly. Release
        # its h5py handle first (so the worker stops trying to read
        # frames), then quit + wait the thread so its event loop
        # exits before we delete its Q objects.
        if self._prefetch_worker is not None:
            self._prefetchRelease.emit()
            # Process the queued release so it lands on the worker's
            # thread before we quit it; otherwise the worker would
            # try to read on a destroyed h5py.File during shutdown.
            QCoreApplication.processEvents()
            self._prefetch_thread.quit()
            self._prefetch_thread.wait()
            self._prefetch_worker.deleteLater()
            self._prefetch_worker = None
            self._prefetch_thread.deleteLater()
            self._prefetch_thread = None
        for s in list(self._sessions):
            s.close()
        self._sessions.clear()
        self._active_session = None
        event.accept()
