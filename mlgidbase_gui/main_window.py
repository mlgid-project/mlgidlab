from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QCloseEvent, QColor, QFont, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.hdf5 import Hdf5TreeView

from mlgidbase_gui import file_model
from mlgidbase_gui.image_viewer import (
    GIWAXSImageViewer,
    MATCHED_PALETTE,
    MATCHED_STYLE,
    ManualPeak,
    OVERLAY_KINDS,
    OVERLAY_STYLE,
    SelectedPeak,
)
from mlgidbase_gui.parameter_panel import ParameterPanel
from mlgidbase_gui.pipeline import (
    PipelineCommand,
    add_peak_kwargs_for,
    is_mlgidbase_available,
)
from mlgidbase_gui.pipeline_panel import PipelinePanel
from mlgidbase_gui.profile_viewer import ProfileViewer
from mlgidbase_gui.conversion_panel import ConversionPanel
from mlgidbase_gui.session import BaseSession, NexusSession, RawSession, Session
from mlgidbase_gui.workers import (
    CifParseWorker,
    ConversionWorker,
    CopyWorker,
    PipelineWorker,
)

APP_NAME = "mlgidBASE GUI"
NEXUS_FILTER = "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;All files (*)"
RAW_FILTER = "HDF5 raw data (*.h5 *.hdf5 *.nxs);;All files (*)"


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
    """Solid-line swatch in the given color — used for matched-structure rows
    where the line style is fixed and only the color distinguishes structures.
    """
    return _make_pen_swatch(
        {"color": color, "style": MATCHED_STYLE["style"]}, width, height
    )


class MainWindow(QMainWindow):
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

        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        self._build_menu()
        self._build_central()
        self._build_docks()
        # View menu is built last because it pulls toggleViewAction()s from
        # the docks created in _build_docks.
        self._build_view_menu()
        self._update_title()
        self._update_actions()

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

        self.action_clear_manual = QAction("Manual", self)
        self.action_clear_manual.triggered.connect(self._action_clear_manual)
        clear_menu.addAction(self.action_clear_manual)

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

        self.action_clear_matched = QAction("Matched", self)
        self.action_clear_matched.triggered.connect(
            lambda: self._action_clear_file_peaks("matched")
        )
        clear_menu.addAction(self.action_clear_matched)

        # Export the current frame to PNG. Works for either NeXus or
        # raw mode — pyqtgraph's ImageExporter operates on the active
        # plot item regardless of which stack supplied the data.
        tools_menu.addSeparator()
        self.action_export_png = QAction("Export current frame as PNG…", self)
        self.action_export_png.triggered.connect(self._action_export_png)
        tools_menu.addAction(self.action_export_png)

    def _action_clear_manual(self) -> None:
        """Drop every manual peak. In-memory only, no file write."""
        if not self._confirm_clear("manual"):
            return
        self.viewer.clear_all_manual_peaks()
        self.pipeline_panel.append_log("Cleared all manual peaks")

    def _action_clear_file_peaks(self, kind: str) -> None:
        """Empty every ``<kind>_peaks`` dataset for the active entry.

        Cascade rule: clearing fitted also clears matched, because matched
        rows reference fitted ids — leaving stale matched_* solutions
        pointing at deleted fitted rows is worse than wiping them.
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

    def _confirm_clear(self, kind: str) -> bool:
        descriptions = {
            "manual":   ("manual peaks (in-memory)",
                         "every manual peak in this session"),
            "detected": ("detected peaks",
                         "every row of detected_peaks for the active entry"),
            "fitted":   ("fitted + matched peaks",
                         "every row of fitted_peaks AND every matched_* "
                         "solution for the active entry "
                         "(matched references fitted, so it has to go too)"),
            "matched":  ("matched peaks",
                         "every matched_* solution for the active entry"),
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

        self.action_open = QAction("Open &NeXus…", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._action_open)
        file_menu.addAction(self.action_open)

        # Distinct entry point for raw detector data: routes through pygid
        # conversion before the rest of the pipeline becomes usable. The
        # two flows are kept separate by design — auto-detecting raw vs
        # converted from file content is brittle.
        self.action_open_raw = QAction("Open &raw data…", self)
        self.action_open_raw.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.action_open_raw.triggered.connect(self._action_open_raw)
        file_menu.addAction(self.action_open_raw)

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

    def _build_central(self) -> None:
        self.viewer = GIWAXSImageViewer(self)
        self.data_viewer = DataViewerFrame(self)

        self.tabs = QTabWidget(self)
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

        # Left: HDF5 tree (silx)
        self.tree = Hdf5TreeView(self)
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

        # Frame slider — bidirectional sync with the image viewer's
        # built-in timeline. Hidden for single-frame stacks where it
        # would just take vertical space without any function.
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
        frame_row = QWidget()
        frame_h = QHBoxLayout(frame_row)
        frame_h.setContentsMargins(0, 0, 0, 0)
        frame_h.setSpacing(6)
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
        self._overlay_checks: dict[str, QCheckBox] = {}
        for kind, label in (
            ("detected", "Detected peaks"),
            ("fitted", "Fitted peaks"),
            ("manual", "Manual peaks"),
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
        self._matched_master_check.toggled.connect(
            self.viewer.set_matched_master_visible
        )
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

        self._display_dock = QDockWidget("Display", self)
        self._display_dock.setWidget(panel)
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

    # -- Actions --

    def _action_open(self) -> None:
        # Multi-select supported: every selected file is added to the file
        # browser of THIS window (no new windows are spawned). Opens run
        # serially through one CopyWorker thread — extra paths queue up.
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open NeXus file(s)", "", NEXUS_FILTER
        )
        if not paths:
            return
        self._open_queue.extend(Path(p) for p in paths)
        self._process_open_queue()

    def _action_open_raw(self) -> None:
        """Open one or more raw HDF5 detector files for conversion.

        All selected files are bundled into a single ``RawSession`` so the
        Conversion panel can apply one shared config to the whole batch.
        Inputs are read-only — pygid only reads them — so no temp copy is
        made and the open is synchronous (no CopyWorker needed).
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open raw HDF5 file(s)", "", RAW_FILTER
        )
        if not paths:
            return
        try:
            session = RawSession.open([Path(p) for p in paths])
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        # Insert each raw file into the silx tree as a read-only entry so
        # the user can browse the HDF5 structure before configuring the
        # conversion. The tree model accepts the same insertFile() call
        # that ``_on_open_finished`` uses for converted files.
        model = self.tree.findHdf5TreeModel()
        for raw_path in session.raw_paths:
            model.insertFile(str(raw_path))
        self._sessions.append(session)
        self._set_active_session(session)

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
        try:
            self.session.save_as(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Save As failed", str(exc))
            return
        # The active session's temp file may have been renamed to match the
        # new basename; rebuild the tree from all sessions so its label
        # updates while sibling files stay attached.
        self._detach_silx_tree()
        self._reattach_silx_tree()
        self._update_title()

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
            getattr(self, "action_clear_manual", None),
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
        """Release silx's read handles on every loaded temp file.

        Required before any code path opens an HDF5 file ``r+`` (pipeline
        runs, direct h5py edits) since silx's open handle would otherwise
        block the writer.
        """
        self.tree.findHdf5TreeModel().clear()
        self.data_viewer.setData(None)

    def _reattach_silx_tree(self) -> None:
        """Re-insert every session's files into the tree in order.

        NeXus sessions contribute one file (the temp working copy); raw
        sessions contribute every selected raw input so the user can keep
        browsing all of them while configuring conversion.
        """
        model = self.tree.findHdf5TreeModel()
        for s in self._sessions:
            if isinstance(s, RawSession):
                for raw_path in s.raw_paths:
                    model.insertFile(str(raw_path))
            else:
                model.insertFile(str(s.temp_path))

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
        re-emitting valueChanged back into the viewer.
        """
        self.frame_label.setText(self._frame_label_text(frame))
        if self.frame_slider.value() == frame:
            return
        self.frame_slider.blockSignals(True)
        try:
            self.frame_slider.setValue(int(frame))
        finally:
            self.frame_slider.blockSignals(False)

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

    def _on_tree_selection_changed(self, *_: object) -> None:
        nodes = list(self.tree.selectedH5Nodes())
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
        nodes = list(self.tree.selectedH5Nodes())
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
        """Toggle between segment / ring preview without waiting for the
        next profile recompute. Also tells the profile viewer to skip the
        angular Gaussian fit while ring is active — that fit wouldn't be
        saved by Add-to-fitted in ring mode.
        """
        # Drop the angular fit *before* recomputing the preview so the
        # cached afit is None when _update_fitted_preview reads it.
        self.profile_viewer.set_skip_angular_fit(is_ring)
        fits = self.profile_viewer.last_fit_params()
        self._update_fitted_preview(fits.get("radial"), fits.get("angular"))

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
        self._cif_parse_thread = QThread(self)
        self._cif_parse_worker = CifParseWorker(cif_input, nexus_file)
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

        self.pipeline_panel.set_running(True)
        self.parameter_panel.set_busy(True)
        self.viewer.set_busy(True)
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
        # Hand the polar transform to the profile viewer (lazy-computed; same
        # cache the image viewer uses when the user toggles to polar mode).
        polar = self.viewer.polar_data()
        if polar is not None:
            self.profile_viewer.set_polar_stack(*polar)

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

        if not structures:
            self._matched_empty_label = QLabel("<i>No matched solutions for this frame.</i>")
            self._matched_empty_label.setWordWrap(True)
            self._matched_struct_layout.addWidget(self._matched_empty_label)
            return

        for i, s in enumerate(structures):
            color = MATCHED_PALETTE[i % len(MATCHED_PALETTE)]
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            swatch = QLabel()
            swatch.setPixmap(_make_color_swatch(color))
            row.addWidget(swatch)
            chk = QCheckBox(s.label)
            chk.setChecked(self.viewer.matched_visibility(_frame, s.unique_id))
            chk.toggled.connect(
                lambda v, uid=s.unique_id: self.viewer.set_matched_structure_visible(uid, v)
            )
            row.addWidget(chk)
            row.addStretch(1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            self._matched_struct_layout.addWidget(row_widget)

    def _update_title(self) -> None:
        if self.session is None:
            self.setWindowTitle(APP_NAME)
            return
        marker = "*" if self.session.dirty else ""
        self.setWindowTitle(
            f"{self.session.original_path.name}{marker} — {APP_NAME}"
        )

    def _update_actions(self) -> None:
        has_session = self.session is not None
        # Save/Save As only apply to NeXus sessions — raw sessions have no
        # writable temp copy. Close still works either way.
        is_nexus = has_session and self.session.kind == "nexus"
        self.action_save.setEnabled(is_nexus)
        self.action_save_as.setEnabled(is_nexus)
        self.action_close_file.setEnabled(has_session)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Each loaded file may have unsaved changes — prompt per dirty
        # session in load order so the user gets the same per-file save
        # dialog they would on _action_close_file.
        for s in list(self._sessions):
            if not self._confirm_discard_changes(s):
                event.ignore()
                return
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
        for s in list(self._sessions):
            s.close()
        self._sessions.clear()
        self._active_session = None
        event.accept()
