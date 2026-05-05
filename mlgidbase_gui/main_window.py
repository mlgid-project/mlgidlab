from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QCloseEvent, QColor, QKeySequence, QPainter, QPen, QPixmap
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
    QProgressDialog,
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
from mlgidbase_gui.session import Session
from mlgidbase_gui.workers import CopyWorker, PipelineWorker

APP_NAME = "mlgidBASE GUI"
NEXUS_FILTER = "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;All files (*)"


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
        self._sessions: list[Session] = []
        self._active_session: Session | None = None
        # Opens run serially through the existing single-thread CopyWorker
        # plumbing; extra paths from a multi-select dialog wait here.
        self._open_queue: list[Path] = []
        self._thread: QThread | None = None
        self._worker: CopyWorker | None = None
        self._progress: QProgressDialog | None = None
        self._pipe_thread: QThread | None = None
        self._pipe_worker: PipelineWorker | None = None

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

        self.action_open = QAction("&Open…", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._action_open)
        file_menu.addAction(self.action_open)

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
        layout.addLayout(form)

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
            "to resize detected/fitted, <b>Delete</b> to remove. "
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
        self.pipeline_panel.runRequested.connect(self._on_pipeline_run)
        self._pipeline_dock = QDockWidget("Pipeline", self)
        self._pipeline_dock.setWidget(self.pipeline_panel)
        self._pipeline_dock.setObjectName("PipelineDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._pipeline_dock)
        self.tabifyDockWidget(self._display_dock, self._pipeline_dock)
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
        self.parameter_panel.runFittingRequested.connect(self._on_run_fitting_from_panel)
        self.parameter_panel.runMatchingRequested.connect(self._on_run_matching_from_panel)
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

    def _process_open_queue(self) -> None:
        """Kick off the next queued open if no copy is in flight."""
        if self._thread is not None or not self._open_queue:
            return
        self._open_path(self._open_queue.pop(0))

    def _action_save(self) -> None:
        self._save(confirm=True)

    def _save(self, confirm: bool, session: Session | None = None) -> bool:
        """Overwrite the original from the temp. Returns True on success."""
        target = session if session is not None else self._active_session
        if target is None:
            return False
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
        if self.session is None:
            return
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

        self._progress = QProgressDialog("Opening file…", "", 0, 0, self)
        self._progress.setWindowTitle(APP_NAME)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setCancelButton(None)
        self._progress.setMinimumDuration(0)
        self._progress.show()

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
        if self._progress is not None:
            self._progress.close()
            self._progress = None

        if error is not None:
            QMessageBox.critical(self, "Open failed", str(error))
        elif session is not None:
            self._sessions.append(session)
            self.tree.findHdf5TreeModel().insertFile(str(session.temp_path))
            # Newly-opened file becomes the active one — the user almost
            # always wants to inspect what they just opened.
            self._set_active_session(session)

        # Keep draining the queue regardless of this open's outcome so a
        # single bad file in a batch doesn't strand the rest.
        self._process_open_queue()

    def _close_session(self, session: Session) -> None:
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

    def _set_active_session(self, session: Session | None) -> None:
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
        if session is not None:
            self._populate_entries()
        self._update_title()
        self._update_actions()

    def _confirm_discard_changes(self, session: Session | None = None) -> bool:
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
        """Re-insert every session's temp file into the tree in order."""
        model = self.tree.findHdf5TreeModel()
        for s in self._sessions:
            model.insertFile(str(s.temp_path))

    # -- Entry / viewer wiring --

    def _populate_entries(self) -> None:
        if self.session is None:
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
        if entries:
            self._load_entry_into_viewer(entries[0])

    def _on_entry_changed(self, entry: str) -> None:
        if not entry or self.session is None:
            return
        self._load_entry_into_viewer(entry)

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

    def _on_tree_activated(self, *_: object) -> None:
        nodes = list(self.tree.selectedH5Nodes())
        if not nodes:
            return
        node = nodes[0]
        self.data_viewer.setData(node)
        self.tabs.setCurrentWidget(self.data_viewer)
        self._activate_session_for_node(node)

    def _activate_session_for_node(self, node) -> None:
        """If ``node`` lives in a non-active session's file, swap active."""
        fname = self._node_filename(node)
        if fname is None:
            return
        for s in self._sessions:
            if s.temp_path == fname:
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
        a manual peak (the preview only makes sense as a what-if for
        ``Add to fitted``, which is manual-only).
        """
        if sel is None or sel.kind != "manual":
            self.viewer.set_fitted_preview(None, None, None, None)

    def _update_fitted_preview(self, rfit, afit) -> None:
        """Sync the viewer's fitted-preview box to the latest 1D fits.

        Only relevant for manual selections — file-resident peaks already
        have a stored fitted box, so previewing a refit there would be
        confusing. ``rfit`` / ``afit`` may be ``None`` (no convergence) →
        clear the preview.
        """
        sel = self.viewer.selected_peak
        if sel is None or sel.kind != "manual":
            self.viewer.set_fitted_preview(None, None, None, None)
            return
        if rfit is None or afit is None:
            self.viewer.set_fitted_preview(None, None, None, None)
            return
        self.viewer.set_fitted_preview(
            float(rfit.center), float(rfit.fwhm),
            float(afit.center), float(afit.fwhm),
        )

    # -- Pipeline --

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
        self.pipeline_panel.append_log(f"--- {command.op_name} ---")

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

    def _on_pipeline_finished(self, _result: object, error: Exception | None) -> None:
        if self._pipe_thread is not None:
            self._pipe_thread.quit()
            self._pipe_thread.wait()
            self._pipe_thread.deleteLater()
            self._pipe_thread = None
        if self._pipe_worker is not None:
            self._pipe_worker.deleteLater()
            self._pipe_worker = None

        self.pipeline_panel.set_running(False)
        self.parameter_panel.set_busy(False)
        self.viewer.set_busy(False)

        if error is not None:
            self.pipeline_panel.append_log(f"ERROR - {error}")
            QMessageBox.critical(self, "Pipeline error", str(error))
        else:
            self.pipeline_panel.append_log("DONE")

        # The manual peak is intentionally NOT dropped after Add-to-detected
        # so the user can also commit it to fitted_peaks (or keep tweaking
        # the box and re-commit). The new detected/fitted overlay just
        # appears alongside the still-selected manual box.

        # Reattach silx tree (every session), refresh viewer, mark dirty.
        self._reattach_silx_tree()
        if self.session is not None:
            if error is None:
                self.session.mark_dirty()
            entry = self.entry_combo.currentText()
            if entry:
                self._load_entry_into_viewer(entry)
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

        Geometry comes from the radial / angular Gaussian centers (and FWHMs);
        amplitude from the radial fit. Fields not measurable from 1D fits
        (theta, A, B, C, score) get zeroed — downstream code that needs them
        should re-run the proper 2D fit.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind != "manual" or sel.manual_ref is None or not entry:
            return
        manual_peak = sel.manual_ref
        frame = self.viewer.current_frame

        fits = self.profile_viewer.last_fit_params()
        rfit = fits.get("radial")
        afit = fits.get("angular")
        if rfit is None or afit is None:
            QMessageBox.warning(
                self, "Add to fitted",
                "No 1D Gaussian fit is available for this peak. Drag the box "
                "until the pink fit curves appear in both profile plots, "
                "then try again.",
            )
            return

        self._detach_silx_tree()
        try:
            new_id = file_model.add_fitted_peak_row(
                self.session.temp_path, entry, frame,
                radius=float(rfit.center),
                radius_width=float(rfit.fwhm),
                angle=float(afit.center),
                # Box convention: radial border = FWHM, azimuthal border =
                # 2 × FWHM. The wider azimuthal box gives the next refit
                # enough context to converge on the same Gaussian; the
                # radial box hugs the FWHM tightly because that's where
                # peak position matters most for downstream matching.
                angle_width=float(2.0 * afit.fwhm),
                amplitude=float(rfit.amplitude),
                is_ring=manual_peak.is_ring,
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

        # Manual peak stays selected so the user can also commit it to
        # detected_peaks or keep editing the box; the cyan fitted overlay
        # simply appears alongside the yellow manual box.
        self.session.mark_dirty()
        self._update_title()
        # File-level mutation invalidates pending FileGeomActions whose ids
        # were ordered before the new row.
        self.viewer.clear_history()
        # Pull the fresh fitted_peaks (and matched, which references it) back
        # into the viewer.
        self._load_entry_into_viewer(entry)
        self.pipeline_panel.append_log(
            f"Added fitted peak id={new_id} on {entry}/frame{frame:05d}"
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

    def _on_run_fitting_from_panel(self) -> None:
        if self.session is None or self._pipe_thread is not None:
            return
        self._on_pipeline_run(PipelineCommand("run_fitting", {}))

    def _on_run_matching_from_panel(self) -> None:
        if self.session is None or self._pipe_thread is not None:
            return
        if not is_mlgidbase_available():
            return
        # Reuse whatever the Pipeline dock currently has configured. The CIF
        # pickle is required; nudge the user to the Pipeline tab if it's missing.
        cif = getattr(self.pipeline_panel, "cif_path", None)
        if cif is None or not cif.text().strip():
            QMessageBox.information(
                self,
                "Matching configuration",
                "Set the CIF preprocessed pickle in the Pipeline panel "
                "before running matching.",
            )
            return
        cmd = PipelineCommand(
            "run_matching",
            {
                "cif_prepr": cif.text().strip(),
                "peaks_type": self.pipeline_panel.peaks_type.currentText(),
                "threshold": float(self.pipeline_panel.threshold.value()),
                "device": self.pipeline_panel.device.currentText(),
            },
        )
        self._on_pipeline_run(cmd)

    def _load_entry_into_viewer(self, entry: str) -> None:
        assert self.session is not None
        try:
            stack = file_model.load_entry(self.session.temp_path, entry)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"Could not load {entry}: {exc}")
            return
        self.viewer.show_stack(stack)
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
        self.action_save.setEnabled(has_session)
        self.action_save_as.setEnabled(has_session)
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
        for s in list(self._sessions):
            s.close()
        self._sessions.clear()
        self._active_session = None
        event.accept()
