"""Pipeline launcher: collapsible Detection / Fitting / Matching sections.

Each section exposes the full kwarg surface of the underlying ``mlgidBASE``
method (see ``mlgidbase/main.py``: ``run_detection``, ``run_fitting``,
``run_matching``). The panel knows nothing about the active session — the
host wires a ``get_active_entry`` callback so the entry-scope dropdowns can
resolve "Active entry" at click time.

Defaults intentionally scope every run to the *active* entry rather than to
all entries (mlgidBASE's own default). The viewer shows one entry at a time
and per-entry runs sidestep failures on incompatible sibling entries — the
user can still pick "All entries" explicitly when they want a sweep.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidbase_gui.pipeline import PipelineCommand, is_mlgidbase_available


# Sentinels for the entry-scope and frame-scope dropdowns. The panel resolves
# them to mlgidBASE-shaped kwargs at click time so the host MainWindow stays
# the single source of truth for "what's active right now".
ENTRY_ACTIVE = "Active entry"
ENTRY_ALL = "All entries"

FRAME_ACTIVE = "Active frame"
FRAME_ALL = "All frames"


class _CollapsibleSection(QWidget):
    """Section header (clickable) + body widget that hides on collapse.

    Qt has no built-in expander, so this is a small QToolButton + QFrame
    combo. Hosts add controls to ``body_layout``. ``expandedChanged`` lets
    a coordinator (e.g. an accordion group) react when the user opens or
    closes the section.
    """

    expandedChanged = Signal(bool)

    def __init__(self, title: str, *, expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        # Section header: no border, bold, full width — matches dark theme.
        self._toggle.setStyleSheet(
            "QToolButton { border: none; padding: 4px 0px; font-weight: bold; }"
        )
        self._toggle.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle)

        self._body = QFrame(self)
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(16, 0, 4, 6)
        self.body_layout.setSpacing(4)
        self._body.setVisible(expanded)
        outer.addWidget(self._body)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        """Open or close without re-emitting (used by accordion peers)."""
        if self._toggle.isChecked() == expanded:
            return
        # blockSignals avoids a re-entrant accordion update — the coordinator
        # already knows it caused this state change.
        self._toggle.blockSignals(True)
        try:
            self._toggle.setChecked(expanded)
        finally:
            self._toggle.blockSignals(False)
        self._apply_state(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self._apply_state(checked)
        self.expandedChanged.emit(checked)

    def _apply_state(self, expanded: bool) -> None:
        self._body.setVisible(expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class PipelinePanel(QWidget):
    """Buttons + parameter controls for the three mlgidbase pipeline stages.

    Emits ``runRequested(PipelineCommand)`` when the user clicks a Run
    button; the main window owns the threading and file-handle juggling.
    """

    runRequested = Signal(PipelineCommand)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._available = is_mlgidbase_available()
        # Resolved by the host so panel stays decoupled from MainWindow.
        # If unset (or returns None) "Active entry/frame" silently falls
        # back to mlgidBASE's own None-default (all entries / all frames).
        self._get_active_entry: Callable[[], str | None] | None = None
        self._get_active_frame: Callable[[], int | None] | None = None
        self._build_ui()

    # -- Public API used by MainWindow --

    def set_active_entry_resolver(
        self, fn: Callable[[], str | None]
    ) -> None:
        self._get_active_entry = fn

    def set_active_frame_resolver(
        self, fn: Callable[[], int | None]
    ) -> None:
        self._get_active_frame = fn

    def append_log(self, msg: str) -> None:
        if hasattr(self, "log_view"):
            self.log_view.appendPlainText(msg)

    def clear_log(self) -> None:
        if hasattr(self, "log_view"):
            self.log_view.clear()

    def set_running(self, running: bool) -> None:
        if not self._available:
            return
        self.btn_detect.setEnabled(not running)
        self.btn_fit.setEnabled(not running)
        # Match button additionally requires a CIF path:
        if running:
            self.btn_match.setEnabled(False)
        else:
            self.btn_match.setEnabled(bool(self.cif_path.text().strip()))

    # -- UI construction --

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        if not self._available:
            hint = QLabel(
                "<b>mlgidbase</b> is not installed in this environment.<br><br>"
                "Install it to enable detection, fitting, and matching:"
                "<pre>  pip install mlgidbase</pre>"
            )
            hint.setWordWrap(True)
            outer.addWidget(hint)
            outer.addStretch(1)
            return

        # Accordion: only one section open at a time. Detection starts open.
        self._sections: list[_CollapsibleSection] = [
            self._build_detection_section(),
            self._build_fitting_section(),
            self._build_matching_section(),
        ]
        for s in self._sections:
            outer.addWidget(s)
            s.expandedChanged.connect(
                lambda opened, src=s: self._on_section_toggled(src, opened)
            )

        # Logs — kept as a regular GroupBox so the textarea always shows.
        log_box = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("monospace"))
        self.log_view.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log_view)
        outer.addWidget(log_box, 1)

    def _on_section_toggled(
        self, source: "_CollapsibleSection", opened: bool
    ) -> None:
        """Accordion: opening a section collapses every other one.

        Closing a section is left alone — the user can have all three closed
        at once if they want the panel out of the way entirely.
        """
        if not opened:
            return
        for s in self._sections:
            if s is not source and s.is_expanded():
                s.set_expanded(False)

    def _build_detection_section(self) -> QWidget:
        section = _CollapsibleSection("Detection", expanded=True)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        # Entry scope — "Active entry" is the default to keep runs aligned
        # with what's on screen and avoid surprises on multi-entry files.
        self.det_entry_scope = QComboBox()
        self.det_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.det_entry_scope)

        self.det_frame_scope = QComboBox()
        self.det_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        form.addRow("Frames:", self.det_frame_scope)

        # YAML config picker — passed straight through to mlgidBASE's
        # ``config_detect`` argument when non-empty.
        self.det_config_path = QLineEdit()
        self.det_config_path.setPlaceholderText("(default config)")
        self.det_config_path.setToolTip(
            "Optional YAML config file passed to mlgidDETECT as "
            "config_detect. Leave blank to use the built-in defaults."
        )
        det_browse = QPushButton("Browse…")
        det_browse.clicked.connect(self._browse_detect_config)
        det_clear = QPushButton("Clear")
        det_clear.clicked.connect(lambda: self.det_config_path.setText(""))
        det_cfg_row = QWidget()
        det_cfg_h = QHBoxLayout(det_cfg_row)
        det_cfg_h.setContentsMargins(0, 0, 0, 0)
        det_cfg_h.setSpacing(4)
        det_cfg_h.addWidget(self.det_config_path, 1)
        det_cfg_h.addWidget(det_browse)
        det_cfg_h.addWidget(det_clear)
        form.addRow("Config (yaml):", det_cfg_row)

        # Model type — empty string means "use mlgidbase default".
        self.det_model_type = QComboBox()
        self.det_model_type.addItems(["(default)", "faster_rcnn", "dino"])
        self.det_model_type.setToolTip(
            "Detection model architecture. Leave on (default) unless your "
            "config selects a different backbone."
        )
        form.addRow("Model:", self.det_model_type)

        section.body_layout.addLayout(form)
        self.btn_detect = QPushButton("Run detection")
        self.btn_detect.clicked.connect(self._on_run_detection)
        section.body_layout.addWidget(self.btn_detect)
        return section

    def _build_fitting_section(self) -> QWidget:
        section = _CollapsibleSection("Fitting", expanded=False)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.fit_entry_scope = QComboBox()
        self.fit_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.fit_entry_scope)

        self.fit_frame_scope = QComboBox()
        self.fit_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        form.addRow("Frames:", self.fit_frame_scope)

        # Match mlgidbase defaults exactly so an unedited form yields the
        # same behaviour as a bare ``analysis.run_fitting()`` call.
        self.fit_crit_angle = QDoubleSpinBox()
        self.fit_crit_angle.setDecimals(3)
        self.fit_crit_angle.setRange(0.0, 90.0)
        self.fit_crit_angle.setSingleStep(0.5)
        self.fit_crit_angle.setValue(0.0)
        self.fit_crit_angle.setSuffix(" °")
        self.fit_crit_angle.setToolTip(
            "Maximum allowed misorientation angle between peaks within a cluster."
        )
        form.addRow("Critical angle:", self.fit_crit_angle)

        self.fit_dist_peaks = QDoubleSpinBox()
        self.fit_dist_peaks.setDecimals(2)
        self.fit_dist_peaks.setRange(0.0, 1000.0)
        self.fit_dist_peaks.setSingleStep(1.0)
        self.fit_dist_peaks.setValue(10.0)
        self.fit_dist_peaks.setToolTip(
            "Distance threshold for peak clustering (px in detector frame)."
        )
        form.addRow("Cluster dist (peaks):", self.fit_dist_peaks)

        self.fit_dist_rings = QDoubleSpinBox()
        self.fit_dist_rings.setDecimals(2)
        self.fit_dist_rings.setRange(0.0, 1000.0)
        self.fit_dist_rings.setSingleStep(1.0)
        self.fit_dist_rings.setValue(10.0)
        self.fit_dist_rings.setToolTip("Distance threshold for ring clustering.")
        form.addRow("Cluster dist (rings):", self.fit_dist_rings)

        self.fit_cluster_extend = QSpinBox()
        self.fit_cluster_extend.setRange(0, 100)
        self.fit_cluster_extend.setValue(2)
        self.fit_cluster_extend.setToolTip(
            "Number of neighboring peaks to include in cluster expansion."
        )
        form.addRow("Cluster extend:", self.fit_cluster_extend)

        self.fit_theta_fixed = QCheckBox()
        self.fit_theta_fixed.setChecked(True)
        self.fit_theta_fixed.setToolTip(
            "Hold theta fixed during clustering. Default in mlgidBASE."
        )
        form.addRow("Theta fixed:", self.fit_theta_fixed)

        self.fit_use_pool = QCheckBox()
        self.fit_use_pool.setChecked(False)
        self.fit_use_pool.setToolTip(
            "Use multiprocessing for fitting (faster on large stacks, "
            "but interleaves logs unpredictably)."
        )
        form.addRow("Use pool:", self.fit_use_pool)

        self.fit_debug = QCheckBox()
        self.fit_debug.setChecked(False)
        form.addRow("Debug:", self.fit_debug)

        section.body_layout.addLayout(form)
        self.btn_fit = QPushButton("Run fitting")
        self.btn_fit.clicked.connect(self._on_run_fitting)
        section.body_layout.addWidget(self.btn_fit)
        return section

    def _build_matching_section(self) -> QWidget:
        section = _CollapsibleSection("Matching", expanded=False)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.match_entry_scope = QComboBox()
        self.match_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.match_entry_scope)

        self.match_frame_scope = QComboBox()
        self.match_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        form.addRow("Frames:", self.match_frame_scope)

        self.cif_path = QLineEdit()
        self.cif_path.setPlaceholderText("Select preprocessed CIF pickle…")
        cif_browse = QPushButton("Browse…")
        cif_browse.clicked.connect(self._browse_cif)
        cif_row = QWidget()
        cif_h = QHBoxLayout(cif_row)
        cif_h.setContentsMargins(0, 0, 0, 0)
        cif_h.addWidget(self.cif_path, 1)
        cif_h.addWidget(cif_browse)
        form.addRow("CIF pickle:", cif_row)

        self.peaks_type = QComboBox()
        self.peaks_type.addItems(["segments", "rings"])
        form.addRow("Peaks type:", self.peaks_type)

        # Probability threshold (mlgidBASE alias: ``threshold``). The two
        # are equivalent on the mlgidBASE side; we send ``threshold`` since
        # that's what the underlying _run_matching prefers.
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.5)
        self.threshold.setToolTip(
            "Minimum probability for a CIF candidate to be accepted."
        )
        form.addRow("Probability threshold:", self.threshold)

        self.intensity_threshold = QDoubleSpinBox()
        self.intensity_threshold.setRange(0.0, 1e9)
        self.intensity_threshold.setSingleStep(1.0)
        self.intensity_threshold.setDecimals(3)
        self.intensity_threshold.setValue(0.0)
        self.intensity_threshold.setToolTip(
            "Minimum peak intensity to consider during matching."
        )
        form.addRow("Intensity threshold:", self.intensity_threshold)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        form.addRow("Device:", self.device)

        section.body_layout.addLayout(form)

        self.btn_match = QPushButton("Run matching")
        self.btn_match.setEnabled(False)
        self.btn_match.clicked.connect(self._on_run_matching)
        # Gate run button on a CIF path being set — matching can't proceed
        # without it.
        self.cif_path.textChanged.connect(
            lambda t: self.btn_match.setEnabled(bool(t.strip()))
        )
        section.body_layout.addWidget(self.btn_match)
        return section

    # -- Click handlers --

    def _on_run_detection(self) -> None:
        kwargs: dict = {}
        self._inject_entry_scope(self.det_entry_scope, kwargs)
        self._inject_frame_scope(self.det_frame_scope, kwargs)
        cfg = self.det_config_path.text().strip()
        if cfg:
            kwargs["config_detect"] = cfg
        model = self.det_model_type.currentText()
        if model and not model.startswith("("):
            kwargs["model_type"] = model
        self.runRequested.emit(PipelineCommand("run_detection", kwargs))

    def _on_run_fitting(self) -> None:
        kwargs: dict = {}
        self._inject_entry_scope(self.fit_entry_scope, kwargs)
        self._inject_frame_scope(self.fit_frame_scope, kwargs)
        kwargs["crit_angle"] = float(self.fit_crit_angle.value())
        kwargs["clustering_distance_peaks"] = float(self.fit_dist_peaks.value())
        kwargs["clustering_distance_rings"] = float(self.fit_dist_rings.value())
        kwargs["clustering_extend"] = int(self.fit_cluster_extend.value())
        kwargs["theta_fixed"] = bool(self.fit_theta_fixed.isChecked())
        kwargs["use_pool"] = bool(self.fit_use_pool.isChecked())
        kwargs["debug"] = bool(self.fit_debug.isChecked())
        self.runRequested.emit(PipelineCommand("run_fitting", kwargs))

    def _on_run_matching(self) -> None:
        cif = self.cif_path.text().strip()
        if not cif:
            return
        kwargs: dict = {
            "cif_prepr": cif,
            "peaks_type": self.peaks_type.currentText(),
            "threshold": float(self.threshold.value()),
            "intensity_threshold": float(self.intensity_threshold.value()),
            "device": self.device.currentText(),
        }
        self._inject_entry_scope(self.match_entry_scope, kwargs)
        self._inject_frame_scope(self.match_frame_scope, kwargs)
        self.runRequested.emit(PipelineCommand("run_matching", kwargs))

    # -- Internals --

    def _inject_entry_scope(self, combo: QComboBox, kwargs: dict) -> None:
        """Translate the entry-scope dropdown into mlgidBASE's ``entry`` kwarg.

        - ``ENTRY_ACTIVE``: insert ``entry=<active>`` (skip if no resolver
          or no active entry — mlgidBASE will then iterate all).
        - ``ENTRY_ALL``: leave ``entry`` out of kwargs so mlgidBASE
          defaults to all entries.
        """
        if combo.currentText() != ENTRY_ACTIVE:
            return
        if self._get_active_entry is None:
            return
        active = self._get_active_entry()
        if active:
            kwargs["entry"] = active

    def _inject_frame_scope(self, combo: QComboBox, kwargs: dict) -> None:
        if combo.currentText() != FRAME_ACTIVE:
            return
        if self._get_active_frame is None:
            return
        active = self._get_active_frame()
        if active is not None:
            kwargs["frame_num"] = int(active)

    def _browse_detect_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select detection config (YAML)",
            "",
            "YAML (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.det_config_path.setText(path)

    def _browse_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CIF preprocessed pickle",
            "",
            "Pickle (*.pickle *.pkl);;All files (*)",
        )
        if path:
            self.cif_path.setText(path)
