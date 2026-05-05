from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mlgidbase_gui.pipeline import PipelineCommand, is_mlgidbase_available


class PipelinePanel(QWidget):
    """Buttons + parameter controls for the three mlgidbase pipeline stages.

    Emits runRequested(PipelineCommand) when the user clicks a Run button;
    the main window owns the threading and file-handle juggling.
    """

    runRequested = Signal(PipelineCommand)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._available = is_mlgidbase_available()
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

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

        # Detection
        det_box = QGroupBox("Detection")
        det_layout = QVBoxLayout(det_box)
        self.btn_detect = QPushButton("Run detection")
        self.btn_detect.clicked.connect(
            lambda: self.runRequested.emit(PipelineCommand("run_detection", {}))
        )
        det_layout.addWidget(self.btn_detect)
        outer.addWidget(det_box)

        # Fitting
        fit_box = QGroupBox("Fitting")
        fit_layout = QVBoxLayout(fit_box)
        self.btn_fit = QPushButton("Run fitting")
        self.btn_fit.clicked.connect(
            lambda: self.runRequested.emit(PipelineCommand("run_fitting", {}))
        )
        fit_layout.addWidget(self.btn_fit)
        outer.addWidget(fit_box)

        # Matching
        match_box = QGroupBox("Matching")
        match_layout = QFormLayout(match_box)

        self.cif_path = QLineEdit()
        self.cif_path.setPlaceholderText("Select preprocessed CIF pickle…")
        cif_browse = QPushButton("Browse…")
        cif_browse.clicked.connect(self._browse_cif)
        cif_row = QWidget()
        cif_h = QHBoxLayout(cif_row)
        cif_h.setContentsMargins(0, 0, 0, 0)
        cif_h.addWidget(self.cif_path, 1)
        cif_h.addWidget(cif_browse)
        match_layout.addRow("CIF pickle:", cif_row)

        self.peaks_type = QComboBox()
        self.peaks_type.addItems(["segments", "rings"])
        match_layout.addRow("Peaks type:", self.peaks_type)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.5)
        match_layout.addRow("Threshold:", self.threshold)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        match_layout.addRow("Device:", self.device)

        self.btn_match = QPushButton("Run matching")
        self.btn_match.setEnabled(False)
        self.btn_match.clicked.connect(self._on_run_matching)
        self.cif_path.textChanged.connect(
            lambda t: self.btn_match.setEnabled(bool(t.strip()))
        )
        match_layout.addRow(self.btn_match)
        outer.addWidget(match_box)

        # Logs
        log_box = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("monospace"))
        self.log_view.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log_view)
        outer.addWidget(log_box, 1)

    # -- Public API --

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

    # -- Internals --

    def _browse_cif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CIF preprocessed pickle",
            "",
            "Pickle (*.pickle *.pkl);;All files (*)",
        )
        if path:
            self.cif_path.setText(path)

    def _on_run_matching(self) -> None:
        cif = self.cif_path.text().strip()
        if not cif:
            return
        cmd = PipelineCommand(
            "run_matching",
            {
                "cif_prepr": cif,
                "peaks_type": self.peaks_type.currentText(),
                "threshold": float(self.threshold.value()),
                "device": self.device.currentText(),
            },
        )
        self.runRequested.emit(cmd)
