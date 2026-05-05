"""Live readout of the currently selected manual peak, plus commit actions.

The three buttons (Add to detected / Run fitting / Run matching) are gated
on mlgidbase being installed; they emit signals that MainWindow turns into
``PipelineCommand``s on the existing worker thread.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mlgidbase_gui.image_viewer import ManualPeak
from mlgidbase_gui.pipeline import is_mlgidbase_available

EMPTY = "—"


class ParameterPanel(QGroupBox):
    addToDetectedRequested = Signal()
    runFittingRequested = Signal()
    runMatchingRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Selected peak", parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        outer.addWidget(form_widget)

        self._radius_label = self._make_value_label()
        self._radius_width_label = self._make_value_label()
        self._angle_label = self._make_value_label()
        self._angle_width_label = self._make_value_label()
        self._type_label = self._make_value_label()
        self._id_label = self._make_value_label()

        form.addRow("Radius:", self._radius_label)
        form.addRow("Δ radius:", self._radius_width_label)
        form.addRow("Angle:", self._angle_label)
        form.addRow("Δ angle:", self._angle_width_label)
        form.addRow("Type:", self._type_label)
        form.addRow("ID:", self._id_label)

        self._mlgidbase_available = is_mlgidbase_available()

        self.btn_add_detected = QPushButton("Add to detected")
        self.btn_add_detected.clicked.connect(self.addToDetectedRequested)
        self.btn_run_fitting = QPushButton("Run fitting")
        self.btn_run_fitting.clicked.connect(self.runFittingRequested)
        self.btn_run_matching = QPushButton("Run matching")
        self.btn_run_matching.clicked.connect(self.runMatchingRequested)
        for btn in (
            self.btn_add_detected,
            self.btn_run_fitting,
            self.btn_run_matching,
        ):
            outer.addWidget(btn)

        if not self._mlgidbase_available:
            note = QLabel("<i>mlgidbase not installed — actions disabled.</i>")
            note.setWordWrap(True)
            outer.addWidget(note)

        self.set_peak(None)

    @staticmethod
    def _make_value_label() -> QLabel:
        lbl = QLabel(EMPTY)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return lbl

    # Both selectionChanged and peakGeometryChanged emit ManualPeak | None.
    # One slot handles both — when peak is None (deselect) we blank the row;
    # otherwise we re-render the values.
    @Slot(object)
    def set_peak(self, peak: ManualPeak | None) -> None:
        self._last_peak = peak
        self._update_actions_enabled(peak)
        if peak is None:
            for lbl in (
                self._radius_label,
                self._radius_width_label,
                self._angle_label,
                self._angle_width_label,
                self._type_label,
                self._id_label,
            ):
                lbl.setText(EMPTY)
            return
        self._radius_label.setText(f"{peak.radius:.3f} Å⁻¹")
        self._radius_width_label.setText(f"{peak.radius_width:.3f} Å⁻¹")
        self._angle_label.setText(f"{peak.angle:.2f}°")
        self._angle_width_label.setText(f"{peak.angle_width:.2f}°")
        self._type_label.setText("Ring" if peak.is_ring else "Segment")
        self._id_label.setText(str(peak.temp_id))

    def set_busy(self, busy: bool) -> None:
        """Disable buttons while a pipeline run is in flight."""
        if not self._mlgidbase_available:
            return
        if busy:
            for btn in (
                self.btn_add_detected,
                self.btn_run_fitting,
                self.btn_run_matching,
            ):
                btn.setEnabled(False)
        else:
            self._update_actions_enabled(self._current_peak())

    def _update_actions_enabled(self, peak: ManualPeak | None) -> None:
        if not self._mlgidbase_available:
            for btn in (
                self.btn_add_detected,
                self.btn_run_fitting,
                self.btn_run_matching,
            ):
                btn.setEnabled(False)
            return
        # Add-to-detected requires a selected peak; fitting/matching can run
        # independently (mirroring the Pipeline tab) but only make sense when
        # there's at least something on the file — keep enabled regardless and
        # let the worker surface errors.
        self.btn_add_detected.setEnabled(peak is not None)
        self.btn_run_fitting.setEnabled(True)
        self.btn_run_matching.setEnabled(True)

    def _current_peak(self) -> ManualPeak | None:
        # Re-derive the peak the panel is currently showing (if any) so we can
        # restore button state after a busy spell.
        return getattr(self, "_last_peak", None)
