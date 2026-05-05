"""Live readout of the currently selected peak, plus commit/delete actions.

The buttons emit signals that MainWindow turns into ``PipelineCommand``s on
the existing worker thread. Add-to-detected only makes sense for manual
peaks (the others are already on file). Delete-peak is the inverse: only
file-resident peaks can be deleted from here — manual peaks use the
Delete shortcut.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mlgidbase_gui.fit import GaussianFit
from mlgidbase_gui.image_viewer import SelectedPeak
from mlgidbase_gui.pipeline import is_mlgidbase_available

EMPTY = "—"

_SOURCE_LABEL = {
    "manual": "Manual",
    "detected": "Detected",
    "fitted": "Fitted",
    "matched": "Matched",
}


class ParameterPanel(QGroupBox):
    addToDetectedRequested = Signal()
    addToFittedRequested = Signal()
    runFittingRequested = Signal()
    runMatchingRequested = Signal()
    deletePeakRequested = Signal()

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

        self._source_label = self._make_value_label()
        self._radius_label = self._make_value_label()
        self._radius_width_label = self._make_value_label()
        self._angle_label = self._make_value_label()
        self._angle_width_label = self._make_value_label()
        self._type_label = self._make_value_label()
        self._id_label = self._make_value_label()
        # Fit-derived rows. Populated from the profile viewer's last 1D
        # Gaussian fits (manual peaks: real refit; non-manual: synthetic
        # Gaussian honoring the FWHM_r / 2·FWHM_a box convention).
        self._fit_radius_label = self._make_value_label()
        self._fit_fwhm_r_label = self._make_value_label()
        self._fit_angle_label = self._make_value_label()
        self._fit_fwhm_a_label = self._make_value_label()
        self._fit_amp_label = self._make_value_label()

        # Source / Type / ID describe the peak itself and apply to every
        # kind, so they sit above the kind-specific Detected/Fitted blocks.
        form.addRow("Source:", self._source_label)
        form.addRow("Type:", self._type_label)
        form.addRow("ID:", self._id_label)
        form.addRow(self._make_section_label("Detected peak"))
        form.addRow("Radius:", self._radius_label)
        form.addRow("Δ radius:", self._radius_width_label)
        form.addRow("Angle:", self._angle_label)
        form.addRow("Δ angle:", self._angle_width_label)
        form.addRow(self._make_section_label("Fitted peak"))
        form.addRow("Center r:", self._fit_radius_label)
        form.addRow("FWHM r:", self._fit_fwhm_r_label)
        form.addRow("Center a:", self._fit_angle_label)
        form.addRow("FWHM a:", self._fit_fwhm_a_label)
        form.addRow("Amplitude:", self._fit_amp_label)

        self._mlgidbase_available = is_mlgidbase_available()

        # "Add to detected" and "Add to fitted" are mutually exclusive choices
        # the user picks per manual peak — sit them side by side.
        self.btn_add_detected = QPushButton("Add to detected")
        self.btn_add_detected.clicked.connect(self.addToDetectedRequested)
        self.btn_add_fitted = QPushButton("Add to fitted")
        self.btn_add_fitted.setToolTip(
            "Append a row to fitted_peaks using the 1D Gaussian fit "
            "parameters from the radial / angular profile."
        )
        self.btn_add_fitted.clicked.connect(self.addToFittedRequested)
        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.setSpacing(6)
        add_row.addWidget(self.btn_add_detected)
        add_row.addWidget(self.btn_add_fitted)
        add_row_widget = QWidget()
        add_row_widget.setLayout(add_row)
        outer.addWidget(add_row_widget)

        self.btn_run_fitting = QPushButton("Run fitting")
        self.btn_run_fitting.clicked.connect(self.runFittingRequested)
        self.btn_run_matching = QPushButton("Run matching")
        self.btn_run_matching.clicked.connect(self.runMatchingRequested)
        self.btn_delete_peak = QPushButton("Delete peak")
        self.btn_delete_peak.clicked.connect(self.deletePeakRequested)
        for btn in (
            self.btn_run_fitting,
            self.btn_run_matching,
            self.btn_delete_peak,
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

    @staticmethod
    def _make_section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        font = QFont(lbl.font())
        font.setBold(True)
        lbl.setFont(font)
        lbl.setContentsMargins(0, 4, 0, 0)
        return lbl

    # Both selectionChanged and peakGeometryChanged emit SelectedPeak | None.
    # One slot handles both — when peak is None (deselect) we blank every
    # row; otherwise we populate only the section(s) that apply to the
    # peak's kind:
    #
    #   manual           → Detected + Fitted (user is choosing what to commit)
    #   detected         → Detected only
    #   fitted / matched → Fitted only
    #
    # The opposite section stays blank so the same parameters never appear
    # twice for a single peak.
    @Slot(object)
    def set_peak(self, peak: SelectedPeak | None) -> None:
        self._last_peak = peak
        self._update_actions_enabled(peak)
        if peak is None:
            for lbl in (
                self._source_label,
                self._type_label,
                self._id_label,
                self._radius_label,
                self._radius_width_label,
                self._angle_label,
                self._angle_width_label,
                self._fit_radius_label,
                self._fit_fwhm_r_label,
                self._fit_angle_label,
                self._fit_fwhm_a_label,
                self._fit_amp_label,
            ):
                lbl.setText(EMPTY)
            return
        source = _SOURCE_LABEL.get(peak.kind, peak.kind.capitalize())
        if peak.kind == "matched" and peak.structure_uid:
            source = f"{source} ({peak.structure_uid})"
        self._source_label.setText(source)
        self._type_label.setText("Ring" if peak.is_ring else "Segment")
        self._id_label.setText(str(peak.peak_id))

        show_detected = peak.kind in ("manual", "detected")
        if show_detected:
            self._radius_label.setText(f"{peak.radius:.3f} Å⁻¹")
            self._radius_width_label.setText(f"{peak.radius_width:.3f} Å⁻¹")
            self._angle_label.setText(f"{peak.angle:.2f}°")
            self._angle_width_label.setText(f"{peak.angle_width:.2f}°")
        else:
            for lbl in (
                self._radius_label, self._radius_width_label,
                self._angle_label, self._angle_width_label,
            ):
                lbl.setText(EMPTY)

        # If the new selection has no Fitted section, blank those rows
        # immediately so a stale value from the previous selection doesn't
        # linger until set_fits fires.
        if peak.kind == "detected":
            for lbl in (
                self._fit_radius_label, self._fit_fwhm_r_label,
                self._fit_angle_label, self._fit_fwhm_a_label,
                self._fit_amp_label,
            ):
                lbl.setText(EMPTY)

    @Slot(object, object)
    def set_fits(
        self, rfit: GaussianFit | None, afit: GaussianFit | None,
    ) -> None:
        """Update the Fitted-peak rows from the profile viewer's 1D fits.

        Skipped (and blanked) for detected selections — those don't have a
        meaningful fitted-peak readout. Either fit may be ``None`` (no
        convergence, ring with inf width, no selection) → blank that row.
        """
        peak = self._last_peak
        if peak is not None and peak.kind == "detected":
            for lbl in (
                self._fit_radius_label, self._fit_fwhm_r_label,
                self._fit_angle_label, self._fit_fwhm_a_label,
                self._fit_amp_label,
            ):
                lbl.setText(EMPTY)
            return
        if rfit is not None:
            self._fit_radius_label.setText(f"{rfit.center:.3f} Å⁻¹")
            self._fit_fwhm_r_label.setText(f"{rfit.fwhm:.3f} Å⁻¹")
            self._fit_amp_label.setText(f"{rfit.amplitude:.3g}")
        else:
            self._fit_radius_label.setText(EMPTY)
            self._fit_fwhm_r_label.setText(EMPTY)
            self._fit_amp_label.setText(EMPTY)
        if afit is not None:
            self._fit_angle_label.setText(f"{afit.center:.2f}°")
            self._fit_fwhm_a_label.setText(f"{afit.fwhm:.2f}°")
        else:
            self._fit_angle_label.setText(EMPTY)
            self._fit_fwhm_a_label.setText(EMPTY)

    def set_busy(self, busy: bool) -> None:
        """Disable buttons while a pipeline run is in flight."""
        if not self._mlgidbase_available:
            return
        if busy:
            for btn in (
                self.btn_add_detected,
                self.btn_add_fitted,
                self.btn_run_fitting,
                self.btn_run_matching,
                self.btn_delete_peak,
            ):
                btn.setEnabled(False)
        else:
            self._update_actions_enabled(self._current_peak())

    def _update_actions_enabled(self, peak: SelectedPeak | None) -> None:
        if not self._mlgidbase_available:
            for btn in (
                self.btn_add_detected,
                self.btn_add_fitted,
                self.btn_run_fitting,
                self.btn_run_matching,
                self.btn_delete_peak,
            ):
                btn.setEnabled(False)
            return
        # Add-to-detected only makes sense for manual peaks (committing the
        # in-memory candidate). Add-to-fitted accepts manual *and* detected
        # selections — a detected box is the natural input for a fit, and
        # this lets the user promote a detected row into fitted_peaks
        # using the live 1D Gaussian fit. Delete-peak only applies to
        # non-manual peaks (manual uses the Delete shortcut). Fitting /
        # matching always available — the worker surfaces errors if no
        # peaks exist.
        is_manual = peak is not None and peak.kind == "manual"
        is_addable_to_fitted = peak is not None and peak.kind in ("manual", "detected")
        is_file_peak = peak is not None and peak.kind != "manual"
        self.btn_add_detected.setEnabled(is_manual)
        self.btn_add_fitted.setEnabled(is_addable_to_fitted)
        self.btn_delete_peak.setEnabled(is_file_peak)
        self.btn_run_fitting.setEnabled(True)
        self.btn_run_matching.setEnabled(True)

    def _current_peak(self) -> SelectedPeak | None:
        # Re-derive the peak the panel is currently showing (if any) so we can
        # restore button state after a busy spell.
        return getattr(self, "_last_peak", None)
