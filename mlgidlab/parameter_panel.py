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
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from mlgidlab.fit import GaussianFit
from mlgidlab.image_viewer import SelectedPeak
from mlgidlab.pipeline import is_mlgidbase_available

EMPTY = "—"

_SOURCE_LABEL = {
    "manual": "Manual",
    "detected": "Detected",
    "fitted": "Fitted",
    "matched": "Matched",
}


class ParameterPanel(QGroupBox):
    # Mode tokens for the Add-to-fitted dispatch. ``"scipy_1d"`` runs
    # the legacy 1D scipy + zero-fill code path that pre-dated the F-06
    # work; ``"pygidfit_2d"`` routes through ``manual_fit.fit_one_peak``
    # and matches what the pipeline ``run_fitting`` writes. Kept as
    # module-level string constants so callers can compare cleanly
    # (no enum import, no magic strings spread across files).
    FIT_MODE_1D = "scipy_1d"
    FIT_MODE_2D = "pygidfit_2d"

    addToDetectedRequested = Signal()
    addToFittedRequested = Signal()
    # Emits the new state of the "Save fitted as ring" checkbox so the host
    # can refresh the cyan fitted-preview overlay (rings render as a full
    # angular sweep) without waiting for the next selection change.
    saveAsRingChanged = Signal(bool)
    # Emits the active fit-mode token (``"scipy_1d"`` or
    # ``"pygidfit_2d"``) when the user flips the radio pair. The host
    # connects this to a preview-refresh shim so the dashed cyan
    # preview redraws immediately with the new mode's box widths
    # (radial / angular convention differs per mode — see
    # ``_update_fitted_preview`` in main_window).
    fitModeChanged = Signal(str)
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
        self._score_label = self._make_value_label()
        self._type_label = self._make_value_label()
        self._id_label = self._make_value_label()
        # Color swatch shown next to the ID for matched selections —
        # matches the matched-overlay palette so the user can map the
        # readout back to the box on screen at a glance. Lives on the
        # ID row (not Source) because the structure ID is what the
        # colour identifies.
        self._source_swatch = QLabel()
        self._source_swatch.setFixedSize(14, 14)
        self._source_swatch.setVisible(False)
        self._id_row = QWidget()
        _id_h = QHBoxLayout(self._id_row)
        _id_h.setContentsMargins(0, 0, 0, 0)
        _id_h.setSpacing(6)
        _id_h.addWidget(self._id_label, 1)
        _id_h.addWidget(self._source_swatch)
        # Fit-derived rows. Populated from the profile viewer's last 1D
        # Gaussian fits (manual peaks: real refit; non-manual: synthetic
        # Gaussian honoring the unified ``2σ`` box convention shared by
        # the 1D and 2D Add-to-fitted code paths).
        self._fit_radius_label = self._make_value_label()
        self._fit_fwhm_r_label = self._make_value_label()
        self._fit_angle_label = self._make_value_label()
        self._fit_fwhm_a_label = self._make_value_label()
        self._fit_amp_label = self._make_value_label()

        # Source / Type / ID describe the peak itself and apply to every
        # kind, so they sit above the kind-specific Detected/Fitted blocks.
        form.addRow("Source:", self._source_label)
        form.addRow("Type:", self._type_label)
        form.addRow("ID:", self._id_row)
        # Track row indices for the section blocks so set_peak can hide
        # the irrelevant section wholesale (header + 4-5 value rows).
        # Only the section the peak's kind actually populates stays
        # visible; the other is removed from the layout flow rather
        # than just blanked.
        self._detected_section_label = self._make_section_label("Detected peak")
        form.addRow(self._detected_section_label)
        self._row_detected_header = form.rowCount() - 1
        form.addRow("Radius:", self._radius_label)
        form.addRow("Δ radius:", self._radius_width_label)
        form.addRow("Angle:", self._angle_label)
        form.addRow("Δ angle:", self._angle_width_label)
        # mlgidDETECT confidence score. Populated from the SelectedPeak
        # for detected/fitted/matched rows; the row stays hidden for
        # manual peaks (no model provenance, so showing 0.000 would
        # be misleading).
        form.addRow("Score:", self._score_label)
        self._row_score = form.rowCount() - 1
        self._detected_rows = list(range(
            self._row_detected_header, form.rowCount()
        ))
        self._fitted_section_label = self._make_section_label("Fitted peak")
        form.addRow(self._fitted_section_label)
        self._row_fitted_header = form.rowCount() - 1
        form.addRow("Center r:", self._fit_radius_label)
        form.addRow("FWHM r:", self._fit_fwhm_r_label)
        form.addRow("Center a:", self._fit_angle_label)
        form.addRow("FWHM a:", self._fit_fwhm_a_label)
        form.addRow("Amplitude:", self._fit_amp_label)
        self._fitted_rows = list(range(
            self._row_fitted_header, form.rowCount()
        ))
        self._form = form

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

        # Fit-mode selector for Add-to-fitted. Two radios, mutually
        # exclusive via a QButtonGroup. Default is 2D pygidfit (matches
        # what the pipeline run_fitting writes). 1D scipy is the legacy
        # mode that pre-dated the F-06 work — kept available because
        # narrow / off-shape peaks sometimes look better with scipy's
        # quick 1D model than with the full 2D Gaussian. Greyed out
        # when "Save fitted as ring" is on because pygidfit's segment
        # model can't fit a ring cleanly; the ring storage convention
        # bypasses both fit paths anyway.
        self.rb_fit_2d = QRadioButton("2D fit (pygidfit)")
        self.rb_fit_2d.setToolTip(
            "Save through pygidfit's 2D Gaussian fit — same model the "
            "pipeline 'run_fitting' uses. Stores real A/B/C/theta shape "
            "coefficients on the row."
        )
        self.rb_fit_1d = QRadioButton("1D fit (scipy)")
        self.rb_fit_1d.setToolTip(
            "Save through the legacy 1D scipy Gaussian fit on the radial "
            "and angular profile slices. 2D shape coefficients are "
            "zero-filled. Useful when the 2D fit doesn't converge on "
            "the active box."
        )
        self.rb_fit_2d.setChecked(True)
        self._fit_mode_group = QButtonGroup(self)
        self._fit_mode_group.setExclusive(True)
        self._fit_mode_group.addButton(self.rb_fit_2d)
        self._fit_mode_group.addButton(self.rb_fit_1d)
        # Either toggled signal fires for both buttons in an exclusive
        # group — fan in to one emit per user click.
        self._fit_mode_group.buttonToggled.connect(
            lambda *_: self.fitModeChanged.emit(self.fit_mode())
        )
        fit_mode_row = QWidget()
        fit_mode_h = QHBoxLayout(fit_mode_row)
        fit_mode_h.setContentsMargins(0, 0, 0, 0)
        fit_mode_h.setSpacing(12)
        fit_mode_h.addWidget(self.rb_fit_2d)
        fit_mode_h.addWidget(self.rb_fit_1d)
        fit_mode_h.addStretch(1)
        outer.addWidget(fit_mode_row)

        # Ring/segment toggle — applies to whichever box "Add to fitted"
        # would commit. State is sticky: once the user (un)checks it the
        # value persists across selection changes; only Add-to-fitted
        # itself resets it (back to unchecked) after a successful commit.
        # When checked, the saved row uses the canonical ring convention
        # (angle = 45°, angle_width = ∞), the angular profile fit is
        # skipped, and the cyan preview renders as a full-sweep ring.
        # Ring also forces the legacy 1D code path — pygidfit's segment
        # model has no ring analogue — so the fit-mode radios above are
        # greyed out while the ring box is checked (see
        # ``_sync_fit_mode_enabled``).
        self.chk_save_as_ring = QCheckBox("Save fitted as ring")
        self.chk_save_as_ring.toggled.connect(self.saveAsRingChanged)
        self.chk_save_as_ring.toggled.connect(self._sync_fit_mode_enabled)
        outer.addWidget(self.chk_save_as_ring)

        # Run fitting / Run matching used to live here too, but they're
        # already exposed in the Pipeline dock with their full kwarg
        # surface — duplicating them in the per-peak panel just confused
        # the user about what each call would do. Removed.
        self.btn_delete_peak = QPushButton("Delete peak")
        self.btn_delete_peak.clicked.connect(self.deletePeakRequested)
        outer.addWidget(self.btn_delete_peak)

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
                self._score_label,
                self._fit_radius_label,
                self._fit_fwhm_r_label,
                self._fit_angle_label,
                self._fit_fwhm_a_label,
                self._fit_amp_label,
            ):
                lbl.setText(EMPTY)
            self._source_swatch.setVisible(False)
            # No selection → collapse both kind-specific sections so
            # the panel doesn't show stale section headers above empty
            # rows.
            for r in self._detected_rows:
                self._form.setRowVisible(r, False)
            for r in self._fitted_rows:
                self._form.setRowVisible(r, False)
            return
        source = _SOURCE_LABEL.get(peak.kind, peak.kind.capitalize())
        if peak.kind == "matched":
            # Prefer the human-readable structure label (CIF + (hkl) +
            # probability) when the viewer attached one; fall back to the
            # raw structure_uid only if the label wasn't populated.
            tag = peak.structure_label or peak.structure_uid
            if tag:
                source = f"{source} ({tag})"
            if peak.structure_color:
                self._source_swatch.setStyleSheet(
                    f"background-color: {peak.structure_color};"
                    " border: 1px solid #444;"
                )
                self._source_swatch.setVisible(True)
            else:
                self._source_swatch.setVisible(False)
        else:
            self._source_swatch.setVisible(False)
        self._source_label.setText(source)
        self._type_label.setText("Ring" if peak.is_ring else "Segment")
        self._id_label.setText(str(peak.peak_id))

        # Ring/segment toggle is sticky across selection changes (set by
        # the user, reset only by Add-to-fitted) — see chk_save_as_ring.

        # Show only the section(s) relevant to this peak's kind:
        #   manual           → Detected + Fitted (user is choosing what to commit)
        #   detected         → Detected only
        #   fitted / matched → Fitted only
        show_detected = peak.kind in ("manual", "detected")
        show_fitted = peak.kind in ("manual", "fitted", "matched")
        for r in self._detected_rows:
            self._form.setRowVisible(r, show_detected)
        for r in self._fitted_rows:
            self._form.setRowVisible(r, show_fitted)

        if show_detected:
            self._radius_label.setText(f"{peak.radius:.3f} Å⁻¹")
            self._radius_width_label.setText(f"{peak.radius_width:.3f} Å⁻¹")
            self._angle_label.setText(f"{peak.angle:.2f}°")
            self._angle_width_label.setText(f"{peak.angle_width:.2f}°")
        # Score is meaningful only for model-derived peaks. Manual
        # peaks have no score; hide the row entirely so the user
        # doesn't read "0.000" as an actual confidence. Detected /
        # fitted / matched all carry score; the row is only visible
        # when the Detected section is on (matched/fitted hide the
        # whole Detected block).
        has_score = peak.score is not None and peak.kind != "manual"
        self._form.setRowVisible(self._row_score, show_detected and has_score)
        if has_score and show_detected:
            self._score_label.setText(f"{peak.score:.3f}")
        else:
            self._score_label.setText(EMPTY)
        # Detected rows are hidden when ``show_detected`` is False, so
        # we don't blank them — set_fits / next show will refresh.

        # If the new selection has no Fitted section, blank those rows
        # so a stale value from the previous selection can't linger if
        # the section is later re-shown without set_fits running.
        if not show_fitted:
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

    def save_as_ring(self) -> bool:
        """Whether the next Add-to-fitted should commit a ring row."""
        return self.chk_save_as_ring.isChecked()

    def fit_mode(self) -> str:
        """Return the active Add-to-fitted dispatch mode.

        ``FIT_MODE_1D`` (``"scipy_1d"``) → legacy 1D scipy + zero-fill
        path. ``FIT_MODE_2D`` (``"pygidfit_2d"``) → pygidfit's 2D fit
        via ``mlgidlab.manual_fit.fit_one_peak``. Returns the 1D token
        whenever the ring toggle is on, since pygidfit's segment model
        can't fit a ring cleanly — the host should respect this and
        skip the 2D dispatch even if the radio is on.
        """
        if self.chk_save_as_ring.isChecked():
            return self.FIT_MODE_1D
        return (
            self.FIT_MODE_2D if self.rb_fit_2d.isChecked() else self.FIT_MODE_1D
        )

    def _sync_fit_mode_enabled(self, ring_checked: bool) -> None:
        """Grey out the fit-mode radios when ring storage is on.

        pygidfit doesn't model rings, so the 2D option would
        misconverge or fail; the ring code path uses the legacy 1D
        machinery regardless. Disabling the radios while ring is on
        makes that constraint visible — the user can see at a glance
        that they can't pick a different mode here.
        """
        enabled = not bool(ring_checked)
        self.rb_fit_2d.setEnabled(enabled)
        self.rb_fit_1d.setEnabled(enabled)

    def reset_save_as_ring(self) -> None:
        """Force the ring toggle back to unchecked.

        Called by MainWindow after a successful Add-to-fitted commit so
        the user has to opt back in for each new ring row. Emits
        saveAsRingChanged via the standard toggled connection so the
        cyan preview / angular fit refresh follow.
        """
        if self.chk_save_as_ring.isChecked():
            self.chk_save_as_ring.setChecked(False)

    def set_busy(self, busy: bool) -> None:
        """Disable buttons while a pipeline run is in flight."""
        if not self._mlgidbase_available:
            return
        if busy:
            for btn in (
                self.btn_add_detected,
                self.btn_add_fitted,
                self.btn_delete_peak,
            ):
                btn.setEnabled(False)
            self.chk_save_as_ring.setEnabled(False)
        else:
            self._update_actions_enabled(self._current_peak())

    def _update_actions_enabled(self, peak: SelectedPeak | None) -> None:
        if not self._mlgidbase_available:
            for btn in (
                self.btn_add_detected,
                self.btn_add_fitted,
                self.btn_delete_peak,
            ):
                btn.setEnabled(False)
            self.chk_save_as_ring.setEnabled(False)
            return
        # Add-to-detected only makes sense for manual peaks (committing the
        # in-memory candidate). Add-to-fitted accepts manual *and* detected
        # selections — a detected box is the natural input for a fit, and
        # this lets the user promote a detected row into fitted_peaks
        # using the live 1D Gaussian fit. Delete-peak only applies to
        # non-manual peaks (manual uses the Delete shortcut).
        is_manual = peak is not None and peak.kind == "manual"
        is_addable_to_fitted = peak is not None and peak.kind in ("manual", "detected")
        is_file_peak = peak is not None and peak.kind != "manual"
        self.btn_add_detected.setEnabled(is_manual)
        self.btn_add_fitted.setEnabled(is_addable_to_fitted)
        self.chk_save_as_ring.setEnabled(is_addable_to_fitted)
        self.btn_delete_peak.setEnabled(is_file_peak)

    def _current_peak(self) -> SelectedPeak | None:
        # Re-derive the peak the panel is currently showing (if any) so we can
        # restore button state after a busy spell.
        return getattr(self, "_last_peak", None)
