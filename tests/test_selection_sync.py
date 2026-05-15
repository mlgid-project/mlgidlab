"""Image-driven selection mirrored onto the peaks table tabs.

``PeaksTablePanel`` constructs standalone (peaks_table_panel.py:182).
``set_external_selection`` (:320-355) clears all three tables first,
then for a non-manual/non-None selection switches to the matching tab
*before* row selection — so the tab switch is observable even with
empty models. ``_TAB_BY_KIND`` is {detected:0, fitted:1, matched:2}
(:52-61). Wiring is at main_window.py:2299.
"""

from __future__ import annotations

import pytest

from mlgidlab.image_viewer import SelectedPeak
from mlgidlab.peaks_table_panel import PeaksTablePanel
from mlgidlab.session import NexusSession

pytestmark = pytest.mark.gui


def _sel(kind):
    return SelectedPeak(
        kind=kind, frame=0, peak_id=0,
        radius=1.0, angle=45.0, radius_width=0.1, angle_width=10.0,
    )


def _no_selection(panel) -> bool:
    return not any(
        t.selectionModel().hasSelection()
        for t in (
            panel._detected_table, panel._fitted_table, panel._matched_table,
        )
    )


def test_set_external_selection_switches_tab_and_clears(qtbot):
    panel = PeaksTablePanel()
    qtbot.addWidget(panel)

    assert panel._tabs.currentIndex() == 0

    # None: clears every table, no tab switch.
    panel.set_external_selection(None)
    assert panel._tabs.currentIndex() == 0
    assert _no_selection(panel)

    # Fitted → tab index 1.
    panel.set_external_selection(_sel("fitted"))
    assert panel._tabs.currentIndex() == 1

    # None again: selection cleared, current tab left where it was.
    panel.set_external_selection(None)
    assert panel._tabs.currentIndex() == 1
    assert _no_selection(panel)

    # Manual peaks have no table row → all cleared, no tab switch.
    panel.set_external_selection(_sel("manual"))
    assert panel._tabs.currentIndex() == 1
    assert _no_selection(panel)

    # Detected → tab index 0.
    panel.set_external_selection(_sel("detected"))
    assert panel._tabs.currentIndex() == 0


def test_viewer_selection_signal_is_wired_to_panel(
    main_window, synthetic_nexus
):
    session = NexusSession.open(synthetic_nexus)
    main_window._set_active_session(session)

    main_window.viewer.selectionChanged.emit(_sel("fitted"))
    assert main_window.peaks_table_panel._tabs.currentIndex() == 1

    main_window.viewer.selectionChanged.emit(_sel("detected"))
    assert main_window.peaks_table_panel._tabs.currentIndex() == 0
