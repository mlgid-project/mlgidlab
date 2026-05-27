"""Locks in the backend-free claim for the pure-logic path.

Importing ``file_model`` / ``fit`` must not drag in the heavy analysis
stack — those are imported lazily only by ``pipeline.py`` when a
pipeline command actually runs (and by ``manual_fit.py`` when the
"Add to fitted" path's 2D fit is invoked).

The assertion is order-independent: it snapshots ``sys.modules``
before the imports and confirms the two specific imports add no
heavy backend modules. This survives running the test suite in any
order (earlier tests may have already loaded pygid via
``test_manual_fit.py`` or ``test_energy_guard.py``).
"""

from __future__ import annotations

import sys


_HEAVY_BACKENDS = {"mlgidbase", "pygid", "pygidfit", "pygidsim", "torch"}


def test_pure_path_has_no_heavy_backend():
    before = set(sys.modules)
    import mlgidlab.file_model  # noqa: F401
    import mlgidlab.fit  # noqa: F401
    newly_loaded = set(sys.modules) - before
    heavy_added = _HEAVY_BACKENDS & newly_loaded
    assert not heavy_added, (
        f"Importing file_model + fit pulled in heavy backend modules "
        f"{heavy_added!r} — both modules must stay backend-free so the "
        f"pure-logic path runs on environments without the private "
        f"upstream stack installed."
    )
