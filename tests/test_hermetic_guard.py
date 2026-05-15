"""Locks in the backend-free claim for the pure-logic path.

Importing ``file_model`` / ``fit`` must not drag in the heavy analysis
stack — those are imported lazily only by ``pipeline.py`` when a
pipeline command actually runs.
"""

from __future__ import annotations

import sys


def test_pure_path_has_no_heavy_backend():
    import mlgidlab.file_model  # noqa: F401
    import mlgidlab.fit  # noqa: F401

    assert {"mlgidbase", "pygid", "torch"}.isdisjoint(sys.modules)
