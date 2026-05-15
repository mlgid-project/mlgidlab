"""Smoke harness — workstream C: the `_detached_silx_tree` CM.

`test_smoke_session.test_save_writes_back_and_clears_dirty` had to
stamp the *original* file out-of-band because the viewer's FrameSource
holds `temp_path` open read-only and h5py refuses an in-process `r+`
reopen — the exact silx/h5py coupling C's context manager exists to
manage. These are the positive-path proof: inside
`with main_window._detached_silx_tree():` the working copy *is*
writable, and on exit (normal or exception) the FrameSource is
reacquired so the viewer keeps working.
"""

from __future__ import annotations

import h5py
import pytest

from mlgidlab.session import NexusSession


def _open(window, path) -> NexusSession:
    """Open + activate, mirroring `_on_open_finished`: the session is
    appended to `_sessions` *before* activation so `_reattach_silx_tree`
    reinserts its file into the silx tree."""
    session = NexusSession.open(path)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_temp_path_writable_inside_detached_scope(
    main_window, synthetic_nexus, qtbot
):
    """Inside the detached scope the working copy can be opened `r+`
    (no 'already open for read-only'), and the edit round-trips through
    save after the context manager reacquires on exit.

    `_detach_silx_tree` closes the viewer's FrameSource handle
    synchronously but asks the background PrefetchWorker to drop *its*
    handle via a queued cross-thread signal. In the running GUI the
    event loop is always spinning so that release is effectively
    immediate; a headless test must pump the loop (`qtbot.wait`) to
    model the same thing. This is the CM's real behaviour, not a
    workaround for it.
    """
    session = _open(main_window, synthetic_nexus)
    assert main_window.viewer.n_frames == 3  # FrameSource acquired

    with main_window._detached_silx_tree():
        qtbot.wait(150)  # let the queued PrefetchWorker release land
        with h5py.File(session.temp_path, "r+") as f:
            f["entry_0000/data"].attrs["cm_marker"] = "in_scope"

    # Exit reacquired the FrameSource: the viewer still reads frames.
    main_window.viewer.set_frame(1)
    assert main_window.viewer.current_frame == 1

    # The in-scope edit survives the reattach and a normal save.
    session.mark_dirty()
    assert main_window._save(confirm=False, session=session) is True
    with h5py.File(synthetic_nexus, "r") as f:
        assert f["entry_0000/data"].attrs.get("cm_marker") == "in_scope"


def test_detached_scope_reattaches_on_exception(
    main_window, synthetic_nexus
):
    """An exception raised inside the scope still triggers the
    finally-reattach; the exception propagates and the viewer remains
    usable afterwards."""
    _open(main_window, synthetic_nexus)

    with pytest.raises(RuntimeError, match="boom"):
        with main_window._detached_silx_tree():
            raise RuntimeError("boom")

    main_window.viewer.set_frame(2)
    assert main_window.viewer.current_frame == 2
