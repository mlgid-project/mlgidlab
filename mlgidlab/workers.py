from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from mlgidlab.conversion import execute as conversion_execute
from mlgidlab.conversion_panel import ConversionConfig, RawScan
from mlgidlab.pipeline import (
    PipelineCommand,
    execute,
    parse_cif_input,
)
from mlgidlab.session import Session

logger = logging.getLogger(__name__)


def _trigger_pipeline_imports() -> None:
    """Force ``mlgidbase`` (and its transitive ``pygid`` import) to load.

    Several ``pygid`` submodules (``coordmaps``, ``datasaver``,
    ``conversion``, ``dataloader``) call ``logging.basicConfig`` at
    module top — which first **removes every handler already attached
    to the root logger** before installing their own ``StreamHandler``.
    If we attached our ``_SignalLogHandler`` and *then* triggered the
    lazy import, our handler would be ripped out before the first log
    line ever reaches it. Calling this helper before installing our
    handler means pygid's destructive `removeHandler` sequence has
    already run; module bodies don't re-execute on subsequent imports
    so our handler stays attached for the rest of the worker's run.

    Failures are swallowed: a missing ``mlgidbase`` install just means
    the worker will surface the actual ``ImportError`` from
    ``execute`` later, with the proper exception channel.
    """
    try:
        import mlgidbase  # noqa: F401
    except Exception:
        logger.debug("suppressed exception in _trigger_pipeline_imports", exc_info=True)
        pass


def _trigger_conversion_imports() -> None:
    """Same idea as ``_trigger_pipeline_imports`` for raw conversion.

    ``conversion.execute`` lazily imports ``pygid``; make that happen
    before the worker attaches its log handler so pygid's basicConfig
    side effect doesn't strip our sink.
    """
    try:
        import pygid  # noqa: F401
    except Exception:
        logger.debug("suppressed exception in _trigger_conversion_imports", exc_info=True)
        pass


class CifParseWorker(QObject):
    """Parses CIF input (raw .cif files / folder / pickle) into a CifPattern.

    Runs on a worker thread because raw CIF parsing simulates a 2D
    diffraction pattern per CIF and can take several seconds for a
    typical batch — we don't want the GUI thread blocked while it works.
    Emits ``finished(CifPattern | str | None, Exception | None)`` —
    the result is the cached object (CifPattern for raw, str path for a
    pickle, None for empty input).
    """

    finished = Signal(object, object)

    def __init__(
        self,
        cif_input: str,
        nexus_file: Path,
        entry: str | None = None,
    ) -> None:
        super().__init__()
        self._cif_input = cif_input
        self._nexus_file = nexus_file
        self._entry = entry

    def run(self) -> None:
        try:
            result = parse_cif_input(
                self._cif_input, self._nexus_file, self._entry
            )
            self.finished.emit(result, None)
        except Exception as exc:
            logger.debug("suppressed exception in CifParseWorker.run", exc_info=True)
            self.finished.emit(None, exc)


class CopyWorker(QObject):
    """Runs Session.open in a worker thread."""

    finished = Signal(object, object)  # (Session | None, Exception | None)

    def __init__(self, original_path: Path):
        super().__init__()
        self._original_path = original_path

    def run(self) -> None:
        try:
            session = Session.open(self._original_path)
            self.finished.emit(session, None)
        except Exception as exc:
            logger.debug("suppressed exception in CopyWorker.run", exc_info=True)
            self.finished.emit(None, exc)


class _SignalLogHandler(logging.Handler):
    """Forwards log records to a Qt signal for cross-thread display."""

    def __init__(self, sink: Signal) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.emit(self.format(record))
        except Exception:
            logger.debug("suppressed exception in _SignalLogHandler.emit", exc_info=True)
            pass


class ConversionWorker(QObject):
    """Runs ``pygid`` raw → NeXus conversion off the GUI thread.

    Mirrors ``PipelineWorker``: streams every ``pygid`` log record through
    ``log`` and emits ``finished(list[Path] | None, Exception | None)``
    with the produced output files (or the failure exception). One
    progress-style ``progress`` signal fires per scan boundary so the
    host's QProgressDialog can show batch progress.
    """

    finished = Signal(object, object)  # (list[Path] | None, Exception | None)
    log = Signal(str)
    progress = Signal(int, int)        # (done, total)

    def __init__(
        self,
        scans: list[RawScan],
        cfg: ConversionConfig,
    ) -> None:
        super().__init__()
        self._scans = list(scans)
        self._cfg = cfg

    def run(self) -> None:
        # Force pygid's import-time basicConfig side effect to run
        # before we install our handler — see the helper docstring
        # for the gory details.
        _trigger_conversion_imports()

        handler = _SignalLogHandler(self.log)
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        # Attach to the root logger so propagated records from pygid
        # (which logs through class-named loggers like ``CoordMaps``,
        # ``DataLoader``, ``Datasaver``, plus the unnamed root logger)
        # all reach our sink. Every pygid logger has the default
        # ``propagate=True``.
        root = logging.getLogger()
        root.addHandler(handler)
        prev_level = root.level
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

        total = len(self._scans)
        try:
            self.progress.emit(0, total)
            # ``conversion.execute`` runs all scans internally; per-scan
            # progress would require slicing it open, which we don't do
            # in v1. Fire the start + end progress events so the dialog
            # shows the work range; v2 can break this into per-scan
            # callbacks.
            outputs = conversion_execute(self._scans, self._cfg)
            self.progress.emit(total, total)
            self.finished.emit(outputs, None)
        except Exception as exc:
            logger.debug("suppressed exception in ConversionWorker.run", exc_info=True)
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit(None, exc)
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)


class PipelineWorker(QObject):
    """Runs one mlgidbase pipeline command and streams its log records."""

    finished = Signal(object, object)  # (result, Exception | None)
    log = Signal(str)

    def __init__(self, file_path: Path, command: PipelineCommand) -> None:
        super().__init__()
        self._file_path = file_path
        self._command = command

    def run(self) -> None:
        # Force pygid / mlgidbase import-time basicConfig side effects
        # to run before we install our handler — see the helper
        # docstring for why this ordering matters.
        _trigger_pipeline_imports()

        handler = _SignalLogHandler(self.log)
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        # Attach to the root logger. Pipeline modules use a mix of
        # class-named loggers (``mlgidBASE``, ``DataLoader``,
        # ``CoordMaps``) and the unnamed root logger
        # (``logging.getLogger()`` in mlgidbase.peak_operations,
        # mlgidbase.nexus_operations, mlgidbase.mlgiddetect_functions,
        # pygidfit.process_scans, …). Hooking every one of those by
        # name is fragile; attaching to root catches them all via
        # default propagation.
        root = logging.getLogger()
        root.addHandler(handler)
        prev_level = root.level
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

        try:
            result = execute(self._file_path, self._command)
            self.finished.emit(result, None)
        except Exception as exc:
            # Stream the traceback through the log channel so the user
            # can see *where* mlgidBASE failed — the modal dialog only
            # shows the bare exception message which is often opaque
            # ("invalid index to scalar variable" doesn't tell anyone
            # which dataset / function tripped). The traceback lands in
            # the panel log alongside the mlgidbase log lines.
            logger.debug("suppressed exception in PipelineWorker.run", exc_info=True)
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit(None, exc)
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)


class PrefetchWorker(QObject):
    """Background filler for the FrameSource's per-frame LRU caches.

    Lives on its own ``QThread``. Owns an **independent**
    ``h5py.File`` handle so cross-thread h5py access doesn't share
    state with the main thread's ``FrameSource``. Reads one
    Cartesian frame, computes its polar resample, and emits both to
    the main thread; the main thread then deposits the result into
    the FrameSource's LRU via ``warm_cartesian`` / ``warm_polar``.

    The worker walks frames ahead of the current play-head using a
    **sliding window**: prefetches up to ``window`` frames forward
    of ``head``, pauses when caught up, restarts when ``head``
    advances. ``window`` is set by the host to ``LRU_size - 1`` so
    the prefetcher never overflows the LRU (which would evict
    frames the play-head still needs to reach).

    The internal QTimer ticks at 15 ms and prefetches **one frame
    per tick** — yields between frames so other Qt work on the
    worker's thread (signal delivery, shutdown) doesn't starve.

    Lifecycle slots (all run on the worker's thread via queued
    connections from the host):

    - ``configure(file_path, entry, n_frames, window)`` — (re)open
      the h5py handle for a new entry. Clears the ``_done`` set.
    - ``update_state(head, active, step)`` — push the current play-
      head, the active flag, and the playback frame-stride. Starts /
      stops the internal timer. ``step > 1`` means the host is
      skipping frames during playback (sub-50 ms requested interval);
      the worker walks the same stride so it doesn't prefetch frames
      the player will skip.
    - ``release()`` — close the h5py handle. Called when MainWindow
      detaches silx for a pipeline run.
    """

    # Emitted on every successful frame prefetch. Args:
    # (frame_idx, cart_frame, polar_frame, polar_radius, polar_angle)
    # The two axis arrays are sent so the host can prime
    # FrameSource.polar_axes() on the first emit without a duplicate
    # resample.
    prefetched = Signal(int, object, object, object, object)
    # Emitted when the worker has nothing left to do for the current
    # window — diagnostic only.
    idle = Signal()

    # How aggressively the worker schedules itself. 15 ms is fast
    # enough that one warm frame queues up nearly every play-tick,
    # but slow enough that the worker can't starve the main thread
    # on slow disks.
    _TICK_INTERVAL_MS = 15

    def __init__(self) -> None:
        super().__init__()
        self._file = None
        self._dataset = None
        self._q_xy: np.ndarray | None = None
        self._q_z: np.ndarray | None = None
        self._n_frames: int = 0
        self._head: int = 0
        self._prev_head: int = 0
        self._window: int = 0
        self._active: bool = False
        # Stride to walk when prefetching. Mirrors the player's
        # frame-step so we don't burn disk + CPU on frames the player
        # will skip. Set via ``update_state``; defaults to 1.
        self._step: int = 1
        # Indices we've already sent to the main thread. Cleared on
        # configure() and whenever the head jumps backwards (user
        # scrubbed back or wrapped to 0).
        self._done: set[int] = set()
        self._timer: QTimer | None = None

    @Slot(str, str, int, int)
    def configure(
        self, file_path: str, entry: str, n_frames: int, window: int,
    ) -> None:
        """(Re)open the h5py handle for a new entry."""
        self._release_file()
        try:
            import h5py
            self._file = h5py.File(file_path, "r")
            self._dataset = self._file[f"{entry}/data/img_gid_q"]
            self._q_xy = np.asarray(
                self._file[f"{entry}/data/q_xy"][()], dtype=float
            )
            self._q_z = np.asarray(
                self._file[f"{entry}/data/q_z"][()], dtype=float
            )
            self._n_frames = int(n_frames)
            self._window = max(0, int(window))
            self._done.clear()
            self._prev_head = self._head
        except Exception:
            # Anything fails → leave the worker in a clean closed
            # state. The host will reconfigure on the next entry
            # load; quietly dropping a bad configure beats raising
            # across a queued connection.
            logger.debug("suppressed exception in PrefetchWorker.configure", exc_info=True)
            self._release_file()
            return
        if self._timer is None:
            self._timer = QTimer()
            self._timer.setInterval(self._TICK_INTERVAL_MS)
            self._timer.timeout.connect(self._tick)
        # update_state controls whether the timer is actually running.

    @Slot(int, bool, int)
    def update_state(self, head: int, active: bool, step: int = 1) -> None:
        """Set the play-head, active flag, and prefetch stride.

        If ``head`` moves backwards by more than one frame (user
        scrubbed back or wrapped), clear ``_done`` so the worker
        re-prefetches the new window. Forward advance by 1 (normal
        playback) doesn't reset — the previous frame's spot in
        ``_done`` is harmless. A ``step`` change also clears
        ``_done`` because the set of "useful" frames changes.

        ``step`` mirrors the host's playback frame-step (1 in normal
        playback, >1 when the host is skipping frames to honour a
        sub-50 ms requested interval).
        """
        head = int(head)
        step = max(1, int(step))
        if head < self._prev_head - 1 or step != self._step:
            self._done.clear()
        self._prev_head = head
        self._head = head
        self._active = bool(active)
        self._step = step
        if self._timer is None:
            return
        if self._active and not self._timer.isActive():
            self._timer.start()
        elif not self._active and self._timer.isActive():
            self._timer.stop()

    @Slot()
    def release(self) -> None:
        """Close the h5py handle. Idempotent."""
        if self._timer is not None and self._timer.isActive():
            self._timer.stop()
        self._release_file()

    def _release_file(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                logger.debug("suppressed exception in PrefetchWorker._release_file", exc_info=True)
                pass
        self._file = None
        self._dataset = None
        self._q_xy = None
        self._q_z = None
        self._n_frames = 0
        self._done.clear()

    def _tick(self) -> None:
        if (
            not self._active
            or self._dataset is None
            or self._n_frames <= 1
            or self._window <= 0
            or self._q_xy is None
            or self._q_z is None
        ):
            return
        head = self._head
        step = max(1, self._step)
        # Window covers ``self._window`` *future stops* of the player,
        # not raw frames — so a step>1 schedule still spans the LRU
        # without overshooting into unreachable territory.
        window_end = min(head + step * self._window + 1, self._n_frames)
        # Find the next frame in the window that hasn't been sent yet.
        # One per tick so other queued slots (update_state, release)
        # get a chance to run.
        for i in range(head + step, window_end, step):
            if i in self._done:
                continue
            try:
                cart = np.asarray(self._dataset[i])
                from mlgidlab.polar import cartesian_to_polar
                pol = cartesian_to_polar(cart, self._q_xy, self._q_z)
            except Exception:
                # File closed under us (silx-dance race), dataset
                # moved, polar resample failed. Bail quietly — the
                # host will reconfigure when the dance settles.
                logger.debug("suppressed exception in PrefetchWorker._tick", exc_info=True)
                return
            self.prefetched.emit(i, cart, pol.image, pol.radius, pol.angle)
            self._done.add(i)
            return
        # No work left in the window — idle until the head advances.
        self.idle.emit()
