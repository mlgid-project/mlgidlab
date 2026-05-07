from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from mlgidbase_gui.conversion import execute as conversion_execute
from mlgidbase_gui.conversion_panel import ConversionConfig, RawScan
from mlgidbase_gui.pipeline import (
    PipelineCommand,
    execute,
    parse_cif_input,
)
from mlgidbase_gui.session import Session


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
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit(None, exc)
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)
