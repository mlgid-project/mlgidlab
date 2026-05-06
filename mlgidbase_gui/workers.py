from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from mlgidbase_gui.pipeline import (
    PIPELINE_LOGGERS,
    PipelineCommand,
    execute,
    parse_cif_input,
)
from mlgidbase_gui.session import Session


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

    def __init__(self, cif_input: str, nexus_file: Path) -> None:
        super().__init__()
        self._cif_input = cif_input
        self._nexus_file = nexus_file

    def run(self) -> None:
        try:
            result = parse_cif_input(self._cif_input, self._nexus_file)
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


class PipelineWorker(QObject):
    """Runs one mlgidbase pipeline command and streams its log records."""

    finished = Signal(object, object)  # (result, Exception | None)
    log = Signal(str)

    def __init__(self, file_path: Path, command: PipelineCommand) -> None:
        super().__init__()
        self._file_path = file_path
        self._command = command

    def run(self) -> None:
        handler = _SignalLogHandler(self.log)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        loggers = [logging.getLogger(name) for name in PIPELINE_LOGGERS]
        prev_levels = [lg.level for lg in loggers]
        for lg in loggers:
            lg.addHandler(handler)
            if lg.level == logging.NOTSET or lg.level > logging.INFO:
                lg.setLevel(logging.INFO)

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
            for lg, prev in zip(loggers, prev_levels):
                lg.removeHandler(handler)
                lg.setLevel(prev)
