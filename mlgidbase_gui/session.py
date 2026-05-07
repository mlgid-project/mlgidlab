from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal


SessionKind = Literal["nexus", "raw"]


class BaseSession:
    """Common state for any opened-file session.

    Subclasses carry mode-specific paths. ``display_path`` is the path the
    UI title and tree label should show; ``temp_path`` is whatever HDF5
    file the rest of the app should treat as the active file (the writable
    NeXus working copy, or the read-only first raw input for a raw batch).
    """

    kind: SessionKind
    display_path: Path
    temp_path: Path

    def __init__(self) -> None:
        self.dirty: bool = False

    def mark_dirty(self) -> None:
        self.dirty = True

    @property
    def original_path(self) -> Path:
        """Back-compat shim — old call sites read ``session.original_path``."""
        return self.display_path

    def close(self) -> None:
        raise NotImplementedError


class NexusSession(BaseSession):
    """Working copy of a converted NeXus file.

    The original is copied into a fresh per-session temp directory on open,
    keeping the original basename so the silx tree shows the right filename.
    All edits target the temp copy; the original is only touched on Save.
    """

    kind: SessionKind = "nexus"

    def __init__(self, original_path: Path, temp_path: Path) -> None:
        super().__init__()
        self._original_path = original_path
        self.temp_path = temp_path

    @property
    def display_path(self) -> Path:  # type: ignore[override]
        return self._original_path

    @display_path.setter
    def display_path(self, value: Path) -> None:
        self._original_path = value

    @classmethod
    def open(cls, original_path: Path | str) -> NexusSession:
        original = Path(original_path).resolve()
        if not original.is_file():
            raise FileNotFoundError(original)

        temp_dir = Path(tempfile.mkdtemp(prefix="mlgidbase_gui_"))
        temp_path = temp_dir / original.name
        try:
            shutil.copy2(original, temp_path)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return cls(original_path=original, temp_path=temp_path)

    def save(self) -> None:
        """Overwrite the original from the temp file."""
        shutil.copy2(self.temp_path, self._original_path)
        self.dirty = False

    def save_as(self, new_path: Path | str) -> None:
        """Write the temp to a new path; adopt it as the new original.

        Renames the temp file in place so its basename matches the new path —
        callers that re-open the silx tree afterward will see the new name.
        """
        new = Path(new_path).resolve()
        shutil.copy2(self.temp_path, new)

        new_temp = self.temp_path.parent / new.name
        if new_temp != self.temp_path:
            self.temp_path.rename(new_temp)
            self.temp_path = new_temp

        self._original_path = new
        self.dirty = False

    def close(self) -> None:
        """Delete the temp file and its per-session directory. Idempotent."""
        parent = self.temp_path.parent
        self.temp_path.unlink(missing_ok=True)
        try:
            parent.rmdir()
        except (OSError, FileNotFoundError):
            pass


class RawSession(BaseSession):
    """A batch of raw HDF5 detector files awaiting conversion.

    Unlike a NeXus session, the raw inputs are read-only — pygid only reads
    them. ``raw_paths`` is ordered as the user selected them in the open
    dialog. ``temp_path`` exposes the first raw path so generic call sites
    that ask "which file is this?" still work; mode-aware code should
    iterate ``raw_paths`` directly.
    """

    kind: SessionKind = "raw"

    def __init__(self, raw_paths: list[Path]) -> None:
        super().__init__()
        if not raw_paths:
            raise ValueError("RawSession requires at least one raw file path")
        self._raw_paths = list(raw_paths)
        # The "temp" of a raw session is just the first raw file: no
        # writable copy is made. Saving to a raw input is meaningless;
        # the user produces output via the Conversion panel instead.
        self.temp_path = self._raw_paths[0]
        self.output_paths: list[Path] = []

    @property
    def display_path(self) -> Path:  # type: ignore[override]
        # Show the first file's name as the session label; multi-file
        # batches read as "<first.h5>" with the rest visible in the tree.
        return self._raw_paths[0]

    @property
    def raw_paths(self) -> list[Path]:
        return list(self._raw_paths)

    @classmethod
    def open(cls, raw_paths: list[Path | str]) -> RawSession:
        resolved: list[Path] = []
        for p in raw_paths:
            path = Path(p).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            resolved.append(path)
        if not resolved:
            raise ValueError("RawSession.open requires at least one path")
        return cls(raw_paths=resolved)

    def close(self) -> None:
        """Raw inputs are not owned by the GUI; nothing to delete.

        Defined for parity with NexusSession.close so callers can treat
        every session uniformly.
        """
        return None


# Back-compat alias: existing call sites import ``Session`` from this
# module. Keep the name pointing at the converted-NeXus subclass since
# that's what every legacy call site expects (it carries a writable
# temp_path and a save() method).
Session = NexusSession
