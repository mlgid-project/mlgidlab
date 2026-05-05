from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Session:
    """Working copy of a NeXus file.

    The original is copied into a fresh per-session temp directory on open,
    keeping the original basename so the silx tree shows the right filename.
    All edits target the temp copy; the original is only touched on Save.
    """

    original_path: Path
    temp_path: Path
    dirty: bool = False

    @classmethod
    def open(cls, original_path: Path | str) -> Session:
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

    def mark_dirty(self) -> None:
        self.dirty = True

    def save(self) -> None:
        """Overwrite the original from the temp file."""
        shutil.copy2(self.temp_path, self.original_path)
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

        self.original_path = new
        self.dirty = False

    def close(self) -> None:
        """Delete the temp file and its per-session directory. Idempotent."""
        parent = self.temp_path.parent
        self.temp_path.unlink(missing_ok=True)
        try:
            parent.rmdir()
        except (OSError, FileNotFoundError):
            pass
