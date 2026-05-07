"""I/O helpers for SIGMA."""

from __future__ import annotations

from pathlib import Path
from typing import List


def ensure_dir(path: Path) -> None:
    """Create a directory path if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def list_frames(frames_dir: Path) -> List[Path]:
    """Return sorted frame paths within a directory."""
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)
    return sorted(p for p in frames_dir.glob("*.jpg"))
