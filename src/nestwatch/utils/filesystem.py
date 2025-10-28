"""Filesystem helper utilities."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist and return the path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["ensure_directory"]
