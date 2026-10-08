"""SQLite backend (zero-dependency fallback)."""

from __future__ import annotations

from pathlib import Path

from .base import SQLStore


def sqlite_url(path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{p.as_posix()}"


class SQLiteStore(SQLStore):
    def __init__(self, path: str | Path):
        super().__init__(sqlite_url(path))
        self.path = str(path)
