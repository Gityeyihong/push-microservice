"""Postgres backend (SQLAlchemy async + asyncpg)."""

from __future__ import annotations

from .base import SQLStore


class PostgresStore(SQLStore):
    def __init__(self, dsn: str):
        if dsn.startswith("postgresql://"):
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif dsn.startswith("postgres://"):
            dsn = dsn.replace("postgres://", "postgresql+asyncpg://", 1)
        super().__init__(dsn)
