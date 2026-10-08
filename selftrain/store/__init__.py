"""Training store factory: Postgres/Redis primary, SQLite fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import TrainingStore
from .redis import RedisHelper
from .sqlite import SQLiteStore

logger = logging.getLogger(__name__)


def _sqlite_path(cfg: dict[str, Any]) -> str:
    rel = (cfg.get("storage") or {}).get("sqlite_path", "data/training/training.db")
    if os.path.isabs(rel):
        return rel
    from ..constants import PROJECT_ROOT

    return str(PROJECT_ROOT / rel)


def create_store(cfg: dict[str, Any] | None = None) -> TrainingStore:
    cfg = cfg or {}
    storage = cfg.get("storage") or {}
    backend = (storage.get("backend") or os.getenv("TRAINING_BACKEND", "sqlite")).strip().lower()
    dsn = os.getenv(storage.get("dsn_env", "DATABASE_URL"), "").strip()

    if backend == "postgres" and dsn:
        try:
            from .postgres import PostgresStore

            return PostgresStore(dsn)
        except Exception as exc:
            logger.warning("postgres store unavailable (%s); falling back to sqlite", exc)

    return SQLiteStore(_sqlite_path(cfg))


def create_redis(cfg: dict[str, Any] | None = None) -> RedisHelper:
    cfg = cfg or {}
    storage = cfg.get("storage") or {}
    url = os.getenv(storage.get("redis_url_env", "REDIS_URL"), "")
    return RedisHelper(url or None)


__all__ = ["TrainingStore", "create_store", "create_redis", "RedisHelper", "SQLiteStore"]
