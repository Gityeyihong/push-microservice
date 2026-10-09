"""Capture push/feedback records into the training store.

Collection is cheap and toggled via TRAINING_ENABLED (disabled by default so
serving is unaffected). Uses Postgres/Redis when configured, SQLite otherwise.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from .constants import load_training_config

if TYPE_CHECKING:
    from .store.base import TrainingStore

logger = logging.getLogger(__name__)


def _label_from_classified(classified: Optional[str]) -> str:
    if classified == "positive":
        return "success"
    if classified in ("negative", "strong_negative"):
        return "failure"
    return "unknown"


class TrainingCollector:
    def __init__(self, enabled: bool | None = None, store: "TrainingStore | None" = None, config: dict | None = None):
        self.config = config or load_training_config()
        if enabled is None:
            enabled = os.getenv("TRAINING_ENABLED", "0").strip().lower() in ("1", "true", "yes")
        self.enabled = bool(enabled) or bool(self.config.get("enabled"))
        self._store = store

    def _get_store(self) -> "TrainingStore":
        if self._store is None:
            from .store import create_store

            self._store = create_store(self.config)
        return self._store

    async def record_push(
        self,
        *,
        session_id: str | None,
        context: str,
        item_ids: list[str],
        intent_query: str = "",
    ) -> None:
        if not self.enabled or not item_ids:
            return
        anchor = (intent_query or "").strip() or _last_user(context)
        items = [{"item_id": iid, "shown": True} for iid in item_ids]
        try:
            await self._get_store().record_push(
                session_id=session_id or "default", anchor=anchor, items=items
            )
        except Exception as exc:
            logger.warning("record_push failed: %s", exc)

    async def record_feedback(
        self,
        *,
        session_id: str | None,
        item_id: str,
        event: str,
        watch_duration_sec: float = 0.0,
        item_duration_sec: float = 60.0,
        classified: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        ratio = watch_duration_sec / item_duration_sec if item_duration_sec else 0.0
        try:
            await self._get_store().record_feedback(
                session_id=session_id or "default",
                item_id=item_id,
                event=event,
                watch_ratio=round(ratio, 4),
                reward=0.0,
                label=_label_from_classified(classified),
                propensity=1.0,
            )
        except Exception as exc:
            logger.warning("record_feedback failed: %s", exc)

    async def close(self) -> None:
        if self._store is not None:
            await self._store.close()


def _last_user(context: str) -> str:
    from policy.keywords import last_user_utterance

    return last_user_utterance(context) or (context or "").strip()[:200]
