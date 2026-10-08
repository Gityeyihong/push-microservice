"""Capture push/engagement records used to build training samples.

Records are append-only JSONL under data/training/. Collection is cheap and can
be toggled via TRAINING_ENABLED; disabled by default so serving stays unaffected.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from .constants import training_dir

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("TRAINING_ENABLED", "0").strip().lower() in ("1", "true", "yes")


class TrainingCollector:
    def __init__(self, enabled: bool | None = None, base_dir: Path | None = None):
        self.enabled = _enabled() if enabled is None else enabled
        self.base_dir = base_dir or training_dir()

    def _append(self, name: str, row: dict) -> None:
        if not self.enabled:
            return
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            path = self.base_dir / name
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("collector append failed: %s", exc)

    def record_push(
        self,
        *,
        session_id: str | None,
        context: str,
        video_ids: list[str],
        intent_query: str = "",
        interaction_mode: str = "auto",
    ) -> None:
        if not video_ids:
            return
        self._append(
            "pushes.jsonl",
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id or "default",
                "context": context,
                "intent_query": intent_query,
                "video_ids": list(video_ids),
                "interaction_mode": interaction_mode,
            },
        )

    def record_engagement(
        self,
        *,
        session_id: str | None,
        video_id: str,
        event: str,
        watch_duration_sec: float = 0.0,
        video_duration_sec: float = 60.0,
        classified: str | None = None,
    ) -> None:
        self._append(
            "engagement.jsonl",
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id or "default",
                "video_id": video_id,
                "event": event,
                "watch_duration_sec": watch_duration_sec,
                "video_duration_sec": video_duration_sec,
                "classified": classified,
            },
        )
