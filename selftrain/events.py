"""Append-only training event stream (JSONL), mirroring a lightweight message bus."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import training_dir

logger = logging.getLogger(__name__)

EVENT_BUILD_START = "training:build_dataset"
EVENT_TUNE_START = "training:tune_start"
EVENT_TUNE_DONE = "training:tune_done"
EVENT_TUNE_SKIPPED = "training:tune_skipped"
EVENT_ROLLBACK = "training:rollback"
EVENT_COLLAPSE = "training:collapse"
EVENT_ACTIVATED = "training:activated"


class TrainingEvents:
    def __init__(self, path: Path | None = None):
        self.path = path or (training_dir() / "events.jsonl")

    def publish(self, event_type: str, payload: dict | None = None) -> dict:
        entry = {
            "at": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload or {},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("failed to write training event: %s", exc)
        level = logging.WARNING if "skipped" in event_type or "rollback" in event_type else logging.INFO
        logger.log(level, "%s %s", event_type, json.dumps(payload or {}, ensure_ascii=False))
        return entry

    def tail(self, limit: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines[-max(1, limit):]:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
