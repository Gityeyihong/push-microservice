"""Generic push events emitted by a decision policy."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PushEvent:
    data: dict
    type: str = "event"


@dataclass
class ItemPushEvent(PushEvent):
    type: str = "item_push"


@dataclass
class HandoffPushEvent(PushEvent):
    type: str = "handoff_push"
