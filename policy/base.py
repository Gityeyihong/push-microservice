"""Decision-policy interface (extensible / inheritable)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .events import PushEvent
from .state import StateManager
from .config import PolicyConfig


class DecisionPolicy(ABC):
    """Decide which items (or a handoff) to push for a user turn.

    Subclass and register via config `policy.provider = "module:ClassName"`.
    """

    def __init__(self, state: Optional[StateManager] = None, config: Optional[PolicyConfig] = None):
        self.state = state or StateManager()
        cfg = config or PolicyConfig()
        self._default_config = cfg
        self._active_config = cfg

    @abstractmethod
    def process(
        self,
        *,
        user_message: str,
        candidates: Optional[list[dict]] = None,
        positive_rate: float = 0.6,
        handoff_suggested: bool = False,
        signals: Optional[dict[str, Any]] = None,
        state: str = "INIT",
        config: Optional[PolicyConfig] = None,
    ) -> list[PushEvent]:
        ...

    @abstractmethod
    def record_engagement(
        self,
        *,
        item_id: str,
        event: str,
        watch_duration_sec: float = 0,
        item_duration_sec: float = 60,
        config: Optional[PolicyConfig] = None,
    ) -> dict:
        ...

    def state_summary(self) -> dict:
        return {}
