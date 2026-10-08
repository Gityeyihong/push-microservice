"""Human-handoff policy interface (extensible / inheritable)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class HandoffPolicy(ABC):
    """Decide whether to suggest connecting the user to a human, and build the payload."""

    @abstractmethod
    def should_handoff(
        self,
        *,
        context: str,
        signals: Optional[dict[str, Any]] = None,
        state: str = "INIT",
        positive_rate: float = 0.6,
        context_vector: Any = None,
    ) -> bool:
        ...

    @abstractmethod
    def build_payload(self) -> dict:
        ...
