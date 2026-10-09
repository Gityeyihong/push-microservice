"""Default handoff policy: urgent keywords + help-topic vector similarity."""

from __future__ import annotations

import numpy as np

from .base import HandoffPolicy
from policy.config import HANDOFF_LINK
from policy.keywords import signal_forces_handoff, urgent_keywords

HELP_SIMILARITY_THRESHOLD = 0.6
POSITIVE_RATE_THRESHOLD = 0.5


class BuiltinHandoffPolicy(HandoffPolicy):
    def __init__(self, help_topic_vec=None):
        self.help_topic_vec = help_topic_vec

    def should_handoff(
        self,
        *,
        context: str,
        signals: dict | None = None,
        state: str = "INIT",
        positive_rate: float = 0.6,
        context_vector=None,
    ) -> bool:
        signals = signals or {}
        if signal_forces_handoff(signals):
            return True
        if any(kw in context for kw in urgent_keywords()):
            return True
        if self.help_topic_vec is None or context_vector is None:
            return False
        ctx = np.asarray(context_vector, dtype=np.float32)
        help_vec = np.asarray(self.help_topic_vec, dtype=np.float32)
        ctx = ctx / (np.linalg.norm(ctx) + 1e-8)
        help_vec = help_vec / (np.linalg.norm(help_vec) + 1e-8)
        similarity = float(np.dot(ctx, help_vec))
        return similarity > HELP_SIMILARITY_THRESHOLD and positive_rate < POSITIVE_RATE_THRESHOLD

    def build_payload(self) -> dict:
        return {
            "title": "Talk to a human specialist?",
            "subtitle": "Connect with a live agent for 1:1 help",
            "link": HANDOFF_LINK,
        }
