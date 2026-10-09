"""Test double loaded via POLICY_PROVIDER."""

from policy.base import DecisionPolicy
from policy.events import ItemPushEvent


class EchoPolicy(DecisionPolicy):
    def process(
        self,
        *,
        user_message,
        candidates=None,
        positive_rate=0.6,
        handoff_suggested=False,
        signals=None,
        state="INIT",
        config=None,
    ):
        return [ItemPushEvent(data={"item_id": "from-custom", "trigger": "custom"})]

    def record_engagement(
        self,
        *,
        item_id,
        event,
        watch_duration_sec=0,
        item_duration_sec=60,
        config=None,
    ):
        return {"feedback_type": None, "reason": "noop"}
