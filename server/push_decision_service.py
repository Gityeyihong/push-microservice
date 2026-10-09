"""Run the decision policy on the push server and serialize push events."""

from __future__ import annotations

from typing import Any, List, Optional

from policy import load_policy
from policy.config import PolicyConfig, load_policy_config

from .models import PushEventModel
from .push_session_store import get_session_state


def _candidate_to_dict(candidate: Any) -> dict:
    if hasattr(candidate, "model_dump"):
        return candidate.model_dump()
    if isinstance(candidate, dict):
        return candidate
    return dict(candidate)


def run_push_decision(
    *,
    session_id: Optional[str],
    user_message: str,
    candidates: list,
    positive_rate: float,
    handoff_suggested: bool,
    signals: dict | None,
    state: str,
    policy_config: Optional[dict] = None,
) -> tuple[List[PushEventModel], dict]:
    cfg = PolicyConfig.from_dict(policy_config) if policy_config else load_policy_config()
    session_state = get_session_state(session_id or "default")
    engine = load_policy(state=session_state, config=cfg)

    events = engine.process(
        user_message=user_message,
        candidates=[_candidate_to_dict(c) for c in candidates],
        positive_rate=positive_rate,
        handoff_suggested=handoff_suggested,
        signals=signals or {},
        state=state,
        config=cfg,
    )
    push_events = [PushEventModel(type=e.type, data=e.data) for e in events]
    return push_events, engine.state_summary()
