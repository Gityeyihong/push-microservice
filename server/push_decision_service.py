"""Run DecisionEngine on the push server and serialize push events."""

from __future__ import annotations

from typing import Any, List, Optional

from mobile.decision_config import DecisionConfig, load_decision_config
from mobile.decision_engine import DecisionEngine

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
    advisor_suggestion: bool,
    risk_alert: bool,
    conversation_state: str,
    interaction_mode: str,
    decision_config: Optional[dict] = None,
) -> tuple[List[PushEventModel], dict]:
    cfg = (
        DecisionConfig.from_dict(decision_config)
        if decision_config
        else load_decision_config()
    )
    state = get_session_state(session_id or "default")
    engine = DecisionEngine(state=state, config=cfg)

    candidate_dicts = [_candidate_to_dict(c) for c in candidates]
    events = engine.process_user_message(
        user_message=user_message,
        candidates=candidate_dicts,
        positive_rate=positive_rate,
        advisor_suggestion=advisor_suggestion,
        risk_alert=risk_alert,
        conversation_state=conversation_state,
        interaction_mode=interaction_mode,
        decision_config=cfg,
    )

    push_events = [
        PushEventModel(type=e.type, data=e.data)
        for e in events
    ]
    return push_events, engine.get_state_summary()
