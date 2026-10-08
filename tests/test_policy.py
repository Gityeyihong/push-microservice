"""Unit tests for the generic decision policy, ranking and keywords."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy.builtin import BuiltinThresholdPolicy
from policy.config import PolicyConfig
from policy.engagement import classify_feedback
from policy.keywords import is_handoff_request, is_item_request
from policy.state import StateManager
from server.models import CandidateItem
from server.ranking import apply_ranking, extract_last_user_message


def _engine():
    fd, path = tempfile.mkstemp(suffix=".json")
    import os

    os.close(fd)
    os.remove(path)
    return BuiltinThresholdPolicy(state=StateManager(state_file=path))


def test_classify_positive_watch():
    fb, reason = classify_feedback("watch_end", watch_duration_sec=35, item_duration_sec=60)
    assert fb == "positive"
    assert reason == "watch_above_threshold"


def test_classify_negative_no_click():
    fb, _ = classify_feedback("no_click", was_clicked=False)
    assert fb == "negative"


def test_classify_strong_negative_streak():
    fb, reason = classify_feedback("no_click", consecutive_negative=2)
    assert fb == "strong_negative"
    assert reason == "consecutive_negative_streak"


def test_is_item_request():
    assert is_item_request("can you recommend a tutorial?")
    assert is_item_request("video")
    assert not is_item_request("hello there")


def test_is_handoff_request():
    assert is_handoff_request("I want to talk to a human agent")
    assert is_handoff_request("customer service please")
    assert not is_handoff_request("tell me about billing")


def test_policy_active_item_push():
    engine = _engine()
    candidates = [
        {"item_id": "i_001", "title": "Tutorial", "similarity": 0.9, "prob_base": 0.5}
    ]
    events = engine.process(user_message="give me a tutorial", candidates=candidates)
    assert len(events) == 1
    assert events[0].type == "item_push"
    assert events[0].data["item_id"] == "i_001"
    assert events[0].data["trigger"] == "active"


def test_policy_handoff_on_active_request():
    engine = _engine()
    events = engine.process(user_message="connect me to a human agent", candidates=[])
    assert any(e.type == "handoff_push" for e in events)


def test_apply_ranking_orders_by_relevance():
    metadata = [
        {"item_id": "i_001", "title": "Billing and invoices", "tags": ["billing"]},
        {"item_id": "i_002", "title": "Getting started setup", "tags": ["getting_started"]},
    ]
    candidates = [
        CandidateItem(item_id="i_002", similarity=0.50, prob_base=0.5, title="Getting started setup"),
        CandidateItem(item_id="i_001", similarity=0.52, prob_base=0.5, title="Billing and invoices"),
    ]
    ranked = apply_ranking("how does billing and invoice work", candidates, metadata)
    assert ranked[0].item_id == "i_001"


def test_extract_last_user_message():
    assert extract_last_user_message("user: hi\n用户: 你好") == "你好"
