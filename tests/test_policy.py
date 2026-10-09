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


def test_dismiss_uses_persisted_click():
    engine = _engine()
    missed = engine.record_engagement(item_id="i_001", event="dismiss")
    assert missed["reason"] == "dismiss_without_click"
    engine.state.mark_clicked("i_009")
    later = engine.record_engagement(
        item_id="i_009", event="dismiss", watch_duration_sec=30, item_duration_sec=60
    )
    assert later["reason"] == "dismiss_after_partial_watch"


def test_classify_positive_watch():
    fb, reason = classify_feedback("watch_end", watch_duration_sec=35, item_duration_sec=60)
    assert fb == "positive"
    assert reason == "dwell_above_threshold"


def test_classify_dwell_end_alias():
    fb, reason = classify_feedback("dwell_end", watch_duration_sec=35, item_duration_sec=60)
    assert fb == "positive"
    assert reason == "dwell_above_threshold"


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


def test_ranking_honors_type_topic_and_exclusion():
    metadata = [
        {"item_id": "i_001", "title": "Billing and invoices", "tags": ["billing"], "item_type": "article"},
        {"item_id": "i_002", "title": "Getting started setup", "tags": ["getting_started"], "item_type": "video"},
    ]
    candidates = [
        CandidateItem(item_id="i_002", item_type="video", similarity=0.50, prob_base=0.5, title="Getting started setup"),
        CandidateItem(item_id="i_001", item_type="article", similarity=0.50, prob_base=0.5, title="Billing and invoices"),
    ]
    boosted = apply_ranking(
        "hello",
        candidates,
        metadata,
        intent={"primary_topic": "billing", "confidence": 0.0},
    )
    assert boosted[0].item_id == "i_001"

    excluded = apply_ranking(
        "hello",
        candidates,
        metadata,
        intent={"exclude_topics": ["billing"]},
    )
    assert [c.item_id for c in excluded] == ["i_002"]

    typed = apply_ranking("hello", candidates, metadata, item_type="article")
    assert [c.item_id for c in typed] == ["i_001"]


def test_trigger_mode_and_content_need():
    active = _engine()
    events = active.process(
        user_message="hello",
        candidates=[{"item_id": "i_001", "title": "A", "similarity": 0.9}],
        signals={"trigger_mode": "active"},
    )
    assert events[0].data["trigger"] == "active"
    assert events[0].data["item_id"] == "i_001"

    no_handoff = _engine()
    events = no_handoff.process(
        user_message="connect me to a human agent",
        candidates=[{"item_id": "i_001", "title": "A", "similarity": 0.9}],
        signals={"content_need": "item"},
    )
    assert all(e.type != "handoff_push" for e in events)

    handoff = _engine()
    events = handoff.process(user_message="hello", candidates=[], signals={"content_need": "handoff"})
    assert any(e.type == "handoff_push" for e in events)


def test_experiment_id_selects_profile(monkeypatch, tmp_path):
    import json

    from server import ab_resolver

    profiles = tmp_path / "profiles.json"
    experiment = tmp_path / "ab.json"
    profiles.write_text(json.dumps({"profiles": {"tight": {"cooldown_ms": 1000}}}), encoding="utf-8")
    experiment.write_text(json.dumps({"active": False}), encoding="utf-8")
    monkeypatch.setattr(ab_resolver, "profiles_path", lambda: profiles)
    monkeypatch.setattr(ab_resolver, "ab_experiment_path", lambda: experiment)

    chosen = ab_resolver.resolve_config_for_session("s-1", experiment_id="tight")
    assert chosen["cooldown_ms"] == 1000
    try:
        ab_resolver.resolve_config_for_session("s-1", experiment_id="missing")
        raise AssertionError("unknown experiment_id should fail")
    except ValueError:
        pass


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
    assert ranked[0].similarity == 0.52
    assert ranked[1].similarity == 0.50
    assert ranked[0].prob_base >= ranked[1].prob_base


def test_extract_last_user_message():
    assert extract_last_user_message("user: hi\n用户: 你好") == "你好"


def test_user_prefix_from_config(monkeypatch, tmp_path):
    import json

    from policy import keywords as kw

    path = tmp_path / "keywords.json"
    path.write_text(json.dumps({"context": {"user_prefixes": ["human:"]}}), encoding="utf-8")
    monkeypatch.setenv("KEYWORDS_CONFIG_PATH", str(path))
    kw.load_keywords(reload=True)
    try:
        assert extract_last_user_message("human: hello there") == "hello there"
    finally:
        monkeypatch.delenv("KEYWORDS_CONFIG_PATH", raising=False)
        kw.load_keywords(reload=True)


def test_run_push_decision_uses_provider(monkeypatch):
    monkeypatch.setenv("POLICY_PROVIDER", "tests.fake_policy:EchoPolicy")
    from server.push_decision_service import run_push_decision

    events, _summary = run_push_decision(
        session_id="t-custom-policy",
        user_message="hello",
        candidates=[],
        positive_rate=0.6,
        handoff_suggested=False,
        signals={},
        state="INIT",
        policy_config={"provider": "builtin:default"},
    )
    assert len(events) == 1
    assert events[0].type == "item_push"
    assert events[0].data["trigger"] == "custom"
