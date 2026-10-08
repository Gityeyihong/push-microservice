"""Tests for click/watch-duration engagement classification."""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mobile.decision_engine import DecisionEngine
from mobile.state_manager import StateManager
from mobile.engagement import classify_engagement
from mobile.config import STRONG_COOLDOWN_MSGS


def _fresh_engine() -> DecisionEngine:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    return DecisionEngine(StateManager(state_file=path))


def test_classify_positive_watch():
    fb, reason = classify_engagement("watch_end", watch_duration_sec=35, video_duration_sec=60)
    assert fb == "positive"
    assert reason == "watch_above_threshold"


def test_classify_negative_no_click():
    fb, reason = classify_engagement("no_click")
    assert fb == "negative"
    assert reason == "exposure_no_click"


def test_classify_negative_fast_swipe():
    fb, reason = classify_engagement(
        "watch_end", watch_duration_sec=3, video_duration_sec=60, was_clicked=True
    )
    assert fb == "negative"
    assert reason == "fast_swipe"


def test_classify_share_favorite():
    fb, _ = classify_engagement("share")
    assert fb == "positive"
    fb, _ = classify_engagement("favorite")
    assert fb == "positive"


def test_classify_not_interested():
    fb, reason = classify_engagement("not_interested")
    assert fb == "strong_negative"
    assert reason == "not_interested"


def test_classify_consecutive_strong_negative():
    fb, reason = classify_engagement("no_click", consecutive_negative=2)
    assert fb == "strong_negative"
    assert reason == "consecutive_negative_streak"


def test_engine_watch_end_positive_updates_tau():
    engine = _fresh_engine()
    cfg = engine._default_config
    tau_before = engine.state.get("tau")
    result = engine.record_video_engagement("v_001", "watch_end", 40, 60)
    assert result["feedback_type"] == "positive"
    tau_after = engine.state.get("tau")
    assert tau_after == max(cfg.tau_min, tau_before * cfg.tau_positive_mult)
    assert engine.state.get("positive_feedbacks") == 1
    assert engine.state.get("consecutive_negative") == 0


def test_engine_no_click_negative_updates_tau():
    engine = _fresh_engine()
    cfg = engine._default_config
    tau_before = engine.state.get("tau")
    result = engine.record_video_engagement("v_002", "no_click")
    assert result["feedback_type"] == "negative"
    tau_after = engine.state.get("tau")
    assert tau_after == min(cfg.tau_max, tau_before * cfg.tau_negative_mult)
    assert engine.state.get("consecutive_negative") == 1


def test_engine_click_only_no_feedback():
    engine = _fresh_engine()
    tau_before = engine.state.get("tau")
    result = engine.record_video_engagement("v_003", "click")
    assert result["feedback_type"] is None
    assert engine.state.get("tau") == tau_before
    assert engine.state.get("total_clicks") == 1


def test_engine_three_consecutive_negatives_strong():
    engine = _fresh_engine()
    for vid in ("v_001", "v_002", "v_003"):
        result = engine.record_video_engagement(vid, "no_click")
    assert result["feedback_type"] == "strong_negative"
    assert engine.state.get("cool_down_remain") == STRONG_COOLDOWN_MSGS
    assert engine.state.get("urge") == 0


def test_engine_dismiss_without_click():
    engine = _fresh_engine()
    result = engine.record_video_engagement("v_004", "dismiss")
    assert result["feedback_type"] == "negative"
    assert result["reason"] == "dismiss_without_click"


def test_engine_click_then_fast_swipe():
    engine = _fresh_engine()
    engine.record_video_engagement("v_005", "click")
    result = engine.record_video_engagement("v_005", "dismiss", watch_duration_sec=2, video_duration_sec=60)
    assert result["feedback_type"] == "negative"
    assert result["reason"] == "fast_swipe"


def test_record_video_feedback_backward_compat():
    engine = _fresh_engine()
    cfg = engine._default_config
    tau_before = engine.state.get("tau")
    engine.record_video_feedback("v_006", "positive")
    assert engine.state.get("tau") == max(cfg.tau_min, tau_before * cfg.tau_positive_mult)


def test_get_state_summary_click_rate():
    engine = _fresh_engine()
    engine.state.add_push_history("v_001")
    engine.record_video_engagement("v_001", "click")
    summary = engine.get_state_summary()
    assert summary["click_rate"] == 1.0


import pytest


@pytest.mark.slow
def test_server_engagement_endpoint():
    from fastapi.testclient import TestClient
    from server.main import app

    with TestClient(app) as client:
        resp = client.post("/api/v1/engagement", json={
            "video_id": "v_001",
            "event": "watch_end",
            "watch_duration_sec": 40,
            "video_duration_sec": 60,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert data["classified_feedback"] == "positive"


def main():
    tests = [
        test_classify_positive_watch,
        test_classify_negative_no_click,
        test_classify_negative_fast_swipe,
        test_classify_share_favorite,
        test_classify_not_interested,
        test_classify_consecutive_strong_negative,
        test_engine_watch_end_positive_updates_tau,
        test_engine_no_click_negative_updates_tau,
        test_engine_click_only_no_feedback,
        test_engine_three_consecutive_negatives_strong,
        test_engine_dismiss_without_click,
        test_engine_click_then_fast_swipe,
        test_record_video_feedback_backward_compat,
        test_get_state_summary_click_rate,
        test_server_engagement_endpoint,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {passed}/{len(tests)} engagement tests passed.")


if __name__ == "__main__":
    main()
