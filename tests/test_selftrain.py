"""Unit tests for selftrain reward/gate/safety/registry/store."""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selftrain.gate import classify_sample, gate_batch
from selftrain.registry import rollback as registry_rollback
from selftrain.reviewer import ExternalReviewer, ManualReviewStore
from selftrain.rewards import GenericFunnelAdapter
from selftrain.safety import TrainingSafety, cosine
from selftrain.store.sqlite import SQLiteStore


def _success_sample():
    return {
        "anchor": "how to get started",
        "positive": "Getting started guide",
        "positive_id": "i_001",
        "negatives": ["Billing FAQ"],
        "reward": 0.9,
        "propensity": 0.5,
        "label": "success",
    }


def test_reward_adapter_success_failure_unknown():
    cfg = {"reward": {"events": {
        "watch_end": {"role": "success", "weight": 1.0, "min_ratio": 0.5},
        "no_click": {"role": "failure"},
    }}}
    adapter = GenericFunnelAdapter(cfg)
    success = adapter.outcome(exposure={"item_id": "i_001", "rank": 0}, events=[{"item_id": "i_001", "event": "watch_end", "watch_ratio": 0.8}])
    failure = adapter.outcome(exposure={"item_id": "i_001", "rank": 1}, events=[{"item_id": "i_001", "event": "no_click"}])
    unknown = adapter.outcome(exposure={"item_id": "i_001", "rank": 0}, events=[])
    assert success.label == "success" and success.reward > 0
    assert failure.label == "failure" and failure.reward == 0
    assert unknown.label == "unknown"


def test_gate_requires_external_review():
    reviewer = ExternalReviewer({"review": {"provider": "none"}})
    decision = asyncio.run(classify_sample(_success_sample(), reviewer))
    assert decision.label == "drop"
    assert decision.reason == "review_not_configured"


def test_gate_manual_import_approves():
    with tempfile.TemporaryDirectory() as tmp:
        store = ManualReviewStore(Path(tmp) / "manual.jsonl")
        sample = _success_sample()
        store.import_review(sample, True, "ok")
        reviewer = ExternalReviewer({"review": {"provider": "none"}}, manual=store)
        decision = asyncio.run(classify_sample(sample, reviewer))
        assert decision.label == "positive"
        assert decision.review.by == "manual"


def test_gate_batch_flags_review_unavailable():
    reviewer = ExternalReviewer({"review": {"provider": "none"}})
    result = asyncio.run(gate_batch([_success_sample()], reviewer))
    assert result.review_unavailable


def test_safety_diversity_floor():
    safety = TrainingSafety(min_positive=2, max_sim=0.95)
    approved = {**_success_sample(), "review": {"status": "approved"}}

    def same_embedder(_text):
        return [1.0, 0.0, 0.0]

    verdict = safety.pre_tune_check([approved, {**approved}], same_embedder)
    assert not verdict.ok
    assert verdict.reason == "low_diversity"


def test_safety_commit_rolls_back_on_regression():
    safety = TrainingSafety(epsilon=0.02)
    with tempfile.TemporaryDirectory() as tmp:
        version = Path(tmp) / "v1"
        version.mkdir()
        (version / "model.bin").write_text("x")
        result = safety.commit_adapter(new_path=version, heldout_before=0.8, heldout_after=0.5)
        assert result["applied"] is False
        assert safety.consecutive_failures == 1
        assert (Path(tmp) / "failed_v1").exists()


def test_safety_commit_applies_on_improvement():
    safety = TrainingSafety(epsilon=0.02)
    with tempfile.TemporaryDirectory() as tmp:
        version = Path(tmp) / "v1"
        version.mkdir()
        result = safety.commit_adapter(new_path=version, heldout_before=0.5, heldout_after=0.6)
        assert result["applied"] is True


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_registry_activate_and_rollback():
    import selftrain.registry as registry

    with tempfile.TemporaryDirectory() as tmp:
        original = registry.versions_dir
        registry.versions_dir = lambda: Path(tmp)
        try:
            v1 = registry.next_version_path(0)
            (v1 / "config.json").write_text("{}")
            registry.activate(v1)
            v2 = registry.next_version_path(0)
            registry.activate(v2)
            assert registry.get_active() == v2
            assert registry_rollback()["ok"] is True
            assert registry.get_active() == v1
        finally:
            registry.versions_dir = original


def test_sqlite_store_roundtrip():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "t.db")
            await store.init_schema()
            push_id = await store.record_push(
                session_id="s1",
                anchor="how to get started",
                items=[{"item_id": "i_001", "shown": True}, {"item_id": "i_002", "shown": True}],
            )
            assert push_id >= 1
            await store.record_feedback(session_id="s1", item_id="i_001", event="watch_end", watch_ratio=0.8, reward=0.8, label="success", propensity=1.0)
            exposures = await store.get_new_exposures(0, 10)
            assert len(exposures) == 1 and len(exposures[0]["items"]) == 2
            fb = await store.get_feedback_for_sessions(["s1"])
            assert "s1" in fb
            inserted = await store.upsert_samples([
                {"anchor": "a", "positive_id": "i_001", "negatives": [], "reward": 1.0, "propensity": 1.0, "label": "success", "review_status": "approved", "dedup_hash": "h1"}
            ])
            assert inserted == 1
            assert await store.upsert_samples([{"anchor": "a", "positive_id": "i_001", "label": "success", "dedup_hash": "h1"}]) == 0
            assert await store.count_samples() == 1
            await store.save_heldout([{"anchor": "a", "candidates": [{"item_id": "i_001"}], "labels": {"i_001": "success"}}])
            assert await store.has_heldout() is True
            funnel = await store.funnel_counts()
            assert funnel["successes"] == 1
            await store.close()

    asyncio.run(scenario())
