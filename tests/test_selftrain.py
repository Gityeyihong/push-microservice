"""Unit tests for the selftrain safety/gate/registry primitives."""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selftrain.gate import classify_sample, gate_batch
from selftrain.reviewer import ExternalReviewer, ManualReviewStore
from selftrain.safety import TrainingSafety, cosine


def _positive_sample():
    return {
        "anchor": "怎么打破冷战",
        "positive": "冷战破冰的沟通方法",
        "negatives": ["异地恋维系"],
        "source": "positive",
        "behavior": "positive",
        "session_id": "s1",
    }


def test_gate_requires_external_review():
    reviewer = ExternalReviewer({"review": {"provider": "none"}})
    decision = asyncio.run(classify_sample(_positive_sample(), reviewer))
    assert decision.label == "drop"
    assert decision.reason == "review_not_configured"


def test_gate_manual_import_approves():
    with tempfile.TemporaryDirectory() as tmp:
        store = ManualReviewStore(Path(tmp) / "manual.jsonl")
        sample = _positive_sample()
        store.import_review(sample, True, "looks good")
        reviewer = ExternalReviewer({"review": {"provider": "none"}}, manual=store)
        decision = asyncio.run(classify_sample(sample, reviewer))
        assert decision.label == "positive"
        assert decision.review.by == "manual"


def test_gate_batch_flags_review_unavailable():
    samples = [_positive_sample(), {**_positive_sample(), "session_id": "s2"}]
    reviewer = ExternalReviewer({"review": {"provider": "none"}})
    result = asyncio.run(gate_batch(samples, reviewer))
    assert result.review_unavailable
    assert not result.positives


def test_safety_diversity_floor():
    safety = TrainingSafety(min_positive=2, max_sim=0.95)
    dup = _positive_sample()

    def same_embedder(_text):
        return [1.0, 0.0, 0.0]

    verdict = safety.pre_tune_check([dup, {**dup}], same_embedder)
    assert not verdict.ok
    assert verdict.reason == "low_diversity"


def test_safety_commit_rolls_back_on_regression():
    safety = TrainingSafety(epsilon=0.02, max_tune_failures=3)
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
        assert safety.consecutive_failures == 0


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_registry_activate_and_rollback():
    import selftrain.registry as registry

    with tempfile.TemporaryDirectory() as tmp:
        original = registry.versions_dir
        registry.versions_dir = lambda: Path(tmp)
        try:
            v1 = registry.next_version_path(0)
            v1.mkdir(parents=True, exist_ok=True)
            (v1 / "config.json").write_text("{}")
            registry.activate(v1)
            assert registry.get_active() == v1

            v2 = registry.next_version_path(0)
            v2.mkdir(parents=True, exist_ok=True)
            registry.activate(v2)
            assert registry.get_active() == v2

            result = registry.rollback()
            assert result["ok"] is True
            assert registry.get_active() == v1
            assert len(registry.list_versions()) == 2
        finally:
            registry.versions_dir = original
