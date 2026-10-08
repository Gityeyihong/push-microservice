"""Orchestration of one embedding self-training cycle.

collect (incremental) -> build samples -> external review gate -> safety pre-check
-> freeze/load held-out -> isolated train -> push-success eval -> commit/rollback.
"""

from __future__ import annotations

import logging
from typing import Any

from .constants import load_training_config, training_dir, versions_dir
from .datagen import build_candidate_rows, load_catalog, save_pairs
from .evaluator import evaluate_model, freeze_heldout, write_report
from .events import (
    EVENT_ACTIVATED,
    EVENT_BUILD_START,
    EVENT_TUNE_DONE,
    EVENT_TUNE_SKIPPED,
    EVENT_TUNE_START,
    TrainingEvents,
)
from .gate import gate_batch
from . import registry
from .reviewer import ExternalReviewer
from .rewards import load_reward_adapter
from .safety import TrainingSafety
from .store import create_store
from .trainer import run_isolated

logger = logging.getLogger(__name__)

BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _make_embedder():
    try:
        from .evaluator import build_encoder

        active = registry.get_active()
        return build_encoder(str(active) if active else None)
    except Exception as exc:
        logger.info("diversity embedder unavailable: %s", exc)
        return None


def _heldout_keys(heldout: list[dict]) -> set[tuple[str, str]]:
    keys = set()
    for row in heldout:
        for c in row.get("candidates", []):
            keys.add((row.get("anchor", ""), str(c.get("item_id"))))
    return keys


def _exclude_heldout(positives: list[dict], keys: set[tuple[str, str]]) -> list[dict]:
    return [s for s in positives if (s.get("anchor", ""), str(s.get("positive_id"))) not in keys]


def _next_generation() -> int:
    root = versions_dir()
    gens = [int(p.name.split("_")[1]) for p in root.glob("gen_*") if p.name.split("_")[-1].isdigit()]
    return (max(gens) + 1) if gens else 0


async def run_cycle(config: dict[str, Any] | None = None, *, force: bool = False) -> dict:
    cfg = config or load_training_config()
    events = TrainingEvents()
    safety = TrainingSafety(
        events=events,
        positive_threshold=float(cfg.get("positive_threshold", 0.5)),
        min_positive=int(cfg.get("min_positive", 2)),
        max_sim=float(cfg.get("max_sim", 0.95)),
        epsilon=float(cfg.get("epsilon", 0.02)),
        max_tune_failures=int(cfg.get("max_tune_failures", 3)),
        max_versions=int(cfg.get("max_versions", 5)),
        source_ratio_cap=float(cfg.get("source_ratio_cap", 0.7)),
    )

    if not cfg.get("enabled") and not force:
        return {"ok": False, "reason": "training_disabled"}
    if not safety.can_tune(force):
        return {"ok": False, "reason": "auto_tune_paused"}

    store = create_store(cfg)
    await store.init_schema()
    reward = load_reward_adapter(cfg)
    catalog = load_catalog()
    scale = cfg.get("scale", {})
    eval_cfg = cfg.get("eval", {})

    rows, new_watermark, stats = await build_candidate_rows(
        store,
        reward,
        catalog,
        negatives_per_anchor=int(scale.get("negatives_per_anchor", 8)),
        page_size=int(scale.get("page_size", 5000)),
        max_rows=int(scale.get("sample_cap", 200000)),
    )
    events.publish(EVENT_BUILD_START, {"candidates": len(rows), "catalog": len(catalog), **stats})

    if not rows:
        events.publish(EVENT_TUNE_SKIPPED, {"reason": "no_new_success_samples"})
        return {"ok": False, "reason": "no_new_success_samples"}

    reviewer = ExternalReviewer(cfg)
    gated = await gate_batch(rows, reviewer)
    if gated.review_unavailable and not gated.positives:
        events.publish(EVENT_TUNE_SKIPPED, {"reason": "review_not_configured"})
        return {"ok": False, "reason": "review_not_configured", "candidates": len(rows)}
    if not gated.positives:
        events.publish(EVENT_TUNE_SKIPPED, {"reason": "no_approved_positives", "dropped": len(gated.dropped)})
        return {"ok": False, "reason": "no_approved_positives"}

    await store.upsert_samples(
        [
            {
                "anchor": s.get("anchor", ""),
                "positive_id": s.get("positive_id", ""),
                "negatives": s.get("negatives", []),
                "reward": s.get("reward", 1.0),
                "propensity": s.get("propensity", 1.0),
                "label": "success",
                "review_status": "approved",
                "dedup_hash": s.get("dedup_hash"),
            }
            for s in gated.positives
        ]
    )

    if not await store.has_heldout():
        await freeze_heldout(store, reward, catalog, size=int(eval_cfg.get("freeze_size", 5000)))
    heldout = await store.load_heldout()
    positives = _exclude_heldout(gated.positives, _heldout_keys(heldout))
    if len(positives) < int(cfg.get("min_positive", 2)):
        events.publish(EVENT_TUNE_SKIPPED, {"reason": "too_few_train_positives"})
        return {"ok": False, "reason": "too_few_train_positives"}

    verdict = safety.pre_tune_check(positives, _make_embedder())
    if not verdict.ok:
        events.publish(EVENT_TUNE_SKIPPED, {"reason": verdict.reason})
        return {"ok": False, "reason": verdict.reason}

    save_pairs(positives)

    generation = _next_generation()
    version_path = registry.next_version_path(generation)
    k = int(eval_cfg.get("k", 3))
    with_recall = bool(eval_cfg.get("catalog_recall", False))

    active_model = str(registry.get_active()) if registry.get_active() else None
    before = evaluate_model(active_model or BASE_MODEL, heldout, catalog, k=k, with_recall=with_recall)

    events.publish(EVENT_TUNE_START, {"generation": generation, "positives": len(positives), "version": str(version_path)})
    try:
        run_isolated(active_model, positives, version_path, cfg.get("training", {}))
    except Exception as exc:
        events.publish(EVENT_TUNE_SKIPPED, {"reason": f"train_error:{exc}"})
        return {"ok": False, "reason": "train_error", "error": str(exc)}

    after = evaluate_model(str(version_path), heldout, catalog, k=k, with_recall=with_recall)

    meta = {
        "generation": generation,
        "base_model": active_model or BASE_MODEL,
        "positives": len(positives),
        "success_before": before.get("success_at_k"),
        "success_after": after.get("success_at_k"),
        "auc_before": before.get("auc"),
        "auc_after": after.get("auc"),
        "training": cfg.get("training", {}),
    }
    registry.write_meta(version_path, meta)
    events.publish(EVENT_TUNE_DONE, meta)

    decision = safety.commit_adapter(
        new_path=version_path,
        heldout_before=float(before.get("success_at_k") or 0.0),
        heldout_after=float(after.get("success_at_k") or 0.0),
    )

    run_dir = training_dir() / "runs" / f"gen_{generation}_{version_path.name}"
    write_report(run_dir, {"before": before, "after": after, **meta})

    await store.set_watermark("push_id", new_watermark)

    if decision.get("applied"):
        entry = registry.activate(version_path, meta)
        registry.prune(safety.max_versions)
        events.publish(EVENT_ACTIVATED, {"path": str(version_path), "entry": entry})
        return {"ok": True, "applied": True, "version": str(version_path), "before": before, "after": after}

    return {
        "ok": True,
        "applied": False,
        "rolled_back": True,
        "failed_path": decision.get("failed_path"),
        "consecutive_failures": decision.get("consecutive_failures"),
        "before": before,
        "after": after,
    }
