"""Orchestration of one embedding self-training cycle.

collect -> build candidates -> external review gate -> safety pre-check
-> freeze/load held-out -> isolated train -> held-out eval -> commit/rollback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .constants import load_training_config, training_dir, versions_dir
from .datagen import build_candidates, load_catalog, save_pairs
from .evaluator import evaluate_model, freeze_heldout, load_heldout, write_report
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
from .safety import TrainingSafety
from .trainer import run_isolated

logger = logging.getLogger(__name__)

BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _make_embedder():
    try:
        from .evaluator import build_encoder

        return build_encoder(str(registry.get_active()) if registry.get_active() else None)
    except Exception as exc:
        logger.info("diversity embedder unavailable: %s", exc)
        return None


def _heldout_keys(heldout: list[dict]) -> set[tuple[str, str]]:
    return {(h.get("anchor", ""), str(h.get("video_id"))) for h in heldout}


def _exclude_heldout(positives: list[dict], heldout_keys: set[tuple[str, str]]) -> list[dict]:
    return [
        s for s in positives
        if (s.get("anchor", ""), str(s.get("video_id"))) not in heldout_keys
    ]


def _next_generation() -> int:
    root = versions_dir()
    gens = [int(p.name.split("_")[1]) for p in root.glob("gen_*") if p.name.split("_")[-1].isdigit()]
    return (max(gens) + 1) if gens else 0


async def run_cycle(
    config: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> dict:
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

    reviewer = ExternalReviewer(cfg)
    catalog = load_catalog()
    candidates = build_candidates(catalog)
    events.publish(EVENT_BUILD_START, {"candidates": len(candidates), "catalog": len(catalog)})

    gated = await gate_batch(candidates, reviewer)
    if gated.review_unavailable and not gated.positives:
        events.publish(EVENT_TUNE_SKIPPED, {"reason": "review_not_configured"})
        return {"ok": False, "reason": "review_not_configured", "candidates": len(candidates)}
    if not gated.positives:
        events.publish(
            EVENT_TUNE_SKIPPED,
            {"reason": "no_approved_positives", "dropped": len(gated.dropped)},
        )
        return {"ok": False, "reason": "no_approved_positives"}

    heldout_path = training_dir() / "heldout.jsonl"
    if not heldout_path.exists():
        freeze_heldout(
            gated.positives,
            path=heldout_path,
            size=int(cfg.get("eval", {}).get("freeze_size", 100)),
        )
    heldout = load_heldout(heldout_path)
    train_positives = _exclude_heldout(gated.positives, _heldout_keys(heldout))
    if len(train_positives) < int(cfg.get("min_positive", 2)):
        events.publish(EVENT_TUNE_SKIPPED, {"reason": "too_few_train_positives"})
        return {"ok": False, "reason": "too_few_train_positives"}

    verdict = safety.pre_tune_check(train_positives, _make_embedder())
    if not verdict.ok:
        events.publish(EVENT_TUNE_SKIPPED, {"reason": verdict.reason})
        return {"ok": False, "reason": verdict.reason}

    save_pairs(train_positives)

    generation = _next_generation()
    version_path = registry.next_version_path(generation)
    k = int(cfg.get("eval", {}).get("k", 5))

    active_model = str(registry.get_active()) if registry.get_active() else None
    heldout_before = evaluate_model(active_model or BASE_MODEL, heldout_path=heldout_path, k=k)

    events.publish(
        EVENT_TUNE_START,
        {"generation": generation, "positives": len(train_positives), "version": str(version_path)},
    )
    try:
        run_isolated(active_model, train_positives, version_path, cfg.get("training", {}))
    except Exception as exc:
        events.publish(EVENT_TUNE_SKIPPED, {"reason": f"train_error:{exc}"})
        return {"ok": False, "reason": "train_error", "error": str(exc)}

    heldout_after = evaluate_model(str(version_path), heldout_path=heldout_path, k=k)

    meta = {
        "generation": generation,
        "base_model": active_model or BASE_MODEL,
        "positives": len(train_positives),
        "heldout_before": heldout_before.get("precision_at_k"),
        "heldout_after": heldout_after.get("precision_at_k"),
        "heldout_before_mrr": heldout_before.get("mrr"),
        "heldout_after_mrr": heldout_after.get("mrr"),
        "training": cfg.get("training", {}),
    }
    registry.write_meta(version_path, meta)
    events.publish(EVENT_TUNE_DONE, meta)

    decision = safety.commit_adapter(
        new_path=version_path,
        heldout_before=float(heldout_before.get("precision_at_k") or 0.0),
        heldout_after=float(heldout_after.get("precision_at_k") or 0.0),
    )

    run_dir = training_dir() / "runs" / f"gen_{generation}_{version_path.name}"
    write_report(run_dir, {"before": heldout_before, "after": heldout_after, **meta})

    if decision.get("applied"):
        entry = registry.activate(version_path, meta)
        registry.prune(safety.max_versions)
        events.publish(EVENT_ACTIVATED, {"path": str(version_path), "entry": entry})
        return {
            "ok": True,
            "applied": True,
            "version": str(version_path),
            "heldout_before": heldout_before,
            "heldout_after": heldout_after,
        }

    return {
        "ok": True,
        "applied": False,
        "rolled_back": True,
        "failed_path": decision.get("failed_path"),
        "consecutive_failures": decision.get("consecutive_failures"),
        "heldout_before": heldout_before,
        "heldout_after": heldout_after,
    }
