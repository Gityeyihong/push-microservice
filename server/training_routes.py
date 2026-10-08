"""Self-training admin API (mounted under /admin/training)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .admin_routes import require_push_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/training", tags=["push-training"])


class RunBody(BaseModel):
    force: bool = False


class ImportSampleBody(BaseModel):
    anchor: str = Field(..., min_length=1)
    positive: str = Field(..., min_length=1)
    negatives: list[str] = Field(default_factory=list)
    approved: bool = True
    reason: str = ""


@router.get("/status")
async def training_status(_: None = Depends(require_push_admin)):
    from selftrain.constants import load_training_config
    from selftrain.events import TrainingEvents
    from selftrain.registry import get_active, list_versions
    from selftrain.store import create_store

    cfg = load_training_config()
    active = get_active()
    store = create_store(cfg)
    try:
        funnel = await store.funnel_counts()
    except Exception:
        funnel = {}
    return {
        "config": cfg,
        "active": str(active) if active else None,
        "version_count": len(list_versions()),
        "funnel": funnel,
        "recent_events": TrainingEvents().tail(20),
    }


@router.get("/versions")
async def training_versions(_: None = Depends(require_push_admin)):
    from selftrain.registry import get_active, list_versions

    active = get_active()
    return {"active": str(active) if active else None, "versions": list_versions()}


@router.get("/funnel")
async def training_funnel(_: None = Depends(require_push_admin)):
    from selftrain.constants import load_training_config
    from selftrain.store import create_store

    store = create_store(load_training_config())
    return await store.funnel_counts()


@router.post("/run")
async def training_run(body: RunBody, _: None = Depends(require_push_admin)):
    from selftrain.loop import run_cycle

    try:
        return await run_cycle(force=body.force)
    except Exception as exc:
        logger.exception("training run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/build-dataset")
async def training_build_dataset(_: None = Depends(require_push_admin)):
    from selftrain.constants import load_training_config
    from selftrain.datagen import build_candidate_rows, load_catalog
    from selftrain.rewards import load_reward_adapter
    from selftrain.store import create_store

    cfg = load_training_config()
    store = create_store(cfg)
    reward = load_reward_adapter(cfg)
    rows, watermark, stats = await build_candidate_rows(store, reward, load_catalog())
    return {"candidates": len(rows), "new_watermark": watermark, **stats}


@router.post("/samples/import")
async def training_import_sample(body: ImportSampleBody, _: None = Depends(require_push_admin)):
    from selftrain.constants import training_dir
    from selftrain.reviewer import ManualReviewStore

    sample = {"anchor": body.anchor, "positive": body.positive, "negatives": body.negatives}
    review = ManualReviewStore().import_review(sample, body.approved, body.reason)

    imports_path = training_dir() / "imports.jsonl"
    imports_path.parent.mkdir(parents=True, exist_ok=True)
    with imports_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return {"ok": True, "review": review}


@router.post("/rollback")
async def training_rollback(_: None = Depends(require_push_admin)):
    from selftrain.registry import rollback

    result = rollback()
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("reason", "rollback_failed"))
    return result


@router.post("/store/compact")
async def training_store_compact(_: None = Depends(require_push_admin)):
    from selftrain.constants import load_training_config
    from selftrain.store import create_store

    return await create_store(load_training_config()).compact()


@router.get("/eval")
async def training_eval(_: None = Depends(require_push_admin)):
    from selftrain.constants import training_dir

    runs_dir = Path(training_dir()) / "runs"
    if not runs_dir.is_dir():
        return {"runs": []}
    runs = [{"run": r.parent.name, "report": r.read_text(encoding="utf-8")} for r in sorted(runs_dir.glob("*/report.md"))]
    return {"runs": runs[-10:]}
