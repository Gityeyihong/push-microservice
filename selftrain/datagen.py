"""Build contrastive samples incrementally from the training store.

Only NEW exposures (past the watermark) are processed, so this scales to large
event volumes without loading everything into memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT, training_dir
from .rewards import RewardAdapter

logger = logging.getLogger(__name__)


def item_text(item: dict) -> str:
    tags = item.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return (
        f"title: {item.get('title', '')}. "
        f"description: {item.get('description', '')}. "
        f"summary: {item.get('summary', '')}. "
        f"tags: {' '.join(tags)}"
    )


def video_text(item: dict) -> str:
    return item_text(item)


def _catalog_path(catalog_file: str | None = None) -> Path:
    name = (
        catalog_file
        or os.getenv("PUSH_CATALOG", "").strip()
        or os.getenv("PUSH_VIDEOS", "").strip()
        or "catalog.json"
    )
    return PROJECT_ROOT / "data" / name


def load_catalog(path: str | Path | None = None) -> dict[str, dict]:
    p = Path(path) if path else _catalog_path()
    if not p.exists():
        logger.warning("catalog not found: %s", p)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("catalog read failed: %s", exc)
        return {}
    return {str(v.get("item_id")): v for v in data if v.get("item_id")}


def dedup_hash(anchor: str, positive_id: str, label: str) -> str:
    raw = f"{anchor}\u0001{positive_id}\u0001{label}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _last_user(context: str) -> str:
    from policy.keywords import last_user_utterance

    return last_user_utterance(context)


async def build_candidate_rows(
    store,
    reward: RewardAdapter,
    catalog: dict[str, dict],
    *,
    negatives_per_anchor: int = 8,
    page_size: int = 5000,
    max_rows: int = 200000,
) -> tuple[list[dict], int, dict]:
    """Return (rows, new_watermark, stats) processing only new exposures."""
    watermark = await store.get_watermark("push_id")
    max_id = watermark
    rows: list[dict] = []
    stats = {"exposures": 0, "success": 0, "failure": 0, "unknown": 0, "capped": False}

    while len(rows) < max_rows:
        exposures = await store.get_new_exposures(watermark, page_size)
        if not exposures:
            break
        sessions = list({e["session_id"] for e in exposures})
        feedback = await store.get_feedback_for_sessions(sessions)

        for exposure in exposures:
            max_id = max(max_id, int(exposure["id"]))
            anchor = (exposure.get("anchor") or "").strip() or _last_user(exposure.get("anchor", ""))
            events = feedback.get(exposure["session_id"], [])
            items = exposure.get("items", [])
            stats["exposures"] += len(items)

            success_items = []
            failure_ids = []
            for item in items:
                outcome = reward.outcome(exposure=item, events=events)
                if outcome.label == "success":
                    success_items.append((item, outcome))
                elif outcome.label == "failure":
                    failure_ids.append(item["item_id"])
                    stats["failure"] += 1
                else:
                    stats["unknown"] += 1

            for item, outcome in success_items:
                pos_id = str(item["item_id"])
                if pos_id not in catalog:
                    continue
                neg_ids = [i for i in failure_ids if i != pos_id][:negatives_per_anchor]
                negatives = [video_text(catalog[i]) for i in neg_ids if i in catalog]
                rows.append(
                    {
                        "anchor": anchor,
                        "positive_id": pos_id,
                        "positive": video_text(catalog[pos_id]),
                        "negatives": negatives,
                        "reward": outcome.reward,
                        "propensity": outcome.propensity,
                        "label": "success",
                        "review_status": "pending",
                        "dedup_hash": dedup_hash(anchor, pos_id, "success"),
                    }
                )
                stats["success"] += 1
                if len(rows) >= max_rows:
                    stats["capped"] = True
                    break

        watermark = max_id
        if stats["capped"] or len(exposures) < page_size:
            break

    return rows, max_id, stats


def save_pairs(rows: list[dict], path: Path | None = None) -> Path:
    out = path or (training_dir() / "pairs.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out
