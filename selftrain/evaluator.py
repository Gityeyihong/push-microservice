"""Held-out push-success evaluation for embedding models.

The held-out set is a frozen snapshot of (anchor, candidate pool, success labels)
captured before training and must never be written into the training corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from .constants import training_dir
from .datagen import video_text
from .push_metrics import evaluate_success
from .rewards import RewardAdapter

logger = logging.getLogger(__name__)


def catalog_text_map(catalog: dict[str, dict]) -> dict[str, str]:
    return {iid: video_text(v) for iid, v in catalog.items()}


async def freeze_heldout(
    store,
    reward: RewardAdapter,
    catalog: dict[str, dict],
    *,
    size: int = 5000,
    page_size: int = 5000,
) -> dict:
    rows: list[dict] = []
    watermark = 0
    while len(rows) < size:
        exposures = await store.get_new_exposures(watermark, page_size)
        if not exposures:
            break
        sessions = list({e["session_id"] for e in exposures})
        feedback = await store.get_feedback_for_sessions(sessions)
        for exposure in exposures:
            watermark = max(watermark, int(exposure["id"]))
            events = feedback.get(exposure["session_id"], [])
            candidates = []
            labels = {}
            for item in exposure.get("items", []):
                iid = str(item.get("item_id"))
                if iid not in catalog:
                    continue
                outcome = reward.outcome(exposure=item, events=events)
                candidates.append({"item_id": iid, "score": float(item.get("score", 0.0))})
                labels[iid] = outcome.label
            if candidates and any(v == "success" for v in labels.values()):
                rows.append({"anchor": exposure.get("anchor", ""), "candidates": candidates, "labels": labels})
            if len(rows) >= size:
                break
        if len(exposures) < page_size:
            break

    await store.save_heldout(rows)
    content = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return {"count": len(rows), "hash": hashlib.sha256(content.encode("utf-8")).hexdigest()}


async def load_heldout(store) -> list[dict]:
    return await store.load_heldout()


def build_encoder(model_path: str | None) -> Callable[[list[str]], list[list[float]]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path) if model_path else SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    def encode(texts):
        return model.encode(texts, normalize_embeddings=True).tolist()

    return encode


def evaluate_rows(
    encode,
    heldout: list[dict],
    catalog: dict[str, dict],
    *,
    k: int = 3,
    with_recall: bool = False,
) -> dict:
    return evaluate_success(encode, heldout, catalog_text_map(catalog), k=k, with_recall=with_recall)


def evaluate_model(
    model_path: str | None,
    heldout: list[dict],
    catalog: dict[str, dict],
    *,
    k: int = 3,
    with_recall: bool = False,
) -> dict:
    try:
        encode = build_encoder(model_path)
    except Exception as exc:
        logger.warning("evaluate_model: encoder unavailable (%s)", exc)
        return {"n": 0, "error": str(exc), "success_at_k": 0.0, "auc": 0.0, "ndcg": 0.0, "k": k}
    return evaluate_rows(encode, heldout, catalog, k=k, with_recall=with_recall)


def write_report(run_dir: Path, metrics: dict, extra: dict | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {**metrics, **(extra or {})}
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    lines = ["# Training run report", ""]
    for key, value in record.items():
        lines.append(f"- **{key}**: {value}")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
