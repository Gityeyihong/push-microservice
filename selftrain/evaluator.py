"""Held-out retrieval evaluation for embedding models.

Metrics: precision@k, hit_rate (top-1), MRR. The held-out set is frozen before
training and must never be written into the training corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT, training_dir

logger = logging.getLogger(__name__)


def _heldout_path(path: Path | None = None) -> Path:
    return path or (training_dir() / "heldout.jsonl")


def freeze_heldout(
    samples: list[dict],
    *,
    path: Path | None = None,
    size: int | None = None,
) -> dict:
    rows = [
        {"anchor": s["anchor"], "video_id": s["video_id"]}
        for s in samples
        if s.get("anchor") and s.get("video_id")
    ]
    if size:
        rows = rows[:size]
    out = _heldout_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    out.write_text(content, encoding="utf-8")
    return {
        "path": str(out),
        "count": len(rows),
        "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def load_heldout(path: Path | None = None) -> list[dict]:
    out = _heldout_path(path)
    if not out.exists():
        return []
    rows = []
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def build_encoder(model_path: str | None):
    """Return encode(texts) -> list[list[float]] using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path) if model_path else SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    def encode(texts):
        return model.encode(texts, normalize_embeddings=True).tolist()

    return encode


def _catalog_map(catalog_file: str | None = None) -> dict[str, dict]:
    from .datagen import load_catalog, video_text

    catalog = load_catalog()
    return {vid: {"text": video_text(v)} for vid, v in catalog.items()}


def evaluate(
    encode,
    heldout: list[dict],
    catalog: dict[str, Any],
    *,
    k: int = 5,
) -> dict:
    import numpy as np

    if not heldout or not catalog:
        return {"n": 0, "precision_at_k": 0.0, "hit_rate": 0.0, "mrr": 0.0, "k": k}

    catalog_ids = list(catalog.keys())
    catalog_texts = [catalog[c]["text"] for c in catalog_ids]
    cat_emb = np.asarray(encode(catalog_texts), dtype=np.float32)

    anchors = [h["anchor"] for h in heldout]
    targets = [h["video_id"] for h in heldout]
    anchor_emb = np.asarray(encode(anchors), dtype=np.float32)

    hits = 0
    rr_sum = 0.0
    valid = 0
    for i, target in enumerate(targets):
        if target not in catalog_ids:
            continue
        valid += 1
        sims = cat_emb @ anchor_emb[i]
        order = np.argsort(-sims)
        ranked_ids = [catalog_ids[j] for j in order[: max(k, 1)]]
        if target in ranked_ids:
            hits += 1
            rank = ranked_ids.index(target) + 1
            rr_sum += 1.0 / rank

    if valid == 0:
        return {"n": 0, "precision_at_k": 0.0, "hit_rate": 0.0, "mrr": 0.0, "k": k}

    return {
        "n": valid,
        "k": k,
        "precision_at_k": round(hits / valid, 4),
        "hit_rate": round(hits / valid, 4),
        "mrr": round(rr_sum / valid, 4),
    }


def evaluate_model(
    model_path: str | None,
    *,
    heldout_path: Path | None = None,
    catalog_file: str | None = None,
    k: int = 5,
) -> dict:
    heldout = load_heldout(heldout_path)
    catalog = _catalog_map(catalog_file)
    try:
        encode = build_encoder(model_path)
    except Exception as exc:
        logger.warning("evaluate_model: encoder unavailable (%s)", exc)
        return {"n": 0, "error": str(exc), "precision_at_k": 0.0, "hit_rate": 0.0, "mrr": 0.0, "k": k}
    return evaluate(encode, heldout, catalog, k=k)


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
