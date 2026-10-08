"""Push-success metrics: success@k, AUC, NDCG@k, funnel, catalog recall (ANN)."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import numpy as np


def _cosine_matrix(anchor_vecs: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
    a = anchor_vecs / (np.linalg.norm(anchor_vecs, axis=1, keepdims=True) + 1e-8)
    c = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
    return a @ c.T


def success_at_k(
    encode: Callable[[list[str]], list[list[float]]],
    heldout: list[dict],
    catalog_text: dict[str, str],
    *,
    k: int = 3,
) -> dict:
    precisions = []
    hits = 0
    n = 0
    for row in heldout:
        candidates = [c for c in row.get("candidates", []) if c.get("item_id") in catalog_text]
        labels = row.get("labels", {})
        if not candidates:
            continue
        n += 1
        anchor_vec = np.asarray(encode([row["anchor"]]), dtype=np.float32)
        cand_vecs = np.asarray(encode([catalog_text[c["item_id"]] for c in candidates]), dtype=np.float32)
        sims = _cosine_matrix(anchor_vec, cand_vecs)[0]
        order = np.argsort(-sims)[: max(k, 1)]
        top_ids = [candidates[i]["item_id"] for i in order]
        success_flags = [1 if labels.get(iid) == "success" else 0 for iid in top_ids]
        precisions.append(sum(success_flags) / max(len(top_ids), 1))
        if any(success_flags):
            hits += 1
    if n == 0:
        return {"n": 0, "k": k, "success_at_k": 0.0, "hit_rate": 0.0}
    return {
        "n": n,
        "k": k,
        "success_at_k": round(sum(precisions) / n, 4),
        "hit_rate": round(hits / n, 4),
    }


def rank_auc(
    encode: Callable[[list[str]], list[list[float]]],
    heldout: list[dict],
    catalog_text: dict[str, str],
) -> float:
    pairs = 0
    correct = 0.0
    for row in heldout:
        candidates = [c for c in row.get("candidates", []) if c.get("item_id") in catalog_text]
        labels = row.get("labels", {})
        pos = [c for c in candidates if labels.get(c["item_id"]) == "success"]
        neg = [c for c in candidates if labels.get(c["item_id"]) == "failure"]
        if not pos or not neg:
            continue
        anchor_vec = np.asarray(encode([row["anchor"]]), dtype=np.float32)
        texts = [catalog_text[c["item_id"]] for c in candidates]
        sims = _cosine_matrix(anchor_vec, np.asarray(encode(texts), dtype=np.float32))[0]
        id_to_sim = {c["item_id"]: float(sims[i]) for i, c in enumerate(candidates)}
        for p in pos:
            for ng in neg:
                pairs += 1
                sp, sn = id_to_sim[p["item_id"]], id_to_sim[ng["item_id"]]
                correct += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
    return round(correct / pairs, 4) if pairs else 0.0


def ndcg_at_k(
    encode: Callable[[list[str]], list[list[float]]],
    heldout: list[dict],
    catalog_text: dict[str, str],
    *,
    k: int = 5,
) -> float:
    scores = []
    for row in heldout:
        candidates = [c for c in row.get("candidates", []) if c.get("item_id") in catalog_text]
        labels = row.get("labels", {})
        if not candidates:
            continue
        anchor_vec = np.asarray(encode([row["anchor"]]), dtype=np.float32)
        cand_vecs = np.asarray(encode([catalog_text[c["item_id"]] for c in candidates]), dtype=np.float32)
        sims = _cosine_matrix(anchor_vec, cand_vecs)[0]
        order = np.argsort(-sims)[: max(k, 1)]
        gains = [1.0 if labels.get(candidates[i]["item_id"]) == "success" else 0.0 for i in order]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted((1.0 if labels.get(c["item_id"]) == "success" else 0.0) for c in candidates)[::-1][: max(k, 1)]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def catalog_recall(
    encode: Callable[[list[str]], list[list[float]]],
    heldout: list[dict],
    catalog_text: dict[str, str],
    *,
    k: int = 10,
) -> float:
    import faiss

    ids = list(catalog_text.keys())
    if not ids:
        return 0.0
    cat_vecs = np.asarray(encode([catalog_text[i] for i in ids]), dtype=np.float32)
    cat_vecs = cat_vecs / (np.linalg.norm(cat_vecs, axis=1, keepdims=True) + 1e-8)
    index = faiss.IndexFlatIP(cat_vecs.shape[1])
    index.add(cat_vecs)
    hits = 0
    n = 0
    for row in heldout:
        targets = [iid for iid, lab in (row.get("labels") or {}).items() if lab == "success" and iid in catalog_text]
        if not targets:
            continue
        n += 1
        q = np.asarray(encode([row["anchor"]]), dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        _, idx = index.search(q, k)
        retrieved = {ids[j] for j in idx[0] if j >= 0}
        if any(t in retrieved for t in targets):
            hits += 1
    return round(hits / n, 4) if n else 0.0


def evaluate_success(
    encode: Callable[[list[str]], list[list[float]]],
    heldout: list[dict],
    catalog_text: dict[str, str],
    *,
    k: int = 3,
    with_recall: bool = False,
) -> dict:
    out = success_at_k(encode, heldout, catalog_text, k=k)
    out["auc"] = rank_auc(encode, heldout, catalog_text)
    out["ndcg"] = ndcg_at_k(encode, heldout, catalog_text, k=k)
    if with_recall:
        out["catalog_recall"] = catalog_recall(encode, heldout, catalog_text, k=max(k, 10))
    return out
