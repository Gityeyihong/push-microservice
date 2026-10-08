"""Generic candidate ranking / reranking against last user intent.

Intent rules and exclusion rules are loaded from config/keywords.json.
"""

from __future__ import annotations

from typing import Any

from .models import CandidateItem
from policy.keywords import (
    exclusion_rules,
    intent_rules,
    meta_request_keywords,
)

LEXICAL_WEIGHT_MOCK = 0.82
VECTOR_WEIGHT_MOCK = 0.18
LEXICAL_WEIGHT_REAL = 0.35
VECTOR_WEIGHT_REAL = 0.65
LEXICAL_WEIGHT_INTENT = 0.50
VECTOR_WEIGHT_INTENT = 0.50
LEXICAL_RESCUE_MIN = 0.30
LEXICAL_RESCUE_MAX = 3


def extract_last_user_message(context: str) -> str:
    msgs = extract_user_messages(context)
    return msgs[-1] if msgs else ""


def extract_user_messages(context: str) -> list[str]:
    msgs: list[str] = []
    for line in (context or "").strip().split("\n"):
        line = line.strip()
        if line.startswith("用户:"):
            text = line.removeprefix("用户:").strip()
            if text:
                msgs.append(text)
        elif line.lower().startswith("user:"):
            text = line[5:].strip()
            if text:
                msgs.append(text)
    return msgs


def _all_intent_user_kws() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for rule in intent_rules():
        for k in rule.get("user_kws") or []:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
    return ordered


def is_meta_content_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    meta_kws = [k.lower() for k in meta_request_keywords()]
    if not any(k in t for k in meta_kws):
        return False
    substantive = [k for k in _all_intent_user_kws() if k in t and k not in meta_kws]
    if substantive:
        return False
    if len(t) <= 36:
        return True
    return sum(1 for k in meta_kws if k in t) >= 2


def build_intent_query(context: str) -> str:
    msgs = extract_user_messages(context)
    if not msgs:
        return ""
    if len(msgs) == 1:
        return msgs[0]
    if is_meta_content_request(msgs[-1]):
        prior = [m for m in msgs[:-1] if m.strip()]
        if prior:
            return " ".join(prior)
    return " ".join(msgs)


def _intent_confidence(intent: Any | None) -> float:
    if intent is None:
        return 0.0
    try:
        if hasattr(intent, "confidence"):
            return float(intent.confidence)
        if isinstance(intent, dict):
            return float(intent.get("confidence") or 0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def resolve_intent_query(context: str, intent: Any | None = None) -> str:
    if intent is not None and _intent_confidence(intent) >= 0.5:
        iq = getattr(intent, "query", None)
        if iq is None and isinstance(intent, dict):
            iq = intent.get("query")
        if iq and str(iq).strip():
            return str(iq).strip()
    return build_intent_query(context) or extract_last_user_message(context) or ""


def resolve_is_meta_request(last_user: str, intent: Any | None = None) -> bool:
    if intent is not None and _intent_confidence(intent) >= 0.5:
        if hasattr(intent, "is_meta"):
            return bool(intent.is_meta)
        if isinstance(intent, dict):
            return bool(intent.get("is_meta", False))
    return is_meta_content_request(last_user)


def _meta_blob(meta: dict[str, Any]) -> str:
    tags = meta.get("tags") or []
    return " ".join(
        [
            str(meta.get("title") or ""),
            str(meta.get("summary") or ""),
            str(meta.get("description") or ""),
            " ".join(str(t) for t in tags),
        ]
    ).lower()


def _active_rules(user: str) -> list[dict]:
    low = (user or "").lower()
    return [r for r in intent_rules() if any(str(k).lower() in low for k in (r.get("user_kws") or []))]


def compute_lexical_score(user: str, meta: dict[str, Any]) -> float:
    if not (user or "").strip():
        return 0.0
    blob = _meta_blob(meta)
    tags = {str(t).lower() for t in (meta.get("tags") or [])}
    active = _active_rules(user)

    if not active:
        meta_kws = [str(k).lower() for k in meta_request_keywords()]
        hits = sum(1 for k in meta_kws if k in user.lower() and k in blob)
        return min(0.35, hits * 0.08)

    scores: list[float] = []
    for rule in active:
        rule_score = 0.0
        video_hits = sum(1 for k in (rule.get("item_kws") or []) if str(k).lower() in blob)
        tag_hits = sum(1 for t in (rule.get("tag_boost") or []) if str(t).lower() in tags)
        if video_hits:
            rule_score += min(0.75, 0.25 * video_hits)
        if tag_hits:
            rule_score += min(0.35, 0.12 * tag_hits)
        if rule_score > 0:
            scores.append(min(1.0, rule_score * float(rule.get("weight", 1.0))))

    if not scores:
        return 0.05

    base = max(scores)
    low = user.lower()
    for ex in exclusion_rules():
        if any(str(k).lower() in low for k in (ex.get("user_kws") or [])) and any(
            str(k).lower() in blob for k in (ex.get("item_kws") or [])
        ):
            base = max(0.0, base - float(ex.get("penalty", 0.2)))
    return round(min(1.0, max(0.0, base)), 4)


def keyword_overlap_boost(user: str, meta: dict[str, Any]) -> float:
    blob = _meta_blob(meta)
    low = user.lower()
    boost = 0.0
    for rule in _active_rules(user):
        if any(str(k).lower() in blob for k in (rule.get("item_kws") or [])):
            boost += 0.06
    for ex in exclusion_rules():
        if any(str(k).lower() in low for k in (ex.get("user_kws") or [])) and any(
            str(k).lower() in blob for k in (ex.get("item_kws") or [])
        ):
            boost -= float(ex.get("penalty", 0.2))
    return boost


def _normalize_vector_scores(candidates: list[CandidateItem]) -> dict[str, float]:
    sims = [float(c.similarity) for c in candidates]
    lo, hi = min(sims), max(sims)
    span = hi - lo
    out: dict[str, float] = {}
    for c in candidates:
        out[c.item_id] = 0.5 if span < 1e-6 else (float(c.similarity) - lo) / span
    return out


def _blend_score(vec_norm: float, lexical: float, boost: float, *, lexical_primary: bool, strong_intent: bool) -> float:
    if lexical_primary:
        lw, vw = LEXICAL_WEIGHT_MOCK, VECTOR_WEIGHT_MOCK
    elif strong_intent:
        lw, vw = LEXICAL_WEIGHT_INTENT, VECTOR_WEIGHT_INTENT
    else:
        lw, vw = LEXICAL_WEIGHT_REAL, VECTOR_WEIGHT_REAL
    return vw * vec_norm + lw * lexical + boost


def _as_candidate(c: CandidateItem | dict[str, Any]) -> CandidateItem:
    return c if isinstance(c, CandidateItem) else CandidateItem(**c)


def inject_lexical_rescues(
    intent_query: str, candidates: list[CandidateItem], item_metadata: list[dict[str, Any]]
) -> list[CandidateItem]:
    if not intent_query.strip() or not _active_rules(intent_query):
        return candidates
    existing = {c.item_id for c in candidates}
    rescues: list[tuple[float, CandidateItem]] = []
    for meta in item_metadata:
        iid = meta.get("item_id")
        if not iid or iid in existing:
            continue
        lex = compute_lexical_score(intent_query, meta)
        if lex >= LEXICAL_RESCUE_MIN:
            rescues.append(
                (
                    lex,
                    CandidateItem(
                        item_id=iid,
                        similarity=lex,
                        prob_base=0.1,
                        title=str(meta.get("title") or ""),
                        media_url=str(meta.get("media_url") or meta.get("cover_url") or ""),
                        duration_sec=float(meta.get("duration_sec") or 60),
                    ),
                )
            )
    if not rescues:
        return candidates
    rescues.sort(key=lambda x: x[0], reverse=True)
    return candidates + [c for _, c in rescues[:LEXICAL_RESCUE_MAX]]


def apply_ranking(
    context: str,
    candidates: list[CandidateItem | dict[str, Any]],
    item_metadata: list[dict[str, Any]],
    *,
    lexical_primary: bool = False,
    intent: Any | None = None,
) -> list[CandidateItem]:
    """Rerank candidates by blended lexical + vector score."""
    if not candidates:
        return []
    candidates = [_as_candidate(c) for c in candidates]
    meta_by_id = {m.get("item_id"): m for m in item_metadata if m.get("item_id")}
    user = resolve_intent_query(context, intent)
    strong_intent = bool(_active_rules(user))
    candidates = inject_lexical_rescues(user, candidates, item_metadata)
    vec_norm_map = _normalize_vector_scores(candidates)

    scored: list[tuple[float, CandidateItem, dict[str, Any]]] = []
    for cand in candidates:
        meta = meta_by_id.get(cand.item_id, {"title": cand.title})
        lexical = compute_lexical_score(user, meta)
        boost = keyword_overlap_boost(user, meta)
        vec_norm = vec_norm_map[cand.item_id]
        adj = _blend_score(vec_norm, lexical, boost, lexical_primary=lexical_primary, strong_intent=strong_intent)
        scored.append((adj, cand, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    result: list[CandidateItem] = []
    for adj, cand, meta in scored:
        result.append(
            CandidateItem(
                item_id=cand.item_id,
                item_type=cand.item_type,
                similarity=round(adj, 4),
                prob_base=cand.prob_base,
                title=cand.title,
                media_url=cand.media_url or str(meta.get("media_url") or meta.get("cover_url") or ""),
                duration_sec=cand.duration_sec,
                item_ref=cand.item_ref or meta.get("item_ref"),
            )
        )
    return result
