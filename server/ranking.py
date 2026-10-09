"""Generic candidate ranking / reranking against last user intent.

Intent rules and exclusion rules are loaded from config/keywords.json.
"""

from __future__ import annotations

import importlib
import math
import os
from typing import Any

from .config import SERVICE_SOFTMAX_TEMP
from .models import CandidateItem
from policy.keywords import (
    exclusion_rules,
    intent_rules,
    meta_request_keywords,
    strip_user_prefix,
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
        text = strip_user_prefix(line)
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


def _softmax_probs(scores: list[float], temperature: float = SERVICE_SOFTMAX_TEMP) -> list[float]:
    if not scores:
        return []
    temp = temperature if temperature > 0 else 1.0
    peak = max(scores)
    exps = [math.exp((score - peak) / temp) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


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
                        similarity=0.0,
                        prob_base=0.0,
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


def _intent_value(intent: Any | None, name: str, default: Any = None) -> Any:
    if intent is None:
        return default
    if isinstance(intent, dict):
        return intent.get(name, default)
    return getattr(intent, name, default)


def _candidate_allowed(
    cand: CandidateItem,
    meta_by_id: dict[str, dict[str, Any]],
    wanted_type: str,
    excluded: list[str],
) -> bool:
    meta = meta_by_id.get(cand.item_id, {"title": cand.title})
    if wanted_type:
        actual = str(meta.get("item_type") or cand.item_type or "item").lower()
        if actual != wanted_type:
            return False
    return not any(_topic_hit(meta, topic) for topic in excluded)


def _topic_hit(meta: dict[str, Any], topic: str) -> bool:
    needle = (topic or "").strip().lower()
    if not needle:
        return False
    tags = {str(tag).lower() for tag in (meta.get("tags") or [])}
    return needle in tags or needle in _meta_blob(meta)


_PRIMARY_TOPIC_BOOST = 0.15


def apply_ranking(
    context: str,
    candidates: list[CandidateItem | dict[str, Any]],
    item_metadata: list[dict[str, Any]],
    *,
    lexical_primary: bool = False,
    intent: Any | None = None,
    item_type: str | None = None,
) -> list[CandidateItem]:
    """Rerank candidates by blended lexical + vector score."""
    if not candidates:
        return []
    candidates = [_as_candidate(c) for c in candidates]
    meta_by_id = {m.get("item_id"): m for m in item_metadata if m.get("item_id")}
    wanted_type = (item_type or "").strip().lower()
    if wanted_type:
        candidates = [
            cand
            for cand in candidates
            if str(cand.item_type or meta_by_id.get(cand.item_id, {}).get("item_type") or "item").lower()
            == wanted_type
        ]
    excluded = [str(topic) for topic in (_intent_value(intent, "exclude_topics") or []) if str(topic).strip()]
    if excluded:
        candidates = [
            cand
            for cand in candidates
            if not any(_topic_hit(meta_by_id.get(cand.item_id, {"title": cand.title}), topic) for topic in excluded)
        ]
    if not candidates:
        return []
    primary_topic = str(_intent_value(intent, "primary_topic") or "").strip()
    user = resolve_intent_query(context, intent)
    strong_intent = bool(_active_rules(user))
    candidates = inject_lexical_rescues(user, candidates, item_metadata)
    candidates = [
        cand
        for cand in candidates
        if _candidate_allowed(cand, meta_by_id, wanted_type, excluded)
    ]
    if not candidates:
        return []

    scored: list[tuple[float, CandidateItem, dict[str, Any]]] = []
    for cand in candidates:
        meta = meta_by_id.get(cand.item_id, {"title": cand.title})
        lexical = compute_lexical_score(user, meta)
        boost = keyword_overlap_boost(user, meta)
        vec_component = max(0.0, min(1.0, float(cand.similarity)))
        adj = _blend_score(vec_component, lexical, boost, lexical_primary=lexical_primary, strong_intent=strong_intent)
        if primary_topic and _topic_hit(meta, primary_topic):
            adj += _PRIMARY_TOPIC_BOOST
        scored.append((adj, cand, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    probs = _softmax_probs([adj for adj, _, _ in scored])
    result: list[CandidateItem] = []
    for (_adj, cand, meta), prob in zip(scored, probs):
        retrieval = max(-1.0, min(1.0, float(cand.similarity)))
        result.append(
            CandidateItem(
                item_id=cand.item_id,
                item_type=cand.item_type,
                similarity=round(retrieval, 4),
                prob_base=round(prob, 4),
                title=cand.title,
                media_url=cand.media_url or str(meta.get("media_url") or meta.get("cover_url") or ""),
                duration_sec=cand.duration_sec,
                item_ref=cand.item_ref or meta.get("item_ref"),
            )
        )
    return result


class BuiltinRanker:
    """Default ranker: lexical + vector blend in apply_ranking."""

    def rank(
        self,
        context: str,
        candidates: list[CandidateItem | dict[str, Any]],
        item_metadata: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[CandidateItem]:
        return apply_ranking(
            context,
            candidates,
            item_metadata,
            lexical_primary=bool(kwargs.get("lexical_primary", False)),
            intent=kwargs.get("intent"),
            item_type=kwargs.get("item_type"),
        )


def load_ranker(provider: str | None = None) -> BuiltinRanker:
    """Load builtin:default or module.path:ClassName (class implements rank())."""
    provider = (provider or os.environ.get("RANKER_PROVIDER", "builtin:default")).strip() or "builtin:default"
    if provider in ("builtin:default", "builtin", "default"):
        return BuiltinRanker()
    module_name, _, attr = provider.partition(":")
    if not attr or module_name == "builtin":
        return BuiltinRanker()
    module = importlib.import_module(module_name)
    cls = getattr(module, attr)
    return cls()
