"""Filter and rerank push candidates against last user message intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import CandidateVideo

BREAKUP_USER_KWS = ("分手", "失恋", "前任", "离婚", "挽回", "被甩", "放弃了", "抛弃")
BREAKUP_VIDEO_KWS = ("失恋", "分手", "前任", "挽回", "放下前任", "自我重建")

INTEREST_COLD_USER_KWS = (
    "回复", "变慢", "冷淡", "兴趣", "不回", "慢", "变少", "热情", "聊天", "暧昧", "三个月", "凉了", "信号",
)
INTEREST_COLD_VIDEO_KWS = (
    "冷淡", "回复", "兴趣", "变慢", "热情", "暧昧", "聊天", "约会", "追求", "暗恋", "信号", "变淡",
)

# 末轮仅为「要课程/教程/推荐」等交付请求，不含新话题时应沿用前文情境
META_CONTENT_REQUEST_KWS = (
    "课程", "教程", "视频", "推荐", "教学", "学习", "提升", "看看", "有没有",
    "有什么", "想学", "教我", "干货", "资料", "课吗", "教程吗",
)

# Mock 向量无语义时 lexical 权重更高；真实 embedding 时保留向量主导
LEXICAL_WEIGHT_MOCK = 0.82
VECTOR_WEIGHT_MOCK = 0.18
LEXICAL_WEIGHT_REAL = 0.35
VECTOR_WEIGHT_REAL = 0.65
LEXICAL_WEIGHT_INTENT = 0.50
VECTOR_WEIGHT_INTENT = 0.50
LEXICAL_RESCUE_MIN = 0.30
LEXICAL_RESCUE_MAX = 3


@dataclass(frozen=True)
class IntentRule:
    name: str
    user_kws: tuple[str, ...]
    video_kws: tuple[str, ...]
    tag_boost: tuple[str, ...] = ()
    weight: float = 1.0


INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        "interest_cold",
        ("回复", "变慢", "冷淡", "兴趣", "不回", "慢", "变少", "凉了", "信号", "秒回", "礼貌"),
        ("冷淡", "回复", "兴趣", "变慢", "信号", "变淡", "热情", "聊天", "暧昧"),
        ("沟通技巧", "安全感", "第一次约会"),
        1.2,
    ),
    IntentRule(
        "cold_war",
        ("冷战", "不理", "僵", "僵持", "破冰", "冷暴力", "沉默"),
        ("冷战", "破冰", "吵架和好", "沟通"),
        ("冷战", "吵架和好", "沟通技巧"),
        1.1,
    ),
    IntentRule(
        "long_distance",
        ("异地", "异地恋", "见面", "距离"),
        ("异地", "见面", "维系", "新鲜感"),
        ("异地恋",),
        1.0,
    ),
    IntentRule(
        "first_date",
        ("约会", "见面", "第一次", "去哪", "穿搭", "紧张"),
        ("约会", "第一次", "见面", "穿搭", "冷场"),
        ("第一次约会", "表白"),
        1.0,
    ),
    IntentRule(
        "communication_fight",
        ("吵架", "争吵", "沟通", "翻旧账", "和好", "争论"),
        ("沟通", "吵架", "争吵", "和好", "翻旧账"),
        ("沟通技巧", "吵架和好"),
        1.0,
    ),
    IntentRule(
        "breakup_recover",
        ("分手", "失恋", "前任", "挽回", "重建"),
        ("分手", "失恋", "前任", "挽回", "重建", "放下"),
        ("分手挽回", "前任"),
        1.0,
    ),
    IntentRule(
        "jealousy",
        ("吃醋", "嫉妒", "异性", "吃醋了"),
        ("吃醋", "嫉妒", "异性"),
        ("吃醋",),
        0.9,
    ),
    IntentRule(
        "trust",
        ("信任", "隐瞒", "撒谎", "欺骗", "出轨"),
        ("信任", "隐瞒", "出轨", "欺骗"),
        ("信任建立",),
        1.0,
    ),
    IntentRule(
        "confession",
        ("表白", "告白", "喜欢你", "暗恋"),
        ("表白", "告白", "暗恋", "成功率"),
        ("表白",),
        1.0,
    ),
    IntentRule(
        "security",
        ("安全感", "粘人", "依赖", "患得患失"),
        ("安全感", "粘人", "依赖", "患得患失"),
        ("安全感",),
        0.9,
    ),
    IntentRule(
        "parents",
        ("父母", "家长", "反对", "催婚", "相亲"),
        ("父母", "家长", "反对", "催婚", "相亲"),
        ("父母反对",),
        1.0,
    ),
    IntentRule(
        "passion_fade",
        ("倦怠", "没感觉", "激情", "新鲜感", "平淡"),
        ("倦怠", "激情", "新鲜感", "平淡", "维持"),
        ("安全感", "沟通技巧"),
        0.9,
    ),
)


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
    return msgs


def _all_intent_user_kws() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for rule in INTENT_RULES:
        for k in rule.user_kws:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
    return tuple(ordered)


_INTENT_USER_KWS = _all_intent_user_kws()


def is_meta_content_request(text: str) -> bool:
    """True when the utterance only asks for content delivery, not a new relationship topic."""
    t = text.strip()
    if not t:
        return False
    if not any(k in t for k in META_CONTENT_REQUEST_KWS):
        return False
    substantive = [
        k for k in _INTENT_USER_KWS
        if k in t and k not in META_CONTENT_REQUEST_KWS
    ]
    if substantive:
        return False
    if len(t) <= 36:
        return True
    return sum(1 for k in META_CONTENT_REQUEST_KWS if k in t) >= 2


def build_intent_query(context: str) -> str:
    """Aggregate multi-turn user intent; ignore trailing meta-only course/tutorial asks."""
    msgs = extract_user_messages(context)
    if not msgs:
        return ""
    if len(msgs) == 1:
        return msgs[0]
    last = msgs[-1]
    if is_meta_content_request(last):
        prior = [m for m in msgs[:-1] if m.strip()]
        if prior:
            return " ".join(prior)
    return " ".join(msgs)


def _push_intent_confidence(push_intent: Any | None) -> float:
    if push_intent is None:
        return 0.0
    try:
        if hasattr(push_intent, "confidence"):
            return float(push_intent.confidence)
        if isinstance(push_intent, dict):
            return float(push_intent.get("confidence") or 0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def resolve_intent_query(context: str, push_intent: Any | None = None) -> str:
    """Prefer upstream push_intent.intent_query when confidence >= 0.5."""
    if push_intent is not None and _push_intent_confidence(push_intent) >= 0.5:
        iq = getattr(push_intent, "intent_query", None)
        if iq is None and isinstance(push_intent, dict):
            iq = push_intent.get("intent_query")
        if iq and str(iq).strip():
            return str(iq).strip()
    return build_intent_query(context) or extract_last_user_message(context) or ""


def resolve_is_meta_request(last_user: str, push_intent: Any | None = None) -> bool:
    if push_intent is not None and _push_intent_confidence(push_intent) >= 0.5:
        if hasattr(push_intent, "is_meta_request"):
            return bool(push_intent.is_meta_request)
        if isinstance(push_intent, dict):
            return bool(push_intent.get("is_meta_request", False))
    return is_meta_content_request(last_user)


def effective_breakup_intent(user: str, push_intent: Any | None = None) -> bool:
    """User has breakup intent unless upstream explicitly excludes breakup_recover."""
    if not user_has_breakup_intent(user):
        return False
    if push_intent is not None and _push_intent_confidence(push_intent) >= 0.5:
        excludes = getattr(push_intent, "exclude_topics", None)
        if excludes is None and isinstance(push_intent, dict):
            excludes = push_intent.get("exclude_topics") or []
        if "breakup_recover" in (excludes or []):
            return False
    return True


def _meta_blob(meta: dict[str, Any]) -> str:
    tags = meta.get("tags") or []
    return " ".join(
        [
            str(meta.get("title") or ""),
            str(meta.get("summary") or ""),
            str(meta.get("description") or ""),
            " ".join(str(t) for t in tags),
        ]
    )


def user_has_breakup_intent(text: str) -> bool:
    return any(k in text for k in BREAKUP_USER_KWS)


def video_is_breakup_heavy(meta: dict[str, Any]) -> bool:
    return any(k in _meta_blob(meta) for k in BREAKUP_VIDEO_KWS)


def _active_intent_rules(user: str) -> list[IntentRule]:
    return [r for r in INTENT_RULES if any(k in user for k in r.user_kws)]


def compute_lexical_score(user: str, meta: dict[str, Any]) -> float:
    """0~1 intent alignment from tags/title keywords (independent of vector sim)."""
    if not user.strip():
        return 0.0

    blob = _meta_blob(meta)
    tags = set(meta.get("tags") or [])
    active = _active_intent_rules(user)

    if not active:
        # Generic overlap fallback
        hits = sum(1 for k in INTEREST_COLD_USER_KWS if k in user and k in blob)
        return min(0.35, hits * 0.08)

    scores: list[float] = []
    for rule in active:
        rule_score = 0.0
        video_hits = sum(1 for k in rule.video_kws if k in blob)
        tag_hits = sum(1 for t in rule.tag_boost if t in tags)
        if video_hits:
            rule_score += min(0.75, 0.25 * video_hits)
        if tag_hits:
            rule_score += min(0.35, 0.12 * tag_hits)
        if rule_score > 0:
            scores.append(min(1.0, rule_score * rule.weight))

    if not scores:
        return 0.05

    base = max(scores)

    # Hard mismatches
    if not user_has_breakup_intent(user) and video_is_breakup_heavy(meta):
        base = max(0.0, base - 0.55)
    if any(k in user for k in INTEREST_COLD_USER_KWS) and video_is_breakup_heavy(meta):
        base = max(0.0, base - 0.25)
    if any(k in user for k in ("冷战", "破冰")) and "异地" in blob and "冷战" not in blob:
        base = max(0.0, base - 0.15)

    active_names = {r.name for r in active}
    if "interest_cold" in active_names and "communication_fight" not in active_names:
        fight_blob = any(k in blob for k in ("吵架", "争吵", "翻旧账"))
        cold_blob = any(k in blob for k in INTEREST_COLD_VIDEO_KWS)
        if fight_blob and not cold_blob:
            base = max(0.0, base - 0.4)

    return round(min(1.0, max(0.0, base)), 4)


def keyword_overlap_boost(user: str, meta: dict[str, Any]) -> float:
    """Legacy fine-tune bump (kept for backward-compatible tests)."""
    blob = _meta_blob(meta)
    boost = 0.0
    if any(k in user for k in INTEREST_COLD_USER_KWS) and any(
        k in blob for k in INTEREST_COLD_VIDEO_KWS
    ):
        boost += 0.12
    if not user_has_breakup_intent(user) and video_is_breakup_heavy(meta):
        boost -= 0.35
    return boost


def _normalize_vector_scores(candidates: list[CandidateVideo]) -> dict[str, float]:
    sims = [float(c.similarity) for c in candidates]
    lo, hi = min(sims), max(sims)
    span = hi - lo
    out: dict[str, float] = {}
    for c in candidates:
        if span < 1e-6:
            out[c.video_id] = 0.5
        else:
            out[c.video_id] = (float(c.similarity) - lo) / span
    return out


def _blend_score(
    vec_norm: float,
    lexical: float,
    boost: float,
    *,
    lexical_primary: bool,
    strong_intent: bool = False,
) -> float:
    if lexical_primary:
        lw, vw = LEXICAL_WEIGHT_MOCK, VECTOR_WEIGHT_MOCK
    elif strong_intent:
        lw, vw = LEXICAL_WEIGHT_INTENT, VECTOR_WEIGHT_INTENT
    else:
        lw, vw = LEXICAL_WEIGHT_REAL, VECTOR_WEIGHT_REAL
    return vw * vec_norm + lw * lexical + boost


def _as_candidate(c: CandidateVideo | dict[str, Any]) -> CandidateVideo:
    if isinstance(c, CandidateVideo):
        return c
    return CandidateVideo(**c)


def inject_lexical_rescues(
    intent_query: str,
    candidates: list[CandidateVideo],
    video_metadata: list[dict[str, Any]],
) -> list[CandidateVideo]:
    """Pull high-lexical catalog hits into the pool when vector search misses them."""
    if not intent_query.strip() or not _active_intent_rules(intent_query):
        return candidates

    existing = {c.video_id for c in candidates}
    rescues: list[tuple[float, CandidateVideo]] = []

    for meta in video_metadata:
        vid = meta.get("video_id")
        if not vid or vid in existing:
            continue
        lex = compute_lexical_score(intent_query, meta)
        if lex >= LEXICAL_RESCUE_MIN:
            rescues.append(
                (
                    lex,
                    CandidateVideo(
                        video_id=vid,
                        similarity=lex,
                        prob_base=0.1,
                        title=str(meta.get("title") or ""),
                        cover_url=str(meta.get("cover_url") or ""),
                        duration_sec=float(meta.get("duration_sec") or 60),
                    ),
                )
            )

    if not rescues:
        return candidates

    rescues.sort(key=lambda x: x[0], reverse=True)
    return candidates + [c for _, c in rescues[:LEXICAL_RESCUE_MAX]]


def apply_relevance_guard(
    context: str,
    candidates: list[CandidateVideo | dict[str, Any]],
    video_metadata: list[dict[str, Any]],
    *,
    lexical_primary: bool = False,
    push_intent: Any | None = None,
) -> list[CandidateVideo]:
    """Drop severe mismatches; rerank by lexical + vector blended score."""
    if not candidates:
        return []

    candidates = [_as_candidate(c) for c in candidates]

    meta_by_id = {m.get("video_id"): m for m in video_metadata if m.get("video_id")}
    user = resolve_intent_query(context, push_intent)
    strong_intent = bool(_active_intent_rules(user))
    candidates = inject_lexical_rescues(user, candidates, video_metadata)
    vec_norm_map = _normalize_vector_scores(candidates)

    scored: list[tuple[float, CandidateVideo, dict[str, Any], float]] = []
    for cand in candidates:
        meta = meta_by_id.get(cand.video_id, {"title": cand.title})
        lexical = compute_lexical_score(user, meta)
        boost = keyword_overlap_boost(user, meta)
        vec_norm = vec_norm_map[cand.video_id]
        adj = _blend_score(
            vec_norm, lexical, boost,
            lexical_primary=lexical_primary,
            strong_intent=strong_intent,
        )
        scored.append((adj, cand, meta, lexical))

    has_breakup = effective_breakup_intent(user, push_intent)
    if not has_breakup:
        filtered = [
            (adj, cand, meta, lex)
            for adj, cand, meta, lex in scored
            if not (video_is_breakup_heavy(meta) and lex < 0.2 and adj < max(s[0] for s in scored) - 0.08)
        ]
        if filtered:
            scored = filtered

    scored.sort(key=lambda x: x[0], reverse=True)

    result: list[CandidateVideo] = []
    for adj, cand, meta, _lex in scored:
        result.append(
            CandidateVideo(
                video_id=cand.video_id,
                similarity=round(adj, 4),
                prob_base=cand.prob_base,
                title=cand.title,
                cover_url=cand.cover_url or str(meta.get("cover_url") or ""),
                duration_sec=cand.duration_sec,
            )
        )

    return result
