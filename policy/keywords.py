"""Configurable keyword/intent provider (config/keywords.json)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = PROJECT_ROOT / "config" / "keywords.json"
_CACHE: dict[str, Any] | None = None


def load_keywords(path: str | Path | None = None, *, reload: bool = False) -> dict[str, Any]:
    global _CACHE
    if not reload and path is None and _CACHE is not None:
        return _CACHE
    p = Path(path) if path else Path(os.environ.get("KEYWORDS_CONFIG_PATH", str(_DEFAULT_PATH)))
    data: dict[str, Any] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    if path is None:
        _CACHE = data
    return data


def _section(name: str) -> dict[str, Any]:
    return dict(load_keywords().get(name) or {})


def meta_request_keywords() -> list[str]:
    return list(load_keywords().get("meta_request_keywords") or [])


def urgent_keywords() -> list[str]:
    return list(load_keywords().get("urgent_keywords") or [])


def sensitive_topics() -> list[str]:
    return list(load_keywords().get("sensitive_topics") or [])


def helpless_keywords() -> list[str]:
    return list(load_keywords().get("helpless_keywords") or [])


def intent_rules() -> list[dict[str, Any]]:
    return list(load_keywords().get("intent_rules") or [])


def exclusion_rules() -> list[dict[str, Any]]:
    return list(load_keywords().get("exclusion_rules") or [])


_DEFAULT_USER_PREFIXES = ["user:", "User:", "用户:", "用户："]


def user_prefixes() -> list[str]:
    ctx = load_keywords().get("context") or {}
    prefixes = ctx.get("user_prefixes")
    if isinstance(prefixes, list) and prefixes:
        return [str(p) for p in prefixes if str(p)]
    return list(_DEFAULT_USER_PREFIXES)


def force_handoff_signal() -> str:
    signals = load_keywords().get("signals") or {}
    name = str(signals.get("force_handoff") or "risk_alert").strip()
    return name or "risk_alert"


def signal_forces_handoff(signals: dict[str, Any] | None) -> bool:
    signals = signals or {}
    return bool(signals.get(force_handoff_signal()))


def strip_user_prefix(line: str) -> str | None:
    text = (line or "").strip()
    if not text:
        return None
    for prefix in user_prefixes():
        if text.startswith(prefix):
            return text[len(prefix):].strip() or None
        if prefix.isascii() and text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip() or None
    return None


def last_user_utterance(context: str) -> str:
    for line in reversed((context or "").splitlines()):
        stripped = strip_user_prefix(line)
        if stripped:
            return stripped
    return ""


def _has_any(text: str, words: list[str]) -> bool:
    return any(w and w in text for w in words)


def _has_question_phrase(msg: str) -> bool:
    return bool(re.search(r"(?:do you (?:have|offer)|is there|any)\b", msg, re.IGNORECASE))


def _is_garbled(msg: str) -> bool:
    if re.search(r"\*{2,}", msg):
        return True
    if "//" in msg or "\\\\" in msg:
        return True
    return False


def _normalize(text: str, normalizations: dict[str, str]) -> str:
    for src, dst in (normalizations or {}).items():
        text = text.replace(src, dst)
    return text


def is_item_request(text: str) -> bool:
    msg = (text or "").strip()
    if not msg:
        return False
    cfg = _section("item_request")
    short = [t.lower() for t in (cfg.get("short_triggers") or [])]
    lower = msg.lower().strip(" ?!.,")
    if any(t == lower for t in short):
        return True
    keywords = cfg.get("keywords") or []
    if any(k.lower() in lower for k in keywords):
        return True
    has_topic = any(str(t).lower() in lower for t in (cfg.get("topics") or []))
    has_verb = any(str(v).lower() in lower for v in (cfg.get("verbs") or []))
    return has_topic and has_verb


def is_handoff_request(text: str) -> bool:
    cfg = _section("handoff_request")
    msg = _normalize((text or "").strip(), cfg.get("normalizations") or {})
    if not msg or _is_garbled(msg):
        return False
    lower = msg.lower().strip(" ?!.")
    short = [t.lower() for t in (cfg.get("short_triggers") or [])]
    if any(t == lower for t in short):
        return True
    markers = cfg.get("service_markers") or []
    if any(m.lower() in lower for m in markers):
        return True
    keywords = cfg.get("keywords") or []
    if any(k.lower() in lower for k in keywords):
        return True
    has_topic = any(str(t).lower() in lower for t in (cfg.get("topics") or []))
    has_verb = any(str(v).lower() in lower for v in (cfg.get("verbs") or []))
    if has_topic and (has_verb or _has_question_phrase(lower)):
        return True
    return False
