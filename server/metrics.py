"""Request counters. Memory always; Redis INCR when the cache client is up."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

METRIC_NAMES = (
    "push",
    "cache_hit",
    "cache_miss",
    "handoff",
    "feedback_positive",
    "feedback_negative",
    "feedback_strong_negative",
    "feedback_other",
)

_lock = threading.Lock()
_counts = {name: 0 for name in METRIC_NAMES}


def _memory_inc(name: str) -> None:
    with _lock:
        _counts[name] = _counts.get(name, 0) + 1


def memory_snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counts)


async def hit(name: str, cache_service=None) -> None:
    _memory_inc(name)
    if cache_service is None or not cache_service.available:
        return
    try:
        await cache_service.redis.incr(f"push:metric:{name}")
    except Exception as exc:
        logger.warning("metric incr failed (%s): %s", name, exc)


async def snapshot(cache_service=None) -> dict[str, int]:
    local = memory_snapshot()
    if cache_service is None or not cache_service.available:
        return local
    out: dict[str, int] = {}
    for name in METRIC_NAMES:
        try:
            raw = await cache_service.redis.get(f"push:metric:{name}")
            out[name] = int(raw or 0)
        except Exception as exc:
            logger.warning("metric read failed (%s): %s", name, exc)
            out[name] = local.get(name, 0)
    return out
