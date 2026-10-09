"""Per-minute limit for /api/v1. Redis when the cache is up, else this process."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_buckets: dict[str, tuple[int, int]] = {}


def limit_per_minute() -> int:
    try:
        return int(os.getenv("PUSH_RATE_LIMIT_PER_MIN", "120"))
    except ValueError:
        return 120


def max_body_bytes() -> int:
    try:
        return int(os.getenv("PUSH_MAX_BODY_BYTES", "262144"))
    except ValueError:
        return 262144


def trusted_proxies() -> set[str]:
    raw = os.getenv("PUSH_TRUSTED_PROXIES", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def client_address(forwarded_for: str | None, host: str | None) -> str:
    """Use X-Forwarded-For only when the immediate peer is a trusted proxy.

    ``PUSH_TRUSTED_PROXIES=*`` trusts the header from any peer. Leave the
    variable empty when clients connect directly.
    """
    peer = (host or "").strip()
    proxies = trusted_proxies()
    trust_header = "*" in proxies or (peer and peer in proxies)
    if trust_header and forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return peer or "unknown"


def client_key(forwarded_for: str | None, host: str | None) -> str:
    return hashlib.sha256(client_address(forwarded_for, host).encode("utf-8")).hexdigest()[:32]


def _production() -> bool:
    return os.getenv("PUSH_ENV", "development").strip().lower() == "production"


def _memory_allow(key: str, limit: int) -> bool:
    minute = int(time.time()) // 60
    with _lock:
        prev = _buckets.get(key)
        if not prev or prev[0] != minute:
            _buckets[key] = (minute, 1)
            return True
        count = prev[1] + 1
        _buckets[key] = (minute, count)
        return count <= limit


async def allow(key: str, cache_service=None) -> bool:
    limit = limit_per_minute()
    if limit <= 0:
        return True
    if cache_service is not None and cache_service.available:
        minute = int(time.time()) // 60
        redis_key = f"push:rl:{key}:{minute}"
        try:
            count = int(await cache_service.redis.incr(redis_key))
            if count == 1:
                await cache_service.redis.expire(redis_key, 120)
            return count <= limit
        except Exception as exc:
            logger.warning("rate limit redis failed: %s", exc)
            if _production():
                return False
    if _production():
        return False
    return _memory_allow(key, limit)
