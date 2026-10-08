"""Optional Redis helpers: distributed lock, dedup set, hot counters."""

from __future__ import annotations

import os
from typing import Optional


class RedisHelper:
    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("REDIS_URL", "")
        self._client = None

    async def connect(self) -> bool:
        if not self.url:
            return False
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self.url, decode_responses=True)
            await self._client.ping()
            return True
        except Exception:
            self._client = None
            return False

    @property
    def available(self) -> bool:
        return self._client is not None

    async def acquire_lock(self, name: str, ttl_sec: int = 600) -> bool:
        if not self._client:
            return True  # no redis => best-effort
        return bool(await self._client.set(f"lock:{name}", "1", nx=True, ex=ttl_sec))

    async def release_lock(self, name: str) -> None:
        if self._client:
            await self._client.delete(f"lock:{name}")

    async def add_dedup(self, key: str) -> bool:
        """Return True if newly added (not seen before)."""
        if not self._client:
            return True
        return bool(await self._client.sadd("dedup:samples", key))

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
