import hashlib
import hmac
import json
import logging
import os
from typing import Optional, List, Dict, Any

import redis.asyncio as aioredis

from .config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL, HMAC_SECRET_KEY


def _redis_password() -> str | None:
    password = os.getenv("REDIS_PASSWORD", "").strip()
    return password or None


logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self.redis = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=_redis_password(),
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await self.redis.ping()
        except Exception as exc:
            logger.warning("redis cache connect failed: %s", exc)
            self.redis = None

    async def disconnect(self):
        if self.redis:
            await self.redis.close()
            self.redis = None

    @staticmethod
    def generate_key(context: str) -> str:
        digest = hmac.new(
            HMAC_SECRET_KEY.encode("utf-8"),
            context.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"push:cache:{digest}"

    async def get(self, context: str) -> Optional[Dict[str, Any]]:
        if not self.redis:
            return None
        try:
            key = self.generate_key(context)
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as exc:
            logger.warning("redis cache get failed: %s", exc)
        return None

    async def set(self, context: str, data: Dict[str, Any]):
        if not self.redis:
            return
        try:
            key = self.generate_key(context)
            await self.redis.setex(key, REDIS_TTL, json.dumps(data, ensure_ascii=False))
        except Exception as exc:
            logger.warning("redis cache set failed: %s", exc)

    async def flush_all(self) -> bool:
        if not self.redis:
            return False
        removed = 0
        async for key in self.redis.scan_iter(match="push:cache:*", count=200):
            removed += int(await self.redis.delete(key))
        logger.info("flushed %d cache keys", removed)
        return True

    @property
    def available(self) -> bool:
        return self.redis is not None
