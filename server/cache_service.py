import hashlib
import hmac
import json
from typing import Optional, List, Dict, Any

import redis.asyncio as aioredis

from .config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL, HMAC_SECRET_KEY


class CacheService:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self.redis = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await self.redis.ping()
        except Exception:
            self.redis = None

    async def disconnect(self):
        if self.redis:
            await self.redis.close()
            self.redis = None

    @staticmethod
    def generate_key(context: str) -> str:
        return hmac.new(
            HMAC_SECRET_KEY.encode("utf-8"),
            context.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def get(self, context: str) -> Optional[Dict[str, Any]]:
        if not self.redis:
            return None
        try:
            key = self.generate_key(context)
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    async def set(self, context: str, data: Dict[str, Any]):
        if not self.redis:
            return
        try:
            key = self.generate_key(context)
            await self.redis.setex(key, REDIS_TTL, json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    async def flush_all(self) -> bool:
        if not self.redis:
            return False
        await self.redis.flushdb()
        return True

    @property
    def available(self) -> bool:
        return self.redis is not None
