"""Per-session StateManager. Redis when reachable, otherwise a locked JSON file."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Optional

from policy.state import StateManager

from .config import REDIS_DB, REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

_SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sessions")
_redis_client = None
_redis_checked = False


def _session_ttl_sec() -> int:
    try:
        return int(os.getenv("PUSH_SESSION_TTL_SEC", "604800"))
    except ValueError:
        return 604800


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^\w\-]", "_", (session_id or "default").strip())[:64]
    return cleaned or "default"


def _shared_redis():
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        import redis

        password = os.getenv("REDIS_PASSWORD", "").strip() or None
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=password,
            socket_connect_timeout=0.4,
            socket_timeout=1,
            decode_responses=True,
        )
        client.ping()
        _redis_client = client
        logger.info("Session state using Redis %s:%s", REDIS_HOST, REDIS_PORT)
    except Exception as exc:
        logger.info("Session state using files (%s)", exc)
        _redis_client = None
    return _redis_client


class RedisStateBackend:
    def __init__(self, client, key: str):
        self.client = client
        self.key = key
        self._lock_key = f"{key}:lock"

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        token = uuid.uuid4().hex
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.client.set(self._lock_key, token, nx=True, ex=5):
                try:
                    yield
                finally:
                    current = self.client.get(self._lock_key)
                    if current == token:
                        self.client.delete(self._lock_key)
                return
            time.sleep(0.02)
        raise TimeoutError(f"session lock timeout: {self.key}")

    def load(self) -> Optional[dict]:
        raw = self.client.get(self.key)
        if not raw:
            return None
        return json.loads(raw)

    def save(self, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        ttl = _session_ttl_sec()
        if ttl > 0:
            self.client.set(self.key, payload, ex=ttl)
        else:
            self.client.set(self.key, payload)


@lru_cache(maxsize=256)
def get_session_state(session_id: str) -> StateManager:
    safe = _safe_session_id(session_id)
    client = _shared_redis()
    if client is not None:
        return StateManager(backend=RedisStateBackend(client, f"push:session:{safe}"))
    os.makedirs(_SESSION_DIR, exist_ok=True)
    path = os.path.join(_SESSION_DIR, f"{safe}.json")
    return StateManager(state_file=path)
