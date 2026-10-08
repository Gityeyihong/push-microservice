"""Per-session StateManager for server-side push decisions."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from mobile.state_manager import StateManager

_SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sessions")


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^\w\-]", "_", (session_id or "default").strip())[:64]
    return cleaned or "default"


@lru_cache(maxsize=256)
def get_session_state(session_id: str) -> StateManager:
    os.makedirs(_SESSION_DIR, exist_ok=True)
    safe = _safe_session_id(session_id)
    path = os.path.join(_SESSION_DIR, f"{safe}.json")
    return StateManager(state_file=path)
