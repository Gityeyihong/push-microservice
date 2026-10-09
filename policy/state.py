"""Per-session state for the decision policy.

Every read-modify-write reloads under an exclusive lock so multiple processes
cannot overwrite each other's updates.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator, Optional


class FileStateBackend:
    """JSON file plus a cross-platform exclusive lock."""

    def __init__(self, path: str):
        self.path = path
        self._lock_path = path + ".lock"

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        lock_dir = os.path.dirname(self._lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        handle = open(self._lock_path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if sys.platform == "win32":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def load(self) -> Optional[dict]:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, data: dict) -> None:
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


class StateManager:
    """Session state. Pass a backend (file or Redis); default is a JSON file."""

    def __init__(self, state_file: Optional[str] = None, backend: Any = None):
        from .config import load_policy_config

        if backend is None:
            path = state_file or os.path.join(
                os.path.dirname(__file__), "..", "data", "policy_state.json"
            )
            backend = FileStateBackend(path)
        self._backend = backend
        _cfg = load_policy_config()
        self._defaults = {
            "urge": _cfg.urge_initial,
            "tau": _cfg.tau_init,
            "last_push_ts": 0,
            "push_history": [],
            "total_exposures": 0,
            "positive_feedbacks": 0,
            "total_clicks": 0,
            "consecutive_negative": 0,
            "msg_ts_list": [],
            "cool_down_remain": 0,
            "session_msg_count": 0,
            "last_handoff_push_ts": 0,
            "total_handoff_push_count": 0,
            "auto_item_push_count": 0,
            "last_auto_item_push_msg": 0,
            "auto_handoff_push_count": 0,
            "last_auto_handoff_push_msg": 0,
            "push_ts_list": [],
            "clicked_ids": [],
        }
        self._state = dict(self._defaults)
        with self._backend.exclusive():
            saved = self._backend.load()
            if saved:
                self._reload_from(saved)
            else:
                self._backend.save(self._state)

    def _reload_from(self, saved: Optional[dict]) -> None:
        self._state = dict(self._defaults)
        if not saved:
            return
        for key, value in self._defaults.items():
            self._state[key] = saved.get(key, value)

    def _reload(self) -> None:
        self._reload_from(self._backend.load())

    def _mutate(self, apply):
        with self._backend.exclusive():
            self._reload()
            result = apply()
            self._backend.save(self._state)
            return result

    def get(self, key: str):
        with self._backend.exclusive():
            self._reload()
            return self._state.get(key, self._defaults.get(key))

    def set(self, key: str, value):
        def apply():
            self._state[key] = value

        self._mutate(apply)

    def reset_session(self, *, isolated: bool = False):
        def apply():
            self._state["session_msg_count"] = 0
            self._state["msg_ts_list"] = []
            self._state["last_handoff_push_ts"] = 0
            self._state["total_handoff_push_count"] = 0
            self._state["auto_item_push_count"] = 0
            self._state["last_auto_item_push_msg"] = 0
            self._state["auto_handoff_push_count"] = 0
            self._state["last_auto_handoff_push_msg"] = 0
            if isolated:
                self._state["last_push_ts"] = 0
                self._state["cool_down_remain"] = 0
                self._state["push_history"] = []
                self._state["push_ts_list"] = []
                self._state["clicked_ids"] = []

        self._mutate(apply)

    def add_message(self, ts_seconds: float):
        def apply():
            self._state["msg_ts_list"].append(ts_seconds)
            if len(self._state["msg_ts_list"]) > 15:
                self._state["msg_ts_list"] = self._state["msg_ts_list"][-15:]
            self._state["session_msg_count"] += 1

        self._mutate(apply)

    def add_push_history(self, item_id: str):
        def apply():
            self._state["push_history"].append(item_id)
            if len(self._state["push_history"]) > 5:
                self._state["push_history"] = self._state["push_history"][-5:]
            self._state["total_exposures"] += 1

        self._mutate(apply)

    def record_feedback(self, is_positive: bool):
        def apply():
            if is_positive:
                self._state["positive_feedbacks"] += 1

        self._mutate(apply)

    def record_click(self):
        def apply():
            self._state["total_clicks"] += 1

        self._mutate(apply)

    def bump_urge(self, delta: float, cap: float = 1.0) -> float:
        def apply():
            value = min(float(self._state.get("urge") or 0) + float(delta), cap)
            self._state["urge"] = value
            return value

        return self._mutate(apply)

    def scale_urge(self, factor: float) -> float:
        def apply():
            value = float(self._state.get("urge") or 0) * float(factor)
            self._state["urge"] = value
            return value

        return self._mutate(apply)

    def consume_cooldown(self) -> bool:
        """Decrement a remaining cooldown. True means this turn should not push."""

        def apply():
            remain = int(self._state.get("cool_down_remain") or 0)
            if remain <= 0:
                return False
            self._state["cool_down_remain"] = remain - 1
            return True

        return self._mutate(apply)

    def count_recent_pushes(self, now_ms: int, window_ms: int) -> int:
        def apply():
            stamps = [
                int(ts)
                for ts in (self._state.get("push_ts_list") or [])
                if now_ms - int(ts) < window_ms
            ]
            self._state["push_ts_list"] = stamps
            return len(stamps)

        return self._mutate(apply)

    def note_push(self, now_ms: int, window_ms: int) -> None:
        def apply():
            stamps = [
                int(ts)
                for ts in (self._state.get("push_ts_list") or [])
                if now_ms - int(ts) < window_ms
            ]
            stamps.append(int(now_ms))
            self._state["push_ts_list"] = stamps[-30:]
            self._state["last_push_ts"] = int(now_ms)

        self._mutate(apply)

    def mark_clicked(self, item_id: str) -> None:
        def apply():
            clicked = [str(item) for item in (self._state.get("clicked_ids") or [])]
            if item_id not in clicked:
                clicked.append(item_id)
            self._state["clicked_ids"] = clicked[-50:]
            self._state["total_clicks"] += 1

        self._mutate(apply)

    def has_click(self, item_id: str) -> bool:
        clicked = self.get("clicked_ids") or []
        return str(item_id) in {str(item) for item in clicked}

    def forget_click(self, item_id: str) -> None:
        def apply():
            self._state["clicked_ids"] = [
                str(item) for item in (self._state.get("clicked_ids") or []) if str(item) != str(item_id)
            ]

        self._mutate(apply)

    def record_feedback_outcome(self, is_positive: bool, is_strong_negative: bool = False):
        def apply():
            if is_positive:
                self._state["positive_feedbacks"] += 1
                self._state["consecutive_negative"] = 0
            else:
                self._state["consecutive_negative"] += 1

        self._mutate(apply)

    def get_click_rate(self) -> float:
        with self._backend.exclusive():
            self._reload()
            exposures = max(self._state.get("total_exposures", 0), 1)
            return self._state.get("total_clicks", 0) / exposures
