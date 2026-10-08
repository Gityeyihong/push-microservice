import json
import os
import time
from typing import List, Optional
from dataclasses import dataclass, field

from .decision_config import load_decision_config


@dataclass
class PushEvent:
    data: dict
    type: str = "push"


@dataclass
class VideoPushEvent(PushEvent):
    type: str = "video_push"


@dataclass
class AdvisorPushEvent(PushEvent):
    type: str = "advisor_push"


class StateManager:
    """Simulates MMKV local storage for the mobile decision engine."""

    def __init__(self, state_file: Optional[str] = None):
        self._state_file = state_file or os.path.join(
            os.path.dirname(__file__), "..", "data", "mobile_state.json"
        )
        _cfg = load_decision_config()
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
            "last_advisor_push_ts": 0,
            "total_advisor_push_count": 0,
            "auto_video_push_count": 0,
            "last_auto_video_push_msg": 0,
            "auto_advisor_push_count": 0,
            "last_auto_advisor_push_msg": 0,
        }
        self._state = dict(self._defaults)

        if os.path.exists(self._state_file):
            self._load()
        else:
            self._save()

    def _load(self):
        with open(self._state_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for k, v in self._defaults.items():
            self._state[k] = saved.get(k, v)

    def _save(self):
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def get(self, key: str):
        return self._state.get(key, self._defaults.get(key))

    def set(self, key: str, value):
        self._state[key] = value
        self._save()

    def reset_session(self, *, isolated: bool = False):
        """重置会话计数与导师配额；isolated=True 时一并清空冷却时间戳（压测/新用户隔离）。"""
        self._state["session_msg_count"] = 0
        self._state["msg_ts_list"] = []
        self._state["last_advisor_push_ts"] = 0
        self._state["total_advisor_push_count"] = 0
        self._state["auto_video_push_count"] = 0
        self._state["last_auto_video_push_msg"] = 0
        self._state["auto_advisor_push_count"] = 0
        self._state["last_auto_advisor_push_msg"] = 0
        if isolated:
            self._state["last_push_ts"] = 0
            self._state["cool_down_remain"] = 0
            self._state["push_history"] = []
        self._save()

    def add_message(self, ts_seconds: float):
        self._state["msg_ts_list"].append(ts_seconds)
        if len(self._state["msg_ts_list"]) > 15:
            self._state["msg_ts_list"] = self._state["msg_ts_list"][-15:]
        self._state["session_msg_count"] += 1
        self._save()

    def add_push_history(self, video_id: str):
        self._state["push_history"].append(video_id)
        if len(self._state["push_history"]) > 5:
            self._state["push_history"] = self._state["push_history"][-5:]
        self._state["total_exposures"] += 1
        self._save()

    def record_feedback(self, is_positive: bool):
        if is_positive:
            self._state["positive_feedbacks"] += 1
        self._save()

    def record_click(self):
        self._state["total_clicks"] += 1
        self._save()

    def record_feedback_outcome(self, is_positive: bool, is_strong_negative: bool = False):
        if is_positive:
            self._state["positive_feedbacks"] += 1
            self._state["consecutive_negative"] = 0
        elif is_strong_negative:
            self._state["consecutive_negative"] += 1
        else:
            self._state["consecutive_negative"] += 1
        self._save()

    def get_click_rate(self) -> float:
        exposures = max(self._state.get("total_exposures", 0), 1)
        return self._state.get("total_clicks", 0) / exposures
