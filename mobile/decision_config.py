"""Server-side decision parameters for mobile DecisionEngine."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "decision_config.json"
_CONFIG_CACHE: Optional["DecisionConfig"] = None

# 导师引导链接；由环境变量 ADVISOR_LINK 配置（默认占位）
DEFAULT_ADVISOR_LINK = os.environ.get("ADVISOR_LINK", "https://example.com/advisor")


@dataclass
class AWeights:
    I: float = 0.18
    L: float = 0.22
    F: float = 0.32
    S: float = 0.10
    U: float = 0.18

    @classmethod
    def from_dict(cls, data: dict | None) -> "AWeights":
        if not data:
            return cls()
        return cls(
            I=float(data.get("I", 0.18)),
            L=float(data.get("L", 0.22)),
            F=float(data.get("F", 0.32)),
            S=float(data.get("S", 0.10)),
            U=float(data.get("U", 0.18)),
        )


@dataclass
class DecisionConfig:
    version: str = "1"
    target_push_rate_per_10: float = 2.5
    min_push_rate_per_10: float = 0.6
    max_push_rate_per_10: float = 2.5
    tau_init: float = 0.55
    tau_min: float = 0.3
    tau_max: float = 2.0
    tau_positive_mult: float = 0.92
    tau_negative_mult: float = 1.08
    a_threshold_0: Optional[float] = None
    a_threshold_1: float = 0.75
    a_threshold_2: float = 0.88
    a_weights: AWeights = field(default_factory=AWeights)
    f_default_new_user: float = 0.6
    urge_initial: float = 0.5
    urge_delta: float = 0.07
    urge_active_video_decay: float = 0.7
    cooldown_ms: int = 90000
    cooldown_ms_max: int = 90000
    state_modulation: dict[str, float] = field(
        default_factory=lambda: {
            "INIT": -0.04,
            "GATHERING": -0.02,
            "DEEPENING": 0.0,
            "TAILORING": 0.06,
        }
    )
    recent_push_window_ms: int = 300000
    recent_push_load_divisor: int = 3
    session_msg_cap: int = 30
    auto_push_min_session_msgs: int = 2
    auto_video_target_rate: Optional[float] = None
    auto_video_quota_min_similarity: float = 0.40
    auto_video_quota_cooldown_msgs: int = 3
    auto_advisor_target_rate: Optional[float] = None
    auto_advisor_effective_rate: Optional[float] = None
    auto_advisor_quota_cooldown_msgs: int = 3
    auto_advisor_quota_min_session_msgs: int = 2
    advisor_push_cooldown_ms: int = 300000
    advisor_session_push_cap: int = 3
    mentor_a_boost: float = 0.08
    mentor_cooldown_ms: int = 55000
    mentor_urge_delta_bonus: float = 0.02

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionConfig":
        cfg = cls(
            version=str(data.get("version", "1")),
            target_push_rate_per_10=float(data.get("target_push_rate_per_10", 2.5)),
            min_push_rate_per_10=float(data.get("min_push_rate_per_10", 0.6)),
            max_push_rate_per_10=float(data.get("max_push_rate_per_10", 2.5)),
            tau_init=float(data.get("tau_init", 0.55)),
            tau_min=float(data.get("tau_min", 0.3)),
            tau_max=float(data.get("tau_max", 2.0)),
            tau_positive_mult=float(data.get("tau_positive_mult", 0.92)),
            tau_negative_mult=float(data.get("tau_negative_mult", 1.08)),
            a_threshold_1=float(data.get("a_threshold_1", 0.75)),
            a_threshold_2=float(data.get("a_threshold_2", 0.88)),
            a_weights=AWeights.from_dict(data.get("a_weights")),
            f_default_new_user=float(data.get("f_default_new_user", 0.6)),
            urge_initial=float(data.get("urge_initial", 0.5)),
            urge_delta=float(data.get("urge_delta", 0.07)),
            urge_active_video_decay=float(data.get("urge_active_video_decay", 0.7)),
            cooldown_ms=int(data.get("cooldown_ms", 90000)),
            cooldown_ms_max=int(data.get("cooldown_ms_max", 90000)),
            state_modulation=dict(data.get("state_modulation") or {}),
            recent_push_window_ms=int(data.get("recent_push_window_ms", 300000)),
            recent_push_load_divisor=int(data.get("recent_push_load_divisor", 3)),
            session_msg_cap=int(data.get("session_msg_cap", 30)),
            auto_push_min_session_msgs=int(data.get("auto_push_min_session_msgs", 2)),
            auto_video_quota_min_similarity=float(
                data.get("auto_video_quota_min_similarity", 0.40)
            ),
            auto_video_quota_cooldown_msgs=int(
                data.get("auto_video_quota_cooldown_msgs", 3)
            ),
            auto_advisor_quota_cooldown_msgs=int(
                data.get("auto_advisor_quota_cooldown_msgs", 3)
            ),
            auto_advisor_quota_min_session_msgs=int(
                data.get("auto_advisor_quota_min_session_msgs", 2)
            ),
            advisor_push_cooldown_ms=int(data.get("advisor_push_cooldown_ms", 300000)),
            advisor_session_push_cap=int(data.get("advisor_session_push_cap", 3)),
            mentor_a_boost=float(data.get("mentor_a_boost", 0.08)),
            mentor_cooldown_ms=int(data.get("mentor_cooldown_ms", 55000)),
            mentor_urge_delta_bonus=float(data.get("mentor_urge_delta_bonus", 0.02)),
        )
        raw_th0 = data.get("a_threshold_0")
        cfg.a_threshold_0 = float(raw_th0) if raw_th0 is not None else None
        raw_avtr = data.get("auto_video_target_rate")
        cfg.auto_video_target_rate = float(raw_avtr) if raw_avtr is not None else None
        raw_aatr = data.get("auto_advisor_target_rate")
        cfg.auto_advisor_target_rate = float(raw_aatr) if raw_aatr is not None else None
        raw_aaer = data.get("auto_advisor_effective_rate")
        cfg.auto_advisor_effective_rate = float(raw_aaer) if raw_aaer is not None else None
        return validate(cfg)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["a_weights"] = asdict(self.a_weights)
        return d


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resolve_a_threshold_0(config: DecisionConfig) -> float:
    if config.a_threshold_0 is not None:
        return config.a_threshold_0
    rate = _clamp(
        config.target_push_rate_per_10,
        config.min_push_rate_per_10,
        config.max_push_rate_per_10,
    )
    span = config.max_push_rate_per_10 - config.min_push_rate_per_10
    if span <= 0:
        return 0.60
    th_high = 0.73
    th_low = 0.60
    return th_high - (rate - config.min_push_rate_per_10) / span * (th_high - th_low)


def validate(config: DecisionConfig) -> DecisionConfig:
    config.target_push_rate_per_10 = _clamp(
        config.target_push_rate_per_10,
        config.min_push_rate_per_10,
        config.max_push_rate_per_10,
    )
    config.cooldown_ms = min(config.cooldown_ms, config.cooldown_ms_max)
    return config


def _get_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    custom = os.environ.get("DECISION_CONFIG_PATH", "")
    if custom and os.path.exists(custom):
        return Path(custom)
    return _DEFAULT_CONFIG_PATH


def load_decision_config(path: str | Path | None = None, *, reload: bool = False) -> DecisionConfig:
    global _CONFIG_CACHE
    if not reload and path is None and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = _get_config_path(path)
    if not config_path.exists():
        cfg = validate(DecisionConfig())
        if path is None:
            _CONFIG_CACHE = cfg
        return cfg

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = DecisionConfig.from_dict(raw)
    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def reload_decision_config() -> DecisionConfig:
    return load_decision_config(reload=True)
