"""Generic decision-policy configuration (loaded from config/policy.json)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "policy.json"
_CONFIG_CACHE: Optional["PolicyConfig"] = None

HANDOFF_LINK = os.environ.get("HANDOFF_LINK", "https://example.com/handoff")


def default_state_modulation() -> dict[str, float]:
    return {
        "start": -0.04,
        "explore": -0.02,
        "focus": 0.0,
        "close": 0.06,
        "INIT": -0.04,
        "GATHERING": -0.02,
        "DEEPENING": 0.0,
        "TAILORING": 0.06,
    }


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
class PolicyConfig:
    version: str = "2"
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
    urge_active_item_decay: float = 0.7
    cooldown_ms: int = 90000
    cooldown_ms_max: int = 90000
    state_modulation: dict[str, float] = field(default_factory=default_state_modulation)
    recent_push_window_ms: int = 300000
    recent_push_load_divisor: int = 3
    session_msg_cap: int = 30
    auto_push_min_session_msgs: int = 2
    auto_item_target_rate: Optional[float] = None
    auto_item_quota_min_similarity: float = 0.40
    auto_item_quota_cooldown_msgs: int = 3
    auto_handoff_target_rate: Optional[float] = None
    auto_handoff_effective_rate: Optional[float] = None
    auto_handoff_quota_cooldown_msgs: int = 3
    auto_handoff_quota_min_session_msgs: int = 2
    handoff_push_cooldown_ms: int = 300000
    handoff_session_push_cap: int = 3
    provider: str = "builtin:default"
    positive_dwell_ratio: float = 0.50
    short_dwell_ratio: float = 0.10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyConfig":
        cfg = cls(
            version=str(data.get("version", "2")),
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
            urge_active_item_decay=float(data.get("urge_active_item_decay", 0.7)),
            cooldown_ms=int(data.get("cooldown_ms", 90000)),
            cooldown_ms_max=int(data.get("cooldown_ms_max", 90000)),
            state_modulation=dict(data.get("state_modulation") or default_state_modulation()),
            provider=str(data.get("provider") or "builtin:default"),
            positive_dwell_ratio=float(data.get("positive_dwell_ratio", 0.50)),
            short_dwell_ratio=float(data.get("short_dwell_ratio", 0.10)),
            recent_push_window_ms=int(data.get("recent_push_window_ms", 300000)),
            recent_push_load_divisor=int(data.get("recent_push_load_divisor", 3)),
            session_msg_cap=int(data.get("session_msg_cap", 30)),
            auto_push_min_session_msgs=int(data.get("auto_push_min_session_msgs", 2)),
            auto_item_quota_min_similarity=float(
                data.get("auto_item_quota_min_similarity", 0.40)
            ),
            auto_item_quota_cooldown_msgs=int(data.get("auto_item_quota_cooldown_msgs", 3)),
            auto_handoff_quota_cooldown_msgs=int(
                data.get("auto_handoff_quota_cooldown_msgs", 3)
            ),
            auto_handoff_quota_min_session_msgs=int(
                data.get("auto_handoff_quota_min_session_msgs", 2)
            ),
            handoff_push_cooldown_ms=int(data.get("handoff_push_cooldown_ms", 300000)),
            handoff_session_push_cap=int(data.get("handoff_session_push_cap", 3)),
        )
        raw_th0 = data.get("a_threshold_0")
        cfg.a_threshold_0 = float(raw_th0) if raw_th0 is not None else None
        for name in ("auto_item_target_rate", "auto_handoff_target_rate", "auto_handoff_effective_rate"):
            raw = data.get(name)
            setattr(cfg, name, float(raw) if raw is not None else None)
        return validate(cfg)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["a_weights"] = asdict(self.a_weights)
        return d


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resolve_a_threshold_0(config: PolicyConfig) -> float:
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


def validate(config: PolicyConfig) -> PolicyConfig:
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
    custom = os.environ.get("POLICY_CONFIG_PATH", "")
    if custom and os.path.exists(custom):
        return Path(custom)
    return _DEFAULT_CONFIG_PATH


def load_policy_config(path: str | Path | None = None, *, reload: bool = False) -> PolicyConfig:
    global _CONFIG_CACHE
    if not reload and path is None and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = _get_config_path(path)
    if not config_path.exists():
        cfg = validate(PolicyConfig())
        if path is None:
            _CONFIG_CACHE = cfg
        return cfg

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = PolicyConfig.from_dict(raw)
    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def reload_policy_config() -> PolicyConfig:
    return load_policy_config(reload=True)
