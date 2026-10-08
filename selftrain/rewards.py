"""Pluggable reward/success abstraction for self-training.

The concept of "success" is configuration-defined and extensible: provide a
custom `RewardAdapter` via `reward.adapter = "module.path:ClassName"`.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RewardOutcome:
    reward: float = 0.0
    label: str = "unknown"  # success | failure | unknown
    strength: float = 0.0
    propensity: float = 1.0


class RewardAdapter(ABC):
    @abstractmethod
    def outcome(self, *, exposure: dict, events: list[dict]) -> RewardOutcome:
        ...


class GenericFunnelAdapter(RewardAdapter):
    """Default adapter: event-role funnel defined entirely in config.

    config["reward"]["events"][event_name] = {"role": success|step|failure,
                                              "weight": float, "min_ratio": float}
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        reward_cfg = cfg.get("reward") or {}
        self.event_roles: dict[str, dict] = dict(reward_cfg.get("events") or {})
        self.window_sec = float(reward_cfg.get("window_sec", 86400) or 86400)
        self.unknown_policy = str(reward_cfg.get("unknown_policy", "exclude"))

    def _propensity(self, rank: int) -> float:
        return 1.0 / (1.0 + max(rank, 0))

    def outcome(self, *, exposure: dict, events: list[dict]) -> RewardOutcome:
        rank = int(exposure.get("rank", 0) or 0)
        item_id = exposure.get("item_id")
        relevant = [e for e in events if str(e.get("item_id")) == str(item_id)]

        reward = 0.0
        has_success = False
        has_failure = False
        for ev in relevant:
            role_cfg = self.event_roles.get(str(ev.get("event"))) or {}
            role = role_cfg.get("role")
            weight = float(role_cfg.get("weight", 1.0))
            if role == "success":
                has_success = True
                ratio = float(ev.get("watch_ratio", 1.0) or 1.0)
                min_ratio = float(role_cfg.get("min_ratio", 0.0) or 0.0)
                strength = weight * (ratio if min_ratio > 0 else 1.0)
                reward = max(reward, max(0.0, min(1.0, strength)))
            elif role == "failure":
                has_failure = True

        prop = self._propensity(rank)
        if has_success:
            return RewardOutcome(reward=reward or 1.0, label="success", strength=reward or 1.0, propensity=prop)
        if has_failure:
            return RewardOutcome(reward=0.0, label="failure", strength=0.0, propensity=prop)
        return RewardOutcome(reward=0.0, label="unknown", strength=0.0, propensity=prop)


def load_reward_adapter(config: dict[str, Any] | None = None, provider: Optional[str] = None) -> RewardAdapter:
    cfg = config or {}
    provider = provider or (cfg.get("reward") or {}).get("adapter") or "builtin:generic_funnel"
    if ":" in provider:
        module_name, _, attr = provider.partition(":")
    else:
        module_name, attr = "builtin", provider
    if module_name == "builtin":
        return GenericFunnelAdapter(cfg)
    module = importlib.import_module(module_name)
    cls = getattr(module, attr)
    try:
        return cls(cfg)
    except TypeError:
        return cls()
