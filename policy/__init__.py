"""Extensible decision-policy package.

Load a policy via `policy.load_policy(provider=...)` where provider is
"builtin:default" (default) or "module.path:ClassName".
"""

from __future__ import annotations

import importlib
import os
from typing import Optional

from .base import DecisionPolicy
from .builtin import BuiltinThresholdPolicy
from .config import PolicyConfig, load_policy_config, reload_policy_config
from .events import HandoffPushEvent, ItemPushEvent, PushEvent
from .engagement import classify_feedback
from .state import StateManager

_REGISTRY = {"builtin": BuiltinThresholdPolicy}


def _resolve_provider(provider: Optional[str], config: Optional[PolicyConfig]) -> str:
    if provider and provider.strip():
        return provider.strip()
    env = os.environ.get("POLICY_PROVIDER", "").strip()
    if env:
        return env
    cfg = config or load_policy_config()
    configured = (getattr(cfg, "provider", "") or "").strip()
    return configured or "builtin:default"


def load_policy(
    state: Optional[StateManager] = None,
    config: Optional[PolicyConfig] = None,
    provider: Optional[str] = None,
) -> DecisionPolicy:
    provider = _resolve_provider(provider, config)
    if ":" in provider:
        module_name, _, attr = provider.partition(":")
    else:
        module_name, attr = "builtin", provider
    if module_name == "builtin":
        return BuiltinThresholdPolicy(state=state, config=config or load_policy_config())
    module = importlib.import_module(module_name)
    cls = getattr(module, attr)
    return cls(state=state, config=config or load_policy_config())


__all__ = [
    "DecisionPolicy",
    "BuiltinThresholdPolicy",
    "PolicyConfig",
    "StateManager",
    "PushEvent",
    "ItemPushEvent",
    "HandoffPushEvent",
    "classify_feedback",
    "load_policy",
    "load_policy_config",
    "reload_policy_config",
]
