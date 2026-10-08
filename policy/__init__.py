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


def load_policy(
    state: Optional[StateManager] = None,
    config: Optional[PolicyConfig] = None,
    provider: Optional[str] = None,
) -> DecisionPolicy:
    provider = (provider or os.environ.get("POLICY_PROVIDER", "builtin:default")).strip()
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
