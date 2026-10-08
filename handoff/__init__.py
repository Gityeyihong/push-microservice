"""Extensible human-handoff component.

Load via `handoff.load_handoff(provider=...)`, provider "builtin:default" or
"module.path:ClassName".
"""

from __future__ import annotations

import importlib
import os
from typing import Optional

from .base import HandoffPolicy
from .builtin import BuiltinHandoffPolicy

_REGISTRY = {"builtin": BuiltinHandoffPolicy}


def load_handoff(help_topic_vec=None, provider: Optional[str] = None) -> HandoffPolicy:
    provider = (provider or os.environ.get("HANDOFF_PROVIDER", "builtin:default")).strip()
    if ":" in provider:
        module_name, _, attr = provider.partition(":")
    else:
        module_name, attr = "builtin", provider
    if module_name == "builtin":
        return BuiltinHandoffPolicy(help_topic_vec=help_topic_vec)
    module = importlib.import_module(module_name)
    cls = getattr(module, attr)
    try:
        return cls(help_topic_vec=help_topic_vec)
    except TypeError:
        return cls()


__all__ = ["HandoffPolicy", "BuiltinHandoffPolicy", "load_handoff"]
