"""A/B experiment and profile resolution for the policy config."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from policy.config import load_policy_config

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _config_dir() -> Path:
    custom = os.environ.get("PUSH_CONFIG_DIR", "").strip()
    return Path(custom) if custom else _CONFIG_DIR


def profiles_path() -> Path:
    return _config_dir() / "policy_profiles.json"


def ab_experiment_path() -> Path:
    return _config_dir() / "ab_experiment.json"


def policy_config_path() -> Path:
    custom = os.environ.get("POLICY_CONFIG_PATH", "").strip()
    if custom:
        return Path(custom)
    return _config_dir() / "policy.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_profiles() -> dict:
    data = _read_json(profiles_path(), {"profiles": {}})
    if not isinstance(data, dict):
        return {"profiles": {}}
    if "profiles" not in data or not isinstance(data["profiles"], dict):
        data["profiles"] = {}
    return data


def save_profiles(data: dict) -> None:
    _write_json(profiles_path(), data)


def load_ab_experiment() -> dict:
    data = _read_json(
        ab_experiment_path(),
        {"active": False, "name": "", "variant_a": "", "variant_b": "", "split_b_percent": 50},
    )
    return data if isinstance(data, dict) else {"active": False}


def save_ab_experiment(data: dict) -> None:
    _write_json(ab_experiment_path(), data)


def _session_variant(session_id: str, split_b_percent: int) -> str:
    digest = hashlib.sha256((session_id or "default").encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "b" if bucket < max(0, min(split_b_percent, 100)) else "a"


def _overlay_profile(base: dict, profiles: dict, profile_name: str | None) -> dict | None:
    if not profile_name:
        return None
    overlay = profiles.get(profile_name)
    if not isinstance(overlay, dict):
        return None
    return {**base, **overlay}


def resolve_config_for_session(session_id: str | None = None, experiment_id: str | None = None) -> dict:
    base = load_policy_config().to_dict()
    profiles = load_profiles().get("profiles", {})
    chosen = (experiment_id or "").strip()
    exp = load_ab_experiment()
    if chosen:
        if chosen in profiles:
            return _overlay_profile(base, profiles, chosen) or base
        if exp.get("active") and chosen == str(exp.get("name") or ""):
            pass
        elif exp.get("active") and chosen in (exp.get("variant_a"), exp.get("variant_b")):
            forced = _overlay_profile(base, profiles, chosen)
            if forced is None:
                raise ValueError(f"unknown experiment_id: {chosen}")
            return forced
        else:
            raise ValueError(f"unknown experiment_id: {chosen}")
    if not exp.get("active"):
        return base
    variant = _session_variant(session_id or "default", int(exp.get("split_b_percent", 50)))
    profile_name = exp.get("variant_b") if variant == "b" else exp.get("variant_a")
    return _overlay_profile(base, profiles, profile_name) or base


def get_ab_status() -> dict:
    exp = load_ab_experiment()
    profiles = load_profiles()
    return {
        "experiment": exp,
        "profile_names": sorted(profiles.get("profiles", {}).keys()),
    }
