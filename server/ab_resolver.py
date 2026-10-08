"""A/B experiment and profile resolution for push decision config."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from mobile.decision_config import load_decision_config

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _config_dir() -> Path:
    custom = os.environ.get("PUSH_CONFIG_DIR", "").strip()
    return Path(custom) if custom else _CONFIG_DIR


def profiles_path() -> Path:
    return _config_dir() / "decision_profiles.json"


def ab_experiment_path() -> Path:
    return _config_dir() / "ab_experiment.json"


def decision_config_path() -> Path:
    custom = os.environ.get("DECISION_CONFIG_PATH", "").strip()
    if custom:
        return Path(custom)
    return _config_dir() / "decision_config.json"


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
    if not isinstance(data, dict):
        return {"active": False}
    return data


def save_ab_experiment(data: dict) -> None:
    _write_json(ab_experiment_path(), data)


def _session_variant(session_id: str, split_b_percent: int) -> str:
    digest = hashlib.sha256((session_id or "default").encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "b" if bucket < max(0, min(split_b_percent, 100)) else "a"


def resolve_config_for_session(session_id: str | None = None) -> dict:
    """Base decision_config merged with A/B profile override when active."""
    base = load_decision_config().to_dict()
    exp = load_ab_experiment()
    if not exp.get("active"):
        return base

    profiles = load_profiles().get("profiles", {})
    variant = _session_variant(session_id or "default", int(exp.get("split_b_percent", 50)))
    profile_name = exp.get("variant_b") if variant == "b" else exp.get("variant_a")
    if not profile_name:
        return base
    overlay = profiles.get(profile_name)
    if not isinstance(overlay, dict):
        return base
    merged = {**base, **overlay}
    return merged


def get_ab_status() -> dict:
    exp = load_ab_experiment()
    profiles = load_profiles()
    return {
        "experiment": exp,
        "profile_names": sorted(profiles.get("profiles", {}).keys()),
    }
