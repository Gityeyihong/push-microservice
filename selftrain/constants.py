"""Self-training configuration: defaults merged with config/training_config.json and env."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "training_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "schedule": {
        "enabled": False,
        "cron": "0 3 * * *",
        "min_new_samples": 50,
        "cooldown_hours": 24,
    },
    "positive_threshold": 0.5,
    "min_positive": 2,
    "max_sim": 0.95,
    "epsilon": 0.02,
    "max_tune_failures": 3,
    "max_versions": 5,
    "source_ratio_cap": 0.7,
    "review": {"provider": "none", "timeout_sec": 30},
    "training": {
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "max_seq_length": 256,
    },
    "eval": {"k": 5, "freeze_size": 100},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides() -> dict:
    overrides: dict[str, Any] = {}
    if os.getenv("TRAINING_ENABLED") is not None:
        overrides["enabled"] = os.getenv("TRAINING_ENABLED", "0").strip().lower() in (
            "1", "true", "yes",
        )
    if os.getenv("TRAINING_SCHEDULE_ENABLED") is not None:
        overrides.setdefault("schedule", {})["enabled"] = os.getenv(
            "TRAINING_SCHEDULE_ENABLED", "0"
        ).strip().lower() in ("1", "true", "yes")
    provider = os.getenv("REVIEW_PROVIDER")
    if provider:
        overrides.setdefault("review", {})["provider"] = provider.strip().lower()
    return overrides


def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = Path(path) if path else Path(
        os.getenv("TRAINING_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    )
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = _deep_merge(cfg, json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    cfg = _deep_merge(cfg, _env_overrides())
    return cfg


def training_dir() -> Path:
    return Path(
        os.getenv("TRAINING_DATA_DIR", str(PROJECT_ROOT / "data" / "training"))
    )


def versions_dir() -> Path:
    return Path(os.getenv("MODEL_VERSIONS_DIR", str(PROJECT_ROOT / "models" / "versions")))
