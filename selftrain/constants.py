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
    "schedule": {"enabled": False, "cron": "0 3 * * *", "min_new_samples": 50, "cooldown_hours": 24},
    "positive_threshold": 0.5,
    "min_positive": 2,
    "max_sim": 0.95,
    "epsilon": 0.02,
    "max_tune_failures": 3,
    "max_versions": 5,
    "source_ratio_cap": 0.7,
    "storage": {
        "backend": "sqlite",
        "dsn_env": "DATABASE_URL",
        "redis_url_env": "REDIS_URL",
        "fallback": "sqlite",
        "sqlite_path": "data/training/training.db",
    },
    "reward": {
        "adapter": "builtin:generic_funnel",
        "window_sec": 86400,
        "unknown_policy": "exclude",
        "events": {
            "click": {"role": "step", "weight": 0.2},
            "watch_end": {"role": "success", "weight": 1.0, "min_ratio": 0.5},
            "share": {"role": "success", "weight": 1.0},
            "favorite": {"role": "success", "weight": 1.0},
            "no_click": {"role": "failure"},
            "not_interested": {"role": "failure"},
        },
    },
    "scale": {
        "negatives_per_anchor": 8,
        "negative_strategy": "ann",
        "sample_cap": 200000,
        "page_size": 5000,
        "ann": {"index": "Flat", "nprobe": 16},
        "shards": 8,
        "max_steps": 2000,
    },
    "eval": {"mode": "push_success", "k": 3, "catalog_recall": True, "propensity": True, "freeze_size": 5000},
    "training": {
        "loss": "mnrl_weighted",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "max_seq_length": 256,
    },
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
        overrides["enabled"] = os.getenv("TRAINING_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    if os.getenv("TRAINING_SCHEDULE_ENABLED") is not None:
        overrides.setdefault("schedule", {})["enabled"] = os.getenv("TRAINING_SCHEDULE_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    if os.getenv("REVIEW_PROVIDER"):
        overrides.setdefault("review", {})["provider"] = os.getenv("REVIEW_PROVIDER").strip().lower()
    if os.getenv("TRAINING_BACKEND"):
        overrides.setdefault("storage", {})["backend"] = os.getenv("TRAINING_BACKEND").strip().lower()
    return overrides


def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = Path(path) if path else Path(os.getenv("TRAINING_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
    if cfg_path.exists():
        try:
            cfg = _deep_merge(cfg, json.loads(cfg_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    cfg = _deep_merge(cfg, _env_overrides())
    cfg.setdefault("review", {})
    cfg["review"].setdefault("provider", os.getenv("REVIEW_PROVIDER", "none"))
    return cfg


def training_dir() -> Path:
    return Path(os.getenv("TRAINING_DATA_DIR", str(PROJECT_ROOT / "data" / "training")))


def versions_dir() -> Path:
    return Path(os.getenv("MODEL_VERSIONS_DIR", str(PROJECT_ROOT / "models" / "versions")))
