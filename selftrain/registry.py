"""Versioned embedding model registry with active pointer, rollback and LRU pruning.

Layout:
    models/versions/
        active.json
        gen_0/v1/{model files, meta.json}
        gen_0/v2/...
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import versions_dir

logger = logging.getLogger(__name__)

ACTIVE_FILE = "active.json"


def _root() -> Path:
    root = versions_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def active_file() -> Path:
    return _root() / ACTIVE_FILE


def next_version_path(generation: int) -> Path:
    gen_dir = _root() / f"gen_{generation}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in gen_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    version_dir = gen_dir / f"v{len(existing) + 1}"
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def write_meta(version_path: Path, meta: dict) -> Path:
    path = Path(version_path) / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_meta(version_path: Path) -> dict:
    path = Path(version_path) / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def list_versions() -> list[dict]:
    root = _root()
    out = []
    for gen_dir in sorted(root.glob("gen_*")):
        if not gen_dir.is_dir():
            continue
        for ver_dir in sorted(gen_dir.glob("v*")):
            if not ver_dir.is_dir() or ver_dir.name.startswith("failed_"):
                continue
            out.append(
                {
                    "path": str(ver_dir),
                    "generation": gen_dir.name,
                    "version": ver_dir.name,
                    "meta": read_meta(ver_dir),
                }
            )
    for failed in sorted(root.glob("gen_*/failed_*")):
        out.append(
            {
                "path": str(failed),
                "generation": failed.parent.name,
                "version": failed.name,
                "failed": True,
                "meta": read_meta(failed),
            }
        )
    return out


def get_active() -> Path | None:
    data = _read_active()
    entry = data.get("active") or {}
    path = entry.get("path")
    if path and Path(path).exists():
        return Path(path)
    return None


def _read_active() -> dict:
    path = active_file()
    if not path.exists():
        return {"active": None, "history": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"active": None, "history": []}
    data.setdefault("history", [])
    return data


def _write_active(data: dict) -> None:
    active_file().write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def activate(version_path: Path, meta: dict | None = None) -> dict:
    version_path = Path(version_path)
    data = _read_active()
    current = data.get("active")
    if current:
        data["history"].insert(0, current)
        data["history"] = data["history"][:20]
    entry = {
        "path": str(version_path),
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or read_meta(version_path),
    }
    data["active"] = entry
    _write_active(data)
    return entry


def rollback() -> dict:
    data = _read_active()
    history = data.get("history") or []
    if not history:
        return {"ok": False, "reason": "no_history"}
    previous = history.pop(0)
    data["history"] = history
    data["active"] = previous
    _write_active(data)
    return {"ok": True, "active": previous}


def prune(max_versions: int) -> list[str]:
    """Keep newest `max_versions` successful versions by mtime; delete older ones."""
    versions = [
        Path(v["path"]) for v in list_versions() if not v.get("failed")
    ]
    if len(versions) <= max_versions:
        return []
    active = get_active()
    versions.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
    removed = []
    keep = versions[-max_versions:]
    if active is not None and active not in keep:
        keep.append(active)
    for path in versions:
        if path in keep:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(str(path))
    return removed
