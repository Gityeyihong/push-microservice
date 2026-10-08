"""Push admin audit log (JSONL)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _audit_path() -> Path:
    custom = os.environ.get("PUSH_ADMIN_AUDIT_PATH", "").strip()
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parent.parent / "data" / "admin_audit.jsonl"


def append_audit(action: str, *, actor: str = "admin", detail: dict | None = None) -> dict:
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "actor": actor,
        "detail": detail or {},
    }
    path = _audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_audit(*, limit: int = 50, offset: int = 0) -> dict:
    path = _audit_path()
    if not path.exists():
        return {"entries": [], "total": 0, "offset": offset, "limit": limit}
    lines = path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    slice_lines = lines[max(0, total - offset - limit) : max(0, total - offset)]
    slice_lines.reverse()
    entries = []
    for line in slice_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"entries": entries, "total": total, "offset": offset, "limit": limit}
