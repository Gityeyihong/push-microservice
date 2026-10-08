"""Build contrastive training candidates from collected push/engagement records.

Sample shape:
    {
      "anchor": <user intent text>,
      "positive": <engaged video text>,
      "negatives": [<rejected video text>, ...],
      "source": "positive|negative|import",
      "behavior": "positive|negative",
      "session_id": "...",
      "video_id": "v_001",
      "review": null
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT, training_dir

logger = logging.getLogger(__name__)

POSITIVE_FEEDBACK = {"positive"}
NEGATIVE_FEEDBACK = {"negative", "strong_negative"}


def video_text(video: dict) -> str:
    return (
        f"标题：{video.get('title', '')}。 "
        f"文案：{video.get('description', '')}。 "
        f"简述：{video.get('summary', '')}。 "
        f"标签：{' '.join(video.get('tags') or [])}"
    )


def _last_user_message(context: str) -> str:
    for line in reversed((context or "").strip().splitlines()):
        stripped = line.strip()
        for prefix in ("用户:", "用户：", "user:", "User:"):
            if stripped.startswith(prefix):
                return stripped[len(prefix):].strip()
    return ""


def _catalog_path(catalog_file: str | None = None) -> Path:
    name = catalog_file or os.getenv("PUSH_VIDEOS", "sample_videos.json")
    return PROJECT_ROOT / "data" / name


def load_catalog(path: str | Path | None = None) -> dict[str, dict]:
    p = Path(path) if path else _catalog_path()
    if not p.exists():
        logger.warning("catalog not found: %s", p)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("catalog read failed: %s", exc)
        return {}
    return {str(v.get("video_id")): v for v in data if v.get("video_id")}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _imports(base_dir: Path) -> list[dict]:
    rows = _read_jsonl(base_dir / "imports.jsonl")
    out = []
    for row in rows:
        anchor = str(row.get("anchor", "")).strip()
        positive = str(row.get("positive", "")).strip()
        if not anchor or not positive:
            continue
        out.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negatives": [str(n) for n in row.get("negatives", []) if str(n).strip()],
                "source": "import",
                "behavior": "positive",
                "session_id": row.get("session_id"),
                "video_id": row.get("video_id"),
                "review": {"status": "approved", "reason": "manual_import", "by": "manual"},
            }
        )
    return out


def build_candidates(
    catalog: dict[str, dict] | None = None,
    *,
    base_dir: Path | None = None,
) -> list[dict]:
    base = base_dir or training_dir()
    catalog = catalog if catalog is not None else load_catalog()

    pushes = _read_jsonl(base / "pushes.jsonl")
    engagements = _read_jsonl(base / "engagement.jsonl")

    by_session: dict[str, list[dict]] = {}
    for eng in engagements:
        by_session.setdefault(str(eng.get("session_id") or "default"), []).append(eng)

    samples: list[dict] = []
    for push in pushes:
        session_id = str(push.get("session_id") or "default")
        context = str(push.get("context", ""))
        anchor = str(push.get("intent_query") or "").strip() or _last_user_message(context)
        if not anchor:
            continue
        pushed_ids = [str(v) for v in push.get("video_ids", [])]

        positive_ids: list[str] = []
        negative_ids: list[str] = []
        for eng in by_session.get(session_id, []):
            vid = str(eng.get("video_id"))
            if vid not in pushed_ids:
                continue
            cls = eng.get("classified")
            if cls in POSITIVE_FEEDBACK:
                positive_ids.append(vid)
            elif cls in NEGATIVE_FEEDBACK:
                negative_ids.append(vid)

        negatives_text = [video_text(catalog[v]) for v in negative_ids if v in catalog]

        for vid in positive_ids:
            if vid not in catalog:
                continue
            samples.append(
                {
                    "anchor": anchor,
                    "positive": video_text(catalog[vid]),
                    "negatives": negatives_text,
                    "source": "positive",
                    "behavior": "positive",
                    "session_id": session_id,
                    "video_id": vid,
                    "review": None,
                }
            )

        for vid in negative_ids:
            if vid not in catalog:
                continue
            samples.append(
                {
                    "anchor": anchor,
                    "positive": video_text(catalog[vid]),
                    "negatives": [],
                    "source": "negative",
                    "behavior": "negative",
                    "session_id": session_id,
                    "video_id": vid,
                    "review": None,
                }
            )

    samples.extend(_imports(base))
    return samples


def save_pairs(samples: list[dict], path: Path | None = None) -> Path:
    out = path or (training_dir() / "pairs.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return out
