#!/usr/bin/env python3
"""Import a generic item catalog from CSV/TSV/JSON into data/catalog.json.

No vendor SDK required. Expected fields (case-insensitive, common aliases):
    item_id, title, description, summary, tags, media_url, duration_sec, item_type

Usage:
    python scripts/import_catalog.py --input items.csv --output data/catalog.json
    python scripts/import_catalog.py --input items.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_ALIASES = {
    "item_id": ["item_id", "id", "video_id", "sku"],
    "title": ["title", "name", "heading"],
    "description": ["description", "desc", "body"],
    "summary": ["summary", "short", "excerpt"],
    "tags": ["tags", "labels", "topics"],
    "media_url": ["media_url", "cover_url", "image", "thumbnail", "url"],
    "duration_sec": ["duration_sec", "duration", "length"],
    "item_type": ["item_type", "type", "kind"],
}


def _pick(row: dict, field: str) -> str:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for alias in _ALIASES[field]:
        if alias in lowered and lowered[alias] not in (None, ""):
            return str(lowered[alias]).strip()
    return ""


def _parse_tags(raw: str) -> list[str]:
    if not raw:
        return []
    return [p for p in (x.strip() for x in re.split(r"[,，;；|\t]+", raw)) if p]


def _parse_duration(raw: str) -> float:
    try:
        return round(float(raw or 0), 3)
    except ValueError:
        return 0.0


def _normalize(rows: list[dict]) -> list[dict]:
    out = []
    for idx, row in enumerate(rows, start=1):
        iid = _pick(row, "item_id") or f"i_{idx:03d}"
        title = _pick(row, "title")
        if not title:
            continue
        out.append(
            {
                "item_id": iid,
                "item_type": _pick(row, "item_type") or "item",
                "title": title,
                "description": _pick(row, "description"),
                "summary": _pick(row, "summary") or title,
                "tags": _parse_tags(_pick(row, "tags")),
                "media_url": _pick(row, "media_url"),
                "duration_sec": _parse_duration(_pick(row, "duration_sec")),
            }
        )
    return out


def _read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("items", [])
    delimiter = "\t" if text.count("\t") > text.count(",") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import item catalog from CSV/TSV/JSON")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "catalog.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    items = _normalize(rows)
    print(f"parsed {len(items)} items from {args.input}")
    if args.dry_run:
        for item in items[:5]:
            print(" ", item["item_id"], "|", item["title"][:50])
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print("next: python -m data.generate_data  (rebuild vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
