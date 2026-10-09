"""External review gate for training positives.

Self-review is intentionally NOT supported: a training positive is admitted only
when an external reviewer returns `approved` (Plasmid doc §15.2). Providers:
    none                -> always `unavailable` (positives are rejected)
    http                -> generic JSON reviewer API
    openai-compatible   -> chat completions reviewer
A manual review store takes precedence (human-in-the-loop import).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import training_dir

logger = logging.getLogger(__name__)

STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_UNAVAILABLE = "unavailable"


@dataclass
class ReviewDecision:
    status: str
    reason: str = ""
    by: str = ""


def sample_key(sample: dict) -> str:
    raw = f"{sample.get('anchor', '')}\u0001{sample.get('positive', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ManualReviewStore:
    def __init__(self, path: Path | None = None):
        self.path = path or (training_dir() / "manual_reviews.jsonl")

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        out: dict[str, dict] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if key:
                out[key] = row
        return out

    def lookup(self, sample: dict) -> ReviewDecision | None:
        row = self._load().get(sample_key(sample))
        if not row:
            return None
        approved = bool(row.get("approved"))
        return ReviewDecision(
            status=STATUS_APPROVED if approved else STATUS_REJECTED,
            reason=str(row.get("reason", "")),
            by="manual",
        )

    def import_review(self, sample: dict, approved: bool, reason: str = "") -> dict:
        row = {
            "key": sample_key(sample),
            "anchor": sample.get("anchor", ""),
            "positive": sample.get("positive", ""),
            "approved": bool(approved),
            "reason": reason,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


class ExternalReviewer:
    def __init__(self, config: dict[str, Any] | None = None, manual: ManualReviewStore | None = None):
        self.config = config or {}
        review_cfg = self.config.get("review", {}) if self.config else {}
        self.provider = str(
            os.getenv("REVIEW_PROVIDER", review_cfg.get("provider", "none"))
        ).strip().lower()
        self.api_url = os.getenv("REVIEW_API_URL", "").strip()
        self.api_key = os.getenv("REVIEW_API_KEY", "").strip()
        self.model = os.getenv("REVIEW_MODEL", "reviewer").strip()
        self.timeout = float(review_cfg.get("timeout_sec", 30))
        self.manual = manual or ManualReviewStore()

    @property
    def available(self) -> bool:
        if self.provider in ("", "none"):
            return False
        return bool(self.api_url)

    async def review(self, sample: dict) -> ReviewDecision:
        manual = self.manual.lookup(sample)
        if manual is not None:
            return manual
        if not self.available:
            return ReviewDecision(STATUS_UNAVAILABLE, "review_not_configured", "none")
        try:
            return await asyncio.to_thread(self._call_api, sample)
        except Exception as exc:  # network/parse failure -> unavailable
            logger.warning("review api failed: %s", exc)
            return ReviewDecision(STATUS_UNAVAILABLE, f"review_error:{exc}", "api")

    def _call_api(self, sample: dict) -> ReviewDecision:
        if self.provider == "openai-compatible":
            return self._call_openai_compatible(sample)
        return self._call_http(sample)

    def _request(self, url: str, body: dict) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _call_http(self, sample: dict) -> ReviewDecision:
        payload = {
            "anchor": sample.get("anchor", ""),
            "positive": sample.get("positive", ""),
            "negatives": sample.get("negatives", []),
        }
        result = self._request(self.api_url, payload)
        approved = bool(result.get("approved"))
        return ReviewDecision(
            STATUS_APPROVED if approved else STATUS_REJECTED,
            str(result.get("reason", "")),
            "api",
        )

    def _call_openai_compatible(self, sample: dict) -> ReviewDecision:
        url = self.api_url
        if not url.rstrip("/").endswith("completions"):
            url = url.rstrip("/") + "/v1/chat/completions"
        prompt = (
            "Decide whether this retrieval training pair is a valid positive "
            "(the user intent matches the target item and the item is acceptable).\n"
            f"intent: {sample.get('anchor', '')}\n"
            f"target item: {sample.get('positive', '')}\n"
            'Return JSON only: {"approved": true|false, "reason": "..."}'
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict training-data reviewer."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        result = self._request(url, body)
        content = ""
        choices = result.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content", "")
        approved = False
        reason = content[:200]
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(content[start : end + 1])
                approved = bool(parsed.get("approved"))
                reason = str(parsed.get("reason", reason))
        except json.JSONDecodeError:
            approved = "approved" in content.lower()
        return ReviewDecision(
            STATUS_APPROVED if approved else STATUS_REJECTED, reason, "api"
        )
