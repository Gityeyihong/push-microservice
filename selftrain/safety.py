"""Training safety net (Plasmid §15): gates, diversity, regression, rollback, collapse."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable

from .events import (
    EVENT_COLLAPSE,
    EVENT_ROLLBACK,
    EVENT_TUNE_SKIPPED,
    TrainingEvents,
)

logger = logging.getLogger(__name__)

DEFAULT_POSITIVE_THRESHOLD = 0.5
DEFAULT_MIN_POSITIVE = 2
DEFAULT_MAX_SIM = 0.95
DEFAULT_EPSILON = 0.02
DEFAULT_MAX_TUNE_FAILURES = 3
DEFAULT_MAX_VERSIONS = 5
DEFAULT_SOURCE_RATIO_CAP = 0.7


def cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SafetyVerdict:
    ok: bool
    reason: str = ""
    effective_positives: int = 0


@dataclass
class TrainingSafety:
    events: TrainingEvents | None = None
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD
    min_positive: int = DEFAULT_MIN_POSITIVE
    max_sim: float = DEFAULT_MAX_SIM
    epsilon: float = DEFAULT_EPSILON
    max_tune_failures: int = DEFAULT_MAX_TUNE_FAILURES
    max_versions: int = DEFAULT_MAX_VERSIONS
    source_ratio_cap: float = DEFAULT_SOURCE_RATIO_CAP
    consecutive_failures: int = 0
    auto_tune_enabled: bool = True
    history: list[dict] = field(default_factory=list)

    def can_tune(self, force: bool = False) -> bool:
        return bool(force) or self.auto_tune_enabled

    def passes_positive_gate(self, sample: dict) -> bool:
        review = sample.get("review") or {}
        return review.get("status") == "approved"

    def filter_positives(self, samples: list[dict]) -> list[dict]:
        return [s for s in samples if self.passes_positive_gate(s)]

    def pre_tune_check(
        self,
        positives: list[dict],
        embedder: Callable[[str], list[float]] | None = None,
    ) -> SafetyVerdict:
        if len(positives) < self.min_positive:
            return SafetyVerdict(False, "too_few_positives", len(positives))

        kept = self._apply_source_cap(positives)
        if len(kept) < self.min_positive:
            return SafetyVerdict(False, "source_cap_left_too_few", len(kept))

        if embedder is not None:
            effective = self._count_unique(kept, embedder)
            if effective < self.min_positive:
                self._emit(
                    EVENT_TUNE_SKIPPED,
                    {"reason": f"effective_positives {effective} < {self.min_positive} (near-duplicates)"},
                )
                return SafetyVerdict(False, "low_diversity", effective)
            return SafetyVerdict(True, "", effective)

        return SafetyVerdict(True, "", len(kept))

    def _apply_source_cap(self, samples: list[dict]) -> list[dict]:
        if not samples:
            return []
        if len(samples) <= self.min_positive:
            return samples
        limit = max(self.min_positive, int(len(samples) * self.source_ratio_cap))
        counts: dict[str, int] = {}
        kept: list[dict] = []
        for sample in samples:
            key = str(sample.get("source") or sample.get("session_id") or "unknown")
            if counts.get(key, 0) >= limit:
                continue
            counts[key] = counts.get(key, 0) + 1
            kept.append(sample)
        return kept

    def _count_unique(self, samples: list[dict], embedder: Callable[[str], list[float]]) -> int:
        embeddings = []
        for sample in samples:
            vec = embedder(sample.get("positive", ""))
            if vec:
                embeddings.append(vec)
        if len(embeddings) < 2:
            return len(embeddings)
        unique = 0
        for i, emb in enumerate(embeddings):
            others = [e for j, e in enumerate(embeddings) if j != i]
            if not others or max(cosine(emb, other) for other in others) <= self.max_sim:
                unique += 1
        return unique

    def commit_adapter(self, *, new_path: str | Path, heldout_before: float, heldout_after: float) -> dict:
        new_path = Path(new_path)
        if heldout_after >= heldout_before - self.epsilon:
            self.consecutive_failures = 0
            return {"applied": True, "path": str(new_path)}

        failed_path = new_path.parent / f"failed_{new_path.name}"
        try:
            if new_path.exists():
                if failed_path.exists():
                    import shutil

                    shutil.rmtree(failed_path, ignore_errors=True)
                os.rename(str(new_path), str(failed_path))
        except OSError as exc:
            logger.warning("rollback rename failed: %s", exc)
            failed_path = new_path

        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_tune_failures:
            self.auto_tune_enabled = False
        self._emit(
            EVENT_ROLLBACK,
            {
                "reason": f"held-out regressed: {heldout_after} < {heldout_before} - {self.epsilon}",
                "failed_path": str(failed_path),
                "consecutive_failures": self.consecutive_failures,
            },
        )
        return {
            "applied": False,
            "failed_path": str(failed_path),
            "consecutive_failures": self.consecutive_failures,
        }

    def detect_collapse(
        self,
        *,
        novelty_median_delta: float,
        metric_median_delta: float,
        heldout_delta: float,
    ) -> bool:
        collapsed = (
            novelty_median_delta < -0.15
            and metric_median_delta > 0
            and heldout_delta <= 0
        )
        if collapsed:
            self.positive_threshold = min(self.positive_threshold + 0.05, 1.0)
            self._emit(EVENT_COLLAPSE, {"new_positive_threshold": self.positive_threshold})
        return collapsed

    def record(self, record: dict) -> None:
        self.history.append(record)

    def _emit(self, event_type: str, payload: dict) -> None:
        if self.events is not None:
            self.events.publish(event_type, payload)


def novelty_median(samples: list[dict], embedder: Callable[[str], list[float]]) -> float:
    embeddings = [embedder(s.get("positive", "")) for s in samples]
    embeddings = [e for e in embeddings if e]
    if len(embeddings) < 2:
        return 1.0
    novelties = []
    for i, emb in enumerate(embeddings):
        others = [e for j, e in enumerate(embeddings) if j != i]
        novelties.append(1.0 - max(cosine(emb, other) for other in others))
    return median(novelties)
