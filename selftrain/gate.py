"""Sample gating: positive | correction | negative | drop.

A success-labeled sample becomes a training positive ONLY when an external
reviewer approves it. Reviewer unavailability => drop (never self-approve).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .reviewer import STATUS_APPROVED, STATUS_REJECTED, ExternalReviewer, ReviewDecision

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    label: str
    reason: str = ""
    review: ReviewDecision | None = None


@dataclass
class GateResult:
    positives: list[dict] = field(default_factory=list)
    negatives: list[dict] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    review_unavailable: bool = False
    rejected: int = 0


def _sample_label(sample: dict) -> str:
    label = sample.get("label")
    if label in ("success", "failure", "unknown"):
        return label
    behavior = sample.get("behavior")
    if behavior == "positive":
        return "success"
    if behavior == "negative":
        return "failure"
    return "unknown"


def _decision_from_sample(sample: dict) -> ReviewDecision | None:
    review = sample.get("review")
    if isinstance(review, dict) and review.get("status") in (STATUS_APPROVED, STATUS_REJECTED):
        return ReviewDecision(status=review["status"], reason=str(review.get("reason", "")), by=str(review.get("by", "manual")))
    status = sample.get("review_status")
    if status in (STATUS_APPROVED, STATUS_REJECTED):
        return ReviewDecision(status=status, reason="review_status", by="manual")
    return None


async def classify_sample(sample: dict, reviewer: ExternalReviewer | None = None) -> GateDecision:
    label = _sample_label(sample)
    if label == "failure":
        return GateDecision("negative", "behavior_failure")
    if label != "success":
        return GateDecision("drop", f"unknown_label:{label}")

    pre = _decision_from_sample(sample)
    if pre is not None:
        decision = pre
    elif reviewer is None:
        return GateDecision("drop", "review_not_configured")
    else:
        decision = await reviewer.review(sample)

    if decision.status == STATUS_APPROVED:
        return GateDecision("positive", decision.reason, decision)
    if decision.status == STATUS_REJECTED:
        return GateDecision("correction", decision.reason, decision)
    return GateDecision("drop", decision.reason or "review_unavailable", decision)


async def gate_batch(samples: list[dict], reviewer: ExternalReviewer | None = None) -> GateResult:
    result = GateResult()
    for sample in samples:
        decision = await classify_sample(sample, reviewer)
        if decision.label == "positive":
            result.positives.append({**sample, "review": _review_dict(decision.review)})
        elif decision.label == "negative":
            result.negatives.append(sample)
        elif decision.label == "correction":
            result.rejected += 1
            result.corrections.append({**sample, "review_reason": decision.reason})
        else:
            if decision.reason == "review_not_configured":
                result.review_unavailable = True
            result.dropped.append(sample)
    return result


def _review_dict(decision: ReviewDecision | None) -> dict | None:
    if decision is None:
        return None
    return {"status": decision.status, "reason": decision.reason, "by": decision.by}
