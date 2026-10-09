"""Feedback classification from raw interaction events."""

from __future__ import annotations

from typing import Optional, Tuple

from .constants import (
    DEFAULT_ITEM_DURATION_SEC,
    FAST_SWIPE_RATIO,
    POSITIVE_WATCH_RATIO,
    STRONG_NEGATIVE_STREAK,
)


def _watch_ratio(watch_duration_sec: float, item_duration_sec: float) -> float:
    if item_duration_sec <= 0:
        return 0.0
    return watch_duration_sec / item_duration_sec


def _maybe_upgrade_strong_negative(
    feedback_type: str, consecutive_negative: int
) -> Tuple[str, str]:
    if feedback_type == "negative" and consecutive_negative >= STRONG_NEGATIVE_STREAK - 1:
        return "strong_negative", "consecutive_negative_streak"
    return feedback_type, ""


def classify_feedback(
    event: str,
    watch_duration_sec: float = 0,
    item_duration_sec: float = DEFAULT_ITEM_DURATION_SEC,
    consecutive_negative: int = 0,
    was_clicked: bool = False,
    positive_ratio: float | None = None,
    short_ratio: float | None = None,
) -> Tuple[Optional[str], str]:
    """Map raw UI events to feedback_type. None = intermediate state only."""
    if positive_ratio is None:
        positive_ratio = POSITIVE_WATCH_RATIO
    if short_ratio is None:
        short_ratio = FAST_SWIPE_RATIO
    item_duration_sec = max(item_duration_sec, 1.0)
    ratio = _watch_ratio(watch_duration_sec, item_duration_sec)

    if event == "click":
        return None, "click_recorded"
    if event == "exposed":
        return None, "exposure_recorded"

    feedback_type: Optional[str] = None
    reason = ""

    if event in ("share", "favorite"):
        feedback_type, reason = "positive", event
    elif event == "not_interested":
        feedback_type, reason = "strong_negative", "not_interested"
    elif event == "no_click":
        feedback_type, reason = "negative", "exposure_no_click"
    elif event == "dismiss":
        if not was_clicked:
            feedback_type, reason = "negative", "dismiss_without_click"
        elif ratio < short_ratio:
            feedback_type, reason = "negative", "short_dwell"
        else:
            feedback_type, reason = "negative", "dismiss_after_partial_watch"
    elif event in ("watch_end", "dwell_end"):
        if ratio > positive_ratio:
            feedback_type, reason = "positive", "dwell_above_threshold"
        elif ratio < short_ratio:
            feedback_type, reason = "negative", "short_dwell"
        else:
            feedback_type, reason = "negative", "dwell_below_threshold"
    else:
        return None, f"unknown_event:{event}"

    if feedback_type == "negative":
        upgraded, upgrade_reason = _maybe_upgrade_strong_negative(
            feedback_type, consecutive_negative
        )
        if upgrade_reason:
            return upgraded, upgrade_reason
        return feedback_type, reason

    return feedback_type, reason
