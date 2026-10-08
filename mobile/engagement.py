from typing import Optional, Tuple

from .config import (
    DEFAULT_VIDEO_DURATION_SEC,
    POSITIVE_WATCH_RATIO,
    FAST_SWIPE_RATIO,
    STRONG_NEGATIVE_STREAK,
)


def _watch_ratio(watch_duration_sec: float, video_duration_sec: float) -> float:
    if video_duration_sec <= 0:
        return 0.0
    return watch_duration_sec / video_duration_sec


def _maybe_upgrade_strong_negative(
    feedback_type: str, consecutive_negative: int
) -> Tuple[str, str]:
    if (
        feedback_type == "negative"
        and consecutive_negative >= STRONG_NEGATIVE_STREAK - 1
    ):
        return "strong_negative", "consecutive_negative_streak"
    return feedback_type, ""


def classify_engagement(
    event: str,
    watch_duration_sec: float = 0,
    video_duration_sec: float = DEFAULT_VIDEO_DURATION_SEC,
    consecutive_negative: int = 0,
    was_clicked: bool = False,
) -> Tuple[Optional[str], str]:
    """Map raw UI events to feedback_type. None = intermediate state only (e.g. click)."""
    video_duration_sec = max(video_duration_sec, 1.0)
    ratio = _watch_ratio(watch_duration_sec, video_duration_sec)

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
        elif ratio < FAST_SWIPE_RATIO:
            feedback_type, reason = "negative", "fast_swipe"
        else:
            feedback_type, reason = "negative", "dismiss_after_partial_watch"

    elif event == "watch_end":
        if ratio > POSITIVE_WATCH_RATIO:
            feedback_type, reason = "positive", "watch_above_threshold"
        elif ratio < FAST_SWIPE_RATIO:
            feedback_type, reason = "negative", "fast_swipe"
        else:
            feedback_type, reason = "negative", "watch_below_threshold"

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
