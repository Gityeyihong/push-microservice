"""Default threshold-based decision policy."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from .base import DecisionPolicy
from .config import HANDOFF_LINK, PolicyConfig, load_policy_config, resolve_a_threshold_0
from .constants import DEFAULT_ITEM_DURATION_SEC, STRONG_COOLDOWN_MSGS
from .engagement import classify_feedback
from .events import HandoffPushEvent, ItemPushEvent, PushEvent
from .keywords import (
    helpless_keywords,
    is_handoff_request,
    is_item_request,
    sensitive_topics,
    signal_forces_handoff,
)
from .state import StateManager


class BuiltinThresholdPolicy(DecisionPolicy):
    def _cfg(self) -> PolicyConfig:
        return self._active_config

    def process(
        self,
        *,
        user_message: str,
        candidates: Optional[List[Dict]] = None,
        positive_rate: float = 0.6,
        handoff_suggested: bool = False,
        signals: Optional[dict[str, Any]] = None,
        state: str = "INIT",
        config: Optional[PolicyConfig] = None,
    ) -> List[PushEvent]:
        self._active_config = config or self._default_config
        cfg = self._cfg()
        signals = signals or {}
        events: List[PushEvent] = []

        self.state.add_message(time.time())

        trigger_mode = str(signals.get("trigger_mode") or "auto")
        content_need = str(signals.get("content_need") or "none")
        active_item = trigger_mode == "active" or is_item_request(user_message)
        active_handoff = is_handoff_request(user_message)
        if content_need == "item":
            active_handoff = False
        elif content_need == "handoff":
            active_handoff = True
            handoff_suggested = True
        if not active_item and not active_handoff and self.state.consume_cooldown():
            return events

        self.state.bump_urge(cfg.urge_delta, 1.0)

        risk_alert = signal_forces_handoff(signals)
        handoff_first = risk_alert or active_handoff
        if content_need == "item" and not risk_alert:
            events.extend(
                self._evaluate_item(user_message, candidates, state=state, force_active=active_item)
            )
            return events
        if handoff_first:
            handoff_events = self._evaluate_handoff(
                user_message,
                positive_rate,
                handoff_suggested=handoff_suggested,
                risk_alert=risk_alert,
                force_active=content_need == "handoff",
            )
            if handoff_events:
                events.extend(handoff_events)
            else:
                events.extend(
                    self._evaluate_item(user_message, candidates, state=state, force_active=active_item)
                )
        else:
            item_events = self._evaluate_item(
                user_message, candidates, state=state, force_active=active_item
            )
            events.extend(item_events)
            if not item_events:
                events.extend(
                    self._evaluate_handoff(
                        user_message, positive_rate, handoff_suggested=handoff_suggested, risk_alert=risk_alert
                    )
                )
        return events

    def _compute_dynamic_k(self) -> int:
        msg_ts = self.state.get("msg_ts_list")
        if len(msg_ts) < 3:
            return 15
        gaps = sorted(msg_ts[i] - msg_ts[i - 1] for i in range(1, len(msg_ts)))
        if len(gaps) >= 3:
            gaps = gaps[1:-1]
        if not gaps:
            return 15
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap <= 30:
            return 15
        if avg_gap >= 300:
            return 5
        return round(5 + 10 * (300 - avg_gap) / (300 - 30))

    def _compute_acceptance_a(
        self, top1_similarity: float, recent_push_count_5min: int = 0, state: str = "INIT"
    ) -> float:
        cfg = self._cfg()
        w = cfg.a_weights
        I = min(max(float(top1_similarity), 0.0), 1.0)
        divisor = max(cfg.recent_push_load_divisor, 1)
        L = 1.0 - min(recent_push_count_5min / float(divisor), 1.0)
        total_exposures = self.state.get("total_exposures")
        F = (
            cfg.f_default_new_user
            if total_exposures == 0
            else max(0.3, self.state.get("positive_feedbacks") / max(total_exposures, 1))
        )
        cap = max(cfg.session_msg_cap, 1)
        S = min(self.state.get("session_msg_count") / float(cap), 1.0)
        U = self.state.get("urge")
        A = w.I * I + w.L * L + w.F * F + w.S * S + w.U * U
        A += cfg.state_modulation.get(state, 0.0)
        return A

    def _check_cool_down(self) -> bool:
        last_push = self.state.get("last_push_ts")
        if last_push == 0:
            return True
        now_ms = int(time.time() * 1000)
        return (now_ms - last_push) >= min(self._cfg().cooldown_ms, self._cfg().cooldown_ms_max)

    def _map_push_count(self, A: float, max_candidates: int) -> int:
        cfg = self._cfg()
        th0 = resolve_a_threshold_0(cfg)
        if A < th0:
            return 0
        if A < cfg.a_threshold_1:
            return 1
        if A < cfg.a_threshold_2:
            return 2
        return min(3, max_candidates)

    def _dedup_candidates(self, candidates: List[Dict]) -> List[Dict]:
        recent = set(self.state.get("push_history"))
        return [c for c in candidates if c.get("item_id") not in recent]

    def _softmax_sample(self, candidates: List[Dict]) -> Optional[Dict]:
        if not candidates:
            return None
        weights = [max(float(c.get("prob_base") or 0), 0.0) for c in candidates]
        total = sum(weights)
        if total <= 0:
            return candidates[0]
        draw = random.random() * total
        cumsum = 0.0
        for cand, weight in zip(candidates, weights):
            cumsum += weight
            if draw <= cumsum:
                return cand
        return candidates[-1]

    def _pick_item(self, candidates: List[Dict]) -> Optional[Dict]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        top = float(candidates[0].get("similarity", 0))
        second = float(candidates[1].get("similarity", 0))
        if top >= second + 0.10:
            return candidates[0]
        return self._softmax_sample(candidates)

    def _auto_item_target_rate(self) -> float:
        cfg = self._cfg()
        if cfg.auto_item_target_rate is not None:
            return float(cfg.auto_item_target_rate)
        return cfg.target_push_rate_per_10 / 10.0

    def _auto_item_quota_status(self) -> tuple[bool, int, int]:
        msg_count = self.state.get("session_msg_count")
        allowed = round(msg_count * self._auto_item_target_rate())
        current = int(self.state.get("auto_item_push_count") or 0)
        return current < allowed, allowed, current

    def _record_auto_item_push(self) -> None:
        msg_count = self.state.get("session_msg_count")
        current = int(self.state.get("auto_item_push_count") or 0)
        self.state.set("auto_item_push_count", current + 1)
        self.state.set("last_auto_item_push_msg", msg_count)

    def _should_quota_force_auto_item(self, top1_sim: float, dedup: List[Dict], filtered: List[Dict]) -> bool:
        cfg = self._cfg()
        ok, allowed, current = self._auto_item_quota_status()
        if not ok:
            return False
        pool = dedup or filtered
        if not pool:
            return False
        if float(top1_sim) < cfg.auto_item_quota_min_similarity:
            return False
        msg_count = self.state.get("session_msg_count")
        last_msg = int(self.state.get("last_auto_item_push_msg") or 0)
        behind = (allowed - current) >= 2
        if last_msg > 0 and (msg_count - last_msg) < cfg.auto_item_quota_cooldown_msgs and not behind:
            return False
        if not behind and not self._check_cool_down():
            return False
        return True

    def _emit_auto_item_push(self, selected: Dict, A: float, N: int, *, quota_forced: bool = False) -> ItemPushEvent:
        now_ms = int(time.time() * 1000)
        self.state.set("urge", 0)
        self.state.note_push(now_ms, self._cfg().recent_push_window_ms)
        self.state.add_push_history(selected["item_id"])
        self._record_auto_item_push()
        data = {
            "item_id": selected["item_id"],
            "title": selected.get("title", ""),
            "media_url": selected.get("media_url", ""),
            "duration_sec": selected.get("duration_sec", DEFAULT_ITEM_DURATION_SEC),
            "trigger": "auto",
            "acceptance": round(A, 4),
            "push_count": N,
        }
        if selected.get("item_ref"):
            data["item_ref"] = selected["item_ref"]
        if quota_forced:
            data["quota_forced"] = True
        return ItemPushEvent(data=data)

    def _evaluate_item(
        self,
        user_message: str,
        candidates: Optional[List[Dict]],
        state: str = "INIT",
        force_active: bool = False,
    ) -> List[PushEvent]:
        cfg = self._cfg()
        events: List[PushEvent] = []
        is_active = force_active or is_item_request(user_message)

        if is_active:
            if candidates:
                selected = candidates[0]
                self.state.scale_urge(cfg.urge_active_item_decay)
                self.state.note_push(int(time.time() * 1000), cfg.recent_push_window_ms)
                self.state.add_push_history(selected["item_id"])
                events.append(
                    ItemPushEvent(
                        data={
                            "item_id": selected["item_id"],
                            "title": selected.get("title", ""),
                            "media_url": selected.get("media_url", ""),
                            "duration_sec": selected.get("duration_sec", DEFAULT_ITEM_DURATION_SEC),
                            "trigger": "active",
                            **({"item_ref": selected["item_ref"]} if selected.get("item_ref") else {}),
                        }
                    )
                )
            return events

        if not candidates:
            return events

        msg_count = self.state.get("session_msg_count")
        if msg_count < max(cfg.auto_push_min_session_msgs, 1):
            return events

        dedup = self._dedup_candidates(candidates)
        pool = dedup if dedup else candidates
        if not pool:
            return events

        top1_sim = pool[0].get("similarity", 0)
        if not self._check_cool_down() and not self._should_quota_force_auto_item(top1_sim, dedup, candidates):
            return events

        now_ms = int(time.time() * 1000)
        recent_count = self.state.count_recent_pushes(now_ms, cfg.recent_push_window_ms)
        A = self._compute_acceptance_a(top1_sim, recent_count, state)
        N = self._map_push_count(A, len(pool))
        quota_forced = False
        if N == 0 and self._should_quota_force_auto_item(top1_sim, dedup, candidates):
            N = 1
            quota_forced = True
        if N == 0:
            return events

        selected = self._pick_item(pool[:N])
        if selected is None:
            return events
        events.append(self._emit_auto_item_push(selected, A, N, quota_forced=quota_forced))
        return events

    def _auto_handoff_target_rate(self) -> float:
        cfg = self._cfg()
        if cfg.auto_handoff_target_rate is not None:
            return float(cfg.auto_handoff_target_rate)
        return 0.25

    def _handoff_quota_effective_rate(self) -> float:
        cfg = self._cfg()
        if cfg.auto_handoff_effective_rate is not None:
            return float(cfg.auto_handoff_effective_rate)
        handoff = self._auto_handoff_target_rate()
        item = self._auto_item_target_rate()
        divisor = max(1.0 - item, 0.25)
        return min(handoff / divisor, 0.5)

    def _auto_handoff_quota_status(self) -> tuple[bool, int, int]:
        cfg = self._cfg()
        msg_count = self.state.get("session_msg_count")
        allowed = round(msg_count * self._handoff_quota_effective_rate())
        current = int(self.state.get("auto_handoff_push_count") or 0)
        min_msgs = max(cfg.auto_handoff_quota_min_session_msgs, cfg.auto_push_min_session_msgs, 1)
        if msg_count < min_msgs:
            return False, allowed, current
        return current < allowed, allowed, current

    def _record_auto_handoff_push(self) -> None:
        msg_count = self.state.get("session_msg_count")
        current = int(self.state.get("auto_handoff_push_count") or 0)
        self.state.set("auto_handoff_push_count", current + 1)
        self.state.set("last_auto_handoff_push_msg", msg_count)

    def _should_quota_force_auto_handoff(self, handoff_suggested: bool) -> bool:
        if self._cfg().auto_handoff_target_rate is None:
            return False
        ok, allowed, current = self._auto_handoff_quota_status()
        if not ok:
            return False
        msg_count = self.state.get("session_msg_count")
        last_msg = int(self.state.get("last_auto_handoff_push_msg") or 0)
        behind = (allowed - current) >= 1
        if last_msg > 0 and (msg_count - last_msg) < self._cfg().auto_handoff_quota_cooldown_msgs and not behind:
            return False
        return True

    def _emit_auto_handoff_push(self, *, is_active_request: bool, quota_forced: bool = False) -> HandoffPushEvent:
        now_ms = int(time.time() * 1000)
        self.state.set("last_handoff_push_ts", now_ms)
        count = self.state.get("total_handoff_push_count")
        self.state.set("total_handoff_push_count", count + 1)
        self._record_auto_handoff_push()
        data = {
            "title": "Talk to a human specialist?",
            "subtitle": "Connect with a live agent for 1:1 help",
            "link": HANDOFF_LINK,
            "trigger": "active" if is_active_request else "auto",
        }
        if quota_forced:
            data["quota_forced"] = True
        return HandoffPushEvent(data=data)

    def _handoff_session_cap(self) -> int:
        cfg = self._cfg()
        base = max(int(cfg.handoff_session_push_cap), 1)
        if cfg.auto_handoff_target_rate is None:
            return base
        _, allowed, _ = self._auto_handoff_quota_status()
        if allowed <= 0:
            return 1
        return min(base, allowed)

    def _evaluate_handoff(
        self,
        user_message: str,
        positive_rate: float,
        handoff_suggested: bool = False,
        risk_alert: bool = False,
        force_active: bool = False,
    ) -> List[PushEvent]:
        events: List[PushEvent] = []
        last_ts = self.state.get("last_handoff_push_ts")
        now_ms = int(time.time() * 1000)
        is_active_request = force_active or is_handoff_request(user_message)
        quota_forced = False
        under_quota, _, _ = self._auto_handoff_quota_status()

        if not is_active_request and not risk_alert:
            if not under_quota:
                if (now_ms - last_ts) < self._cfg().handoff_push_cooldown_ms:
                    return events
                if self.state.get("total_handoff_push_count") >= self._handoff_session_cap():
                    return events

        triggered = False
        if risk_alert:
            triggered = True
        elif is_active_request:
            triggered = True
        elif self._check_helpless(user_message) and self.state.get("urge") >= 0.25 and under_quota:
            triggered = True
        elif self._check_sensitive_urge(user_message) and under_quota:
            triggered = True
        elif handoff_suggested and self.state.get("urge") >= 0.15 and under_quota:
            triggered = True
        elif self._should_quota_force_auto_handoff(handoff_suggested):
            triggered = True
            quota_forced = True

        if triggered:
            if is_active_request or risk_alert:
                self.state.set("last_handoff_push_ts", now_ms)
                if not risk_alert:
                    count = self.state.get("total_handoff_push_count")
                    self.state.set("total_handoff_push_count", count + 1)
                    self._record_auto_handoff_push()
                events.append(
                    HandoffPushEvent(
                        data={
                            "title": "Talk to a human specialist?",
                            "subtitle": "Connect with a live agent for 1:1 help",
                            "link": HANDOFF_LINK,
                            "trigger": "active" if is_active_request else "auto",
                        }
                    )
                )
            else:
                events.append(self._emit_auto_handoff_push(is_active_request=is_active_request, quota_forced=quota_forced))
        return events

    def _check_helpless(self, user_message: str) -> bool:
        return any(kw in user_message for kw in helpless_keywords())

    def _check_sensitive_urge(self, user_message: str) -> bool:
        topics = sensitive_topics()
        has_sensitive = any(t in user_message for t in topics) if topics else False
        return has_sensitive and self.state.get("urge") >= 0.7

    def _apply_feedback(self, feedback_type: str):
        cfg = self._cfg()
        tau = self.state.get("tau") or cfg.tau_init
        if tau <= 0:
            tau = cfg.tau_init
        if feedback_type == "positive":
            tau = max(cfg.tau_min, tau * cfg.tau_positive_mult)
            self.state.record_feedback_outcome(is_positive=True)
        elif feedback_type == "negative":
            tau = min(cfg.tau_max, tau * cfg.tau_negative_mult)
            self.state.record_feedback_outcome(is_positive=False)
        elif feedback_type == "strong_negative":
            self.state.set("urge", 0)
            self.state.set("cool_down_remain", STRONG_COOLDOWN_MSGS)
            self.state.record_feedback_outcome(is_positive=False, is_strong_negative=True)
        self.state.set("tau", tau)

    def record_item_feedback(self, item_id: str, feedback_type: str):
        self._apply_feedback(feedback_type)

    def record_video_feedback(self, item_id: str, feedback_type: str):
        self.record_item_feedback(item_id, feedback_type)

    def record_engagement(
        self,
        *,
        item_id: str,
        event: str,
        watch_duration_sec: float = 0,
        item_duration_sec: float = DEFAULT_ITEM_DURATION_SEC,
        config: Optional[PolicyConfig] = None,
    ) -> dict:
        if config is not None:
            self._active_config = config
        if event == "click":
            self.state.mark_clicked(item_id)
            return {
                "feedback_type": None,
                "reason": "click_recorded",
                "click_rate": round(self.state.get_click_rate(), 4),
                "watch_ratio": 0.0,
            }

        consecutive_negative = self.state.get("consecutive_negative")
        cfg = self._cfg()
        feedback_type, reason = classify_feedback(
            event=event,
            watch_duration_sec=watch_duration_sec,
            item_duration_sec=item_duration_sec,
            consecutive_negative=consecutive_negative,
            was_clicked=self.state.has_click(item_id),
            positive_ratio=cfg.positive_dwell_ratio,
            short_ratio=cfg.short_dwell_ratio,
        )
        item_duration_sec = max(item_duration_sec, 1.0)
        watch_ratio = round(watch_duration_sec / item_duration_sec, 4)
        if feedback_type:
            self._apply_feedback(feedback_type)
            self.state.forget_click(item_id)
        return {
            "feedback_type": feedback_type,
            "reason": reason,
            "click_rate": round(self.state.get_click_rate(), 4),
            "watch_ratio": watch_ratio,
        }

    def state_summary(self) -> dict:
        cfg = self._cfg()
        return {
            "urge": self.state.get("urge"),
            "tau": self.state.get("tau"),
            "feedbacks": (
                self.state.get("positive_feedbacks") / max(self.state.get("total_exposures"), 1)
                if self.state.get("total_exposures") > 0
                else cfg.f_default_new_user
            ),
            "click_rate": round(self.state.get_click_rate(), 4),
            "consecutive_negative": self.state.get("consecutive_negative"),
            "session_msg_count": self.state.get("session_msg_count"),
            "dynamic_k": self._compute_dynamic_k(),
        }
