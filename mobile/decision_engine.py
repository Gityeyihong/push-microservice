import math
import random
import time
from typing import List, Dict, Optional

from .state_manager import StateManager, PushEvent, VideoPushEvent, AdvisorPushEvent
from .config import DEFAULT_VIDEO_DURATION_SEC, STRONG_COOLDOWN_MSGS
from .decision_config import DecisionConfig, DEFAULT_ADVISOR_LINK, load_decision_config, resolve_a_threshold_0
from .engagement import classify_engagement
from .request_keywords import ADVISOR_REQUEST_KEYWORDS, VIDEO_REQUEST_KEYWORDS, is_advisor_request, is_video_request

BREAKUP_USER_KWS = [
    "分手", "失恋", "前任", "离婚", "挽回", "被甩", "放弃了", "抛弃",
]
BREAKUP_VIDEO_KWS = [
    "失恋", "分手", "前任", "挽回", "放下前任", "自我重建",
]

HELPLESS_KEYWORDS = [
    "帮不了我", "还是不行", "没用", "怎么办啊", "我崩溃了", "救救我",
]

SENSITIVE_TOPICS = ["分手", "离婚", "冷暴力", "出轨"]


class DecisionEngine:
    def __init__(
        self,
        state: Optional[StateManager] = None,
        config: Optional[DecisionConfig] = None,
    ):
        self.state = state or StateManager()
        self._default_config = config or load_decision_config()
        self._active_config = self._default_config
        self._video_clicked: set = set()

    def _cfg(self) -> DecisionConfig:
        return self._active_config

    def process_user_message(
        self,
        user_message: str,
        candidates: Optional[List[Dict]] = None,
        positive_rate: float = 0.6,
        advisor_suggestion: bool = False,
        risk_alert: bool = False,
        conversation_state: str = "INIT",
        interaction_mode: str = "auto",
        decision_config: Optional[DecisionConfig] = None,
    ) -> List[PushEvent]:
        self._active_config = decision_config or self._default_config
        cfg = self._cfg()
        events = []
        self._interaction_mode = interaction_mode

        self.state.add_message(time.time())

        active_video = is_video_request(user_message)
        active_advisor = is_advisor_request(user_message)
        if self.state.get("cool_down_remain") > 0 and not active_video and not active_advisor:
            remain = self.state.get("cool_down_remain") - 1
            self.state.set("cool_down_remain", remain)
            return events

        urge = self.state.get("urge")
        urge_delta = cfg.urge_delta
        if interaction_mode == "mentor":
            urge_delta += cfg.mentor_urge_delta_bonus
        urge = min(urge + urge_delta, 1.0)
        self.state.set("urge", urge)

        if interaction_mode != "proxy_chat":
            advisor_first = risk_alert or is_advisor_request(user_message)
            if advisor_first:
                advisor_events = self._evaluate_advisor(
                    user_message,
                    positive_rate,
                    has_video_push=False,
                    advisor_suggestion=advisor_suggestion,
                    risk_alert=risk_alert,
                    interaction_mode=interaction_mode,
                )
                if advisor_events:
                    events.extend(advisor_events)
                else:
                    events.extend(
                        self._evaluate_video(
                            user_message,
                            candidates,
                            conversation_state=conversation_state,
                            interaction_mode=interaction_mode,
                        )
                    )
            else:
                video_events = self._evaluate_video(
                    user_message,
                    candidates,
                    conversation_state=conversation_state,
                    interaction_mode=interaction_mode,
                )
                events.extend(video_events)
                if not video_events:
                    events.extend(
                        self._evaluate_advisor(
                            user_message,
                            positive_rate,
                            has_video_push=False,
                            advisor_suggestion=advisor_suggestion,
                            risk_alert=risk_alert,
                            interaction_mode=interaction_mode,
                        )
                    )

        return events

    def _compute_dynamic_k(self) -> int:
        msg_ts = self.state.get("msg_ts_list")
        if len(msg_ts) < 3:
            return 15

        gaps = [msg_ts[i] - msg_ts[i - 1] for i in range(1, len(msg_ts))]

        if len(gaps) >= 3:
            gaps_sorted = sorted(gaps)
            gaps = gaps_sorted[1:-1]

        if not gaps:
            return 15

        avg_gap = sum(gaps) / len(gaps)

        if avg_gap <= 30:
            return 15
        if avg_gap >= 300:
            return 5

        k = 5 + 10 * (300 - avg_gap) / (300 - 30)
        return round(k)

    def _compute_acceptance_a(
        self,
        top1_similarity: float,
        recent_push_count_5min: int = 0,
        conversation_state: str = "INIT",
        interaction_mode: str = "auto",
    ) -> float:
        cfg = self._cfg()
        w = cfg.a_weights
        I = min(top1_similarity, 1.0)

        divisor = max(cfg.recent_push_load_divisor, 1)
        L = 1.0 - min(recent_push_count_5min / float(divisor), 1.0)

        total_exposures = self.state.get("total_exposures")
        if total_exposures == 0:
            F = cfg.f_default_new_user
        else:
            F = max(0.3, self.state.get("positive_feedbacks") / max(total_exposures, 1))

        cap = max(cfg.session_msg_cap, 1)
        S = min(self.state.get("session_msg_count") / float(cap), 1.0)

        U = self.state.get("urge")

        A = w.I * I + w.L * L + w.F * F + w.S * S + w.U * U
        A += cfg.state_modulation.get(conversation_state, 0.0)
        if interaction_mode == "mentor":
            A += cfg.mentor_a_boost

        return A

    def _effective_cooldown_ms(self, interaction_mode: str = "auto") -> int:
        cfg = self._cfg()
        if interaction_mode == "mentor":
            return min(cfg.mentor_cooldown_ms, cfg.cooldown_ms_max)
        return min(cfg.cooldown_ms, cfg.cooldown_ms_max)

    def _check_cool_down(self, interaction_mode: str = "auto") -> bool:
        last_push = self.state.get("last_push_ts")
        if last_push == 0:
            return True
        now_ms = int(time.time() * 1000)
        return (now_ms - last_push) >= self._effective_cooldown_ms(interaction_mode)

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
        recent_ids = set(self.state.get("push_history"))
        return [c for c in candidates if c.get("video_id") not in recent_ids]

    def _client_softmax_sample(self, candidates: List[Dict]) -> Optional[Dict]:
        cfg = self._cfg()
        tau = self.state.get("tau")
        if tau is None or tau <= 0:
            tau = cfg.tau_init
        scores = [c.get("prob_base", 0) for c in candidates]
        if not scores:
            return None

        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / tau) for s in scores]
        total = sum(exp_scores) + 1e-8
        probs = [e / total for e in exp_scores]

        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return candidates[i]

        return candidates[-1]

    def _pick_video_from_pool(self, candidates: List[Dict]) -> Optional[Dict]:
        """语义明显领先时取 top1，否则 softmax 保持多样性。"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        top_sim = float(candidates[0].get("similarity", 0))
        second_sim = float(candidates[1].get("similarity", 0))
        if top_sim >= second_sim + 0.10:
            return candidates[0]
        return self._client_softmax_sample(candidates)

    def _is_active_video_request(self, user_message: str) -> bool:
        return is_video_request(user_message)

    def _is_advisor_request(self, user_message: str) -> bool:
        return is_advisor_request(user_message)

    def _is_breakup_video(self, candidate: Dict) -> bool:
        title = candidate.get("title") or ""
        return any(k in title for k in BREAKUP_VIDEO_KWS)

    def _user_has_breakup_intent(self, user_message: str) -> bool:
        return any(k in user_message for k in BREAKUP_USER_KWS)

    def _filter_intent_mismatch(
        self, user_message: str, candidates: List[Dict]
    ) -> List[Dict]:
        if self._user_has_breakup_intent(user_message):
            return candidates
        return [c for c in candidates if not self._is_breakup_video(c)]

    def _auto_video_target_rate(self) -> float:
        cfg = self._cfg()
        if cfg.auto_video_target_rate is not None:
            return float(cfg.auto_video_target_rate)
        return cfg.target_push_rate_per_10 / 10.0

    def _auto_video_quota_status(self) -> tuple[bool, int, int]:
        msg_count = self.state.get("session_msg_count")
        allowed = round(msg_count * self._auto_video_target_rate())
        current = int(self.state.get("auto_video_push_count") or 0)
        return current < allowed, allowed, current

    def _record_auto_video_push(self) -> None:
        msg_count = self.state.get("session_msg_count")
        current = int(self.state.get("auto_video_push_count") or 0)
        self.state.set("auto_video_push_count", current + 1)
        self.state.set("last_auto_video_push_msg", msg_count)

    def _should_quota_force_auto_video(
        self,
        top1_sim: float,
        interaction_mode: str,
        dedup: List[Dict],
        filtered: List[Dict],
    ) -> bool:
        cfg = self._cfg()
        ok, allowed, current = self._auto_video_quota_status()
        if not ok:
            return False

        pool = dedup or filtered
        if not pool:
            return False

        min_sim = cfg.auto_video_quota_min_similarity
        if float(top1_sim) < min_sim:
            return False

        msg_count = self.state.get("session_msg_count")
        last_msg = int(self.state.get("last_auto_video_push_msg") or 0)
        cooldown_msgs = cfg.auto_video_quota_cooldown_msgs
        behind = (allowed - current) >= 2

        if last_msg > 0 and (msg_count - last_msg) < cooldown_msgs and not behind:
            return False

        if not behind and not self._check_cool_down(interaction_mode):
            return False

        return True

    def _emit_auto_video_push(
        self,
        selected: Dict,
        A: float,
        N: int,
        *,
        quota_forced: bool = False,
    ) -> VideoPushEvent:
        now_ms = int(time.time() * 1000)
        self.state.set("urge", 0)
        self.state.set("last_push_ts", now_ms)
        self.state.add_push_history(selected["video_id"])
        self._record_auto_video_push()
        data = {
            "video_id": selected["video_id"],
            "title": selected.get("title", ""),
            "cover_url": selected.get("cover_url", ""),
            "duration_sec": selected.get("duration_sec", DEFAULT_VIDEO_DURATION_SEC),
            "trigger": "auto",
            "acceptance": round(A, 4),
            "push_count": N,
        }
        if selected.get("vod_video_id"):
            data["vod_video_id"] = selected["vod_video_id"]
        if quota_forced:
            data["quota_forced"] = True
        return VideoPushEvent(data=data)

    def _evaluate_video(
        self,
        user_message: str,
        candidates: Optional[List[Dict]],
        conversation_state: str = "INIT",
        interaction_mode: str = "auto",
    ) -> List[PushEvent]:
        cfg = self._cfg()
        events = []

        is_active = self._is_active_video_request(user_message)

        if is_active:
            if candidates:
                filtered = self._filter_intent_mismatch(user_message, candidates)
                pool = filtered or candidates
                # Explicit content request: always best relevance-ranked item (allow re-push).
                selected = pool[0]

                urge = self.state.get("urge")
                self.state.set("urge", urge * cfg.urge_active_video_decay)

                self.state.set("last_push_ts", int(time.time() * 1000))
                self.state.add_push_history(selected["video_id"])

                events.append(VideoPushEvent(data={
                    "video_id": selected["video_id"],
                    "title": selected.get("title", ""),
                    "cover_url": selected.get("cover_url", ""),
                    "duration_sec": selected.get("duration_sec", DEFAULT_VIDEO_DURATION_SEC),
                    "trigger": "active",
                    **({"vod_video_id": selected["vod_video_id"]} if selected.get("vod_video_id") else {}),
                }))
            return events

        if not candidates:
            return events

        msg_count = self.state.get("session_msg_count")
        min_msgs = max(cfg.auto_push_min_session_msgs, 1)
        if msg_count < min_msgs:
            return events

        candidates = self._filter_intent_mismatch(user_message, candidates)
        if not candidates:
            return events

        dedup = self._dedup_candidates(candidates)
        pool = dedup if dedup else candidates
        if not pool:
            return events

        top1_sim = pool[0].get("similarity", 0)

        if not self._check_cool_down(interaction_mode):
            if not self._should_quota_force_auto_video(
                top1_sim, interaction_mode, dedup, candidates
            ):
                return events

        last_push = self.state.get("last_push_ts")
        now_ms = int(time.time() * 1000)
        recent_count = 0
        if last_push > 0 and (now_ms - last_push) < cfg.recent_push_window_ms:
            recent_count = 1

        A = self._compute_acceptance_a(
            top1_sim, recent_count, conversation_state, interaction_mode=interaction_mode
        )

        N = self._map_push_count(A, len(pool))
        quota_forced = False
        if N == 0 and self._should_quota_force_auto_video(
            top1_sim, interaction_mode, dedup, candidates
        ):
            N = 1
            quota_forced = True

        if N == 0:
            return events

        candidates_pool = pool[:N]

        selected = self._pick_video_from_pool(candidates_pool)
        if selected is None:
            return events

        events.append(
            self._emit_auto_video_push(selected, A, N, quota_forced=quota_forced)
        )

        return events

    def _auto_advisor_target_rate(self) -> float:
        cfg = self._cfg()
        if cfg.auto_advisor_target_rate is not None:
            return float(cfg.auto_advisor_target_rate)
        return 0.25

    def _advisor_quota_effective_rate(self) -> float:
        """Compensate for video/advisor mutual exclusion on the same turn."""
        cfg = self._cfg()
        if cfg.auto_advisor_effective_rate is not None:
            return float(cfg.auto_advisor_effective_rate)
        advisor = self._auto_advisor_target_rate()
        video = self._auto_video_target_rate()
        divisor = max(1.0 - video, 0.25)
        return min(advisor / divisor, 0.5)

    def _auto_advisor_quota_status(self) -> tuple[bool, int, int]:
        cfg = self._cfg()
        msg_count = self.state.get("session_msg_count")
        allowed = round(msg_count * self._advisor_quota_effective_rate())
        current = int(self.state.get("auto_advisor_push_count") or 0)
        min_msgs = max(cfg.auto_advisor_quota_min_session_msgs, cfg.auto_push_min_session_msgs, 1)
        if msg_count < min_msgs:
            return False, allowed, current
        return current < allowed, allowed, current

    def _record_auto_advisor_push(self) -> None:
        msg_count = self.state.get("session_msg_count")
        current = int(self.state.get("auto_advisor_push_count") or 0)
        self.state.set("auto_advisor_push_count", current + 1)
        self.state.set("last_auto_advisor_push_msg", msg_count)

    def _should_quota_force_auto_advisor(
        self,
        interaction_mode: str,
        advisor_suggestion: bool,
    ) -> bool:
        if interaction_mode == "proxy_chat":
            return False
        if self._cfg().auto_advisor_target_rate is None:
            return False

        ok, allowed, current = self._auto_advisor_quota_status()
        if not ok:
            return False

        cfg = self._cfg()
        msg_count = self.state.get("session_msg_count")
        last_msg = int(self.state.get("last_auto_advisor_push_msg") or 0)
        cooldown_msgs = cfg.auto_advisor_quota_cooldown_msgs
        behind = (allowed - current) >= 1

        if last_msg > 0 and (msg_count - last_msg) < cooldown_msgs and not behind:
            return False

        if advisor_suggestion and (allowed - current) >= 1:
            return True

        return True

    def _emit_auto_advisor_push(
        self,
        *,
        is_active_request: bool,
        quota_forced: bool = False,
    ) -> AdvisorPushEvent:
        now_ms = int(time.time() * 1000)
        self.state.set("last_advisor_push_ts", now_ms)
        count = self.state.get("total_advisor_push_count")
        self.state.set("total_advisor_push_count", count + 1)
        self._record_auto_advisor_push()
        data = {
            "title": "需要更专业的帮助？",
            "subtitle": "一键连线情感导师，获得1对1指导",
            "link": DEFAULT_ADVISOR_LINK,
            "trigger": "active" if is_active_request else "auto",
        }
        if quota_forced:
            data["quota_forced"] = True
        return AdvisorPushEvent(data=data)

    def _advisor_cooldown_ms(self, interaction_mode: str = "auto") -> int:
        cfg = self._cfg()
        base = int(cfg.advisor_push_cooldown_ms)
        if interaction_mode == "mentor":
            return min(int(cfg.mentor_cooldown_ms), base)
        return base

    def _advisor_session_cap(self, interaction_mode: str = "auto") -> int:
        cfg = self._cfg()
        base = max(int(cfg.advisor_session_push_cap), 1)
        if interaction_mode == "proxy_chat" or cfg.auto_advisor_target_rate is None:
            return base
        _, allowed, _ = self._auto_advisor_quota_status()
        if allowed <= 0:
            return 1
        return min(base, allowed)

    def _evaluate_advisor(
        self,
        user_message: str,
        positive_rate: float,
        has_video_push: bool,
        advisor_suggestion: bool = False,
        risk_alert: bool = False,
        interaction_mode: str = "auto",
    ) -> List[PushEvent]:
        events = []

        last_ts = self.state.get("last_advisor_push_ts")
        now_ms = int(time.time() * 1000)

        is_active_request = self._is_advisor_request(user_message)
        quota_forced = False
        under_quota, _, _ = self._auto_advisor_quota_status()

        if not is_active_request and not risk_alert:
            if not under_quota:
                cooldown_ms = self._advisor_cooldown_ms(interaction_mode)
                if (now_ms - last_ts) < cooldown_ms:
                    return events
                if self.state.get("total_advisor_push_count") >= self._advisor_session_cap(
                    interaction_mode
                ):
                    return events

        triggered = False

        if risk_alert:
            triggered = True

        elif is_active_request:
            triggered = True
            self.state.set("last_advisor_push_ts", 0)

        elif self._check_helpless(user_message) and self.state.get("urge") >= 0.25 and under_quota:
            triggered = True

        elif self._check_sensitive_urge(user_message) and under_quota:
            triggered = True

        elif advisor_suggestion and self.state.get("urge") >= 0.15 and under_quota:
            triggered = True

        elif self._should_quota_force_auto_advisor(interaction_mode, advisor_suggestion):
            triggered = True
            quota_forced = True

        if triggered:
            if has_video_push:
                return events

            if is_active_request or risk_alert:
                self.state.set("last_advisor_push_ts", now_ms)
                if not risk_alert:
                    count = self.state.get("total_advisor_push_count")
                    self.state.set("total_advisor_push_count", count + 1)
                    self._record_auto_advisor_push()
                events.append(AdvisorPushEvent(data={
                    "title": "需要更专业的帮助？",
                    "subtitle": "一键连线情感导师，获得1对1指导",
                    "link": DEFAULT_ADVISOR_LINK,
                    "trigger": "active" if is_active_request else "auto",
                }))
            else:
                events.append(
                    self._emit_auto_advisor_push(
                        is_active_request=is_active_request,
                        quota_forced=quota_forced,
                    )
                )

        return events

    def _check_helpless(self, user_message: str) -> bool:
        return any(kw in user_message for kw in HELPLESS_KEYWORDS)

    def _check_sensitive_urge(self, user_message: str) -> bool:
        has_sensitive = any(t in user_message for t in SENSITIVE_TOPICS)
        urge = self.state.get("urge")
        return has_sensitive and urge >= 0.7

    def _check_fixed_round(self, advisor_suggestion: bool = False) -> bool:
        count = self.state.get("session_msg_count")
        threshold = 10 if advisor_suggestion else 20
        return count > 0 and count % threshold == 0

    def _apply_feedback(self, feedback_type: str):
        cfg = self._cfg()
        tau = self.state.get("tau")
        if tau is None or tau <= 0:
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

    def record_video_feedback(self, video_id: str, feedback_type: str):
        """反馈类型：positive, negative, strong_negative"""
        self._apply_feedback(feedback_type)

    def record_video_engagement(
        self,
        video_id: str,
        event: str,
        watch_duration_sec: float = 0,
        video_duration_sec: float = DEFAULT_VIDEO_DURATION_SEC,
        decision_config: Optional[DecisionConfig] = None,
    ) -> dict:
        if decision_config is not None:
            self._active_config = decision_config
        was_clicked = video_id in self._video_clicked

        if event == "click":
            self.state.record_click()
            self._video_clicked.add(video_id)
            return {
                "feedback_type": None,
                "reason": "click_recorded",
                "click_rate": round(self.state.get_click_rate(), 4),
                "watch_ratio": 0.0,
            }

        consecutive_negative = self.state.get("consecutive_negative")
        feedback_type, reason = classify_engagement(
            event=event,
            watch_duration_sec=watch_duration_sec,
            video_duration_sec=video_duration_sec,
            consecutive_negative=consecutive_negative,
            was_clicked=was_clicked or event == "watch_end",
        )

        video_duration_sec = max(video_duration_sec, 1.0)
        watch_ratio = round(watch_duration_sec / video_duration_sec, 4)

        if feedback_type:
            self._apply_feedback(feedback_type)
            self._video_clicked.discard(video_id)

        return {
            "feedback_type": feedback_type,
            "reason": reason,
            "click_rate": round(self.state.get_click_rate(), 4),
            "watch_ratio": watch_ratio,
        }

    def get_state_summary(self) -> dict:
        cfg = self._cfg()
        return {
            "urge": self.state.get("urge"),
            "tau": self.state.get("tau"),
            "feedbacks": (
                self.state.get("positive_feedbacks")
                / max(self.state.get("total_exposures"), 1)
                if self.state.get("total_exposures") > 0
                else cfg.f_default_new_user
            ),
            "click_rate": round(self.state.get_click_rate(), 4),
            "consecutive_negative": self.state.get("consecutive_negative"),
            "session_msg_count": self.state.get("session_msg_count"),
            "dynamic_k": self._compute_dynamic_k(),
        }
