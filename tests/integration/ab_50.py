"""50-round A/B comparison test: Original vs Optimized push parameters."""
import sys
import os
import time
import random
import copy
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.vector_service import VectorService
from server.advisor_service import AdvisorService
from mobile.decision_engine import DecisionEngine
from mobile.state_manager import StateManager

random.seed(42)

# ── 50-round conversation simulating a full relationship arc ──
messages_50 = [
    # Phase 1: Initial conflict (1-8)
    "今天和女朋友吵架了，怎么办",
    "她一直不理我，发了好多消息也不回",
    "异地恋真的好难，感觉坚持不下去了",
    "我们在一起两年多了，最近总是冷战",
    "有什么办法能打破冷战吗",
    "我也想尝试主动沟通，但不知道该说什么",
    "可以推荐一些视频看看吗",
    "刚才看的那个视频确实有道理，谢谢",
    # Phase 2: Emotional turbulence (9-18)
    "但是我还是有点难过，总是想起以前的事",
    "她开始和别人聊天了，我感觉被背叛了",
    "我该怎么办，感觉走不下去了",
    "帮不了我啊，心里好难受",
    "我梦到她和别人在一起了",
    "我感觉自己很没用，配不上她",
    "我崩溃了，救救我",
    "有没有挽回感情的方法",
    "我想找个真人老师好好聊聊",
    "也许我太敏感了，应该放下过去",
    # Phase 3: Reflection & recovery (19-28)
    "要学会爱自己才能爱别人对吧",
    "要不要就这样算了，开始新生活",
    "最近开始去健身了，心情好了一些",
    "朋友约我出去吃饭，感觉好多了",
    "今天工作也很顺利，状态不错",
    "想起她还是有点难受，但能控制住了",
    "看了你推荐的那个关于自我成长的视频",
    "确实，先把自己过好才能有好的感情",
    "她主动找我聊天了，有点意外",
    "不知道该不该回复她",
    # Phase 4: Reconciliation attempt (29-38)
    "我还是很想她，要不要主动一点",
    "我们约好周末见面了，有点紧张",
    "该怎么准备这次见面",
    "穿什么衣服好呢",
    "见面之前我应该说什么",
    "万一又吵架怎么办",
    "我不想再失去她了",
    "有推荐视频吗",
    "谢谢，这个建议很实用",
    "明天就要见面了，睡不着",
    # Phase 5: New dynamics (39-50)
    "见面还不错，我们聊了很多",
    "她说她也想了很多，我们都需要改变",
    "这算是复合了吗",
    "但我担心又会回到以前的样子",
    "怎么让这次不一样",
    "你推荐的那个沟通技巧视频很有用",
    "我们约定了每周视频一次",
    "感觉关系比以前更好了",
    "谢谢你一直以来的帮助",
    "我学会了很多，真的很感谢",
    "以后有什么问题再来找你",
    "今天和她一起看了个电影，很开心",
]


class Config:
    """Parameter config for a test run."""
    def __init__(self, name, **kw):
        self.name = name
        self._kw = kw

    def get(self, key, default=None):
        return self._kw.get(key, default)


# ── Original config (pre-optimization) ──
CONFIG_A = Config("A-原始参数",
    # A formula
    w_I=0.30, w_L=0.20, w_F=0.20, w_S=0.10, w_U=0.20,
    # push count thresholds
    th_0=0.30, th_1=0.55, th_2=0.80,
    # urge
    urge_delta=0.15,
    # cooldown
    cooldown_ms=45000,
    # tau
    tau_init=0.60,
    tau_positive=0.92,
    tau_negative=1.08,
    # cool down after strong negative
    cool_down_after_strong=3,
)

# ── Optimized config ──
CONFIG_B = Config("B-强化优化",
    w_I=0.18, w_L=0.22, w_F=0.32, w_S=0.10, w_U=0.18,
    th_0=0.68, th_1=0.80, th_2=0.90,
    urge_delta=0.05,
    cooldown_ms=90000,
    tau_init=0.32,
    tau_positive=0.95,
    tau_negative=1.04,
    cool_down_after_strong=6,
)


class ConfigurableDecisionEngine(DecisionEngine):
    """DecisionEngine with overridable parameters via config."""

    def __init__(self, config: Config, state: Optional[StateManager] = None):
        super().__init__(state)
        self.cfg = config

    def _compute_acceptance_a(self, top1_similarity, recent_push_count_5min=0):
        I = min(top1_similarity, 1.0)
        L = 1.0 - min(recent_push_count_5min / 3.0, 1.0)
        total_exposures = self.state.get("total_exposures")
        F = 0.5 if total_exposures == 0 else max(0.3, self.state.get("positive_feedbacks") / max(total_exposures, 1))
        S = min(self.state.get("session_msg_count") / 30.0, 1.0)
        U = self.state.get("urge")
        A = (self.cfg.get("w_I") * I + self.cfg.get("w_L") * L +
             self.cfg.get("w_F") * F + self.cfg.get("w_S") * S +
             self.cfg.get("w_U") * U)
        return A

    def _map_push_count(self, A, max_candidates):
        if A < self.cfg.get("th_0"):
            return 0
        if A < self.cfg.get("th_1"):
            return 1
        if A < self.cfg.get("th_2"):
            return 2
        return min(3, max_candidates)

    def process_user_message(self, user_message, candidates=None, positive_rate=0.6):
        events = []
        self.state.add_message(time.time())
        if self.state.get("cool_down_remain") > 0:
            remain = self.state.get("cool_down_remain") - 1
            self.state.set("cool_down_remain", remain)
            return events

        urge = self.state.get("urge")
        urge = min(urge + self.cfg.get("urge_delta"), 1.0)
        self.state.set("urge", urge)

        video_events = self._evaluate_video(user_message, candidates)
        events.extend(video_events)
        advisor_events = self._evaluate_advisor(user_message, positive_rate, len(video_events) > 0)
        events.extend(advisor_events)
        return events

    def _check_cool_down(self):
        last_push = self.state.get("last_push_ts")
        if last_push == 0:
            return True
        now_ms = int(time.time() * 1000)
        return (now_ms - last_push) >= self.cfg.get("cooldown_ms")

    def record_video_feedback(self, video_id, feedback_type):
        tau = self.state.get("tau")
        if feedback_type == "positive":
            tau = max(0.3, tau * self.cfg.get("tau_positive"))
            self.state.record_feedback(True)
        elif feedback_type == "negative":
            tau = min(2.0, tau * self.cfg.get("tau_negative"))
            self.state.record_feedback(False)
        elif feedback_type == "strong_negative":
            self.state.set("urge", 0)
            self.state.set("cool_down_remain", self.cfg.get("cool_down_after_strong"))
            self.state.record_feedback(False)
        self.state.set("tau", tau)


def run_simulation(config: Config, vs: VectorService, advisor: AdvisorService, seed: int):
    """Run 50-round simulation with given config."""
    random.seed(seed)

    state = StateManager()
    state._state_file += f".{config.name.replace(' ', '_')}"
    # Reset to clean defaults with config tau
    state._state.update(state._defaults)
    state.set("tau", config.get("tau_init"))
    state.set("urge", 0.5)
    state.reset_session()

    engine = ConfigurableDecisionEngine(config, state)

    total_video = 0
    total_advisor = 0
    round_records = []
    push_timestamps = []
    tau_series = [config.get("tau_init")]
    urge_series = [0.5]

    for i, msg in enumerate(messages_50):
        r = i + 1

        # Advance time to clear cooldown (simulate real chatting pace)
        last_push = state.get("last_push_ts")
        if last_push > 0:
            state.set("last_push_ts", last_push - (config.get("cooldown_ms") + 5000))

        candidates, ctx_vec, top1_sim = vs.process_request(msg)
        positive_rate = engine.get_state_summary()["feedbacks"]
        advisor_suggest = advisor.check_advisor_suggestion(msg, ctx_vec, positive_rate)

        candidate_dicts = [
            {"video_id": c.video_id, "similarity": c.similarity,
             "prob_base": c.prob_base, "title": c.title, "cover_url": c.cover_url}
            for c in candidates
        ]

        events = engine.process_user_message(msg, candidate_dicts, positive_rate)
        video_evts = [e for e in events if e.type == "video_push"]
        advisor_evts = [e for e in events if e.type == "advisor_push"]

        # Add advisor events triggered by server-side suggestion
        extra_advisor = engine._evaluate_advisor(
            msg, positive_rate, len(video_evts) > 0, advisor_suggestion=advisor_suggest)
        existing = {e.data.get("trigger") for e in advisor_evts}
        for e in extra_advisor:
            if e.data.get("trigger") not in existing:
                advisor_evts.append(e)

        total_video += len(video_evts)
        total_advisor += len(advisor_evts)

        if video_evts:
            for _ in video_evts:
                push_timestamps.append(r)

        # Simulate feedback
        for ve in video_evts:
            vid = ve.data.get("video_id", "")
            rv = random.random()
            if rv < 0.5:
                engine.record_video_feedback(vid, "positive")
            elif rv < 0.85:
                engine.record_video_feedback(vid, "negative")
            else:
                engine.record_video_feedback(vid, "strong_negative")

        state_s = engine.get_state_summary()
        tau_series.append(state_s["tau"])
        urge_series.append(state_s["urge"])

        round_records.append({
            "r": r, "vp": len(video_evts), "ap": len(advisor_evts),
            "urge": state_s["urge"], "tau": state_s["tau"],
        })

    # Cleanup
    if os.path.exists(state._state_file):
        os.remove(state._state_file)

    return {
        "total_video": total_video,
        "total_advisor": total_advisor,
        "records": round_records,
        "push_timestamps": push_timestamps,
        "tau_final": tau_series[-1],
        "tau_series": tau_series,
        "urge_series": urge_series,
        "push_rate": total_video / 50,
    }


def print_line(char="-", w=90):
    print(char * w)


def main():
    print("Loading VectorService...")
    vs = VectorService()
    vs.load()
    advisor = AdvisorService(vs.help_topic_vec)
    print("Done.\n")

    print_line("=")
    print("  A/B Test: 50-round comparison (A=Original vs B=Strengthened)")
    print_line("=")

    result_a = run_simulation(CONFIG_A, vs, advisor, seed=42)
    result_b = run_simulation(CONFIG_B, vs, advisor, seed=42)

    # ── Summary ──
    print(f"\n{'Metric':<28} {'A-原始':<18} {'B-优化':<18} {'变化':<15}")
    print_line("-")
    for label, key, fmt in [
        ("Total rounds", "N/A", None),
        ("Total video pushes", "total_video", "d"),
        ("Total advisor pushes", "total_advisor", "d"),
        ("Video push rate", "push_rate", ".1%"),
        ("Final tau", "tau_final", ".4f"),
        ("Avg pushes per 10 rounds", "N/A", None),
        ("Max consecutive no-push", "N/A", None),
    ]:
        if key == "N/A":
            if label == "Total rounds":
                print(f"{label:<28} {50:<18} {50:<18}")
            elif label == "Avg pushes per 10 rounds":
                print(f"{label:<28} {result_a['total_video']/5:<18.1f} {result_b['total_video']/5:<18.1f}")
            elif label == "Max consecutive no-push":
                def max_no_push(recs):
                    mx = cur = 0
                    for r in recs:
                        if r["vp"] == 0:
                            cur += 1
                        else:
                            cur = 0
                        mx = max(mx, cur)
                    return mx
                print(f"{label:<28} {max_no_push(result_a['records']):<18} {max_no_push(result_b['records']):<18}")
            continue
        val_a = result_a[key]
        val_b = result_b[key]
        if fmt == ".1%":
            val_a_str = f"{val_a:.1%}"
            val_b_str = f"{val_b:.1%}"
            change = f"{(val_b - val_a):.1%}"
        elif fmt == "d":
            val_a_str = str(val_a)
            val_b_str = str(val_b)
            change = f"{(val_b - val_a):+d}"
        elif fmt == ".4f":
            val_a_str = f"{val_a:.4f}"
            val_b_str = f"{val_b:.4f}"
            change = f"{(val_b - val_a):+.4f}"
        print(f"{label:<28} {val_a_str:<18} {val_b_str:<18} {change:<15}")

    # ── Round-by-round comparison ──
    print(f"\n{'R':<4} {'Phase':<20} {'A-VP':<6} {'A-AP':<6} {'A-urge':<8} {'A-tau':<8} | {'B-VP':<6} {'B-AP':<6} {'B-urge':<8} {'B-tau':<8}")
    print_line("-")
    phases = [
        (1, 8, "Initial Conflict"),
        (9, 18, "Emotional Turbulence"),
        (19, 28, "Reflection & Recovery"),
        (29, 38, "Reconciliation"),
        (39, 50, "New Dynamics"),
    ]

    for i in range(50):
        ra = result_a["records"][i]
        rb = result_b["records"][i]
        r = ra["r"]
        phase_name = ""
        for ps, pe, pn in phases:
            if ps <= r <= pe:
                if r == ps:
                    phase_name = pn
                break
        print(f"{r:<4} {phase_name:<20} {ra['vp']:<6} {ra['ap']:<6} "
              f"{ra['urge']:<8.2f} {ra['tau']:<8.4f} | "
              f"{rb['vp']:<6} {rb['ap']:<6} "
              f"{rb['urge']:<8.2f} {rb['tau']:<8.4f}")

    # ── Phase summary ──
    print(f"\n{'Phase':<25} {'A-pushes':<10} {'B-pushes':<10} {'B/A':<8}")
    print_line("-")
    for ps, pe, pn in phases:
        a_cnt = sum(1 for rd in result_a["records"][ps-1:pe] if rd["vp"] > 0)
        b_cnt = sum(1 for rd in result_b["records"][ps-1:pe] if rd["vp"] > 0)
        ratio = f"{b_cnt/a_cnt:.1%}" if a_cnt > 0 else "N/A"
        print(f"{pn:<25} {a_cnt:<10} {b_cnt:<10} {ratio:<8}")


if __name__ == "__main__":
    main()
