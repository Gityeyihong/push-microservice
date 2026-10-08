"""User specificity test: same messages, different user feedback personalities."""
import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.vector_service import VectorService
from server.advisor_service import AdvisorService
from ab_50 import ConfigurableDecisionEngine, Config
from mobile.state_manager import StateManager

random.seed(42)

ROUNDS = 40

NEUTRAL_MESSAGES = [
    "你好，我最近感情遇到了一些问题",
    "和对象吵架了，不知道怎么处理",
    "冷战好几天了，他不理我",
    "我该主动找他吗，还是等他来",
    "朋友说我们性格不合，你说呢",
    "异地恋真的很考验人",
    "有时候觉得感情好累",
    "但又不舍得放弃",
    "有什么建议吗",
    "看了你推荐的技巧，尝试了一下",
    "效果还可以，但还是有点波折",
    "我们约了明天见面聊聊",
    "见面聊得还不错，但问题没完全解决",
    "感觉他对我不够上心",
    "我是不是太敏感了",
    "他的前女友最近又出现了",
    "我很不安，但不知道该怎么表达",
    "他总说我想多了",
    "我需要一些专业建议",
    "最近也开始自我反思",
    "也许问题不完全在他",
    "我也有需要改进的地方",
    "能给我一些自我提升的建议吗",
    "谢谢这段时间的帮助",
    "我会继续努力的",
    "今天看了个视频很有启发",
    "试着用新的方式和对象沟通",
    "没想到效果还不错",
    "他开始主动关心我了",
    "但我还是有点担心反弹",
    "感情真的需要经营",
    "以前我太自我了",
    "现在学会换位思考",
    "关系在慢慢变好",
    "但偶尔还会有小摩擦",
    "不过比之前好多了",
    "你推荐的视频我都看了",
    "每个都有收获",
    "想再要一些建议巩固一下",
    "非常感谢，我会继续进步的",
]

ARCHETYPES = {
    "A-理性批判": {
        "desc": "Mostly negative, critical thinker",
        "fb": {"positive": 0.15, "negative": 0.65, "strong_negative": 0.05, "none": 0.15},
    },
    "B-热情满意": {
        "desc": "Mostly positive, happy user",
        "fb": {"positive": 0.72, "negative": 0.08, "strong_negative": 0.02, "none": 0.18},
    },
    "C-沉默被动": {
        "desc": "Rare feedback, mostly lurks",
        "fb": {"positive": 0.05, "negative": 0.05, "strong_negative": 0.02, "none": 0.88},
    },
    "D-极端反复": {
        "desc": "Alternates extreme, volatile",
        "fb": {"positive": 0.28, "negative": 0.22, "strong_negative": 0.22, "none": 0.28},
    },
    "E-渐进接受": {
        "desc": "Starts negative, gradually positive",
        "fb": None,
    },
    "F-快速满意": {
        "desc": "First 3 pushes positive, then silent",
        "fb": None,
    },
}


def get_feedback(uid, push_num, round_num, rng):
    fb_map = ARCHETYPES[uid]["fb"]

    if uid == "E-渐进接受":
        ratio = min(round_num / ROUNDS, 1.0)
        pw = 0.10 + 0.70 * ratio
        nw = 0.65 - 0.55 * ratio
        sw = 0.10 * (1 - ratio)
        now = 0.15
        total = pw + nw + sw + now
        r = rng.random()
        if r < pw / total: return "positive"
        if r < (pw + nw) / total: return "negative"
        if r < (pw + nw + sw) / total: return "strong_negative"
        return "none"

    if uid == "F-快速满意":
        if push_num <= 3:
            return "positive"
        return "none"

    r = rng.random()
    cum = 0
    for fb_type in ["positive", "negative", "strong_negative", "none"]:
        cum += fb_map[fb_type]
        if r < cum:
            return fb_type
    return "none"


def run_user(uid, vs, advisor, cfg, seed):
    rng = random.Random(seed)
    state = StateManager()
    sf = os.path.join(os.path.dirname(state._state_file),
                      f"test_user_{uid.replace('-','_')}.json")
    state._state_file = sf
    state._state.update(state._defaults)
    state.set("tau", cfg.get("tau_init"))
    state.set("urge", 0.5)
    state.reset_session()

    engine = ConfigurableDecisionEngine(cfg, state)

    total_video = 0
    total_advisor = 0
    push_rounds = []
    advisor_rounds = []
    tau_trace = [cfg.get("tau_init")]
    urge_trace = [0.5]
    a_trace = []
    fb_trace = []
    push_count_local = 0

    for i, msg in enumerate(NEUTRAL_MESSAGES):
        r = i + 1
        lp = state.get("last_push_ts")
        if lp > 0:
            state.set("last_push_ts", lp - (cfg.get("cooldown_ms") + 5000))

        candidates, ctx_vec, top1_sim = vs.process_request(msg)
        pos_rate = engine.get_state_summary()["feedbacks"]
        ad_suggest = advisor.check_advisor_suggestion(msg, ctx_vec, pos_rate)

        cd = [{"video_id": c.video_id, "similarity": c.similarity,
               "prob_base": c.prob_base, "title": c.title, "cover_url": c.cover_url}
              for c in candidates]

        events = engine.process_user_message(msg, cd, pos_rate)
        ve = [e for e in events if e.type == "video_push"]
        ae = [e for e in events if e.type == "advisor_push"]

        extra = engine._evaluate_advisor(msg, pos_rate, len(ve) > 0,
                                         advisor_suggestion=ad_suggest)
        et = {e.data.get("trigger") for e in ae}
        for e in extra:
            if e.data.get("trigger") not in et:
                ae.append(e)

        vp = len(ve)
        ap = len(ae)
        total_video += vp
        total_advisor += ap
        if vp > 0:
            push_rounds.append(r)
        if ap > 0:
            advisor_rounds.append(r)

        for ev in ve:
            av = ev.data.get("acceptance")
            if av is not None:
                a_trace.append(av)

        for ev in ve:
            push_count_local += 1
            fb = get_feedback(uid, push_count_local, r, rng)
            fb_trace.append(fb)
            vid = ev.data.get("video_id", "")
            if fb == "positive":
                engine.record_video_feedback(vid, "positive")
            elif fb == "negative":
                engine.record_video_feedback(vid, "negative")
            elif fb == "strong_negative":
                engine.record_video_feedback(vid, "strong_negative")

        ss = engine.get_state_summary()
        tau_trace.append(ss["tau"])
        urge_trace.append(ss["urge"])

    if os.path.exists(sf):
        os.remove(sf)

    avg_gap = 0
    if total_video >= 2:
        avg_gap = (max(push_rounds) - min(push_rounds)) / (total_video - 1)
    pc = sum(1 for f in fb_trace if f == "positive")
    nc = sum(1 for f in fb_trace if f == "negative")
    sc = sum(1 for f in fb_trace if f == "strong_negative")

    return {
        "uid": uid, "desc": ARCHETYPES[uid]["desc"],
        "total_video": total_video, "total_advisor": total_advisor,
        "push_rate": total_video / ROUNDS,
        "push_rounds": push_rounds, "advisor_rounds": advisor_rounds,
        "tau_start": tau_trace[0], "tau_end": tau_trace[-1],
        "tau_trace": tau_trace, "urge_trace": urge_trace,
        "a_trace": a_trace, "fb_trace": fb_trace,
        "avg_gap": avg_gap, "pos_fb": pc, "neg_fb": nc, "strong_fb": sc,
        "final_f_rate": ss["feedbacks"],
    }


def pl(char="-", w=100):
    print(char * w)


def main():
    print("Loading VectorService (100 videos)...")
    vs = VectorService()
    vs.load()
    advisor = AdvisorService(vs.help_topic_vec)
    print(f"  Loaded: {len(vs.video_metadata)} videos\n")

    pl("=")
    print(f"  User Specificity: 6 Archetypes x Same {ROUNDS}-Round Script")
    pl("=")

    cfg = Config("spec",
        w_I=0.18, w_L=0.22, w_F=0.32, w_S=0.10, w_U=0.18,
        th_0=0.68, th_1=0.80, th_2=0.90,
        urge_delta=0.05, cooldown_ms=90000,
        tau_init=0.32, tau_positive=0.95, tau_negative=1.04,
        cool_down_after_strong=6,
    )

    results = {}
    for idx, (uid, info) in enumerate(ARCHETYPES.items()):
        print(f"  [{uid}] {info['desc']} ... ", end="", flush=True)
        start = time.time()
        r = run_user(uid, vs, advisor, cfg, seed=100 + idx * 7)
        elapsed = time.time() - start
        print(f"{r['total_video']} pushes, tau={r['tau_start']:.2f}->{r['tau_end']:.3f} ({elapsed:.2f}s)")
        results[uid] = r

    # ── MAIN TABLE ──
    pl("=")
    print("  COMPARISON TABLE")
    pl("-")
    print(f"  {'User':<14} {'Pushes':<8} {'Rate':<7} {'Tau':<13} {'F-Rate':<8} "
          f"{'AvgA':<7} {'Gap':<6} {'FB(+/-/! )':<15} {'PushRounds (first 8)'}")
    pl("-")
    for uid in ARCHETYPES:
        r = results[uid]
        fb = f"{r['pos_fb']}/{r['neg_fb']}/{r['strong_fb']}"
        pr = ",".join(str(p) for p in r["push_rounds"][:8])
        if len(r["push_rounds"]) > 8:
            pr += f" +{len(r['push_rounds'])-8}"
        apv = sum(r["a_trace"]) / max(len(r["a_trace"]), 1)
        print(f"  {uid:<14} {r['total_video']:<8} {r['push_rate']:<7.0%} "
              f"{r['tau_start']:.2f}->{r['tau_end']:.3f}  "
              f"{r['final_f_rate']:<8.2f} "
              f"{apv:<7.4f} {r['avg_gap']:<6.1f} {fb:<15} {pr}")

    # ── DIVERGENCE ──
    pl("=")
    print("  DIVERGENCE METRICS")
    pl("-")
    av = [r["total_video"] for r in results.values()]
    at = [r["tau_end"] for r in results.values()]
    af = [r["final_f_rate"] for r in results.values()]
    print(f"  Push spread:    {min(av)} - {max(av)}  (range={max(av)-min(av)}, CV={max(av)/max(min(av),1):.1f}x)")
    print(f"  Tau spread:     {min(at):.4f} - {max(at):.4f}  (delta={max(at)-min(at):.4f})")
    print(f"  F-rate spread:  {min(af):.2f} - {max(af):.2f}  (delta={max(af)-min(af):.2f})")

    # ── FEEDBACK EFFECT ──
    pl("=")
    print("  FEEDBACK IMPACT")
    pl("-")
    print(f"  {'User':<14} {'FB events':<11} {'Pos%':<8} {'Neg%':<8} {'A-range':<15} {'Effect'}")
    pl("-")
    for uid in ARCHETYPES:
        r = results[uid]
        fb_total = r["pos_fb"] + r["neg_fb"] + r["strong_fb"]
        if fb_total > 0:
            pp = f"{r['pos_fb']/fb_total:.0%}"
            np_ = f"{r['neg_fb']/fb_total:.0%}"
        else:
            pp = "N/A"
            np_ = "N/A"
        a_vals = r["a_trace"]
        ar = f"{min(a_vals):.3f}-{max(a_vals):.3f}" if a_vals else "N/A"

        if r["strong_fb"] >= 3:
            fx = "heavily penalized"
        elif r["neg_fb"] > r["pos_fb"] * 2:
            fx = "suppressed"
        elif r["pos_fb"] > r["neg_fb"] * 3:
            fx = "amplified"
        elif fb_total == 0:
            fx = "neutral drift"
        elif r["pos_fb"] > r["neg_fb"]:
            fx = "mild boost"
        else:
            fx = "mild suppress"

        print(f"  {uid:<14} {fb_total:<11} {pp:<8} {np_:<8} {ar:<15} {fx}")

    # ── STATE TRAJECTORY ──
    pl("=")
    print("  STATE TRAJECTORIES (every 5th round)")
    pl("-")
    print(f"  {'R':<4}", end="")
    for uid in ARCHETYPES:
        print(f"{uid.split('-')[0]:>9}", end=" ")
    print()
    for j in range(0, ROUNDS + 1, 5):
        print(f"  {j:<4}", end="")
        for uid in ARCHETYPES:
            r = results[uid]
            v = r["tau_trace"][j] if j < len(r["tau_trace"]) else r["tau_trace"][-1]
            print(f"{v:>9.4f}", end=" ")
        print(f"  (tau)")
    for j in range(0, ROUNDS + 1, 5):
        print(f"  {j:<4}", end="")
        for uid in ARCHETYPES:
            r = results[uid]
            v = r["urge_trace"][j] if j < len(r["urge_trace"]) else r["urge_trace"][-1]
            print(f"{v:>9.2f}", end=" ")
        print(f"  (urge)")

    pl("=")
    print(f"  Done. 6 users x {ROUNDS} rounds, same script.")


if __name__ == "__main__":
    main()
