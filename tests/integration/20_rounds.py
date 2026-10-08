"""20-round real-flow push system test with time simulation."""
import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.vector_service import VectorService
from server.advisor_service import AdvisorService
from mobile.decision_engine import DecisionEngine

random.seed(42)

messages = [
    "今天和女朋友吵架了，怎么办",
    "她一直不理我，发了好多消息也不回",
    "异地恋真的好难，感觉坚持不下去了",
    "我们在一起两年多了，最近总是冷战",
    "有什么办法能打破冷战吗",
    "我也想尝试主动沟通，但不知道该说什么",
    "可以推荐一些视频看看吗",
    "刚才看的那个视频确实有道理，谢谢",
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
    "要学会爱自己才能爱别人对吧",
    "要不要就这样算了，开始新生活",
]


def print_line(char="-", width=80):
    print(char * width)


def main():
    print("Loading VectorService...")
    vs = VectorService()
    vs.load()
    advisor = AdvisorService(vs.help_topic_vec)
    print("Done.\n")

    engine = DecisionEngine()
    engine.state.reset_session()

    initial = engine.get_state_summary()
    print(f"Initial state | urge={initial['urge']:.2f} tau={initial['tau']:.2f}")
    print_line("=")

    total_video = 0
    total_advisor = 0
    results = []

    for i, msg in enumerate(messages):
        r = i + 1
        print(f"\n--- Round {r:02d} ---")
        print(f"  [User] {msg}")

        # Simulate real time (advance by 65s per round to clear 60s cooldown naturally)
        now_ms = int(time.time() * 1000)
        last_push = engine.state.get("last_push_ts")
        if last_push > 0:
            engine.state.set("last_push_ts", last_push - 65000)

        # Server-side processing
        candidates, ctx_vec, top1_sim = vs.process_request(msg)

        positive_rate = engine.get_state_summary()["feedbacks"]
        advisor_suggest = advisor.check_advisor_suggestion(msg, ctx_vec, positive_rate)
        print(f"  [Server] candidates={len(candidates)} top1_sim={top1_sim:.4f} advisor_suggest={advisor_suggest}")

        # Mobile-side processing
        candidate_dicts = [
            {"video_id": c.video_id, "similarity": c.similarity,
             "prob_base": c.prob_base, "title": c.title, "cover_url": c.cover_url}
            for c in candidates
        ]

        events = engine.process_user_message(msg, candidate_dicts, positive_rate)

        video_evts = [e for e in events if e.type == "video_push"]
        advisor_evts_base = [e for e in events if e.type == "advisor_push"]

        # Also check advisor with the correct advisor_suggestion flag (engine defaults to False)
        extra_advisor = engine._evaluate_advisor(
            msg, positive_rate,
            has_video_push=len(video_evts) > 0,
            advisor_suggestion=advisor_suggest,
        )
        existing_triggers = {e.data.get("trigger", "") for e in advisor_evts_base}
        for e in extra_advisor:
            if e.data.get("trigger", "") not in existing_triggers:
                advisor_evts_base.append(e)

        # Display video pushes
        for ve in video_evts:
            total_video += 1
            d = ve.data
            a = d.get("acceptance")
            n = d.get("push_count")
            extra = f" A={a:.4f} N={n}" if a is not None else ""
            print(f"  [PUSH-VIDEO] trigger={d.get('trigger')} | {d.get('title', '')[:35]}{extra}")

        # Display advisor pushes
        for ae in advisor_evts_base:
            total_advisor += 1
            d = ae.data
            print(f"  [PUSH-ADVISOR] trigger={d.get('trigger')} | {d.get('title', '')}")

        if not video_evts and not advisor_evts_base:
            print(f"  [No Push]")

        # State after round
        state = engine.get_state_summary()
        print(f"  [State] urge={state['urge']:.2f} tau={state['tau']:.4f} fb={state['feedbacks']:.2f} msgs={state['session_msg_count']}")

        # Simulate feedback for video pushes
        for ve in video_evts:
            vid = ve.data.get("video_id", "")
            rv = random.random()
            if rv < 0.5:
                engine.record_video_feedback(vid, "positive")
                print(f"  [Feedback] positive ({vid})")
            elif rv < 0.85:
                engine.record_video_feedback(vid, "negative")
                print(f"  [Feedback] negative ({vid})")
            else:
                engine.record_video_feedback(vid, "strong_negative")
                print(f"  [Feedback] strong_negative ({vid})")

        results.append({
            "r": r, "msg": msg[:30], "sim": top1_sim,
            "as": advisor_suggest, "vp": len(video_evts), "ap": len(advisor_evts_base),
            "urge": state["urge"], "tau": state["tau"],
        })

    # Final summary
    print_line("=")
    print("SUMMARY")
    print_line("-")
    final = engine.get_state_summary()
    print(f"  {'Total rounds':<28} {20}")
    print(f"  {'Total video pushes':<28} {total_video}")
    print(f"  {'Total advisor pushes':<28} {total_advisor}")
    print(f"  {'Final urge':<28} {final['urge']:.2f}")
    print(f"  {'Final tau':<28} {final['tau']:.4f}")
    print(f"  {'Final feedback rate':<28} {final['feedbacks']:.2f}")
    print(f"  {'Message count':<28} {final['session_msg_count']}")
    print(f"  {'Dynamic k':<28} {final['dynamic_k']}")

    print(f"\n  {'R':<4} {'Message':<32} {'sim':<8} {'AdS':<5} {'VP':<4} {'AP':<4} {'urge':<6} {'tau':<8}")
    print_line("-", 80)
    for rd in results:
        print(f"  {rd['r']:<4} {rd['msg']:<32} {rd['sim']:<8.4f} {'Y' if rd['as'] else '-':<5} "
              f"{rd['vp']:<4} {rd['ap']:<4} {rd['urge']:<6.2f} {rd['tau']:<8.4f}")


if __name__ == "__main__":
    main()
