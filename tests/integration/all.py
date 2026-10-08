"""Quick test script for the Push Microservice."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.vector_service import VectorService
from server.advisor_service import AdvisorService
from server.models import PushRequest


def test_vector_service():
    print("=" * 60)
    print("Testing VectorService...")
    vs = VectorService()
    vs.load()
    print(f"  Model: {'mock' if vs._use_mock else 'real'}")
    print(f"  Videos: {len(vs.video_metadata)}")
    print(f"  Topics: {vs.topic_vectors.shape[0]}")
    print(f"  Index size: {vs.index.ntotal}")

    test_context = "我和男朋友冷战了三天，他都不主动找我，我该怎么办"
    print(f"\n  Test context: {test_context}")

    candidates, ctx_vec, top1_sim = vs.process_request(test_context)

    print(f"\n  Top-1 similarity: {top1_sim:.4f}")
    print(f"  MMR result count: {len(candidates)}")
    print(f"\n  Top 5 candidates:")
    for i, c in enumerate(candidates[:5]):
        print(f"    {i+1}. [{c.video_id}] {c.title}")
        print(f"        sim={c.similarity:.4f}  prob={c.prob_base:.4f}")

    return vs


def test_advisor(vs: VectorService):
    print("\n" + "=" * 60)
    print("Testing AdvisorService...")

    advisor = AdvisorService(vs.help_topic_vec)

    test_cases = [
        ("I want to break up, I feel so hopeless and suicidal", 0.3),
        ("We had a nice date today, everything is great", 0.9),
        ("感觉走不下去了，我该怎么办，没人能帮我", 0.3),
        ("今天天气真好", 0.8),
    ]

    for ctx, pr in test_cases:
        vec = vs._encode_text(ctx)
        result = advisor.check_advisor_suggestion(ctx, vec, pr)
        print(f"  context='{ctx[:40]}...' positive_rate={pr} → advisor_suggestion={result}")


def test_decision_engine():
    print("\n" + "=" * 60)
    print("Testing DecisionEngine (mobile side)...")

    from mobile.decision_engine import DecisionEngine

    engine = DecisionEngine()

    initial = engine.get_state_summary()
    print(f"  Initial state: urge={initial['urge']:.2f}, tau={initial['tau']:.2f}")

    sample_candidates = [
        {
            "video_id": "v_003",
            "similarity": 0.89,
            "prob_base": 0.32,
            "title": "异地恋维系感情的三个技巧",
            "cover_url": "https://cdn.example.com/thumbs/v003.webp",
            "duration_sec": 60,
        },
        {
            "video_id": "v_009",
            "similarity": 0.72,
            "prob_base": 0.25,
            "title": "恋爱中的高效沟通技巧",
            "cover_url": "https://cdn.example.com/thumbs/v009.webp",
            "duration_sec": 90,
        },
        {
            "video_id": "v_001",
            "similarity": 0.65,
            "prob_base": 0.20,
            "title": "冷战期如何破冰沟通",
            "cover_url": "https://cdn.example.com/thumbs/v001.webp",
            "duration_sec": 75,
        },
    ]

    print(f"\n  Processing 5 user messages...")
    test_messages = [
        "今天和女朋友吵架了，怎么办",
        "想求推荐视频",
        "异地恋好难",
        "帮不了我啊",
        "我崩溃了",
    ]

    for msg in test_messages:
        events = engine.process_user_message(msg, sample_candidates)
        if events:
            for evt in events:
                print(f"    [{evt.type}] msg='{msg[:20]}...' → {evt.data.get('trigger', '?')} | {evt.data.get('title', '')[:30]}")
        else:
            print(f"    [no push] msg='{msg[:20]}...'")


    summary = engine.get_state_summary()
    print(f"\n  Final state: urge={summary['urge']:.2f}, tau={summary['tau']:.2f}, k={summary['dynamic_k']}")

    engine.record_video_feedback("v_003", "positive")
    engine.record_video_feedback("v_009", "negative")

    summary = engine.get_state_summary()
    print(f"  After feedback: tau={summary['tau']:.4f}")

    print("\n  Testing engagement (click + watch_end)...")
    engine.record_video_engagement("v_001", "click")
    eng = engine.record_video_engagement("v_001", "watch_end", 45, 60)
    print(f"  watch_end → feedback={eng['feedback_type']} ratio={eng['watch_ratio']}")
    summary = engine.get_state_summary()
    print(f"  click_rate={summary['click_rate']}, consecutive_negative={summary['consecutive_negative']}")


if __name__ == "__main__":
    vs = test_vector_service()
    test_advisor(vs)
    test_decision_engine()
    print("\n" + "=" * 60)
    print("All tests completed!")
