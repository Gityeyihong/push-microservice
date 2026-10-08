"""Mentor-mode push E2E scripts (A + C) — vector pipeline only, no LLM.

Runs full stack: VectorService → relevance_guard → DecisionEngine.
Writes report to reports/mentor_push_e2e_*.json
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.vector_service import VectorService
from server.relevance_guard import apply_relevance_guard, extract_last_user_message
from mobile.decision_engine import DecisionEngine
from mobile.decision_config import load_decision_config
from mobile.state_manager import StateManager

# Script A: 暗恋回复变慢 → 要课程 (from 一、导师模式首轮.md)
SCRIPT_A_TURNS = [
    "我喜欢一个女生，不知道怎么办",
    "认识三个月了，聊天变少",
    "平时发消息她回很慢，是不是我太敏感了",
    "要课程",
]

# Script C: active video request
SCRIPT_C_MESSAGES = [
    "推荐视频，来个视频看看",
    "要视频",  # after simulated strong_negative cool_down
]

FIGHT_TITLE = "情侣沟通为什么总是吵架"
COLD_SIGNAL_TITLES = ("冷淡", "回复", "信号", "兴趣", "约会")


def _mentor_context(lines: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role}: {text}" for role, text in lines)


def _run_push_round(
    vs: VectorService,
    engine: DecisionEngine,
    context: str,
    cfg,
    interaction_mode: str = "mentor",
) -> dict[str, Any]:
    raw, _ctx_vec, _top1 = vs.process_request(context)
    guarded = apply_relevance_guard(
        context,
        raw,
        vs.video_metadata,
        lexical_primary=vs._use_mock,
    )
    last_user = extract_last_user_message(context)
    events = engine.process_user_message(
        last_user,
        candidates=[c.model_dump() for c in guarded],
        conversation_state="GATHERING",
        interaction_mode=interaction_mode,
        decision_config=cfg,
    )
    video_events = [e for e in events if e.type == "video_push"]
    top_title = guarded[0].title if guarded else None
    selected = video_events[0].data.get("title") if video_events else None
    return {
        "last_user": last_user,
        "candidates_n": len(guarded),
        "top_candidate_title": top_title,
        "video_pushed": bool(video_events),
        "selected_title": selected,
        "trigger": video_events[0].data.get("trigger") if video_events else None,
        "top3_titles": [c.title for c in guarded[:3]],
    }


def run_script_a(vs: VectorService, cfg) -> dict[str, Any]:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    engine = DecisionEngine(StateManager(state_file=path), config=cfg)
    engine.state.set("last_push_ts", 0)
    engine.state.set("urge", 0.6)

    transcript: list[tuple[str, str]] = []
    rounds: list[dict[str, Any]] = []
    for turn in SCRIPT_A_TURNS:
        transcript.append(("用户", turn))
        ctx = _mentor_context(transcript)
        result = _run_push_round(vs, engine, ctx, cfg, "mentor")
        rounds.append({"user_turn": turn, **result})
        transcript.append(("导师", "（导师回复占位）"))

    final = rounds[-1]
    selected = final.get("selected_title") or ""
    top = final.get("top_candidate_title") or ""
    check_title = selected or top
    fight_leak = FIGHT_TITLE in check_title
    cold_align = any(k in check_title for k in COLD_SIGNAL_TITLES) if check_title else False
    return {
        "script": "A",
        "description": "暗恋回复变慢多轮后 要课程",
        "rounds": rounds,
        "pass": final["video_pushed"] and not fight_leak and cold_align,
        "checks": {
            "video_pushed_on_yao_kecheng": final["video_pushed"],
            "not_fight_communication_top": not fight_leak,
            "aligned_cold_or_interest": cold_align,
        },
    }


def run_script_c(vs: VectorService, cfg) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    engine = DecisionEngine(StateManager(state_file=path), config=cfg)
    engine.state.set("last_push_ts", 0)

    ctx1 = _mentor_context([("用户", SCRIPT_C_MESSAGES[0])])
    r1 = _run_push_round(vs, engine, ctx1, cfg, "mentor")
    results.append({"case": "active_recommend_video", **r1})

    engine.record_video_feedback("dummy", "strong_negative")
    engine.state.set("last_push_ts", 0)
    ctx2 = _mentor_context([("用户", SCRIPT_C_MESSAGES[1])])
    r2 = _run_push_round(vs, engine, ctx2, cfg, "mentor")
    results.append({"case": "active_after_strong_negative_cooldown", **r2})

    return {
        "script": "C",
        "description": "主动要视频 + 强负反馈冷却后仍要视频",
        "cases": results,
        "pass": all(c["video_pushed"] for c in results),
        "checks": {
            "recommend_video_pushes": results[0]["video_pushed"],
            "yao_shipin_after_cooldown": results[1]["video_pushed"],
            "active_trigger": results[0].get("trigger") == "active",
        },
    }


def main() -> None:
    print("Mentor push E2E (vector-only)...")
    t0 = time.time()
    vs = VectorService()
    vs.load()
    cfg = load_decision_config(reload=True)

    report = {
        "meta": {
            "test": "mentor_push_e2e",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "encoder": "mock" if vs._use_mock else "real",
            "vector_only": True,
        },
        "script_a": run_script_a(vs, cfg),
        "script_c": run_script_c(vs, cfg),
    }
    report["summary"] = {
        "all_pass": report["script_a"]["pass"] and report["script_c"]["pass"],
        "script_a_pass": report["script_a"]["pass"],
        "script_c_pass": report["script_c"]["pass"],
        "elapsed_sec": round(time.time() - t0, 2),
    }

    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = reports_dir / f"mentor_push_e2e_{ts}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Script A pass: {report['script_a']['pass']}")
    print(f"  Script C pass: {report['script_c']['pass']}")
    print(f"  Encoder: {report['meta']['encoder']}")
    print(f"  Report: {out}")
    if not report["summary"]["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
