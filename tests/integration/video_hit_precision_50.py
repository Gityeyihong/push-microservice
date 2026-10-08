"""50 video-push precision / relevance regression check.

Runs the full push pipeline (VectorService → relevance_guard → DecisionEngine)
until 50 video_push events are collected, then emits detailed JSON for regression.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.vector_service import VectorService
from server.advisor_service import AdvisorService
from server.relevance_guard import (
    apply_relevance_guard,
    compute_lexical_score,
    extract_last_user_message,
    keyword_overlap_boost,
    user_has_breakup_intent,
    video_is_breakup_heavy,
    INTEREST_COLD_USER_KWS,
    INTEREST_COLD_VIDEO_KWS,
    BREAKUP_USER_KWS,
    BREAKUP_VIDEO_KWS,
)
from mobile.decision_engine import DecisionEngine, BREAKUP_VIDEO_KWS as CLIENT_BREAKUP_KWS
from mobile.decision_config import load_decision_config, resolve_a_threshold_0

TARGET_PUSHES = 50
random.seed(42)

# Curated messages: relevance edge cases + general relationship dialogue
MESSAGE_POOL = [
    # Interest / reply slowdown (must NOT push breakup-heavy when no breakup intent)
    "我们认识三个月了，最近一周她回复明显变慢，是不是对我兴趣下降了？",
    "她最近回复消息从秒回变成了几个小时，怎么判断是忙还是凉了",
    "平时发消息她回很慢，是不是我太敏感了",
    "暧昧期她回复变慢了，是不是对我没兴趣了",
    "认识三个月了，聊天变少，回复也冷淡了很多",
    "她对我忽冷忽热的，回复时快时慢",
    "暗恋的女生几乎不主动联系，但线上会秒回",
    "第一次约会后她回复变慢了，是什么信号",
    # Breakup intent — breakup videos allowed
    "我们分手了，我该怎么自我重建",
    "失恋后走不出来，有什么方法",
    "前任又联系我了，要不要挽回",
    # Active video request
    "可以推荐一些视频看看吗",
    "有推荐视频吗",
    "来个视频看看",
    "你推荐的那个沟通技巧视频很有用",
    # Conflict / communication
    "今天和女朋友吵架了，怎么办",
    "冷战好几天了，他不理我",
    "有什么办法能打破冷战吗",
    "异地恋真的好难，感觉坚持不下去了",
    "吵架时对方总是翻旧账，怎么引导就事论事",
    "感情进入倦怠期了，有什么科学的恢复热情的方法",
    # Emotional / general
    "我还是很想她，要不要主动一点",
    "帮不了我啊，心里好难受",
    "最近开始去健身了，心情好了一些",
    "我们约了明天见面聊聊",
    "见面聊得还不错，但问题没完全解决",
    "她主动找我聊天了，有点意外",
    "不知道该不该回复她",
    "怎么让这次不一样",
    "我学会了很多，真的很感谢",
    "感情真的需要经营",
    "关系在慢慢变好",
    "想再要一些建议巩固一下",
    "有没有挽回感情的方法",
    "对象和异性同事走得太近，怎么表达不满又不控制欲太强",
    "双方都是回避型依恋，怎么建立一个安全的沟通模式",
    "快30了家里催婚，但现任条件不符合父母预期，怎么选",
    "怎么判断一个人是真心喜欢你还是只是习惯你的存在",
    "和兵哥哥谈恋爱需要做什么心理准备",
    "恋爱中保持自我和互相迁就的边界在哪里",
    "她今天终于回我消息了，虽然只有一个嗯字但我好开心",
    "我愿意等她一辈子，哪怕她喜欢的是别人",
    "最近和对象关系有点奇怪，说不出来哪里不对",
    "要不要主动去问问他呢，可是又怕显得太敏感",
    "我真的好迷茫，每天都很纠结",
    "我害怕做错决定，后悔一辈子",
    "交往3个月了，什么时候表白比较合适",
    "发现伴侣在社交平台上频繁和前任互动，应该直接摊牌吗",
    "伴侣说不想同居但想继续恋爱，是什么心理",
    "双方经济差距大，怎么在恋爱中保持平等的心理地位",
    "她说不喜欢我这种类型的，我马上去换了发型买了新衣服",
    "她从来没主动给我发过消息，都是我找她",
    "想到她还是有点难受，但能控制住了",
    "我们约定了每周视频一次",
    "感觉关系比以前更好了",
    "谢谢你一直以来的帮助",
    "以后有什么问题再来找你",
    "今天和她一起看了个电影，很开心",
    "她说她也想了很多，我们都需要改变",
    "这算是复合了吗",
    "但我担心又会回到以前的样子",
    "明天就要见面了，睡不着",
    "穿什么衣服好呢",
    "见面之前我应该说什么",
    "万一又吵架怎么办",
    "我不想再失去她了",
    "见面还不错，我们聊了很多",
    "有时候觉得一个人也挺好，但又不舍得放下",
    "朋友都觉得我们不合适，但我不知道该听谁的",
    "想分手又舍不得，不分手又难受",
    "我现在每天都很焦虑，睡不好",
    "想到这些事就忍不住哭",
    "是不是每段感情都会经历这种阶段",
    "能不能帮我分析一下现在的情况",
    "我感觉自己被困住了，进退两难",
    "最近工作和感情两边都烦躁",
    "他总说我太粘人了",
    "是不是应该给彼此留点空间",
    "感觉很累，但又不想放弃",
    "帮我分析一下我们的关系",
    "希望能有个好的结果",
    "这段时间的陪伴很有帮助，来点视频",
    "我学会了很多关于感情的事",
    "真的非常感谢你",
    "她最近开始回应我的消息了",
    "我想做一个长期的计划",
    "我们认识两年了，我想认真经营这段关系",
]

STATES = ["INIT", "GATHERING", "DEEPENING", "TAILORING"]
MODES = ["auto", "mentor"]


def _meta_by_id(metadata: list[dict]) -> dict[str, dict]:
    return {m["video_id"]: m for m in metadata if m.get("video_id")}


def _cand_snapshot(candidates: list, meta_map: dict[str, dict], limit: int = 5) -> list[dict]:
    out = []
    for c in candidates[:limit]:
        vid = c.video_id if hasattr(c, "video_id") else c.get("video_id")
        meta = meta_map.get(vid, {})
        out.append(
            {
                "video_id": vid,
                "title": c.title if hasattr(c, "title") else c.get("title", meta.get("title", "")),
                "similarity": float(c.similarity if hasattr(c, "similarity") else c.get("similarity", 0)),
                "prob_base": float(c.prob_base if hasattr(c, "prob_base") else c.get("prob_base", 0)),
                "tags": meta.get("tags", []),
                "summary": meta.get("summary", ""),
                "breakup_heavy": video_is_breakup_heavy(meta or {"title": c.title if hasattr(c, "title") else ""}),
            }
        )
    return out


def _dict_candidates(candidates: list) -> list[dict]:
    return [
        {
            "video_id": c.video_id,
            "similarity": c.similarity,
            "prob_base": c.prob_base,
            "title": c.title,
            "cover_url": c.cover_url,
            "duration_sec": c.duration_sec,
        }
        for c in candidates
    ]


def _interest_kws_in_text(text: str, kws: tuple[str, ...]) -> list[str]:
    return [k for k in kws if k in text]


def _audit_push(
    push_index: int,
    round_num: int,
    user_message: str,
    context: str,
    interaction_mode: str,
    conversation_state: str,
    raw_top: list[dict],
    guarded_top: list[dict],
    guarded_pool: list[dict],
    client_pool: list[dict],
    selected: dict,
    event_data: dict,
    state_summary: dict,
    prev_video_id: str | None,
    meta_map: dict[str, dict],
) -> dict[str, Any]:
    last_user = extract_last_user_message(context) or user_message
    sel_meta = meta_map.get(selected["video_id"], {"title": selected.get("title", "")})
    user_breakup = user_has_breakup_intent(last_user)
    sel_breakup = video_is_breakup_heavy(sel_meta)
    boost = keyword_overlap_boost(last_user, sel_meta)

    user_ic = _interest_kws_in_text(last_user, INTEREST_COLD_USER_KWS)
    blob = " ".join(
        [
            str(sel_meta.get("title") or ""),
            str(sel_meta.get("summary") or ""),
            " ".join(str(t) for t in sel_meta.get("tags") or []),
        ]
    )
    video_ic = _interest_kws_in_text(blob, INTEREST_COLD_VIDEO_KWS)

    guard_top_before = raw_top[0]["video_id"] if raw_top else None
    guard_top_after = guarded_top[0]["video_id"] if guarded_top else None
    breakup_in_raw = [c["video_id"] for c in raw_top if c.get("breakup_heavy")]
    breakup_in_guarded = [c["video_id"] for c in guarded_top if c.get("breakup_heavy")]

    checks: dict[str, dict] = {}

    # R1: no breakup-heavy video when user has no breakup intent
    r1_pass = user_breakup or not sel_breakup
    checks["R1_NO_BREAKUP_MISMATCH"] = {
        "pass": r1_pass,
        "severity": "error",
        "detail": "selected breakup-heavy without user breakup intent"
        if not r1_pass
        else "ok",
    }

    # R2: no consecutive duplicate video_id
    r2_pass = prev_video_id is None or selected["video_id"] != prev_video_id
    checks["R2_NO_CONSECUTIVE_DUP"] = {
        "pass": r2_pass,
        "severity": "error",
        "detail": f"duplicate {selected['video_id']} after {prev_video_id}" if not r2_pass else "ok",
    }

    # R3: if raw top was breakup-heavy and user has no breakup intent, guard should rerank
    r3_pass = True
    r3_detail = "n/a"
    if not user_breakup and guard_top_before and raw_top:
        raw_top_breakup = raw_top[0].get("breakup_heavy")
        if raw_top_breakup and guard_top_after != guard_top_before:
            r3_detail = f"guard reranked {guard_top_before} -> {guard_top_after}"
        elif raw_top_breakup and guard_top_after == guard_top_before:
            r3_pass = False
            r3_detail = f"guard kept breakup-heavy top {guard_top_before}"
    checks["R3_GUARD_RERANK_BREAKUP"] = {
        "pass": r3_pass,
        "severity": "warn",
        "detail": r3_detail,
    }

    # R4: interest-cold alignment via lexical score
    r4_pass = True
    r4_detail = "no interest-cold user keywords"
    if user_ic:
        lex = compute_lexical_score(last_user, sel_meta)
        if lex >= 0.25 or video_ic or boost > 0:
            r4_detail = f"lexical={lex} user={user_ic} video={video_ic}"
        else:
            r4_pass = False
            r4_detail = f"low lexical={lex} for user keywords {user_ic}"
    checks["R4_INTEREST_COLD_ALIGN"] = {
        "pass": r4_pass,
        "severity": "warn",
        "detail": r4_detail,
    }

    # R5: selected must be in guarded candidates passed to the decision engine
    r5_pass = selected["video_id"] in {c["video_id"] for c in guarded_pool}
    checks["R5_SELECTED_FROM_POOL"] = {
        "pass": r5_pass,
        "severity": "error",
        "detail": "selected in guarded/client pool" if r5_pass else "selected not in candidate pool",
    }

    # R6: client breakup filter — pool should not contain breakup titles when no breakup intent
    client_breakup = [
        c["video_id"]
        for c in client_pool
        if any(k in (c.get("title") or "") for k in CLIENT_BREAKUP_KWS)
    ]
    r6_pass = user_breakup or len(client_breakup) == 0
    checks["R6_CLIENT_BREAKUP_FILTER"] = {
        "pass": r6_pass,
        "severity": "error",
        "detail": f"client pool still has breakup videos: {client_breakup}" if not r6_pass else "ok",
    }

    all_pass = all(c["pass"] for c in checks.values() if c["severity"] == "error")

    return {
        "push_index": push_index,
        "round": round_num,
        "user_message": user_message,
        "last_user_message": last_user,
        "interaction_mode": interaction_mode,
        "conversation_state": conversation_state,
        "trigger": event_data.get("trigger", "auto"),
        "decision": {
            "acceptance_a": event_data.get("acceptance"),
            "push_count_n": event_data.get("push_count"),
            "urge": round(state_summary.get("urge", 0), 4),
            "tau": round(state_summary.get("tau", 0), 4),
            "feedbacks_f": round(state_summary.get("feedbacks", 0), 4),
            "session_msg_count": state_summary.get("session_msg_count"),
        },
        "selected": {
            "video_id": selected["video_id"],
            "title": selected.get("title", sel_meta.get("title", "")),
            "tags": sel_meta.get("tags", []),
            "summary": sel_meta.get("summary", ""),
            "similarity": selected.get("similarity"),
            "prob_base": selected.get("prob_base"),
            "breakup_heavy": sel_breakup,
        },
        "candidates_raw_top5": raw_top,
        "candidates_guarded_top5": guarded_top,
        "candidates_client_pool_top5": client_pool[:5],
        "relevance_audit": {
            "user_breakup_intent": user_breakup,
            "user_breakup_kws_hit": _interest_kws_in_text(last_user, BREAKUP_USER_KWS),
            "selected_breakup_heavy": sel_breakup,
            "user_interest_cold_kws": user_ic,
            "video_interest_cold_kws": video_ic,
            "keyword_overlap_boost": boost,
            "lexical_score": compute_lexical_score(last_user, sel_meta),
            "guard_top_before": guard_top_before,
            "guard_top_after": guard_top_after,
            "guard_top_changed": guard_top_before != guard_top_after,
            "breakup_in_raw_top5": breakup_in_raw,
            "breakup_in_guarded_top5": breakup_in_guarded,
            "breakup_filtered_by_guard": [v for v in breakup_in_raw if v not in breakup_in_guarded],
        },
        "checks": checks,
        "all_pass": all_pass,
    }


def run_precision_check(target: int = TARGET_PUSHES) -> dict[str, Any]:
    vs = VectorService()
    vs.load()
    advisor = AdvisorService(vs.help_topic_vec)
    meta_map = _meta_by_id(vs.video_metadata)
    cfg = load_decision_config()

    engine = DecisionEngine()
    engine.state.reset_session()
    engine.state.set("tau", cfg.tau_init)
    engine.state.set("urge", cfg.urge_initial)

    history: list[str] = []
    pushes: list[dict] = []
    rounds_log: list[dict] = []
    prev_video_id: str | None = None
    msg_idx = 0
    round_num = 0

    while len(pushes) < target:
        round_num += 1
        user_message = MESSAGE_POOL[msg_idx % len(MESSAGE_POOL)]
        msg_idx += 1

        interaction_mode = MODES[round_num % len(MODES)]
        if round_num <= 3:
            conversation_state = "INIT"
        elif round_num <= 12:
            conversation_state = "GATHERING"
        elif round_num <= 25:
            conversation_state = "DEEPENING"
        else:
            conversation_state = "TAILORING"

        ctx_lines = [f"用户: {h}" for h in history[-6:]]
        ctx_lines.append(f"用户: {user_message}")
        prefix = "导师: " if interaction_mode == "mentor" else "AI助手: "
        ctx_lines.append(prefix)
        context = "\n".join(ctx_lines)
        history.append(user_message)

        # Bypass cooldown
        for key in ("last_push_ts", "last_advisor_push_ts"):
            ts = engine.state.get(key)
            if ts > 0:
                engine.state.set(key, ts - 105000)

        raw_candidates, ctx_vec, top1_sim = vs.process_request(context)
        raw_top = _cand_snapshot(raw_candidates, meta_map)

        guarded = apply_relevance_guard(
            context, raw_candidates, vs.video_metadata, lexical_primary=vs._use_mock
        )
        guarded_top = _cand_snapshot(guarded, meta_map)
        cds = _dict_candidates(guarded)

        pos_rate = engine.get_state_summary()["feedbacks"]
        adv_suggest = advisor.check_advisor_suggestion(
            context, ctx_vec, pos_rate, risk_alert=False
        )

        events = engine.process_user_message(
            user_message,
            cds,
            pos_rate,
            advisor_suggestion=adv_suggest,
            risk_alert=False,
            conversation_state=conversation_state,
            interaction_mode=interaction_mode,
        )
        video_events = [e for e in events if e.type == "video_push"]

        round_entry = {
            "round": round_num,
            "user_message": user_message,
            "interaction_mode": interaction_mode,
            "conversation_state": conversation_state,
            "top1_similarity": round(top1_sim, 4),
            "video_push_count": len(video_events),
            "guard_top": guarded_top[0]["video_id"] if guarded_top else None,
        }
        rounds_log.append(round_entry)

        for ev in video_events:
            if len(pushes) >= target:
                break
            sel_id = ev.data.get("video_id", "")
            sel_cand = next((c for c in cds if c["video_id"] == sel_id), ev.data)
            client_pool = engine._filter_intent_mismatch(user_message, cds)
            client_pool = engine._dedup_candidates(client_pool)
            client_snap = [
                {
                    "video_id": c["video_id"],
                    "title": c.get("title", ""),
                    "similarity": c.get("similarity", 0),
                    "prob_base": c.get("prob_base", 0),
                }
                for c in client_pool
            ]

            record = _audit_push(
                push_index=len(pushes) + 1,
                round_num=round_num,
                user_message=user_message,
                context=context,
                interaction_mode=interaction_mode,
                conversation_state=conversation_state,
                raw_top=raw_top,
                guarded_top=guarded_top,
                guarded_pool=cds,
                client_pool=client_snap,
                selected={
                    "video_id": sel_id,
                    "title": ev.data.get("title", sel_cand.get("title", "")),
                    "similarity": sel_cand.get("similarity"),
                    "prob_base": sel_cand.get("prob_base"),
                },
                event_data=ev.data,
                state_summary=engine.get_state_summary(),
                prev_video_id=prev_video_id,
                meta_map=meta_map,
            )
            pushes.append(record)
            prev_video_id = sel_id

            # Mild positive feedback to keep pipeline flowing
            engine.record_video_feedback(sel_id, "positive")

    # Aggregate check stats
    check_stats: dict[str, dict] = {}
    for p in pushes:
        for code, chk in p["checks"].items():
            if code not in check_stats:
                check_stats[code] = {"pass": 0, "fail": 0, "severity": chk["severity"]}
            if chk["pass"]:
                check_stats[code]["pass"] += 1
            else:
                check_stats[code]["fail"] += 1

    failed = [p["push_index"] for p in pushes if not p["all_pass"]]
    error_failures = [
        p["push_index"]
        for p in pushes
        if any(not c["pass"] and c["severity"] == "error" for c in p["checks"].values())
    ]

    return {
        "meta": {
            "test": "video_hit_precision_50",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_pushes": target,
            "actual_pushes": len(pushes),
            "total_rounds": round_num,
            "encoder": "mock" if vs._use_mock else "real",
            "report_label": "real" if not vs._use_mock else "mock",
            "video_catalog_size": len(vs.video_metadata),
            "decision_config": {
                "a_threshold_0": resolve_a_threshold_0(cfg),
                "a_threshold_1": cfg.a_threshold_1,
                "a_threshold_2": cfg.a_threshold_2,
                "tau_init": cfg.tau_init,
                "target_push_rate_per_10": cfg.target_push_rate_per_10,
            },
            "regression_rules": {
                "R1_NO_BREAKUP_MISMATCH": "error — no breakup-heavy video without user breakup intent",
                "R2_NO_CONSECUTIVE_DUP": "error — no consecutive duplicate video_id",
                "R3_GUARD_RERANK_BREAKUP": "warn — relevance_guard should rerank breakup-heavy top",
                "R4_INTEREST_COLD_ALIGN": "warn — interest/reply-slow user should get aligned video",
                "R5_SELECTED_FROM_POOL": "error — selected video must come from guarded pool",
                "R6_CLIENT_BREAKUP_FILTER": "error — client pool must filter breakup when no breakup intent",
            },
        },
        "summary": {
            "all_error_checks_pass": len(error_failures) == 0,
            "failed_push_indices": failed,
            "error_failed_push_indices": error_failures,
            "check_stats": check_stats,
            "breakup_mismatch_count": check_stats.get("R1_NO_BREAKUP_MISMATCH", {}).get("fail", 0),
            "consecutive_dup_count": check_stats.get("R2_NO_CONSECUTIVE_DUP", {}).get("fail", 0),
            "interest_cold_miss_count": check_stats.get("R4_INTEREST_COLD_ALIGN", {}).get("fail", 0),
        },
        "pushes": pushes,
        "rounds_sample": rounds_log[:10] + (rounds_log[-5:] if len(rounds_log) > 15 else []),
    }


def main() -> None:
    print("Running video hit precision check (target=50 pushes)...")
    t0 = time.time()
    result = run_precision_check(TARGET_PUSHES)
    elapsed = time.time() - t0

    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = result["meta"].get("report_label", result["meta"]["encoder"])
    out_path = reports_dir / f"video_hit_precision_50_{label}_{ts}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    s = result["summary"]
    print(f"  Pushes: {result['meta']['actual_pushes']} / {TARGET_PUSHES}")
    print(f"  Rounds: {result['meta']['total_rounds']}")
    print(f"  Error checks pass: {s['all_error_checks_pass']}")
    print(f"  R1 breakup mismatch: {s['breakup_mismatch_count']} fails")
    print(f"  R2 consecutive dup: {s['consecutive_dup_count']} fails")
    print(f"  R4 interest-cold miss: {s['interest_cold_miss_count']} fails")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Report: {out_path}")


if __name__ == "__main__":
    main()
