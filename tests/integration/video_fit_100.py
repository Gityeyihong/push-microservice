"""100-case video relevance / fit test (input → output video + key metrics).

Pipeline per case: VectorService → relevance_guard → guarded top-1 as output.
Uses production-aligned real encoder when PUSH_USE_MOCK_VECTORS=0.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.vector_service import VectorService
from server.relevance_guard import (
    apply_relevance_guard,
    compute_lexical_score,
    user_has_breakup_intent,
    video_is_breakup_heavy,
)
from server.config import PUSH_USE_MOCK_VECTORS

# 100 条：意图边界 + 全场景覆盖（去重保序）
_RAW_CASES: list[tuple[str, str]] = [
    ("interest_cold", "我们认识三个月了，最近一周她回复明显变慢，是不是对我兴趣下降了？"),
    ("interest_cold", "她最近回复消息从秒回变成了几个小时，怎么判断是忙还是凉了"),
    ("interest_cold", "平时发消息她回很慢，是不是我太敏感了"),
    ("interest_cold", "暧昧期她回复变慢了，是不是对我没兴趣了"),
    ("interest_cold", "认识三个月了，聊天变少，回复也冷淡了很多"),
    ("interest_cold", "她对我忽冷忽热的，回复时快时慢"),
    ("interest_cold", "暗恋的女生几乎不主动联系，但线上会秒回"),
    ("interest_cold", "第一次约会后她回复变慢了，是什么信号"),
    ("breakup", "我们分手了，我该怎么自我重建"),
    ("breakup", "失恋后走不出来，有什么方法"),
    ("breakup", "前任又联系我了，要不要挽回"),
    ("breakup", "分手后已经断了联系2周，现在复联的最佳时机是什么"),
    ("breakup", "分手后忍不住想联系对方怎么办"),
    ("cold_war", "今天和女朋友吵架了，怎么办"),
    ("cold_war", "冷战好几天了，他不理我"),
    ("cold_war", "有什么办法能打破冷战吗"),
    ("cold_war", "我们在一起两年多了，最近总是冷战"),
    ("cold_war", "我和女朋友冷战3天了，需要破冰沟通的具体方法"),
    ("cold_war", "冷暴力算不算家暴，遇到冷暴力该怎么处理"),
    ("long_distance", "异地恋真的好难，感觉坚持不下去了"),
    ("long_distance", "异地恋怎样安排见面频率比较科学"),
    ("long_distance", "异地恋吵架怎么和好"),
    ("long_distance", "异地恋见面前的准备攻略"),
    ("first_date", "第一次约会选择咖啡厅合适还是看电影合适"),
    ("first_date", "第一次约会聊什么不冷场"),
    ("first_date", "我们约了明天见面聊聊"),
    ("first_date", "该怎么准备这次见面"),
    ("first_date", "穿什么衣服好呢"),
    ("communication", "吵架时对方总是翻旧账，怎么引导就事论事"),
    ("communication", "情侣沟通为什么总是吵架"),
    ("communication", "伴侣总翻旧账怎么办"),
    ("communication", "怎么避免无意义的争吵"),
    ("jealousy", "对象和异性同事走得太近，怎么表达不满又不控制欲太强"),
    ("jealousy", "对象吃醋了怎么办"),
    ("jealousy", "发现伴侣在社交平台上频繁和前任互动，应该直接摊牌吗"),
    ("trust", "发现伴侣对你有所隐瞒怎么办"),
    ("trust", "恋爱中如何建立信任"),
    ("trust", "信任破裂后如何修复"),
    ("confession", "交往3个月了，什么时候表白比较合适，成功率最高"),
    ("confession", "怎么表白成功率最高"),
    ("confession", "表白被拒后如何调整心态"),
    ("security", "如何在恋爱中给足对方安全感"),
    ("security", "缺乏安全感怎么办"),
    ("security", "他总说我太粘人了"),
    ("parents", "父母反对的爱情该如何坚持"),
    ("parents", "快30了家里催婚，但现任条件不符合父母预期，怎么选"),
    ("parents", "被父母安排相亲该如何应对"),
    ("passion_fade", "感情进入倦怠期了，有什么科学的恢复热情的方法"),
    ("passion_fade", "异地恋如何保持新鲜感"),
    ("passion_fade", "恋爱初期如何维持神秘感"),
    ("general", "我还是很想她，要不要主动一点"),
    ("general", "她主动找我聊天了，有点意外"),
    ("general", "不知道该不该回复她"),
    ("general", "怎么让这次不一样"),
    ("general", "有没有挽回感情的方法"),
    ("general", "帮不了我啊，心里好难受"),
    ("general", "我崩溃了，救救我"),
    ("general", "我想找个真人老师好好聊聊"),
    ("general", "帮我分析一下我们的关系"),
    ("general", "双方都是回避型依恋，怎么建立一个安全的沟通模式"),
    ("general", "对方说没有心动的感觉了但还想继续，这是什么意思"),
    ("general", "怎么判断一个人是真心喜欢你还是只是习惯你的存在"),
    ("general", "暧昧期应该持续多久，拖太久会不会变成备胎"),
    ("general", "双方经济差距大，怎么在恋爱中保持平等的心理地位"),
    ("general", "怎么界定感情中的精神出轨，和异性朋友聊到什么程度算越界"),
    ("general", "恋爱中保持自我和互相迁就的边界在哪里"),
    ("general", "当对方说需要空间的时候，应该等多久再联系"),
    ("general", "伴侣说不想同居但想继续恋爱，是什么心理"),
    ("general", "我该不该看对方的手机，请给我一个明确的判断逻辑"),
    ("general", "两个人对未来的规划完全不同，还应该继续吗"),
    ("general", "朋友都觉得我们不合适，但我不知道该听谁的"),
    ("general", "想分手又舍不得，不分手又难受"),
    ("general", "是不是应该给彼此留点空间"),
    ("general", "感觉很累，但又不想放弃"),
    ("general", "感情真的需要经营"),
    ("general", "关系在慢慢变好"),
    ("general", "见面还不错，我们聊了很多"),
    ("general", "她说她也想了很多，我们都需要改变"),
    ("general", "这算是复合了吗"),
    ("general", "但我担心又会回到以前的样子"),
    ("general", "明天就要见面了，睡不着"),
    ("general", "万一又吵架怎么办"),
    ("general", "我不想再失去她了"),
    ("general", "今天和她一起看了个电影，很开心"),
    ("general", "最近工作和感情两边都烦躁"),
    ("general", "她最近开始回应我的消息了"),
    ("general", "我想做一个长期的计划"),
    ("general", "我们认识两年了，我想认真经营这段关系"),
    ("general", "最近和对象关系有点奇怪，说不出来哪里不对"),
    ("general", "我真的好迷茫，每天都很纠结"),
    ("general", "能不能帮我分析一下现在的情况"),
    ("general", "我害怕做错决定，后悔一辈子"),
    ("general", "她今天终于回我消息了，虽然只有一个嗯字但我好开心"),
    ("general", "她从来没主动给我发过消息，都是我找她"),
    ("general", "她让我去很远的地方帮她拿快递我也愿意"),
    ("general", "朋友约我出去吃饭，感觉好多了"),
    ("general", "最近开始去健身了，心情好了一些"),
    ("general", "要学会爱自己才能爱别人对吧"),
    ("general", "要不要就这样算了，开始新生活"),
    ("general", "想起她还是有点难受，但能控制住了"),
    ("general", "确实，先把自己过好才能有好的感情"),
    ("general", "她又开始和别人聊天了，我感觉被背叛了"),
    ("general", "我梦到她和别人在一起了"),
    ("general", "我感觉自己很没用，配不上她"),
    ("general", "也许我太敏感了，应该放下过去"),
    ("general", "性格不合但三观一致，这种组合能长期吗，给出分析"),
    ("general", "朋友说恋爱超过6个月还不公开关系就是有问题，对吗"),
    ("general", "情侣之间财务AA制好还是一方承担好，请给出判断标准"),
    ("general", "和兵哥哥谈恋爱需要做什么心理准备"),
    ("general", "学历差距会影响恋爱关系吗，请给出数据和分析"),
    ("general", "对方的好朋友不喜欢我，会影响到我们的关系吗"),
    ("general", "怎么处理恋人间的金钱问题"),
]

assert len(_RAW_CASES) >= 100, len(_RAW_CASES)
CASES = _RAW_CASES[:100]


def _meta_map(metadata: list[dict]) -> dict[str, dict]:
    return {m["video_id"]: m for m in metadata if m.get("video_id")}


def run_fit_test() -> dict[str, Any]:
    t0 = time.time()
    vs = VectorService()
    vs.load()
    meta = _meta_map(vs.video_metadata)
    enc = vs.encoder_status()

    rows: list[dict[str, Any]] = []
    guard_changed = 0
    breakup_mismatch = 0
    lexical_sum = 0.0
    vec_raw_guard_gap_sum = 0.0

    for i, (category, user_input) in enumerate(CASES, start=1):
        t_case = time.perf_counter()
        ctx = f"用户: {user_input}"

        raw, _, raw_top1_sim = vs.process_request(ctx)
        guarded = apply_relevance_guard(
            ctx, raw, vs.video_metadata, lexical_primary=vs._use_mock
        )

        raw_top = raw[0] if raw else None
        out = guarded[0] if guarded else None
        out_meta = meta.get(out.video_id, {}) if out else {}
        lex = compute_lexical_score(user_input, out_meta) if out else 0.0
        elapsed_ms = round((time.perf_counter() - t_case) * 1000, 2)

        changed = bool(raw_top and out and raw_top.video_id != out.video_id)
        if changed:
            guard_changed += 1

        user_breakup = user_has_breakup_intent(user_input)
        sel_breakup = video_is_breakup_heavy(out_meta) if out else False
        mismatch = bool(out and not user_breakup and sel_breakup)
        if mismatch:
            breakup_mismatch += 1

        lexical_sum += lex
        if raw_top and out:
            vec_raw_guard_gap_sum += abs(float(raw_top.similarity) - float(out.similarity))

        fit_ok = not mismatch and lex >= 0.15

        rows.append({
            "case_id": i,
            "category": category,
            "input": user_input,
            "output_video": {
                "video_id": out.video_id if out else None,
                "title": out.title if out else None,
                "tags": out_meta.get("tags", []),
                "summary": out_meta.get("summary", ""),
            },
            "runtime": {
                "pipeline": "VectorService → relevance_guard → guarded_top1",
                "encoder": enc.get("encoder"),
                "lexical_primary": vs._use_mock,
                "guard_reranked": changed,
                "elapsed_ms": elapsed_ms,
            },
            "key_data": {
                "raw_top1_id": raw_top.video_id if raw_top else None,
                "raw_top1_title": raw_top.title if raw_top else None,
                "raw_top1_sim": round(float(raw_top.similarity), 4) if raw_top else None,
                "guarded_top1_sim": round(float(out.similarity), 4) if out else None,
                "top1_vector_sim": round(raw_top1_sim, 4),
                "lexical_score": lex,
                "user_breakup_intent": user_breakup,
                "output_breakup_heavy": sel_breakup,
                "fit_ok": fit_ok,
            },
        })

    elapsed = round(time.time() - t0, 2)
    n = len(rows)
    fit_ok_n = sum(1 for r in rows if r["key_data"]["fit_ok"])

    return {
        "meta": {
            "test": "video_fit_100",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case_count": n,
            "encoder": enc,
            "vector_manifest": json.loads(
                (Path(__file__).parent / "data" / "vector_manifest.json").read_text(encoding="utf-8")
            ) if (Path(__file__).parent / "data" / "vector_manifest.json").exists() else {},
            "push_use_mock_vectors": PUSH_USE_MOCK_VECTORS,
            "elapsed_sec": elapsed,
        },
        "summary": {
            "fit_ok_rate": round(fit_ok_n / n, 4),
            "fit_ok_count": fit_ok_n,
            "guard_rerank_count": guard_changed,
            "guard_rerank_rate": round(guard_changed / n, 4),
            "breakup_mismatch_count": breakup_mismatch,
            "avg_lexical_score": round(lexical_sum / n, 4),
            "avg_raw_guard_sim_gap": round(vec_raw_guard_gap_sum / n, 4),
            "by_category": _category_stats(rows),
        },
        "cases": rows,
    }


def _category_stats(rows: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for r in rows:
        cat = r["category"]
        if cat not in stats:
            stats[cat] = {"count": 0, "fit_ok": 0, "guard_changed": 0}
        stats[cat]["count"] += 1
        if r["key_data"]["fit_ok"]:
            stats[cat]["fit_ok"] += 1
        if r["runtime"]["guard_reranked"]:
            stats[cat]["guard_changed"] += 1
    for cat, s in stats.items():
        s["fit_ok_rate"] = round(s["fit_ok"] / s["count"], 4)
    return stats


def main() -> None:
    print("Running 100-case video fit test...")
    result = run_fit_test()
    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = reports / f"video_fit_100_{ts}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    s = result["summary"]
    print(f"  Cases: {result['meta']['case_count']}")
    print(f"  Encoder: {result['meta']['encoder'].get('encoder')} / {result['meta']['encoder'].get('vector_mode')}")
    print(f"  Fit OK: {s['fit_ok_count']}/{result['meta']['case_count']} ({s['fit_ok_rate']:.1%})")
    print(f"  Guard rerank: {s['guard_rerank_count']} ({s['guard_rerank_rate']:.1%})")
    print(f"  Breakup mismatch: {s['breakup_mismatch_count']}")
    print(f"  Avg lexical: {s['avg_lexical_score']}")
    print(f"  Elapsed: {result['meta']['elapsed_sec']}s")
    print(f"  Report: {out}")


if __name__ == "__main__":
    main()
