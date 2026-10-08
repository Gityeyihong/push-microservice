"""5-category x 30-round comprehensive push system test with full monitoring.

Categories:
  1. 迷惘 (Confused)    — hesitant, unsure what to do
  2. 精准 (Precise)     — direct, clear requests
  3. 舔狗 (Simp)        — desperate, self-deprecating
  4. 询问 (Inquiring)   — calm, question-asking
  5. 混乱 (Chaotic)     — rollercoaster, extreme swings

Each category runs 30 rounds through the full server+client pipeline.
"""
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

# ── Conversation scripts for 5 categories ──

SCRIPTS = {
    "迷惘": [
        "最近和对象关系有点奇怪……说不出来哪里不对",
        "他好像对我越来越冷淡了，但我不知道是不是我想多了",
        "要不要主动去问问他呢，可是又怕显得太敏感",
        "有时候觉得一个人也挺好，但又不舍得放下",
        "我不知道自己到底还喜不喜欢他",
        "一会儿觉得很喜欢，一会儿又觉得好累",
        "朋友都觉得我们不合适，但我不知道该听谁的",
        "我是不是太犹豫了，很多人说喜欢就要勇敢",
        "可是万一他不喜欢我了呢……",
        "要不先冷静一段时间想想？",
        "但是冷静久了会不会就真的分手了",
        "我真的好迷茫，每天都很纠结",
        "有没有什么办法可以判断感情值不值得继续",
        "有时候觉得问题在我，是不是我太作了",
        "可有时候又觉得是他不够在乎我",
        "想分手又舍不得，不分手又难受",
        "你能给我一些方向吗，我完全不知道该怎么办",
        "我现在每天都很焦虑，睡不好",
        "想到这些事就忍不住哭",
        "我需要一个人帮我理清思路",
        "也许我该去做心理咨询",
        "但朋友说只是感情问题没必要",
        "你觉得分手的标准是什么",
        "我现在对他已经没什么期待了",
        "但说分手又觉得对不起这段感情",
        "是不是每段感情都会经历这种阶段",
        "我害怕做错决定，后悔一辈子",
        "能不能帮我分析一下现在的情况",
        "我感觉自己被困住了，进退两难",
        "拜托了，我真的需要有人告诉我该怎么做",
    ],
    "精准": [
        "我和女朋友冷战3天了，需要破冰沟通的具体方法",
        "异地恋怎样安排见面频率比较科学",
        "分手后已经断了联系2周，现在复联的最佳时机是什么",
        "第一次约会选择咖啡厅合适还是看电影合适",
        "当对方说需要空间的时候，应该等多久再联系",
        "吵架时对方总是翻旧账，怎么引导就事论事",
        "对象和异性同事走得太近，怎么表达不满又不控制欲太强",
        "交往3个月了，什么时候表白比较合适，成功率最高",
        "对方父母因为地域问题反对，具体怎么做才能让他们放心",
        "发现伴侣在社交平台上频繁和前任互动，应该直接摊牌吗",
        "情侣之间财务AA制好还是一方承担好，请给出判断标准",
        "对方最近回复消息从秒回变成了几个小时，怎么判断"
           "是忙还是凉了",
        "朋友说恋爱超过6个月还不公开关系就是有问题，对吗",
        "性格不合但三观一致，这种组合能长期吗，给出分析",
        "双方都是回避型依恋，怎么建立一个安全的沟通模式",
        "对方说没有心动的感觉了但还想继续，这是什么意思",
        "快30了家里催婚，但现任条件不符合父母预期，怎么选",
        "感情进入倦怠期了，有什么科学的恢复热情的方法",
        "伴侣说不想同居但想继续恋爱，是什么心理",
        "我该不该看对方的手机，请给我一个明确的判断逻辑",
        "两个人对未来的规划完全不同，还应该继续吗",
        "怎么判断一个人是真心喜欢你还是只是习惯你的存在",
        "冷暴力算不算家暴，遇到冷暴力该怎么处理",
        "暧昧期应该持续多久，拖太久会不会变成备胎",
        "对方的好朋友不喜欢我，会影响到我们的关系吗",
        "双方经济差距大，怎么在恋爱中保持平等的心理地位",
        "和兵哥哥谈恋爱需要做什么心理准备",
        "学历差距会影响恋爱关系吗，请给出数据和分析",
        "怎么界定感情中的精神出轨，和异性朋友聊到什么程度算越界",
        "恋爱中保持自我和互相迁就的边界在哪里",
    ],
    "舔狗": [
        "她今天终于回我消息了，虽然只有一个嗯字但我好开心",
        "我每天晚上都给她发晚安，她从来没回过",
        "她朋友圈发的每条动态我都第一个点赞",
        "她让我去很远的地方帮她拿快递我也愿意",
        "她生日我送了个很贵的礼物，她才说了句谢谢",
        "我愿意等她一辈子，哪怕她喜欢的是别人",
        "她对我忽冷忽热的，但我就是放不下她",
        "我为了她改了自己的志愿填报，但她好像一点都不在意",
        "她和别人约会的时候我的心好痛，但我还是喜欢她",
        "我请她吃饭她带了闺蜜，我付了三个人的钱",
        "她说不喜欢我这种类型的，我马上去换了发型买了新衣服",
        "我知道自己是备胎，但做备胎也比没有她好",
        "她失恋了找我倾诉，我安慰了一整夜",
        "第二天她又和她对象和好了，祝我天天开心",
        "我觉得只要我坚持对她好，总有一天她会感动的",
        "朋友都说我舔狗，但我控制不住自己",
        "她让我给她买奶茶，我跑了三家店才买到她要的口味",
        "她说喜欢会唱歌的男生，我马上去报了声乐班",
        "她跟前任复合了我还要装作无所谓的样子",
        "我每天研究她的星座和塔罗牌，想知道怎么让她喜欢我",
        "她哪怕只是对我笑一下我都能开心一整天",
        "她让我删掉所有异性好友的微信，我全都删了",
        "我觉得自己越来越卑微了，但不这样我怕失去她",
        "她从来没主动给我发过消息，都是我找她",
        "我想她的时候会发很多消息，但每次都撤回怕她烦",
        "她去哪我就想去哪，哪怕只是远远看着她",
        "我存了好多她的照片，有几百张了",
        "每次手机响我都希望是她，但基本都是app推送",
        "我觉得这是真爱的表现，不是舔狗",
        "如果真的能和她在一起，我愿意付出任何代价",
    ],
    "询问": [
        "你好，我想了解一下情侣之间如何有效沟通",
        "请问冷战的正确处理方式是什么样的",
        "我想问一个问题：异地恋怎么维系感情",
        "可以请教一下第一次约会要注意什么吗",
        "你好，我对恋爱心理学比较感兴趣，有什么推荐吗",
        "请问怎样判断一个人是否值得信任",
        "我想请教一下，分手后如何走出情绪低谷",
        "可以分享一下处理伴侣吃醋的方法吗",
        "我想问怎么跟女朋友的父母相处比较好",
        "你好，请问怎么才能在恋爱中保持安全感",
        "表白有什么需要注意的时机和场合吗",
        "我想了解吵架之后如何快速修复关系",
        "可以帮我分析一下感情中的常见误区吗",
        "你好，想问关于前任的问题该怎么处理",
        "请问有没有科学的恋爱方法可以参考",
        "我想请教一下怎么避免感情倦怠",
        "请问如何处理伴侣的撒谎问题",
        "你好，想问一下恋爱中的边界感怎么建立",
        "我想了解男性思维和女性思维的差异",
        "可以分享一下健康亲密关系的特征吗",
        "想请教，怎么在吵架时控制情绪",
        "请问信任破裂后有没有修复的可能性",
        "你好，我想问怎么判断对方是不是真心",
        "请问恋爱中应不应该完全坦诚",
        "想请教一下，怎样提高自己的吸引力",
        "你好，我想知道怎么应对催婚的压力",
        "请问什么样的情侣最容易走到最后",
        "我想了解焦虑型依恋和回避型依恋的区别",
        "请问怎样在恋爱中保持独立思考",
        "你好，想请教感情中的付出不平衡怎么办",
    ],
    "混乱": [
        "啊啊啊我快要疯了我和她大吵了一架",
        "我不知道该怎么办现在整个人都是乱的",
        "我们昨晚说到凌晨三点她哭了我喝了一整瓶酒",
        "今天早上她给我发了一个早安，我该回吗还是别回",
        "我想发一个朋友圈暗示我过得很好但又觉得好幼稚",
        "天啊她闺蜜刚刚发消息问我最近怎么样什么意思",
        "我现在好想给她打电话但是又怕她挂掉或者关机",
        "为什么感情这么复杂啊我真的好累好想逃离",
        "我刚刚删了我们的聊天记录现在又后悔了",
        "我觉得我快抑郁了每天情绪都像过山车",
        "一小时前还觉得没有她我活不下去",
        "现在又觉得一个人也挺好的想去旅行",
        "我到底爱不爱她啊我自己都搞不清楚了",
        "帮帮我我真的快撑不住了",
        "刚刚在地铁上看到一个背影很像她差点哭出来",
        "她昨天点赞了一个男生的朋友圈我查了那个人一晚上",
        "我是不是有病啊为什么这么控制不了自己",
        "今天决定彻底放下了！把关于她的东西都扔了！",
        "但是我刚刚又去翻了我们以前的聊天记录",
        "那些对话好甜啊为什么后来变了那么多",
        "我现在躺在床上什么都不想做已经三天没上班了",
        "她发了一条微信撤回了我好想知道是什么但是不敢问",
        "下午心情还不错去健了身回来又崩溃了",
        "到底该不该发最后一条消息然后彻底删除",
        "我列表里有十几个情感博主每天刷到凌晨",
        "理性告诉我应该放下但是感性做不到",
        "我甚至想去她公司楼下等她",
        "不行我不能这样下去这样太丢人了",
        "但我真的好想她疯了疯了疯了",
        "算了今天先睡觉明天再说吧虽然肯定睡不着",
    ],
}


def print_line(char="-", w=85):
    print(char * w)


def run_one_category(name: str, messages: list, vs: VectorService,
                     advisor: AdvisorService, cfg: Config, seed: int):
    random.seed(seed)
    state = StateManager()
    state_file = os.path.join(os.path.dirname(state._state_file),
                              f"test_{name}.json")
    state._state_file = state_file
    state._state.update(state._defaults)
    state.set("tau", cfg.get("tau_init"))
    state.set("urge", 0.5)
    state.reset_session()

    engine = ConfigurableDecisionEngine(cfg, state)

    total_video = 0
    total_advisor = 0
    push_rounds = []
    advisor_rounds = []
    sim_values = []
    tau_trace = [cfg.get("tau_init")]
    urge_trace = [0.5]
    accept_values = []
    no_push_streak = 0
    max_no_push_streak = 0

    for i, msg in enumerate(messages):
        r = i + 1

        last_push = state.get("last_push_ts")
        if last_push > 0:
            state.set("last_push_ts", last_push - (cfg.get("cooldown_ms") + 5000))

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

        extra_adv = engine._evaluate_advisor(
            msg, positive_rate, len(video_evts) > 0, advisor_suggestion=advisor_suggest)
        existing_triggers = {e.data.get("trigger") for e in advisor_evts}
        for e in extra_adv:
            if e.data.get("trigger") not in existing_triggers:
                advisor_evts.append(e)

        vp = len(video_evts)
        ap = len(advisor_evts)
        total_video += vp
        total_advisor += ap
        if vp > 0:
            push_rounds.append(r)
            no_push_streak = 0
        else:
            no_push_streak += 1
            max_no_push_streak = max(max_no_push_streak, no_push_streak)
        if ap > 0:
            advisor_rounds.append(r)

        sim_values.append(top1_sim)

        for ve in video_evts:
            accept = ve.data.get("acceptance")
            if accept is not None:
                accept_values.append(accept)

            vid = ve.data.get("video_id", "")
            rv = random.random()
            if rv < 0.5:
                engine.record_video_feedback(vid, "positive")
            elif rv < 0.85:
                engine.record_video_feedback(vid, "negative")
            else:
                engine.record_video_feedback(vid, "strong_negative")

        ss = engine.get_state_summary()
        tau_trace.append(ss["tau"])
        urge_trace.append(ss["urge"])

    if os.path.exists(state_file):
        os.remove(state_file)

    return {
        "name": name,
        "total_video": total_video,
        "total_advisor": total_advisor,
        "push_rate": total_video / 30,
        "push_rounds": push_rounds,
        "advisor_rounds": advisor_rounds,
        "avg_sim": sum(sim_values) / len(sim_values),
        "avg_accept": sum(accept_values) / len(accept_values) if accept_values else 0,
        "tau_start": tau_trace[0],
        "tau_end": tau_trace[-1],
        "tau_trace": tau_trace,
        "urge_trace": urge_trace,
        "max_no_push": max_no_push_streak,
        "diversity": len(set(
            state.get("push_history")
        )) if state.get("push_history") else 0,
    }


def main():
    print("Loading VectorService (100 videos)...")
    vs = VectorService()
    vs.load()
    advisor = AdvisorService(vs.help_topic_vec)
    print(f"  Loaded: {len(vs.video_metadata)} videos, index size={vs.index.ntotal}")
    print("Done.\n")

    print_line("=")
    print("  5-Category x 30-Round Push System Test")
    print_line("=")

    results = {}
    for cat_name, messages in SCRIPTS.items():
        print(f"\n  Running [{cat_name}] ... ", end="", flush=True)
        start = time.time()
        result = run_one_category(cat_name, messages, vs, advisor, Config(
            "B-test",
            w_I=0.18, w_L=0.22, w_F=0.32, w_S=0.10, w_U=0.18,
            th_0=0.68, th_1=0.80, th_2=0.90,
            urge_delta=0.05, cooldown_ms=90000,
            tau_init=0.32, tau_positive=0.95, tau_negative=1.04,
            cool_down_after_strong=6,
        ), seed=42 + list(SCRIPTS.keys()).index(cat_name))
        elapsed = time.time() - start
        print(f"{result['total_video']} video / {result['total_advisor']} advisor pushes ({elapsed:.2f}s)")
        results[cat_name] = result

    # ── OVERALL SUMMARY ──
    print_line("=")
    print("  OVERALL SUMMARY")
    print_line("=")
    print(f"  {'Category':<12} {'Video':<7} {'Rate':<8} {'Advisor':<8} {'AvgA':<8} "
          f"{'Tau':<11} {'MaxNoPush':<11} {'PushRounds'}")
    print_line("-")
    for cat_name in SCRIPTS:
        r = results[cat_name]
        tau_r = f"{r['tau_start']:.2f}→{r['tau_end']:.3f}"
        push_str = ",".join(str(p) for p in r["push_rounds"][:10])
        if len(r["push_rounds"]) > 10:
            push_str += f" +{len(r['push_rounds'])-10} more"
        print(f"  {cat_name:<12} {r['total_video']:<7} "
              f"{r['push_rate']:<8.0%} {r['total_advisor']:<8} "
              f"{r['avg_accept']:<8.4f} {tau_r:<11} "
              f"{r['max_no_push']:<11} {push_str}")

    # ── CROSS-CATEGORY METRICS ──
    all_video = sum(r["total_video"] for r in results.values())
    all_advisor = sum(r["total_advisor"] for r in results.values())
    print_line("-")
    print(f"  {'TOTAL':<12} {all_video:<7} {all_video/150:<8.0%} "
          f"{all_advisor:<8} {'':<8} {'':<11} {'':<11}")

    # ── DETAIL PER CATEGORY ──
    for cat_name in SCRIPTS:
        r = results[cat_name]
        print_line("=")
        print(f"  [{cat_name}] Detailed Analysis")
        print_line("-")
        print(f"    Video pushes:     {r['total_video']}/30 ({r['push_rate']:.0%})")
        print(f"    Advisor pushes:   {r['total_advisor']}")
        print(f"    Push at rounds:   {r['push_rounds']}")
        print(f"    Advisor rounds:   {r['advisor_rounds']}")
        print(f"    Avg similarity:   {r['avg_sim']:.4f}")
        print(f"    Avg acceptance:   {r['avg_accept']:.4f}")
        print(f"    Tau evolution:     {r['tau_start']:.2f} → {r['tau_end']:.4f}")
        print(f"    Max no-push gap:  {r['max_no_push']} rounds")
        print(f"    Diversity:        {r['diversity']} unique videos pushed")

        # Urge/tau trace
        print(f"    Urge trace:       "
              f"{','.join(f'{u:.2f}' for u in r['urge_trace'][::3])}")
        print(f"    Tau trace:        "
              f"{','.join(f'{t:.3f}' for t in r['tau_trace'][::3])}")

    print_line("=")
    print("  Test complete. 5 categories x 30 rounds each.")


if __name__ == "__main__":
    main()
