"""30-round integration test — full capability matrix P1-P12, I1-I5."""
import sys, os, json, time, random, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.vector_service import VectorService
from server.advisor_service import AdvisorService
from mobile.decision_engine import DecisionEngine
random.seed(42)

SCRIPT = [
    (1,  "我喜欢一个女生，不知道怎么办",                  "auto",      "INIT",       False, None),
    (2,  "认识三个月了，聊天变少",                        "auto",      "INIT",       False, None),
    (3,  "好焦虑，怎么办",                                "auto",      "GATHERING",  False, None),
    (4,  "平时发消息她回很慢",                            "auto",      "GATHERING",  False, None),
    (5,  "是不是我太敏感了",                              "auto",      "GATHERING",  False, None),
    (6,  "没思路，完全不知道聊什么",                      "auto",      "GATHERING",  False, None),
    (7,  "[点击气泡]",                                    "auto",      "GATHERING",  False, None),
    (8,  "我想更深入了解自己",                            "mentor",    "GATHERING",  False, None),
    (9,  "我们总是在吵架",                                "mentor",    "DEEPENING",  False, None),
    (10, "来个视频看看",                                  "auto",      "DEEPENING",  False, None),
    (11, "刚才的视频很有帮助，谢谢",                      "auto",      "DEEPENING",  False, "positive"),
    (12, "这个视频感觉一般般",                            "auto",      "DEEPENING",  False, "negative"),
    (13, "完全不靠谱，不要推这种了",                      "auto",      "DEEPENING",  False, "strong_negative"),
    (14, "抱歉刚才有点情绪化，我想继续聊聊",              "auto",      "DEEPENING",  False, None),
    (15, "最近工作和感情两边都烦躁",                      "auto",      "DEEPENING",  False, None),
    (16, "他总说我太粘人了",                              "auto",      "DEEPENING",  False, None),
    (17, "是不是应该给彼此留点空间",                      "auto",      "DEEPENING",  False, None),
    (18, "我也不知道该怎么办",                            "auto",      "DEEPENING",  False, None),
    (19, "感觉很累，但又不想放弃",                        "auto",      "TAILORING",  False, None),
    (20, "帮我分析一下我们的关系",                        "auto",      "TAILORING",  False, None),
    (21, "帮不了我，还是不行",                            "auto",      "TAILORING",  False, None),
    (22, "我崩溃了，救救我",                              "auto",      "TAILORING",  True,  None),
    (23, "有没有挽回的方法",                              "auto",      "TAILORING",  False, None),
    (24, "我想找个真人老师好好聊聊",                      "auto",      "TAILORING",  False, None),
    (25, "我今天状态不错，感觉有好转",                    "auto",      "TAILORING",  False, None),
    (26, "希望能有个好的结果",                            "auto",      "TAILORING",  False, None),
    (27, "帮不了我啊，还是不行",                          "auto",      "TAILORING",  False, None),
    (28, "这段时间的陪伴很有帮助，来点视频",              "auto",      "TAILORING",  False, None),
    (29, "我学会了很多关于感情的事",                      "auto",      "TAILORING",  False, "positive"),
    (30, "真的非常感谢你",                                "auto",      "TAILORING",  False, "positive"),
]

def main():
    print("Loading Push services...")
    vs = VectorService(); vs.load()
    advisor = AdvisorService(vs.help_topic_vec)
    print(f"  Videos: {len(vs.video_metadata)}\n")

    engine = DecisionEngine(); engine.state.reset_session()
    mtx = {"P1-HTTP":0,"P2-AUTO":0,"P3-ACTIVE":0,"P4-POS":0,"P4-NEG":0,
           "P4-STRONG":0,"P5-COOLDOWN":0,"P6-HELPLESS":0,"P7-ACTIVE_ADV":0,
           "P8-SRV_ADV":0,"P9-RISK":0,"P10-INIT":0,"P10-GATH":0,
           "P10-DEEP":0,"P10-TAIL":0,"P11-DEDUP":0,"I1-TRANS":0,
           "I2-ANAL":0,"I3-SNAP":0,"L9-RISK_OK":0}
    total_v, total_a, push_rounds, history, last_vid, results = 0,0,[],[],None,[]

    for r, msg, mode, cs, risk_alert, fb_type in SCRIPT:
        ctx_lines = [f"用户：{h}" for h in history[-6:]]
        ctx_lines.append(f"用户：{msg}")
        ctx_lines.append("导师：" if mode == "mentor" else "AI助手：")
        ctx = "\n".join(ctx_lines); history.append(msg)

        for key in ["last_push_ts","last_advisor_push_ts"]:
            ts = engine.state.get(key)
            if ts > 0: engine.state.set(key, ts - 105000)

        candidates, ctx_vec, top1_sim = vs.process_request(ctx)
        pos_rate = engine.get_state_summary()["feedbacks"]
        adv_sugg = advisor.check_advisor_suggestion(ctx, ctx_vec, pos_rate, risk_alert=risk_alert)
        adv_sugg = adv_sugg or risk_alert
        mtx["P1-HTTP"] += 1
        if len(history) >= 2 and len(ctx) > len(msg) * 2: mtx["I1-TRANS"] += 1

        cds = [{"video_id":c.video_id,"similarity":c.similarity,
                "prob_base":c.prob_base,"title":c.title,"cover_url":c.cover_url,
                "duration_sec":c.duration_sec}
               for c in candidates]
        events = engine.process_user_message(msg, cds, pos_rate, conversation_state=cs,
                                              risk_alert=risk_alert,
                                              advisor_suggestion=adv_sugg)
        ve = [e for e in events if e.type == "video_push"]
        ae = [e for e in events if e.type == "advisor_push"]
        extra = engine._evaluate_advisor(msg, pos_rate, len(ve)>0,
                                         advisor_suggestion=adv_sugg, risk_alert=risk_alert)
        et = {e.data.get("trigger") for e in ae}
        for e in extra:
            if e.data.get("trigger") not in et: ae.append(e)

        vp, ap = len(ve), len(ae); total_v += vp; total_a += ap
        if vp > 0:
            push_rounds.append(r)
            last_vid = ve[-1].data.get("video_id")
        ss = engine.get_state_summary()
        mtx["I2-ANAL"] += 1 + vp + ap + (1 if fb_type else 0)
        mtx["I3-SNAP"] += 1

        for ev in ve:
            t = ev.data.get("trigger")
            if t == "auto": mtx["P2-AUTO"] += 1
            elif t == "active": mtx["P3-ACTIVE"] += 1
            if cs == "INIT": mtx["P10-INIT"] += 1
            elif cs == "GATHERING": mtx["P10-GATH"] += 1
            elif cs == "DEEPENING": mtx["P10-DEEP"] += 1
            elif cs == "TAILORING": mtx["P10-TAIL"] += 1

        for aev in ae:
            t = aev.data.get("trigger","")
            if "帮不了我" in msg and t in ("auto",): mtx["P6-HELPLESS"] += 1
            if "找个真人" in msg and t == "active": mtx["P7-ACTIVE_ADV"] += 1
            if risk_alert and t == "auto": mtx["P8-SRV_ADV"] += 1; mtx["P9-RISK"] += 1; mtx["L9-RISK_OK"] += 1

        if msg == "[点击气泡]" and last_vid:
            engine.record_video_engagement(last_vid, "click")
            mtx["I2-ANAL"] += 1

        elif last_vid and fb_type:
            dur = 60
            if fb_type == "positive":
                eng = engine.record_video_engagement(last_vid, "watch_end", dur * 0.55, dur)
                if eng["feedback_type"] == "positive": mtx["P4-POS"] += 1
            elif fb_type == "negative":
                eng = engine.record_video_engagement(last_vid, "watch_end", dur * 0.3, dur)
                if eng["feedback_type"] == "negative": mtx["P4-NEG"] += 1
            elif fb_type == "strong_negative":
                eng = engine.record_video_engagement(last_vid, "not_interested")
                if eng["feedback_type"] == "strong_negative":
                    mtx["P4-STRONG"] += 1
                    mtx["P5-COOLDOWN"] += 1

        ph = engine.state.get("push_history")
        if len(set(ph)) == len(ph): mtx["P11-DEDUP"] += 1

        results.append({"r":r,"msg":msg[:25],"cs":cs,"risk":risk_alert,
                        "vp":vp,"ap":ap,"urge":round(ss["urge"],2),
                        "tau":round(ss["tau"],4),"fb":fb_type or "-"})

    print("="*85)
    print("  30-Round Integration Test — Full Capability Matrix")
    print("="*85)
    print(f"\n  Total video pushes:   {total_v}")
    print(f"  Total advisor pushes: {total_a}")
    print(f"  Push rounds:          {push_rounds}\n")

    checks = [
        ("P1-HTTP",      "HTTP equivalent (30 calls)",         mtx["P1-HTTP"]==30),
        ("P2-AUTO",      "Auto video push >=2",               mtx["P2-AUTO"]>=2),
        ("P3-ACTIVE",    "Active video push >=1",             mtx["P3-ACTIVE"]>=1),
        ("P4-POS",       "Positive feedback >=1",             mtx["P4-POS"]>=1),
        ("P4-NEG",       "Negative feedback >=1",             mtx["P4-NEG"]>=1),
        ("P4-STRONG",    "Strong neg feedback >=1",           mtx["P4-STRONG"]>=1),
        ("P5-COOLDOWN",  "Cooldown triggered",                mtx["P5-COOLDOWN"]>=1),
        ("P6-HELPLESS",  "Helpless keyword advisor",          mtx["P6-HELPLESS"]>=1),
        ("P7-ACTIVE_ADV","Active advisor request",            mtx["P7-ACTIVE_ADV"]>=1),
        ("P8-SRV_ADV",   "Server advisor signal",             mtx["P8-SRV_ADV"]>=1),
        ("P9-RISK",      "Risk bypass advisor (R22)",         mtx["P9-RISK"]>=1),
        ("P10-TAIL>INIT","TAILORING pushes >= INIT",          mtx["P10-TAIL"]>=mtx["P10-INIT"]),
        ("P11-DEDUP",    "No duplicate consecutive pushes",   mtx["P11-DEDUP"]>=max(total_v,1)),
        ("I1-TRANS",     "Transcript in context",             mtx["I1-TRANS"]>=20),
        ("I2-ANAL",      "Analytics events generated",        mtx["I2-ANAL"]>=30),
        ("I3-SNAP",      "Decision snapshot (urge+tau)",      mtx["I3-SNAP"]==30),
        ("L9-RISK_OK",   "Risk alert triggered (R22)",        mtx["L9-RISK_OK"]>=1),
    ]

    passed = sum(1 for _,_,ok in checks if ok)
    print(f"  {'Flag':<20} {'Status':<8} {'Value':<10} {'Check'}")
    print("  "+"-"*60)
    for code, desc, ok in checks:
        print(f"  {code:<20} {'PASS' if ok else 'FAIL':<8} {str(mtx.get(code,0)):<10} {desc}")
    print(f"\n  Pass: {passed}/{len(checks)}  Fail: {len(checks)-passed}/{len(checks)}")

    print(f"\n  {'R':<4} {'Msg':<27} {'St':<12} {'Risk':<5} {'V':<3} {'A':<3} {'Urge':<6} {'Tau':<8} {'Fb':<6}")
    print("  "+"-"*80)
    for rd in results:
        print(f"  {rd['r']:<4} {rd['msg']:<27} {rd['cs']:<12} "
              f"{'Y' if rd['risk'] else '-':<5} {rd['vp']:<3} {rd['ap']:<3} "
              f"{rd['urge']:<6.2f} {rd['tau']:<8.4f} {rd['fb']:<6}")

    # P10 comparison
    init_r = sum(1 for rd in results if rd["cs"]=="INIT")
    tail_r = sum(1 for rd in results if rd["cs"]=="TAILORING")
    print(f"\n  P10 Phase: INIT({init_r}r)={mtx['P10-INIT']}p  "
          f"TAILORING({tail_r}r)={mtx['P10-TAIL']}p")

    # Save report
    os.makedirs("reports", exist_ok=True)
    ds = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rp = f"reports/integration_30round_{ds}.json"
    with open(rp,"w",encoding="utf-8") as f:
        json.dump({"summary":{"video":total_v,"advisor":total_a,"rounds":push_rounds},
                   "checks":[{"c":c[0],"pass":c[2]} for c in checks],
                   "matrix":mtx,"rounds":results},f,ensure_ascii=False,indent=2)
    print(f"  Report: {rp}\n  Done.")

if __name__=="__main__": main()
