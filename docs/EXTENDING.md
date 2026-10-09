# 扩展指南

服务的关键能力都是**可继承**接口，通过配置选择内置实现或自定义类。

## 1. 决策策略 `DecisionPolicy`

```python
# myapp/policies.py
from policy.base import DecisionPolicy
from policy.events import ItemPushEvent

class MyPolicy(DecisionPolicy):
    def process(self, *, user_message, candidates=None, positive_rate=0.6,
                handoff_suggested=False, signals=None, state="INIT", config=None):
        if candidates:
            top = candidates[0]
            return [ItemPushEvent(data={"item_id": top["item_id"], "trigger": "custom"})]
        return []

    def record_engagement(self, *, item_id, event, watch_duration_sec=0,
                          item_duration_sec=60, config=None):
        return {"feedback_type": None, "reason": "noop"}
```

启用：环境变量 `POLICY_PROVIDER` 优先；未设置时读 `config/policy.json` 的 `provider`（默认 `builtin:default`）。

## 2. 转人工策略 `HandoffPolicy`

```python
from handoff.base import HandoffPolicy

class MyHandoff(HandoffPolicy):
    def should_handoff(self, *, context, signals=None, state="INIT",
                       positive_rate=0.6, context_vector=None):
        return "escalate" in context.lower()

    def build_payload(self):
        return {"title": "Talk to a human?", "subtitle": "...", "link": "https://..."}
```

启用：`HANDOFF_PROVIDER=myapp.handoff:MyHandoff`。

## 2.1 排序 `Ranker`

```python
class MyRanker:
    def rank(self, context, candidates, item_metadata, **kwargs):
        return candidates
```

启用：`RANKER_PROVIDER=myapp.ranking:MyRanker`（默认 `builtin:default`，即现有词法/向量融合）。

## 3. 奖励适配器 `RewardAdapter`（自训练）

```python
# myapp/rewards.py
from selftrain.rewards import RewardAdapter, RewardOutcome

class MyReward(RewardAdapter):
    def outcome(self, *, exposure, events):
        for e in events:
            if e.get("event") == "purchase":
                return RewardOutcome(reward=1.0, label="success",
                                     strength=1.0, propensity=1.0)
        return RewardOutcome(label="unknown")
```

启用：`config/training_config.json` 的 `reward.adapter = "myapp.rewards:MyReward"`。

## 4. 关键词/意图（配置）

所有关键词、意图规则、排除规则都在 `config/keywords.json`，无需改代码：

- `item_request` / `handoff_request`：请求检测。
- `intent_rules`：`user_kws → item_kws / tag_boost / weight`，用于词法重排。
- `exclusion_rules`：跨主题惩罚。
- `urgent_keywords`、`sensitive_topics`、`helpless_keywords`。
- `context.user_prefixes`：从对话里抽取用户句的行前缀。
- `signals.force_handoff`：为真时强制建议转人工的信号名（默认 `risk_alert`）。

## 5. 目录与向量

- 用 `scripts/import_catalog.py` 导入 CSV/JSON 目录，或直接替换 `data/catalog.json`。
- 重建向量：`python -m data.generate_data`（或 `--mock`）。产物为 `item_embeddings.npy` 与 `item_topic_vecs.npy`；若只存在旧文件名 `video_embeddings.npy` / `video_topic_vecs.npy`，服务会回退读取。
- 目录文件：`PUSH_CATALOG`（优先）或 `PUSH_VIDEOS`。
- 热重载：`POST /admin/items/reload`。

## 6. 存储后端

- 默认 SQLite（零依赖）。生产配置 `TRAINING_BACKEND=postgres` + `DATABASE_URL`，可选 `REDIS_URL`。
- 训练依赖安装：`pip install -r requirements-train.txt`。
