# embedding 自训练

通用、可插拔的检索 embedding 自训练闭环：以**可配置的"成功"定义**从曝光/反馈数据中构建对比样本，
经**外部审查**门控后微调模型，并用 held-out **推送成功率**门闸做提交/回滚，支撑百万级事件、10 万级目录的单机场景。

> 设计借鉴自举式进化系统的训练安全网（正例门槛、多样性下限、held-out 回归、版本化回滚、坍缩检测）。

## 1. 模块

```
selftrain/
  rewards.py     可插拔奖励/成功定义（默认 GenericFunnelAdapter）
  collector.py   曝光/反馈采集（写入存储）
  datagen.py     增量样本构建（watermark）+ 难负例
  push_metrics.py success@k / AUC / NDCG / 因果漏斗 / 目录召回(ANN)
  evaluator.py   held-out 冻结与评估
  gate.py        样本门控 positive|correction|negative|drop
  reviewer.py    外部审查（http/openai-compatible/人工导入；弃自审查）
  safety.py      多样性下限/来源配额/回退/坍缩（ANN 去重）
  trainer.py     全量微调（mnrl_weighted | soft_cosine，配置切换）
  registry.py    版本注册表 + active 指针 + 回滚 + LRU
  loop.py        编排
  events.py      事件流
  store/         存储：base(SQLAlchemy) + postgres + sqlite + redis
```

## 2. 成功/奖励抽象（开源通用）

`rewards.RewardAdapter`：

```python
class RewardAdapter(ABC):
    def outcome(self, *, exposure: dict, events: list[dict]) -> RewardOutcome: ...
# RewardOutcome(reward∈[0,1], label=success|failure|unknown, strength, propensity)
```

默认 `GenericFunnelAdapter` 完全由 `config/training_config.json` 的 `reward.events` 定义：
每个事件映射为 `success | step | failure`，可带 `weight` 与 `min_ratio`。**不含任何领域假设**。
自定义适配器：`reward.adapter = "module.path:ClassName"`。

## 3. 存储（Postgres/Redis 为主 + SQLite 回退）

- `storage.backend=sqlite`（默认，零依赖回退，`data/training/training.db`）或 `postgres`（`DATABASE_URL`）。
- Redis（`REDIS_URL`）用于分布式锁、去重集合、热计数。
- 表：`push_events`、`push_items(rank,score,shown)`、`feedback_events`、`samples(dedup_hash UNIQUE)`、`heldout`、`watermarks`。
- **增量**：以 `watermarks.push_id` 为水位，每轮只处理新增曝光；样本按 `dedup_hash` 去重。

## 4. 数据流

```
采集(collector) → 增量样本(datagen, watermark + 难负例)
  → 外部审查门控(gate+reviewer) [未配置 → training:tune_skipped]
  → 预检(safety: 多样性/来源配额)
  → 冻结/加载 held-out
  → 子进程训练(trainer)
  → 推送成功率评估(push_metrics: success@k/AUC/NDCG/recall)
  → 提交 or 回滚(registry + safety)
  → 通知向量服务重建 FAISS
```

## 5. 外部审查门控（弃自审查）

- 行为成功定标签；**正例必须过外部审查 `approved`**（http / openai-compatible / 人工导入）。
- 未配置审查器 → 正例一律不入训，发 `training:tune_skipped(review_not_configured)`。
- 拒绝 → 降级 `correction`；否则 `drop`。门槛判定单一来源（`safety.py`）。

## 6. 指标与门闸

- `success@k`：在**已展示候选池**内排序（池小、天然可扩展）。
- `AUC` / `NDCG@k`：成功项相对失败项的排序质量。
- `catalog_recall`：FAISS ANN 全目录召回。
- 漏斗：`exposure → click → success`（`/admin/training/funnel`）。
- 提交条件：`success_at_k_after ≥ success_at_k_before − epsilon`，否则回滚。

## 7. 训练与规模化

- 损失可切换：`training.loss = mnrl_weighted`（MNRL + 按 reward 的 WeightedRandomSampler）或 `soft_cosine`（reward→软标签）。
- 位置偏差：奖励适配器输出 `propensity`（按曝光位次），可做 IPS 加权。
- 规模化：增量抽取、ANN 去重、有界难负例、`max_steps`、分页（`scale.page_size`）、样本上限（`scale.sample_cap`）。

## 8. 版本化与回滚

- 输出 `models/versions/gen_{N}/v{k}/`（不覆盖）；`meta.json` 记录 before/after 指标。
- 失败 → 不改 active，产物重命名 `failed_`，发 `training:rollback`。
- 连续失败 `max_tune_failures` → 暂停自动训练（`force` 可恢复）；LRU 保留 `max_versions`。

## 9. 配置（节选）

```json
{
  "storage": { "backend": "sqlite", "sqlite_path": "data/training/training.db" },
  "reward": { "adapter": "builtin:generic_funnel",
    "events": { "watch_end": { "role": "success", "min_ratio": 0.5 },
                "no_click": { "role": "failure" } } },
  "scale": { "negatives_per_anchor": 8, "sample_cap": 200000, "page_size": 5000, "max_steps": 2000 },
  "eval": { "mode": "push_success", "k": 3, "catalog_recall": true, "freeze_size": 5000 },
  "training": { "loss": "mnrl_weighted", "epochs": 3, "batch_size": 32 }
}
```

## 10. 调度进程

定时训练不在推送服务进程里跑。另起：

```powershell
$env:TRAINING_SCHEDULE_ENABLED = "1"
python -m selftrain.scheduler
```

它读取 `schedule.cron`、`min_new_samples`、`cooldown_hours`。有 `REDIS_URL` 时用 `lock:training-cycle` 保证只有一个调度进程进入 `run_cycle`；没有 Redis 时单进程照常跑并打警告。`POST /admin/training/run` 仍是人工触发。

容器：先构建服务镜像，再 `docker compose -f docker-compose.yml -f docker-compose.scheduler.yml up -d scheduler`。只执行 `docker compose up` 不会启动调度。调度镜像由 `Dockerfile.scheduler` 在构建期安装 `requirements-train.txt`。

## 11. 安全底线

- [x] 未配置外部审查器时正例不入训并 `tune_skipped`。
- [x] held-out 退化必回滚，active 不变。
- [x] 样本/报告与 held-out 硬隔离。
- [x] 版本化可回滚，LRU 受 `max_versions` 约束。
- [x] 训练在子进程，主服务不受影响。
