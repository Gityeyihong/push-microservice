# embedding 自训练（Plasmid 借鉴）

> 参考系统：`E:\Plasmid`（Vibe-Plasmid 自举式专家模型进化系统）。
> 本文把其「自生成 → 自评 → 再微调」进化闭环与**训练安全网**迁移为
> 本服务的 **embedding 模型自训练**，并对齐其关键裁决：**正例必须过外部审查，舍弃自审查**。

## 1. 目标

利用服务运行中积累的推送/互动数据，**持续微调检索用 embedding 模型**，提升「给定意图 →
命中用户真正喜欢/需要的视频」的排序质量，同时用硬安全网杜绝
**模式坍缩、奖励黑客、指标虚高**——即「分数上涨」必须对应 held-out 真实能力提升，否则自动回滚。

## 2. 概念映射（Plasmid → 本服务）

| Plasmid（LLM 方案进化） | push-microservice（embedding 自训练） |
|---|---|
| 方案 `Solution` / `final` 分 | 检索样本 `(context ↔ video)` 的互动结果 |
| `build_dataset`（ChatML） | 对比学习三元组：`anchor=意图`，`positive=正反馈视频`，`hard negative=被拒视频` |
| `passes_positive_gate` | 外部审查 `approved`（本文 §4） |
| `gate.py`：positive/correction/negative/drop | 候选样本门控降级 |
| 多样性下限 `max_sim > 0.95` | 同源近重复对过滤 |
| held-out 回归 + `epsilon` | 冻结检索基准（precision@k / hit-rate / MRR）前后对比 |
| LoRA `gen_N/vk` + `adapter_meta.json` | 模型版本 `models/versions/gen_N/vk/` + `meta.json` |
| 回滚 `failed_` + 连续失败暂停 | 版本回滚 + `max_tune_failures` 暂停自动训练 |
| 坍缩检测 | 去重比↑ / 训练指标↑ 但 held-out ↔ |
| `run_tune_isolated`（子进程） | 训练独立子进程；训练时暂停推理加载 |
| R17 消融框架 | `baseline` vs `trained` 在 held-out 出报告 |
| 外部审查优先（§15.2） | **本服务正例唯一准入通道** |

## 3. 模块

```
selftrain/
  constants.py     # 默认参数（对应 Plasmid evolution/constants.py）
  datagen.py       # 由 engagement/session 构建对比样本（三源：positive/negative/correction）
  gate.py          # 样本门控分类 positive | correction | negative | drop
  reviewer.py      # 外部审查器：http / openai-compatible / 人工导入（弃自审查）
  safety.py        # TrainingSafety：门槛/多样性/回滚/坍缩（对应 evolution/safety.py）
  trainer.py       # sentence-transformers 全量微调（MultipleNegativesRankingLoss）
  registry.py      # 版本注册表 + active 指针 + LRU + 回滚
  evaluator.py     # held-out precision@k / hit-rate / MRR
  loop.py          # 编排：采集→构建→门控→预检→训练→评估→提交/回滚→重建 FAISS
  events.py        # 事件流（JSONL，可选 Redis）：training:*
config/training_config.json   # 对齐 Plasmid settings.yaml 的 training_safety 段
data/training/pairs.jsonl     # 门控通过样本（追加，含来源与审查元数据）
data/training/heldout.jsonl   # 冻结基准（与训练集硬隔离）
data/training/runs/<run_id>/  # metrics.jsonl + report.md（R17 消融）
models/versions/gen_N/vk/     # 训练得到的模型版本（禁止原地覆盖）
```

## 4. 正例门槛：外部审查优先，舍弃自审查

对齐 Plasmid §15.2（文本：外部审查 `approved`；未配置审查 API → 禁止作正例）与 §9。

### 4.1 双通道审查器 `reviewer.py`

| 通道 | 配置 | 说明 |
|------|------|------|
| API | `REVIEW_PROVIDER=http` / `openai-compatible`，`REVIEW_API_URL`、`REVIEW_API_KEY`、`REVIEW_MODEL` | 首选的自动审查 |
| 人工导入 | `POST /admin/training/samples/import` | 兜底；带审批标记 |

`reviewer.py` 暴露统一接口：

```python
async def review(sample: TrainingSample) -> ReviewDecision:
    """返回 approved / rejected(reason) / unavailable。"""
```

### 4.2 门槛规则（单一判定来源）

- 判定逻辑**只在 `safety.py` 实现一次**；`gate.py` 通过注入复用，**禁止二次硬编码阈值**（沿用 Plasmid R16.4）。
- 候选先由 `mobile.engagement` 打行为标签（正/负/修正）。
- **正例准入**：行为标签为「正」**且**外部审查 `approved`。
- 审查 `rejected` + 有理由 → 降级 `correction`（无改进）或 `negative`；否则 `drop`。
- **审查器未配置（`none`）或不可用** → 该批次正例全部禁止入训，发
  `training:tune_skipped(review_not_configured)`，候选仅进讨论日志。
- 人工导入的样本视同外部审查结论。

## 5. 数据构建 `datagen.py`

从两类来源产出候选 `TrainingSample`：

- **engagement**：`/api/v1/engagement` 事件 + 会话推送历史。
  - 正反馈（`watch_end` 比率 > 阈值 / `share` / `favorite` / `click`+有效观看）→ 候选正例。
  - `not_interested` / 快速划走 → 候选负例（hard negative）。
- **导入**：admin 导入的标注对。

样本结构：

```json
{
  "anchor": "最近对话意图文本",
  "positive": "正反馈视频的目录文本",
  "negatives": ["被拒视频目录文本", "..."],
  "source": "positive|correction|negative|search|plasmid",
  "behavior": "positive|negative|correction",
  "review": {"status": "approved|rejected|unavailable", "reason": "", "by": "api|manual"},
  "session_id": "…",
  "video_id": "v_001",
  "created_at": "…"
}
```

数据隔离（硬禁止，对齐 Plasmid §15.3）：审查原文/联网片段**不得**写入训练样本内容，只允许结构化分数进入元数据。

## 6. 训练安全网 `safety.py`

参数集中在 `config/training_config.json`，测试可注入。

1. **多样性下限**：正例两两余弦 `max_sim > 0.95` 的近重复过多，使有效正例 < `min_positive`（默认 2）→ 取消本次微调，发 `training:tune_skipped`。
2. **来源配额**：单一 `source` / `session_id` 占比超过上限（默认 ≤70%）→ 按新颖度丢弃过剩样本。
3. **held-out 回归**：训练成功后、切换 active 版本前，在**冻结基线**上评估；通过条件
   `heldout_score >= pre_tune_heldout_score - epsilon`（默认 0.02）。失败 → §7 回滚。
4. **坍缩检测**：滚动窗口（默认最近 2 代）若同时满足
   - 样本新颖度中位数下降 > 0.15；
   - 训练集指标中位数上升但 held-out 持平或下降；
   - 正例近重复对比例 > 0.4；
   则 `log:warn` 并提高正例门槛或暂停微调。
5. **进程隔离**：训练在**独立子进程**执行，主服务不阻塞；训练时 `pause_inference_during_tune`（默认 True）。

## 7. 版本化与回滚 `registry.py`

1. 每次训练输出 `models/versions/gen_{N}/v{k}/`（禁止原地覆盖）；保留至超出 `max_versions`（默认 5）再 LRU 删除。
2. 写 `meta.json`：`generation`、`direction`、`heldout_before`、`heldout_after`、`parent`、`base_model`、`params`。
3. held-out 失败或训练异常：
   - **不切换** active 版本；
   - 产物重命名 `failed_` 前缀保留排查；
   - 发 `training:rollback`（含原因）；
   - 连续 `max_tune_failures`（默认 3）失败 → 暂停自动训练，仅 `--force` / 手动接口恢复。
4. active 版本通过 `models/versions/active.json` 指针切换；`server/embedding_model.py` 优先解析它。

## 8. 编排循环 `loop.py`

```
采集候选 (datagen)
  → 外部审查门控 (gate + reviewer)   [未配置审查器 → tune_skipped]
  → 预检 (safety: 多样性 / 配额)
  → 训练 (trainer, 子进程)
  → held-out 评估 (evaluator)
  → 提交 or 回滚 (registry + safety)
  → 通知向量服务重建 FAISS (vector_service.reload_data)
```

触发方式：

- **手动（默认）**：admin 接口。
- **可选定时器**：`TRAINING_SCHEDULE_ENABLED=1`，满足「新增样本 ≥ 阈值 ∧ 冷却期结束 ∧ 上次训练成功」才跑。

## 9. 微调实现 `trainer.py`

- 基座：`paraphrase-multilingual-MiniLM-L12-v2`。
- 损失：`MultipleNegativesRankingLoss`（in-batch negatives）+ 显式 hard negatives。
- **全量微调**（非 LoRA）：小模型（~118M）CPU 可跑；GPU 存在时自动启用。
- 关键超参入 `config/training_config.json`（epochs、batch、lr、warmup、max_seq_length）。
- 依赖：`requirements-train.txt`（`datasets`、`accelerate` 等），与 serving 依赖隔离。

## 10. 评估 `evaluator.py`

- held-out 集：任务固定用例**或**从历史高分（成功推送）中冻结的 N 条，**训练前快照，不再写入训练集**。
- 指标：`precision@k`、`hit_rate`、`MRR`；可按 topic 分组。
- 报告落 `data/training/runs/<run_id>/metrics.jsonl` + `report.md`；
  复现元数据：`seed`、`git_commit`、`config hash`、`heldout hash`、硬件模式。

## 11. 消融框架（R17）

| 组 | 名称 | 差异 |
|---|---|---|
| A | 基线 | 无自训练（基础模型） |
| B | 完整 | 外部审查 + 安全网 + 微调 |
| C | 无回滚 | 关闭 held-out 门闸与自动回滚 |
| D | 无外部审查（仅行为标签） | 验证外部审查增量 |
| E | 无 hard negatives | 仅随机负例 |

产物与 `evaluator` 同目录；held-out 与报告数据**禁止入训练集**（同源负向断言）。

## 12. 事件表

| 事件 | 时机 |
|------|------|
| `training:build_dataset` | 数据构建开始/结束 |
| `training:tune_start` | 训练启动 |
| `training:tune_done` | 训练完成 |
| `training:tune_skipped` | 门槛/多样性/审查未配置导致跳过 |
| `training:rollback` | held-out 退化或训练异常 |
| `training:collapse` | 坍缩信号 |
| `training:activated` | active 版本切换成功 |

## 13. 配置键（`config/training_config.json`）

```json
{
  "enabled": false,
  "schedule": { "enabled": false, "cron": "0 3 * * *", "min_new_samples": 50, "cooldown_hours": 24 },
  "positive_threshold": 0.5,
  "min_positive": 2,
  "max_sim": 0.95,
  "epsilon": 0.02,
  "max_tune_failures": 3,
  "max_versions": 5,
  "source_ratio_cap": 0.7,
  "review": { "provider": "none", "timeout_sec": 30 },
  "training": {
    "epochs": 3, "batch_size": 32, "learning_rate": 2e-5,
    "warmup_ratio": 0.1, "max_seq_length": 256
  },
  "eval": { "k": 5, "freeze_size": 100 }
}
```

## 14. 安全底线（DoD）

- [ ] 未配置外部审查器时，正例一律不入训，且发出 `training:tune_skipped`。
- [ ] held-out 退化必触发回滚，active 版本不变。
- [ ] 训练样本不含审查原文 / 联网片段（负向断言）。
- [ ] 每次训练产物版本化，可回滚，LRU 受 `max_versions` 约束。
- [ ] 训练在子进程，主服务健康检查不受影响。
- [ ] 至少一次「劣质数据微调 → 观察自动回滚」的演练记录。
