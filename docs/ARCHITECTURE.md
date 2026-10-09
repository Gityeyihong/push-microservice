# 架构

## 1. 定位

通用推荐/推送微服务：

- 输入：对话上下文、通用信号 `signals`、状态 `state`、可选上游意图 `intent`。
- 输出：候选条目、是否建议转人工、策略配置快照、推送事件。
- 边界：服务本身**不调用生成式 LLM**；意图由上游透传。
- 可扩展：决策策略、转人工策略、奖励定义均可继承替换。

## 2. 运行时组件

```
                     ┌──────────────────────────────────────────────┐
   HTTP 客户端 ────► │  FastAPI (server/main.py)                     │
                     │  /api/v1/push  /api/v1/feedback  /health      │
                     │  /admin/*      /internal/*                    │
                     └───┬───────────────┬───────────────┬──────────┘
                         │               │               │
             ┌───────────▼──────┐  ┌─────▼──────┐  ┌─────▼─────────┐
             │ 检索              │  │ 决策策略    │  │ 转人工         │
             │ VectorService     │  │ policy/     │  │ handoff/       │
             │  MiniLM+FAISS+MMR │  │ (可继承)    │  │ (可继承)       │
             │  ranking (配置化) │  └────────────┘  └───────────────┘
             └───────────┬──────┘
                         │ active 版本指针
             ┌───────────▼──────────────────────────────────────────┐
             │ selftrain/  奖励·存储(Postgres/Redis/SQLite)·安全网   │
             └──────────────────────────────────────────────────────┘
```

## 3. 模块边界

| 模块 | 职责 |
|------|------|
| `server/main.py` | 应用装配、端点编排、可插拔组件加载 |
| `server/models.py` | 通用 I/O 模型（`PushRequest/PushResponse/CandidateItem/FeedbackEvent`） |
| `server/vector_service.py` | 编码、FAISS 检索、MMR、主题权重 |
| `server/ranking.py` | 配置化意图规则 + 词法/向量融合重排 |
| `server/cache_service.py` | HMAC 键 + Redis 缓存，可降级 |
| `policy/` | `DecisionPolicy` 抽象 + 默认阈值策略 + 状态/事件/反馈分类/关键词 |
| `handoff/` | `HandoffPolicy` 抽象 + 默认实现 |
| `selftrain/` | 自训练（奖励适配器、存储、指标、训练、注册表、回滚） |
| `config/keywords.json` | 意图/关键词/排除规则（外部化） |
| `config/policy.json` | 决策策略参数 |
| `config/training_config.json` | 自训练配置 |

## 4. 请求流程（`/api/v1/push`）

1. 校验 `context`。
2. 解析策略配置（A/B profile 注入）。
3. `intent.confidence ≥ 0.5` 跳过缓存。
4. 缓存命中：对缓存候选重跑 `ranking`。
5. 未命中：`VectorService.process_request` 编码+检索+MMR → `ranking.apply_ranking`。
6. `HandoffPolicy.should_handoff` 评估转人工。
7. 写缓存；`run_push_decision` 运行策略，产出推送事件与状态摘要。

## 5. 数据与产物

| 路径 | 内容 |
|------|------|
| `data/catalog.json` | 条目目录（通用示例，可替换） |
| `data/*.npy`, `faiss_index.bin` | 向量产物（`data/generate_data.py` 生成） |
| `data/sessions/*.json` | 会话状态（运行期） |
| `data/training/` | 自训练语料/基线/报告 + `training.db`（SQLite 回退） |
| `models/versions/gen_N/vk/` | 训练得到的模型版本 |

## 6. 部署

- 本机：`python run_server.py`。
- 容器：`docker compose up`（Redis 7，需密码，端口只绑 `127.0.0.1`）。服务进程以 uid 10001 运行。编排健康检查打 `/ready`。
- 训练依赖与 serving 分离（`requirements-train.txt`）。
- 生产存储：Postgres（`DATABASE_URL`）+ Redis（`REDIS_URL`）；未配置自动回退 SQLite。

## 7. 可观测性

- 结构化日志；`/health`；`/admin/status`；`/admin/audit`。
- 自训练事件流 `selftrain/events.py` 与 `/admin/training/status`、`/funnel`。
