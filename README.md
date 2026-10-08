# Push Microservice

通用、可扩展的**推荐/推送微服务**（开源）。基于本地句向量 + FAISS + MMR 做内容检索与推送决策，
内置 **Plasmid 启发的 embedding 自训练闭环**（可插拔奖励 / 外部审查门控 / held-out 回归 / 版本回滚）。

服务本身**不调用生成式 LLM** 生成推送内容；意图可由上游预分析后透传。
领域业务（关键词、话题、示例数据、模式）一律**外置为配置**，通过可继承接口替换。

## 特性

- `POST /api/v1/push`：给定上下文/信号，返回候选 + 决策配置 + 推送事件。
- `POST /api/v1/feedback`：回收点击/观看/分享等反馈并分类。
- 本地向量检索：`paraphrase-multilingual-MiniLM-L12-v2` + FAISS(HNSW) + MMR + 主题平滑。
- 可插拔 `DecisionPolicy` / `HandoffPolicy` / `RewardAdapter` / `Ranker`（`builtin:*` 或 `module:Class`）。
- 配置驱动的关键词与意图规则（`config/keywords.json`），零领域耦合。
- 管理接口 `/admin/*`：策略配置、A/B、会话、审计、目录热重载、自训练。
- 内部接口 `/internal/*`：只读目录导出、安全目录同步。
- 自训练 `selftrain/*`：对比学习微调 + 训练安全网 + Postgres/Redis/SQLite 存储。

## 快速开始

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

# 生成向量（离线 mock，无需下载模型）
$env:PUSH_USE_MOCK_VECTORS = "1"
python -m data.generate_data --mock
python run_server.py
```

真实语义向量：

```powershell
python scripts/download_embedding_model.py --method hub   # 约 449MB
$env:PUSH_USE_MOCK_VECTORS = "0"
python -m data.generate_data
python run_server.py
```

Docker：`cp .env.example .env && docker compose up -d --build`

## 目录

| 路径 | 说明 |
|------|------|
| `server/` | FastAPI 服务（入口 `server/main.py`） |
| `policy/` | 可扩展决策策略（`DecisionPolicy`, `BuiltinThresholdPolicy`） |
| `handoff/` | 可扩展转人工组件（`HandoffPolicy`） |
| `selftrain/` | 自训练闭环（奖励/存储/指标/训练） |
| `config/` | `keywords.json`、`policy.json`、`training_config.json` |
| `data/` | `catalog.json`、`topics.py`、离线向量产物 |
| `scripts/` | 启动 / 建数据 / 下载模型 / 通用目录导入 |
| `tests/` | 单元与集成测试 |
| `docs/` | 架构、接口、自训练、扩展、迁移 |

## 文档

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/API.md`](docs/API.md)
- [`docs/SELF-TRAINING.md`](docs/SELF-TRAINING.md)
- [`docs/EXTENDING.md`](docs/EXTENDING.md) — 继承 Policy/Handoff/Reward
- [`docs/MIGRATION.md`](docs/MIGRATION.md)

## 环境变量

见 [`.env.example`](.env.example)。关键项：

| 变量 | 默认 | 说明 |
|------|------|------|
| `HMAC_SECRET_KEY` | 占位符 | Redis 缓存键 HMAC 盐 |
| `PUSH_ADMIN_TOKEN` | 空 | `/admin/*` Bearer 令牌 |
| `PUSH_INTERNAL_TOKEN` | 空 | `/internal/*` 令牌 |
| `PUSH_USE_MOCK_VECTORS` | `0` | `1`=离线随机向量 |
| `PUSH_VIDEOS` | `catalog.json` | 目录文件名（`data/` 下） |
| `POLICY_PROVIDER` | `builtin:default` | 决策策略实现 |
| `HANDOFF_PROVIDER` | `builtin:default` | 转人工策略实现 |
| `TRAINING_BACKEND` | `sqlite` | `sqlite` 或 `postgres` |
| `TRAINING_ENABLED` | `0` | 是否采集训练数据 |
| `REVIEW_PROVIDER` | `none` | 自训练外部审查器 |

## 许可证

待定（开源项目）。
