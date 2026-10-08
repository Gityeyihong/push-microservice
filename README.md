# Push Microservice

独立的智能推送微服务。基于本地 **MiniLM 多语言句向量 + FAISS + MMR** 做视频检索与推送决策，
并内置一套 **Plasmid 启发的 embedding 自训练闭环**（外部审查门控 / held-out 回归 / 版本化 / 自动回滚）。

> 本服务**不调用生成式 LLM** 生成推送内容；检索与打分完全本地。自训练阶段的外部审查为可插拔组件（默认关闭）。

## 特性

- `POST /api/v1/push`：给定对话上下文与状态，返回候选视频 + 决策配置 + 推送事件。
- `POST /api/v1/engagement`：回收曝光/点击/观看/分享/不感兴趣等互动事件并分类。
- 本地向量检索：`paraphrase-multilingual-MiniLM-L12-v2` + FAISS(HNSW) + MMR 去重 + 主题平滑。
- 相关性护栏 `relevance_guard`：多轮意图查询构建 + 词法/向量融合重排。
- 规则决策引擎 `mobile/DecisionEngine`：冲动值/冷却/配额/交互模式（auto/mentor/proxy_chat）。
- 管理接口 `/admin/*`：决策配置、A/B、会话、审计、向量热重载、自训练控制。
- 内部接口 `/internal/*`：只读向量目录导出、安全视频同步。
- 可选阿里云 VOD 批量导出工具。
- 自训练 `selftrain/*`：对比学习微调 + 训练安全网。

## 快速开始

```powershell
# 1. 依赖
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 配置
Copy-Item .env.example .env
#   编辑 .env（至少设置 HMAC_SECRET_KEY / PUSH_ADMIN_TOKEN）

# 3. 生成向量与索引（离线 mock，无需下载模型）
$env:PUSH_USE_MOCK_VECTORS = "1"
python -m data.generate_data --mock

# 4. 启动
python run_server.py
#   或：uvicorn server.main:app --host 0.0.0.0 --port 8000
```

生产（语义向量）模式：

```powershell
# 下载本地 embedding 模型（约 449MB）到 models/
python scripts/download_embedding_model.py --method hub
$env:PUSH_USE_MOCK_VECTORS = "0"
python -m data.generate_data
python run_server.py
```

Docker：

```bash
cp .env.example .env   # 并填写密钥
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

## 目录

| 路径 | 说明 |
|------|------|
| `server/` | FastAPI 服务（入口 `server/main.py`） |
| `mobile/` | 决策引擎与互动分类（`DecisionEngine`, `engagement`, `state_manager`） |
| `selftrain/` | embedding 自训练闭环（见 `docs/SELF-TRAINING.md`） |
| `data/` | 视频目录、离线向量产物、会话状态、训练语料 |
| `config/` | 决策配置、A/B profiles、训练配置 |
| `scripts/` | 启动 / 建数据 / 下载模型 / VOD 导出 / 看门狗 |
| `tests/` | 单元与集成测试 |
| `deploy/` | systemd watchdog 单元文件 |
| `docs/` | 架构、自训练、接口、迁移文档 |

## 文档

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 服务架构与模块边界
- [`docs/SELF-TRAINING.md`](docs/SELF-TRAINING.md) — embedding 自训练与训练安全网
- [`docs/API.md`](docs/API.md) — HTTP 接口
- [`docs/MIGRATION.md`](docs/MIGRATION.md) — 去敏感与自包含迁移记录

## 环境变量

见 [`.env.example`](.env.example)。关键项：

| 变量 | 默认 | 说明 |
|------|------|------|
| `HMAC_SECRET_KEY` | 占位符 | Redis 缓存键 HMAC 盐；生产必须设置 |
| `PUSH_ADMIN_TOKEN` | 空 | `/admin/*` Bearer 令牌；空则返回 503 |
| `PUSH_INTERNAL_TOKEN` | 空 | `/internal/*` 令牌；回退 `PUSH_ADMIN_TOKEN` |
| `REDIS_HOST/PORT/DB` | localhost/6379/0 | 缓存；不可用时自动降级 |
| `PUSH_USE_MOCK_VECTORS` | `0` | `1`=离线随机向量 |
| `PUSH_VIDEOS` | `sample_videos.json` | 视频目录文件名（`data/` 下） |
| `EMBEDDING_MODEL_PATH` | 空 | 覆盖本地模型目录 |
| `ADVISOR_LINK` | 占位符 | 导师引导链接 |
| `REVIEW_PROVIDER` | `none` | 自训练外部审查器（`none`/`http`/`openai-compatible`） |

## 许可证

内部项目，未附许可证。
