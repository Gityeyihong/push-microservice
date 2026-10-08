# 架构

## 1. 定位

Push Microservice 是一个**独立、自包含**的推送检索与决策服务：

- 输入：对话上下文、会话状态、可选的预分析意图（`push_intent`）。
- 输出：候选视频列表、是否建议转人工、决策配置快照、推送事件。
- 边界：服务本身**不调用生成式 LLM**。意图可由上游分析后透传。
- 训练：`selftrain/` 提供可选的 embedding 自训练闭环（外部审查门控）。

## 2. 运行时组件

```
                    ┌──────────────────────────────────────────────┐
   HTTP 客户端 ───► │  FastAPI (server/main.py)                     │
                    │   /api/v1/push   /api/v1/engagement  /health  │
                    │   /admin/*       /internal/*                  │
                    └───────┬───────────────────────────┬──────────┘
                            │                           │
              ┌─────────────▼──────────┐     ┌──────────▼───────────┐
              │ 向量检索                 │     │ 决策 / 状态           │
              │ VectorService           │     │ DecisionEngine        │
              │  - MiniLM 编码          │     │  - 冲动值/冷却/配额    │
              │  - FAISS HNSW           │     │ StateManager(会话)    │
              │  - MMR + 主题平滑       │     │ engagement 分类        │
              │ relevance_guard 护栏    │     │ push_decision_service │
              └─────────────┬──────────┘     └──────────┬───────────┘
                            │                           │
              ┌─────────────▼──────────┐     ┌──────────▼───────────┐
              │ data/ 向量产物 + 目录    │     │ CacheService(Redis)   │
              │ models/(本地模型)        │     │ 可降级                 │
              └────────────────────────┘     └──────────────────────┘
                            ▲
                            │  active 版本指针
              ┌─────────────┴──────────┐
              │ selftrain/ 自训练闭环    │
              │ Registry / Trainer …    │
              └────────────────────────┘
```

## 3. 模块边界

| 模块 | 职责 | 关键约束 |
|------|------|----------|
| `server/main.py` | FastAPI 应用、生命周期、端点编排 | 不在请求线程内训练 |
| `server/vector_service.py` | 编码、FAISS 检索、MMR、主题权重 | 依赖 `data/` 中的 `.npy`/`.bin`/目录 |
| `server/relevance_guard.py` | 意图查询构建 + 词法/向量融合重排 | 纯函数式，无外部 IO |
| `server/embedding_model.py` | 解析模型路径（训练版本 → 基础模型 → Hub） | 训练版本优先于基础模型 |
| `server/cache_service.py` | HMAC 键 + Redis 缓存，可降级 | Redis 不可用不阻塞 |
| `server/admin_routes.py` | 管理 API（Bearer） | `PUSH_ADMIN_TOKEN` 必须配置 |
| `server/internal_routes.py` | 内部 API（Header 令牌） | 视频同步会重建索引 |
| `mobile/decision_config.py` | 决策参数加载/校验/热重载 | 单一配置来源 `config/decision_config.json` |
| `mobile/decision_engine.py` | 推送决策状态机 | 与移动端逻辑对齐 |
| `mobile/engagement.py` | 互动事件分类 | 训练语料的打标来源 |
| `mobile/state_manager.py` | 会话状态持久化 | 文件存储，单机 |
| `data/generate_data.py` | 离线向量/索引生成 | mock 与 real 两种模式 |
| `selftrain/*` | embedding 自训练 | 见 `docs/SELF-TRAINING.md` |

## 4. 请求流程（`/api/v1/push`）

1. 校验 `context` 非空。
2. 解析会话决策配置（`ab_resolver` 注入 A/B profile）。
3. 若 `push_intent.confidence ≥ 0.5` 跳过缓存。
4. 命中缓存：对缓存候选重跑 `relevance_guard`。
5. 未命中：`VectorService.process_request` 编码 + 检索 + MMR → `relevance_guard` 重排。
6. `AdvisorService` 评估是否建议转人工。
7. 写缓存；`run_push_decision` 运行决策引擎，产出推送事件与状态摘要。

## 5. 数据与产物

| 路径 | 内容 | 生成方式 |
|------|------|----------|
| `data/<PUSH_VIDEOS>` | 视频目录 JSON（合成样例） | 手工 / VOD 导出 |
| `data/video_embeddings.npy` | 视频向量 | `data/generate_data.py` |
| `data/topic_vecs.npy` | 主题向量 | 同上 |
| `data/video_topic_vecs.npy` | 视频-主题矩阵 | 同上 |
| `data/faiss_index.bin` | FAISS HNSW 索引 | 同上 |
| `data/vector_manifest.json` | 向量元数据（mode/count/hash） | 同上 |
| `data/sessions/*.json` | 会话状态（运行期，git 忽略） | 服务运行时 |
| `data/training/*` | 自训练语料/基线与报告 | `selftrain/` |
| `models/versions/gen_N/vk/` | 训练得到的模型版本 | `selftrain/trainer.py` |

## 6. 部署形态

- 本机：`python run_server.py`（开发）。
- 容器：`docker compose up`（Redis + 服务）；`docker-compose.cloud.yml` 挂载 `config/`、`server/`、`mobile/`、`data/` 以便热更新。
- 训练依赖与 serving 依赖分离（`requirements-train.txt`）。

## 7. 可观测性

- 结构化日志（`logging`），请求级 `METHOD PATH → status (elapsed)`。
- `/health`：`status`、`redis`、编码器状态。
- `/admin/status`：编码器、决策配置、A/B 状态。
- `/admin/audit`：管理操作审计（JSONL）。
- 自训练事件流：`selftrain/events.py`（JSONL，可选 Redis）。
