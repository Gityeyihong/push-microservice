# HTTP 接口

默认监听 `0.0.0.0:8000`。`/health` 是进程存活。`/ready` 在索引缺失或 Redis 不可用时返回 503，Compose 用它做健康检查。`PUSH_ENV=production` 时，占位 `HMAC_SECRET_KEY`、空的 `PUSH_API_TOKEN` / `PUSH_ADMIN_TOKEN` / `PUSH_INTERNAL_TOKEN`、占位或空的 `REDIS_PASSWORD`、或 `PUSH_USE_MOCK_VECTORS=1` 会使进程拒绝启动。`/api/v1/*` 默认按客户端地址每分钟 120 次（`PUSH_RATE_LIMIT_PER_MIN`）。只有直连对端属于 `PUSH_TRUSTED_PROXIES` 时才使用 `X-Forwarded-For`。请求体超过 `PUSH_MAX_BODY_BYTES`（默认 262144）返回 413。`production` 关闭 `/docs`。

## 鉴权

| 分组 | 头 | 令牌来源 |
|------|----|----------|
| `/api/v1/*` | `Authorization: Bearer <token>` | `PUSH_API_TOKEN`。未设置时开放，便于本地启动 |
| `/admin/*` | `Authorization: Bearer <token>` | `PUSH_ADMIN_TOKEN`（未配置 503） |
| `/internal/*` | `X-Internal-Token: <token>` | `PUSH_INTERNAL_TOKEN`，回退 `PUSH_ADMIN_TOKEN` |

## 公共

### `GET /health`

进程存活。Redis 掉线时这里仍是 `ok`。

```json
{ "status": "ok", "redis": false, "push_use_mock_vectors": "0",
  "encoder": "real", "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "vector_mode": "real", "item_count": 30 }
```

### `GET /ready`

索引未加载或 Redis 不可用返回 503。看门狗探测 `/health`，编排用 `/ready` 决定是否接入流量。

### `POST /api/v1/push`

```json
{
  "context": "user: can you recommend a tutorial on integrations",
  "positive_rate": 0.6,
  "trigger_mode": "auto",
  "item_type": "article",
  "experiment_id": null,
  "signals": { "risk_alert": false },
  "state": "INIT",
  "session_id": "s-1",
  "intent": { "primary_topic": "integrations", "query": "integration tutorial",
              "exclude_topics": ["billing"], "is_meta": false, "content_need": "item", "confidence": 0.8 }
}
```

`trigger_mode=active` 在有候选时直接走主动推荐；`auto` 仍按关键词判断。`item_type` 只保留同类型候选。`experiment_id` 填策略画像名则强制该画像；填当前实验名则仍按 `session_id` 分桶；未知值返回 400。`intent.primary_topic` 给命中标签或正文的候选加分，`exclude_topics` 去掉命中项。`content_need=none` 不改变决策，`item` 在没有强制转人工信号时不转人工，`handoff` 优先转人工。负荷、亲密度和催促由会话状态计算，请求体里不再接受这些字段。

响应：

```json
{
  "candidates": [
    { "item_id": "i_011", "item_type": "article", "similarity": 0.71, "prob_base": 0.2,
      "title": "Integrations Overview", "media_url": "https://cdn.example.com/thumbs/i_011.webp",
      "duration_sec": 60, "item_ref": null, "meta": {} }
  ],
  "handoff_suggested": false,
  "generated_at": "2026-...",
  "policy_config": { "...": "..." },
  "push_events": [ { "type": "item_push", "data": { "item_id": "i_011", "trigger": "active" } } ],
  "state_summary": { "...": "..." }
}
```

### `POST /api/v1/feedback`

```json
{ "item_id": "i_011", "event": "watch_end",
  "watch_duration_sec": 40, "item_duration_sec": 60, "session_id": "s-1" }
```

`event` ∈ `exposed|click|watch_end|dwell_end|dismiss|no_click|share|favorite|not_interested`。`watch_end` 与 `dwell_end` 等价，按停留时长占比分类。

响应：`{ "accepted": true, "classified_feedback": "positive", "reason": "dwell_above_threshold" }`

## 管理（`/admin`，Bearer）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/admin/status` | 编码器 + 策略配置 + A/B + 自训练 |
| GET/PUT | `/admin/policy-config` | 读取/更新策略配置 |
| GET/PUT | `/admin/profiles` | A/B profiles |
| GET/POST/DELETE | `/admin/ab-experiments` | A/B 实验 |
| GET | `/admin/sessions` | 会话列表 |
| DELETE | `/admin/sessions/{id}` | 重置会话 |
| POST | `/admin/items/reload` | 热重载目录/向量 |
| POST | `/admin/cache/flush` | 清空缓存 |
| GET | `/admin/audit` | 审计日志 |

### 自训练

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/admin/training/status` | 状态 + 漏斗 + active 版本 |
| GET | `/admin/training/versions` | 版本列表 |
| GET | `/admin/training/funnel` | 曝光漏斗计数 |
| POST | `/admin/training/run` | 触发一轮（body `{"force": true}`） |
| POST | `/admin/training/build-dataset` | 增量构建样本（不训练） |
| POST | `/admin/training/samples/import` | 人工导入审查样本 |
| POST | `/admin/training/rollback` | 回滚 active 版本 |
| GET | `/admin/training/eval` | 最近评估报告 |
| POST | `/admin/training/store/compact` | 存储压缩 |

## 内部（`/internal`，Header 令牌）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/internal/items/export` | 导出目录 JSON |
| POST | `/internal/items/sync` | 覆盖目录并重建向量（失败回滚） |
