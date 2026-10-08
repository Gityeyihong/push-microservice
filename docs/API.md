# HTTP 接口

默认监听 `0.0.0.0:8000`。除 `/health` 外均需鉴权。

## 鉴权

| 分组 | 头 | 令牌来源 |
|------|----|----------|
| `/admin/*` | `Authorization: Bearer <token>` | `PUSH_ADMIN_TOKEN`（未配置 503） |
| `/internal/*` | `X-Internal-Token: <token>` | `PUSH_INTERNAL_TOKEN`，回退 `PUSH_ADMIN_TOKEN` |

## 公共

### `GET /health`

```json
{ "status": "ok", "redis": false, "push_use_mock_vectors": "0",
  "encoder": "real", "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "vector_mode": "real", "item_count": 30 }
```

### `POST /api/v1/push`

```json
{
  "context": "user: can you recommend a tutorial on integrations",
  "positive_rate": 0.6,
  "signals": { "risk_alert": false },
  "state": "INIT",
  "session_id": "s-1",
  "intent": { "primary_topic": "integrations", "query": "integration tutorial",
              "exclude_topics": [], "is_meta": false, "content_need": "item", "confidence": 0.8 }
}
```

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

`event` ∈ `exposed|click|watch_end|dismiss|no_click|share|favorite|not_interested`。

响应：`{ "accepted": true, "classified_feedback": "positive", "reason": "watch_above_threshold" }`

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
