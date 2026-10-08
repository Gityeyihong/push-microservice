# HTTP 接口

默认监听 `0.0.0.0:8000`。除 `/health` 外均需鉴权。

## 鉴权

| 分组 | 头 | 令牌来源 |
|------|----|----------|
| `/admin/*` | `Authorization: Bearer <token>` | `PUSH_ADMIN_TOKEN`（未配置返回 503） |
| `/internal/*` | `X-Internal-Token: <token>` | `PUSH_INTERNAL_TOKEN`，回退 `PUSH_ADMIN_TOKEN` |

## 公共

### `GET /health`

```json
{ "status": "ok", "redis": true, "push_use_mock_vectors": "0",
  "encoder": "real", "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "model_source": "local", "vector_mode": "real", "video_count": 100 }
```

### `POST /api/v1/push`

请求：

```json
{
  "context": "用户: 我们最近总是冷战...\n助手: ...",
  "load_factor": 0.8, "positive_rate": 0.6, "intimacy": 0.4, "urge": 0.7,
  "trigger_mode": "auto", "resource_type": "video",
  "experiment_id": null, "risk_alert": false,
  "conversation_state": "INIT", "session_id": "s-1",
  "interaction_mode": "auto",
  "push_intent": {
    "primary_topic": "冷战", "intent_query": "怎么打破冷战",
    "exclude_topics": [], "is_meta_request": false,
    "content_need": "video", "confidence": 0.8
  }
}
```

响应：

```json
{
  "candidates": [
    { "video_id": "v_001", "type": "video", "similarity": 0.83, "prob_base": 0.42,
      "title": "冷战期如何破冰沟通", "cover_url": "https://cdn.example.com/thumbs/v001.webp",
      "duration_sec": 60, "vod_video_id": null }
  ],
  "advisor_suggestion": false,
  "generated_at": "2026-10-08T00:00:00+00:00",
  "decision_config": { "...": "..." },
  "push_events": [ { "type": "video_push", "data": { "...": "..." } } ],
  "state_summary": { "...": "..." }
}
```

### `POST /api/v1/engagement`

```json
{ "video_id": "v_001", "event": "watch_end",
  "watch_duration_sec": 40, "video_duration_sec": 60,
  "session_id": "s-1", "timestamp": null }
```

`event` ∈ `exposed|click|watch_end|dismiss|no_click|share|favorite|not_interested`。

响应：

```json
{ "accepted": true, "classified_feedback": "positive", "reason": "watch_above_threshold" }
```

## 管理（`/admin`，Bearer）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/admin/status` | 编码器 + 决策配置 + A/B + 自训练状态 |
| GET/PUT | `/admin/decision-config` | 读取/更新决策配置 |
| GET/PUT | `/admin/profiles` | 读取/更新 A/B profiles |
| GET/POST/DELETE | `/admin/ab-experiments` | A/B 实验启停 |
| GET | `/admin/sessions` | 列出会话 |
| DELETE | `/admin/sessions/{id}` | 重置会话 |
| POST | `/admin/videos/reload` | 热重载目录/向量 |
| POST | `/admin/cache/flush` | 清空缓存 |
| GET | `/admin/audit` | 审计日志 |

### 自训练

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/admin/training/status` | 当前状态、active 版本、上次结果 |
| GET | `/admin/training/versions` | 版本列表（含 held-out 指标、失败版） |
| POST | `/admin/training/run` | 手动触发一轮（body 可带 `{"force": true}`） |
| POST | `/admin/training/build-dataset` | 仅构建并查看门控统计 |
| POST | `/admin/training/samples/import` | 人工导入审查通过的样本对 |
| POST | `/admin/training/rollback` | 回滚到上一 active 版本 |
| GET | `/admin/training/eval` | 最近一次 held-out 评估报告 |

## 内部（`/internal`，Header 令牌）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/internal/videos/export` | 导出视频目录 JSON |
| POST | `/internal/videos/sync` | 覆盖目录并重建向量（失败自动回滚目录） |
