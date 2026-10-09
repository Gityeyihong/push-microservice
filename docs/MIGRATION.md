# 接入与升级

本仓库是通用推荐/推送微服务。接入时替换配置和目录，不必改服务代码。

## 替换什么

| 内容 | 位置 |
|------|------|
| 关键词、意图规则、对话前缀、强制转人工信号名 | `config/keywords.json` |
| 决策阈值、会话状态调制、策略实现名、停留时长阈值 | `config/policy.json` |
| 条目目录 | `data/catalog.json`，或 `PUSH_CATALOG` 指向的文件 |
| 决策 / 转人工 / 排序实现 | `POLICY_PROVIDER`、`HANDOFF_PROVIDER`、`RANKER_PROVIDER` |

目录字段使用 `item_id`、`item_type`、`media_url`、`item_ref`。推送事件为 `item_push` / `handoff_push`。反馈接口是 `POST /api/v1/feedback`。

## 向量文件

`python -m data.generate_data` 写入 `data/item_embeddings.npy` 与 `data/item_topic_vecs.npy`。若目录里只有旧文件名 `video_embeddings.npy`、`video_topic_vecs.npy`，服务启动时会回退读取。

## 会话状态

`state` 是不透明字符串，调制系数在 `policy.json` 的 `state_modulation`。默认同时包含 `start` / `explore` / `focus` / `close` 和 `INIT` / `GATHERING` / `DEEPENING` / `TAILORING`，请求未传状态时仍是 `INIT`。

## 反馈事件

`watch_end` 仍然接受，并与 `dwell_end` 等价。正例原因码为 `dwell_above_threshold`。阈值可用 `positive_dwell_ratio` 与 `short_dwell_ratio` 覆盖。
