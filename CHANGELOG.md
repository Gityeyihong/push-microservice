# 变更记录

## 1.0.0

- 许可证定为 Apache-2.0，并注明默认句向量模型来源。
- 决策策略改为经 `load_policy()` 加载：`POLICY_PROVIDER` 优先，否则读 `policy.json` 的 `provider`。
- 增加 `RANKER_PROVIDER` 与可选的 `PUSH_API_TOKEN`（未设置时公共接口保持开放）。
- 向量产物使用 `item_embeddings.npy` / `item_topic_vecs.npy`，读取时回退旧文件名。目录环境变量 `PUSH_CATALOG` 优先于 `PUSH_VIDEOS`。
- 对话前缀、强制转人工信号名改为配置项。反馈接受 `dwell_end`（与 `watch_end` 等价），原因码改为停留时长表述。
- 生产启动同时拒绝占位 Redis 密码和 `PUSH_USE_MOCK_VECTORS=1`。限流只在 `PUSH_TRUSTED_PROXIES` 命中时采信 `X-Forwarded-For`。
- Redis 7 需要密码，默认只绑定本机。会话键默认 7 天过期。镜像构建排除 `.env`，进程以 uid 10001 运行。
- `production` 关闭 OpenAPI，并拒绝在服务进程里跑自训练（`TRAINING_IN_PROCESS=1` 可显式打开）。调度镜像在构建期安装训练依赖。
