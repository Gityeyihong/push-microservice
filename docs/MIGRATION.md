# 迁移与去敏感记录

来源：`E:\LoveCoach_demo\push-system`（LoveCoach 单体仓库内）。
目标：`E:\push-microservice`（独立、自包含微服务）。

本文记录：(1) 移除的敏感内容；(2) 解除的单体耦合；(3) 未处理的遗留与后续事项。

## 1. 移除的敏感内容

| 类别 | 文件/位置 | 处理 |
|------|-----------|------|
| 真实媒资签名 URL（含 OSS AccessKeyId / bucket） | `data/sample_videos_100.json`、`data/aliyun_videos.json`、`data/aliyun_videos.csv`、`data/aliyun_videos_raw.json` | 删除，改默认合成目录 `data/sample_videos.json`（`cdn.example.com` 占位） |
| 媒资历史备份 | `data/sample_videos_100.json.bak.*` | 删除 |
| 真实用户会话（153 个） | `data/sessions/*.json` | 删除，保留空目录 `.gitkeep` |
| 运行期状态/PII | `data/mobile_state.json`、`reports/engagement_events.jsonl` | 删除 |
| 日志 | `logs/test_log.txt` | 删除 |
| 真实企业微信客服链接 | `mobile/decision_config.py` `DEFAULT_ADVISOR_LINK` | 改为环境变量 `ADVISOR_LINK`（默认占位） |
| 品牌串 | `LoveCoach` / `DeepAttract` / `BFF DeepSeek` / `E:\android` / `深爱` 等 | 统一替换为通用名（Push / 上游意图分析 / 交互模式） |
| 单体根目录调试埋点 | `server/main.py`、`server/relevance_guard.py`、`mobile/decision_engine.py` 的 `#region agent log`、`parents[2]/debug-*.log` | 整段删除 |
| 空/脏文件 | `100`（0 字节） | 删除 |
| 单体指针文档 | `AI_BRIDGE.md` | 删除 |

> 说明：真实视频标题/`vod_video_id` 属于业务数据，随真实媒资一并通过替换为合成目录而移除。

## 2. 解除的单体耦合

| 位置 | 原行为 | 新行为 |
|------|--------|--------|
| `server/config.py` | 读取 `../../.env`（单体根） | 读取本项目 `./.env` |
| `server/main.py` | 调试日志写到 `parents[2]`（单体根） | 删除 |
| `server/relevance_guard.py` | 调试日志写到 `parents[2]` | 删除 |
| `scripts/update_push_videos.sh` | 硬编码 `/opt/lovecoach-push`、`/opt/lovecoach-bff`、`/www/wwwroot/BFF` | 参数化 `PUSH_DIR` 等，去业务路径 |
| `scripts/push_watchdog.sh` | 硬编码 `/opt/lovecoach-push` | 参数化 `PUSH_DIR` |
| `scripts/install_push_watchdog.sh` | 引用缺失的 `deploy/push-watchdog.{service,timer}` | 补齐单元文件 |
| `README.md` / `AI_BRIDGE.md` | 指向单体 `E:\LoveCoach_demo`、`docs/cloud-integration.md`、`E:\android` | 重写为项目内文档 |
| `docker-compose*.yml` / `Dockerfile*` | 镜像名 `lovecoach-push` | 改名 `push-microservice` |
| 构建/启动脚本 | 依赖单体 `scripts\start_push.bat` 等 | 项目内 `scripts/` 提供等价脚本 |
| 测试脚本 | `test_*.py` 位于根、含单体/branding 引用 | 迁入 `tests/`，去 branding |

## 3. 依赖与模型

- 模型 `models/paraphrase-multilingual-MiniLM-L12-v2/`（4.4GB）**不打入**；保留下载脚本
  `scripts/download_embedding_model.py`，安装时下载（核心权重 ~449MB）。
- 向量产物（`*.npy`、`*.bin`）由 `data/generate_data.py` 现场生成，不入版本库。
- 训练依赖与 serving 依赖分离：`requirements.txt`（serving） / `requirements-train.txt`（自训练） / `requirements-vod-export.txt`（VOD 导出）。

## 4. 保留的功能

- 核心：`/api/v1/push`、`/api/v1/engagement`、`/health`。
- 管理：`/admin/*`（决策配置、A/B、会话、审计、向量热重载、自训练控制）。
- 内部：`/internal/*`（目录导出、安全同步）。
- 工具：阿里云 VOD 批量导出（`scripts/export_aliyun_vod_videos.py`）。
- 新增：`selftrain/`（见 `docs/SELF-TRAINING.md`）。

## 5. 遗留与后续

- 会话状态为**文件存储**（单机）。多副本部署需替换为共享存储/Redis。
- `StateManager` 与移动端逻辑对齐，仅服务端使用；跨端同步需另行约定。
- 自训练默认关闭；需配置外部审查器后方可产出正例（见 SELF-TRAINING §4）。
- 若需重新接入真实媒资，使用 VOD 导出工具生成目录后重建向量，勿提交真实签名 URL。
