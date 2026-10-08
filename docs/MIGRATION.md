# 迁移与通用化记录

本项目的早期版本是一个面向特定业务的推送服务，现已**通用化并去领域化**为开源推荐/推送微服务。
本文记录：去领域化范围、接口泛化、可扩展点，以及后续事项。

## 1. 去领域化范围

| 类别 | 处理 |
|------|------|
| 领域关键词/话术 | 全部外置到 `config/keywords.json`（通用默认值），代码零领域词 |
| 领域话题/示例数据 | `data/topics.py` 换为通用话题；`data/catalog.json` 换为中性知识/教程目录 |
| 领域模式（如 mentor/proxy 等） | 移除，交互模式概念删除，改为通用 `signals` + `state` |
| 转人工/导师逻辑 | 抽象为通用 `HandoffPolicy`（`handoff/`），文案/关键词中性化、可配置 |
| 决策引擎 | 抽象为 `DecisionPolicy`（`policy/`），默认阈值策略，去领域规则 |
| 厂商专用工具 | 移除云厂商导出脚本，改为通用 `scripts/import_catalog.py` |
| 品牌/溯源引用 | 全部清除（代码、日志、文档） |
| 领域集成测试 | 移除，改为通用策略/排序/存储测试 |

## 2. 接口泛化（破坏性变更）

| 旧 | 新 |
|----|----|
| `video_id` | `item_id` |
| `CandidateVideo` | `CandidateItem` |
| `vod_video_id` | `item_ref` |
| `cover_url` | `media_url` |
| `advisor_suggestion` | `handoff_suggested` |
| 事件 `video_push` / `advisor_push` | `item_push` / `handoff_push` |
| `interaction_mode` / `risk_alert` / `conversation_state` | `signals`（dict） + `state`（str） |
| `resource_type` | `item_type` |
| `decision_config` / `decision_config.json` | `policy_config` / `policy.json` |
| `POST /api/v1/engagement` | `POST /api/v1/feedback` |
| `/admin/decision-config` | `/admin/policy-config` |
| `/admin/videos/reload` | `/admin/items/reload` |
| `/internal/videos/{export,sync}` | `/internal/items/{export,sync}` |
| 包 `mobile/` | 包 `policy/` |
| `server/advisor_service.py` | 包 `handoff/` |

## 3. 新增能力

- 可继承接口：`DecisionPolicy`、`HandoffPolicy`、`RewardAdapter`（见 `docs/EXTENDING.md`）。
- 通用存储层：`selftrain/store/`（Postgres/Redis 为主 + SQLite 回退）。
- 推送成功率指标体系与门闸（见 `docs/SELF-TRAINING.md`）。
- 通用目录导入：`scripts/import_catalog.py`。

## 4. 遗留与后续

- 会话状态为文件存储（单机）；多副本建议改为共享存储。
- 位置偏差校正已输出 `propensity`，可按需启用 IPS 加权。
- 领域无关的默认关键词为示例，接入时在 `config/keywords.json` 覆盖即可。
