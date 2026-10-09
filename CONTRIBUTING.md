# 贡献

1. 从主分支拉出功能分支。
2. 保持领域内容在配置里：关键词放 `config/keywords.json`，目录放 `data/catalog.json`，策略参数放 `config/policy.json`。
3. 新行为补测试。默认 `pytest` 不跑标记为 `slow` 的用例。
4. 不要提交 `.env`、向量产物（`*.npy`、`*.bin`）或训练出来的模型权重。

本地检查：

```powershell
pip install -r requirements.txt -r requirements-train.txt
pytest
```
