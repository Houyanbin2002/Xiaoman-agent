# 贡献指南

感谢你愿意帮助改进小满。项目仍在快速演进，建议先通过 Issue 描述较大的功能改动，再开始实现。

## 开发环境

```bash
python3.12 -m venv .venv
python -m pip install -r requirements-dev.txt
npm ci
npm run build
```

复制 `config.example.toml` 为 `config.toml`，只填写自己的本地配置。`config.toml`、数据库、日志和工作区数据不得提交。

## 提交前检查

```bash
python -m pytest -q
npm run typecheck
npm run lint
npm run build
```

请同时确认：

- 新行为有对应测试，删除旧模块时同步删除失效测试和文档。
- 核心领域逻辑放在 `core/`，依赖装配放在 `bootstrap/`，外部实现放在 `infra/`。
- 外部写入、发送和敏感数据读取继续遵守审批与访问策略。
- 不提交 API Key、Token、个人聊天、记忆数据库、日志或机器绝对路径。
- 前端改动在桌面与移动端都不存在明显溢出和控件遮挡。

## Pull Request

PR 描述应包含：问题背景、实现方案、验证方式和可见界面变化。涉及数据库结构时说明迁移与回滚方式；涉及用户数据时说明权限、生命周期和删除语义。

建议使用简明的 Conventional Commit 风格，例如：

```text
feat: add calendar signal provider
fix: prevent expired memory from recall
docs: clarify local deployment
```
