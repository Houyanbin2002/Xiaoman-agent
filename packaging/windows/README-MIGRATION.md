# 小满 Windows 私人迁移包

这个包用于把当前电脑上的小满迁移到另一台 **Windows x64** 电脑。它是可继续修改的源码迁移包，不是冻结后的 EXE。包内包含当前工作区代码、Git 历史与未提交修改、已构建前端、个人记忆、会话、任务、主动协助状态、Skill、MCP 配置及 `config.toml`。

## 安装

1. 把整个 ZIP 复制到新电脑。
2. 完整解压到一个固定目录，例如 `D:\Xiaoman-Agent`，不要直接在压缩包内运行。
3. 双击 `install.cmd`。脚本会检查 Python 3.12；缺失时使用 winget 安装，并创建独立虚拟环境、安装依赖、恢复个人数据。
4. 安装结束后双击 `start.cmd`。
5. 浏览器打开 <http://127.0.0.1:2236/>。

如果还要在新电脑继续开发，基础安装完成后再双击 `setup-development.cmd`。它会准备 Git、Node.js、Python 测试依赖和前端依赖；之后可直接用 VS Code 或 Codex 打开 `app` 目录。

## 迁移范围

已迁移：

- 当前分支、Git 提交历史、远程仓库配置和未提交修改在内的完整源码工作区；
- `personal.db`、`sessions.db`、`langgraph-checkpoints.db`、`langgraph-workflows.db`、`langgraph-workflow-index.db`、`proactive.db`、记忆数据库等一致性快照；
- 定时任务、主动数据源、MCP 配置、已安装 Skill；
- `config.toml` 中的模型和频道配置；
- 小红书 MCP 的 Windows 二进制与本地数据。

需要重新建立：

- Notion、Gmail 等 OAuth 授权；
- 微信扫码登录和其他保存在 Windows 凭据管理器中的 Token；
- MarkItDown MCP 的 Python 环境会由安装脚本重新下载，避免复制旧电脑不可用的虚拟环境。
- GitHub 登录凭据；Git 仓库与远程地址仍在，但首次拉取或推送时需在新电脑重新登录。

## 安全提醒

这是**私人迁移包**，不是公开发布包。`migration/config.toml` 可能包含模型 API Key、频道密钥，数据库包含个人记忆和完整会话。

- 不要上传到 GitHub、网盘公开链接或发送给其他人；
- 建议使用可信 U 盘或加密磁盘传输；
- 迁移完成并确认可用后，删除中间复制文件；
- 如果包曾离开你的控制范围，应立即轮换 API Key 和频道密钥。

新电脑已存在 `~/.xiaoman` 时，安装脚本会先改名备份为 `.xiaoman.before-migration-时间戳`，不会直接覆盖后删除。
