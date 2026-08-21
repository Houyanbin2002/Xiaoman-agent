---
name: codex-delegate
description: 把长代码库任务交给本机 Codex CLI，通过小满任务中心持久化执行。当用户说用 codex skill、codex delegate、委托 codex、后台 codex、阻塞 codex exec 时使用。
metadata: {"xiaoman": {"always": false, "requires": {"bins": ["codex"]}}}
---

# Codex Delegate

## 目标

把可独立完成的长任务创建为一个持久化任务。任务先经过用户审批，再由仅开放 `shell` 的 Agent 执行步骤前台等待 `codex exec`；结果由任务中心保存并通知原会话。

## 标准流程

1. 主会话先调用 `shell(command="command -v codex && codex --version", auto_promote=false)` 检查 Codex CLI。检查失败就直接告知用户，不创建任务。
2. 调用 `task_create` 创建两个步骤：先用 `kind="approval"` 请求用户确认运行本机 Codex CLI，再用依赖审批步骤的 `kind="agent"` 执行。
3. 执行步骤必须设置 `executor="agent"`、`allowed_tools=["shell"]`，把权限限制为命令执行；不要使用隔离执行器，因为它的文件系统权限只覆盖步骤目录，无法访问目标代码库。
4. 审批通过后，执行步骤以前台方式运行 `codex exec`。Codex 最终答复、完整日志和会话 ID 应写到目标仓库之外、任务上下文明确指定的工作目录。
5. 任务中心自动记录步骤结果并通知用户。用户询问进度、要求取消或重试时，先用 `task_manage(action="list")` 找到任务，再用 `get`、`events`、`cancel` 或 `retry` 操作。

## 边界规则

- 用户已给出仓库路径时，主会话不要预先探索仓库；把路径与用户目标原样放进步骤描述，让 Codex 自行发现入口、目录和相关文件。
- 只有路径缺失、明显无效，或用户明确要求确认时，主会话才做最小路径检查。
- 不要把主会话猜测的入口文件、候选目录或搜索结果塞进 Codex prompt，除非这些信息来自用户原文。
- 步骤内调用 `shell` 必须设置 `auto_promote=false`，不得设置 `run_in_background=true`。
- `auto_promote=false` 且不传 `timeout` 时会同步等待默认上限；只有需要更短硬截止时才显式设置 `timeout`。
- 使用 `codex exec --cd <repo>` 指定仓库，不依赖 shell 的当前目录。
- 默认从 `prompt.txt` 经 stdin 传入任务说明，避免引号、换行和特殊字符破坏 prompt。
- 必须设置 `--output-last-message <task_dir>/codex-result.md`，不要从临时 shell 日志提取最终答复。
- stdout 与 stderr 一并保存到 `<task_dir>/codex-run.log`，因为会话 ID 可能出现在 stderr。
- prompt、结果、日志和会话 ID 必须写入步骤描述明确指定的委托工作目录，不要污染目标仓库，也不要依赖系统临时日志。
- 不要用 `&`、`nohup` 或 `disown` 包装 `codex exec`。

## 推荐调用

先检查 CLI：

```text
shell(
  command="command -v codex && codex --version",
  description="检查 codex",
  auto_promote=false
)
```

再创建持久化任务：

```text
task_create(
  name="Codex 代码库任务",
  goal="完成用户要求并返回可核验的结果",
  steps=[
    {
      "id": "approve_codex",
      "title": "确认运行 Codex CLI",
      "description": "即将在本机对 /path/to/repo 运行 Codex CLI，并允许其按用户目标读取和修改该仓库。是否批准执行？",
      "kind": "approval"
    },
    {
      "id": "run_codex",
      "title": "运行 Codex",
      "description": "用户仓库路径是 /path/to/repo；委托工作目录是 /path/to/delegate-work；用户目标是：<原样概括目标>。先在委托工作目录准备 prompt.txt，要求 Codex 自行扫描完整仓库并完成任务。用 shell 前台执行 codex exec --cd /path/to/repo --output-last-message /path/to/delegate-work/codex-result.md - < /path/to/delegate-work/prompt.txt，并将 stdout 和 stderr 保存到 /path/to/delegate-work/codex-run.log。提取明确的 session id 到 codex-session.txt。最后通过 shell 读取结果文件，返回完成情况、改动文件和验证结果。",
      "kind": "agent",
      "depends_on": ["approve_codex"],
      "executor": "agent",
      "allowed_tools": ["shell"],
      "max_attempts": 2
    }
  ],
  auto_start=true
)
```

步骤内部推荐命令：

```text
shell(
  command="bash -lc 'set -o pipefail; codex exec --cd /path/to/repo --output-last-message /path/to/step_dir/codex-result.md - < /path/to/step_dir/prompt.txt 2>&1 | tee /path/to/step_dir/codex-run.log; sed -n \"s/^session id: //p\" /path/to/step_dir/codex-run.log | tail -1 > /path/to/step_dir/codex-session.txt'",
  description="运行 codex",
  auto_promote=false
)
```

需要继续同一 Codex 会话时，先用 `task_manage` 读取任务事件和上一步输出，取得明确的 session ID，再创建一个新的持久化步骤任务并执行：

```text
codex exec resume <session_id> --output-last-message /path/to/step_dir/codex-result-2.md - < /path/to/step_dir/prompt-2.txt
```
