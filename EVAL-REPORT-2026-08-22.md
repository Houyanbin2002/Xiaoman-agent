# Xiaoman 个人助手真实评测报告

**评测日期：** 2026-08-22  
**数据集：** `regression_v1_judge.jsonl`  
**执行版本：** `regression-v15-full`  
**执行方式：** 隔离工作区中的真实 AgentLoop + 真实持久化 fixture + Qwen LLM Judge + Langfuse

## 1. 最终结论

本轮已经补齐原先缺失的 24 个真实 fixture，并完成一次全新的 54-case 全量回归：

| 指标 | 结果 |
|---|---:|
| Hard-gate 通过率 | **100%（54/54）** |
| Mean reward | **0.884** |
| Agent 运行错误 | **0** |
| Hard-gate 失败 | **0** |
| Judge 降级 | **0** |
| 全量代码测试 | **1399/1399 通过** |

这说明当前版本化测试集中，记忆写入与纠错、工具失败恢复、Workflow Checkpoint、上下文压缩、安全边界等确定性能力门禁均已通过。这个 100% 只代表当前 54 项 hard gate，不代表开放世界中的所有个人助手任务或自然语言回答质量已经达到 100%。

LLM Judge 的 `response_quality` 均值为 **0.515**，`required_tools` 均值为 **0.700**。因此系统的执行正确性基线已经建立，但日程执行、历史检索和部分主动协助表达仍有可见的软质量优化空间。

## 2. 新增的真实 fixture 链路

评测运行器现在统一采用以下生命周期：

```text
case setup
  ├─ seed existing memory
  ├─ inject tool failure
  ├─ create LangGraph checkpoint
  └─ generate long context
        ↓
真实 AgentLoop 执行
        ↓
检查最终响应、工具轨迹、Checkpoint 与持久化结果
        ↓
case teardown：清理会话、记忆、工具和 Workflow 状态
```

新增的 24 个用例由四组 fixture 组成，每组 6 个：

| Fixture | Setup | 真实检查 | Teardown |
|---|---|---|---|
| 记忆纠错 | 写入带 `user_locked=true` 的旧偏好 | 新值处于 active，旧值被 supersede，版本链和置信度正确 | 删除本 case 新增记忆 |
| 工具失败恢复 | 注册首选失败工具与只读 fallback，并注入真实异常 | Trace 中必须出现 `error → fallback success`，限制重试次数 | 恢复原工具注册表 |
| Workflow Checkpoint | 创建 `WAIT_USER` Workflow，并由 LangGraph 写入 interrupt/checkpoint | 从快照恢复、最终成功，副作用只执行一次 | 删除 Workflow 和 checkpoint thread |
| 长上下文压缩 | 写入旧对话与三轮 Tool Result，降低隔离 case 的触发阈值 | 真实生成摘要，校验关键字段、热后缀和 token 缩减 | 删除测试会话并恢复压缩配置 |

所有 case 使用唯一 session、trace 和隔离 workspace，不读取或污染正式用户会话与长期记忆。

## 3. 分能力结果

| 能力域 | Case | Hard-gate 通过率 | Mean reward |
|---|---:|---:|---:|
| 长期记忆（偏好 + 纠错） | 12 | 100% | 0.948 |
| 工具失败恢复 | 6 | 100% | 0.900 |
| Workflow / Checkpoint / 幂等 | 6 | 100% | 0.941 |
| 主动协助策略 | 6 | 100% | 0.838 |
| Context Compaction / Cache Breakpoint | 6 | 100% | 0.921 |
| 安全与权限 | 6 | 100% | 0.938 |
| 日程与时区 | 6 | 100% | 0.790 |
| 近期会话与记忆召回 | 6 | 100% | 0.737 |

确定性指标：

| 指标 | 均值 |
|---|---:|
| `memory_event` | 1.000 |
| `trajectory` | 1.000 |
| `state_contains` | 1.000 |
| `forbidden_tools` | 1.000 |
| `status` | 1.000 |
| `required_tools` | 0.700 |

## 4. 本轮发现并修复的问题

### 4.1 同一偏好被重复抽取形成中间版本

真实回归发现，“旧项目已经结束，当前主要关注 Xiaoman 项目”可能同时被模型抽成 `project_context`，又被显式规则抽成 `correction`。二者虽然标签不同，但都写入 `active_project` 槽位，原实现会生成：

```text
旧项目 → Xiaoman → Xiaoman 项目
```

现已改为按“来源消息 + 语义槽位”统一去重，不再依赖宽泛标签，版本链变为：

```text
旧项目 → Xiaoman 项目
```

这也回答了“后台再次抽到同一偏好会怎样”：同值候选只增加证据；同一批次同槽位候选只写一次；用户明确纠正会 supersede 冲突旧值，普通后台推断不能覆盖 `user_locked` 记录。

### 4.2 Judge 异常误报为 Agent 失败

原实现中，Judge 偶发返回非 JSON 会覆盖真实 AgentRun，并把用例标成执行错误。现在：

- 非 JSON、缺字段或非数值分数最多重试 3 次；
- 重试时明确要求只返回约定 JSON；
- Judge 最终不可用时保留真实响应、轨迹和状态；
- 确定性 Rubric 继续评分，并写入 `judge_degraded`，不会伪装成 Agent 失败。

### 4.3 用户纠正不再被误当成隐私删除

`forget_memory` 只负责用户明确要求的忘记、删除和隐私擦除。“改为”“规则取消后只在……”等纠正表达统一交给后台语义批次和记忆治理服务，保留可审计的 supersede 版本链。

## 5. 仍需优化的软质量问题

54 个 case 的 hard gate 已通过，但以下 6 个用例没有调用期望工具，当前只作为软扣分：

- `proactive.policy.deadline`
- `proactive.policy.feedback`
- `schedule.intent.one_off`
- `schedule.intent.reschedule`
- `retrieval.history.recent_project`
- `retrieval.history.avoid_noise`

最低 reward 集中在：

- `retrieval.history.avoid_noise`：0.440
- `schedule.intent.reschedule`：0.440
- `retrieval.history.recent_project`：0.500
- `schedule.intent.one_off`：0.500
- `proactive.policy.deadline`：0.500

下一版评测建议把“必须产生持久化副作用”的日程与反馈 case 的 `required_tools` 升为 hard gate，并为改期任务 seed 目标日程、为近期检索 seed 精确 history。这样可以避免“口头答应但没有真正落库”仍被算作能力通过。

## 6. 服务联调状态

| 服务 | 本轮状态 |
|---|---|
| 主模型 `deepseek-v4-pro-0813` | 54 个真实 case 全部完成，0 个运行错误 |
| Judge `qwen3.7-flash` | 完成全量评分，0 次 Judge 降级 |
| Embedding `qwen3.7-text-embedding` | 配置与独立向量调用此前已验证可用 |
| LangGraph Checkpoint | 6/6 真实恢复与幂等检查通过 |
| Langfuse | Rubric score 发布流程完成；出现过一次异步 span batch 5 秒网络超时，不影响本地结果，个别 trace 可能延迟或缺失 |
| QQ/Telegram/飞书/微信/企业微信/MCP | 按要求未纳入本轮联调 |

凭据未写入数据集、报告或代码。已在截图或聊天中暴露过的 Langfuse Secret Key 建议在平台轮换。

## 7. 可复现命令

```powershell
# 完整真实回归
.venv\Scripts\python.exe -m eval.live `
  --dataset eval/datasets/regression_v1_judge.jsonl `
  --limit 54 --judge --publish-langfuse `
  --workspace data/eval-live-v15-full `
  --dataset-name personal-assistant-live `
  --version regression-v15-full `
  --report eval/reports/regression-v15-full.json `
  --markdown eval/reports/regression-v15-full.md `
  --analysis eval/reports/regression-v15-full-analysis.json `
  --store data/eval-live-v15-full.sqlite

# 全量代码测试
.venv\Scripts\python.exe -m pytest -q
```

## 8. 结果文件

- `eval/reports/regression-v15-full.json`：完整结构化结果与真实轨迹
- `eval/reports/regression-v15-full.md`：自动生成的用例报告
- `eval/reports/regression-v15-full-analysis.json`：低分与失败模式分析
- `data/eval-live-v15-full.sqlite`：可查询的本地历史结果

