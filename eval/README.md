# Xiaoman Eval Harness

这是小满的本地 Agent 评测与数据飞轮入口。它把一次执行规范化为
`AgentRun`，把最终回复、工具轨迹、状态变更、记忆事件和延迟统一送进评分器，
因此可以用同一套样例做 CI 回归，也可以接真实 `AgentLoop` 做诊断。

## 先跑内置样例

在 `app` 目录执行：

```powershell
python -m eval.cli run `
  --dataset eval/datasets/smoke.jsonl `
  --report eval/reports/smoke.json `
  --markdown eval/reports/smoke.md `
  --store data/eval.sqlite
```

该命令使用 JSONL 中的 `replay` 字段，不调用模型、不修改真实会话或记忆，适合
本机和 CI 的快速冒烟测试。报告包含每个 case 的 hard gate、各维度分数和总 reward。
需要保留历史分数时，可在 Python 中使用 `EvalResultStore("data/eval.sqlite")`；
它提供低 reward case 查询，作为人工审核和新增 hard case 的入口。

## Case 结构

一个 case 至少包含 `case_id`、`input`、`expected` 和 `rubric`。运行时会把
`expected` 中的断言编译进同一份 Rubric，因此报告不会同时维护两套评价体系。
每个 criterion 可以设置 `evaluator: "deterministic"` 或 `evaluator: "llm"`：
前者检查回复、工具、状态和记忆等精确事实，后者调用注入的 Judge 判断任务完成度
和表达质量。明确危险或持久化正确性的维度应设置 `hard: true`，硬门禁失败时总
reward 直接为 0。

生产轨迹不要原样复制进黄金集。先脱敏、裁剪凭据和个人内容，再标注失败类型，
由人工审核后升格为版本化 JSONL case。`eval.dataset.mine_hard_cases` 提供了候选
挖掘的最小接口。

## 接入真实 AgentLoop

```python
from eval.dataset import load_cases
from eval.runner import EvalHarness, ProcessDirectExecutor

cases = load_cases("eval/datasets/smoke.jsonl")
executor = ProcessDirectExecutor(agent_loop, trace_store=trace_store)
summary = await EvalHarness(dataset_name="smoke", version="v1").run(cases, executor)
```

`ProcessDirectExecutor` 为每个 case 创建隔离的 session/trace，读取现有
`TraceStore` 中的工具事件；因此不会污染用户正常会话。Langfuse 可以继续作为
远程观测和评分展示层，TraceStore 仍是本地可离线运行的事实来源。

真实回归可注入 `LiveEvalFixtureManager`。它为每个 case 执行
`prepare → AgentLoop → observe → cleanup`：目前支持旧记忆 seed、工具失败注入、
LangGraph checkpoint seed 和长上下文生成，并从真实持久化存储与 TraceStore
构造评分状态。fixture 只存在于指定的隔离 workspace，结束后会清理会话、测试记忆、
临时工具、Workflow 和 checkpoint。

## 评分与后续数据飞轮

当前内置：最终回复、必需/禁止工具、工具顺序和调用次数、状态/记忆写入、执行
状态、延迟，以及 Rubric criterion 的加权聚合。`EvalHarness` 接受一个可选的
`judge(case, run) -> {criterion_id: score}` 是 LLM Judge 的注入接口。内置的
`eval.judge.OpenAICompatibleRubricJudge` 会复用 Xiaoman 配置中的 fast 模型和
OpenAI-compatible endpoint（当前为 DashScope `qwen3.7-flash`），只给显式
`evaluator: "llm"` 的 Rubric 维度打 0~1 分，不改变 case、hard gate 和报告格式。
`eval.datasets.generate_regression --with-llm-judge` 会为回归集增加一个软性的
`response_quality` criterion；没有 Judge 时回退到确定性断言，保证离线 CI 不依赖
网络。真实联调时增加 `--judge` 即可启用模型评分。

Judge 对非 JSON、缺失 criterion 和非法分数最多重试 3 次。如果 Judge 最终仍不可用，
运行器会保留真实 AgentRun，以确定性 Rubric 继续评分，并在 case 的 `error` 字段记录
`judge_degraded`；评分服务故障不会再被误报成 Agent 执行故障。

有 Langfuse client 时，可用 `LangfuseScorePublisher(client)` 把每个 criterion 和
聚合 reward 作为 trace score 上报；上报失败通过 `publish_best_effort` 返回错误，
不会阻断本地回归结果。

变更 prompt、模型、工具策略、记忆或压缩逻辑时，先分别得到 baseline 和 candidate
的 `EvalSummary`，再调用 `eval.compare.compare`。它会检查黄金 case 是否退化、hard
gate 是否新增失败，以及总体 pass rate/reward 是否下降，作为提交或发布门禁。

建议流程：本地冒烟 → nightly 全量 → 从低 reward/失败/负反馈轨迹挖掘候选 → 脱敏
和人工确认 → 加入 hard set → 变更 prompt、工具策略、记忆或压缩逻辑后回归。数据
集至少拆成 `memory`、`execution`、`workflow`、`proactive`、`context`、`safety`、
`schedule`、`retrieval` 八个切片，并保留独立 holdout 集防止过拟合。

## 个人助手回归集

生成当前版本的真实场景回放集：

```powershell
python -m eval.datasets.generate_regression `
  --output eval/datasets/regression_v1.jsonl
```

当前包含 54 个 Case，覆盖 `memory`、`execution`、`workflow`、`proactive`、
`context`、`safety`、`schedule` 和 `retrieval`，每个 Case 都包含用户问题、预期
回复、工具轨迹、状态/记忆断言、难度和可能的失败模式。

批量运行并生成切片统计和失败热点：

```powershell
python -m eval.cli run `
  --dataset eval/datasets/regression_v1.jsonl `
  --dataset-name personal-assistant-regression `
  --version regression-v1 `
  --report eval/reports/regression.json `
  --markdown eval/reports/regression.md `
  --analysis eval/reports/regression-hotspots.json `
  --store data/eval.sqlite
```

回放集的 100% 通过只代表评测器和黄金样例契约正常；要发现小满当前实现的真实
问题，需要将同一批 Case 的 executor 换成 `ProcessDirectExecutor`。此时报告会按
场景切片统计 Pass Rate、Mean Reward、Hard Gate 失败率，并将低分结果归类为例如
`preference_missed`、`duplicate_side_effect`、`fact_loss`、`dnd_violation` 等优化
热点。

真实 AgentLoop 联调使用独立命令，默认只跑前 6 个 Case；确认模型额度和配置后，
可通过 `--limit 54` 跑完整批次：

```powershell
python -m eval.live `
  --dataset eval/datasets/regression_v1_judge.jsonl `
  --limit 54 `
  --judge `
  --workspace data/eval-live-workspace
```

带 LLM Judge 和 Langfuse 评分发布的联调：

```powershell
$env:LANGFUSE_PUBLIC_KEY = "..."
$env:LANGFUSE_SECRET_KEY = "..."
$env:LANGFUSE_BASE_URL = "https://jp.cloud.langfuse.com"
python -m eval.live `
  --dataset eval/datasets/regression_v1_judge.jsonl `
  --limit 6 `
  --judge `
  --publish-langfuse `
  --workspace data/eval-live-judge-smoke
```

`--publish-langfuse` 会把每个 Rubric 分数和聚合 reward 绑定到对应的远程 trace；
本地 trace id 会按 recorder 使用的 deterministic seed 映射为 Langfuse trace id。
没有 Langfuse 凭据时命令会明确失败，不会假装已经发布成功。

它不会使用普通用户会话，而是在独立 workspace/session 下执行；但会调用配置中的
模型服务，批量运行前应确认 API 配额和费用。
