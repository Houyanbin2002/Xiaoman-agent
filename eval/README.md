# Xiaoman Eval Harness

当前验收入口为下方“V2 多轮验收与稳定性评测”。旧 `regression_v1` / replay 仍可用于
评分器自测，不应拿它的 hard gate 通过率作为个人助手任务完成率。

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

外部执行适配器可以用 `objective_outcome` / `delivery_status` 声明经过核验的结果；
没有观测结果时必须保持 unknown，不能从模型的“已完成”文字推断。这些字段不会由
现有内核自动提供。对于当前真实场景，推荐使用确定性证据和显式验收契约：

```json
{
  "expected": {
    "evidence": {
      "saved": {"source":"state", "contains":{"artifacts":{"plan.json":{"exists":true,"json":{"budget":900}}}}}
    }
  },
  "metadata": {"acceptance":{"objective_checks":["saved"]}}
}
```

`state` 必须由 fixture 的 observe 从实际存储读取。`required_tool_status` 仅核验
工具状态；需要参数和结果核验时使用 `evidence` 的 `source=tool`，声明 `arguments`、
`status`、`output`。文件落盘不等于网页下载或渠道已发送，不能据此标记 delivered。

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

真实 AgentLoop 联调使用独立命令，默认 dev 切片的前 6 个 Case；`--limit 0` 表示
运行所有已选择用例。旧 54 例完整批次须显式选择 `--split all`：

```powershell
python -m eval.live `
  --dataset eval/datasets/regression_v1_judge.jsonl `
  --limit 0 --split all `
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

为降低模型随机性，可对同一批用例重复运行并保留每次报告：

```powershell
python -m eval.live --dataset eval/datasets/regression_v1_judge.jsonl `
  --judge --repeats 3 --report eval/reports/repeated.json `
  --markdown eval/reports/repeated.md
```

报告会额外写入 `.r1`、`.r2`、`.r3` 文件。也可以使用稳定哈希划分的 holdout 子集：

```powershell
python -m eval.live --dataset eval/datasets/regression_v1_judge.jsonl `
  --split holdout --holdout-percent 20 --limit 0 --judge
```

默认 dev 自动排除 holdout；数据集可显式指定 `metadata.split`，否则按 family（缺省
case_id）做稳定哈希划分，同一场景的改写归入同一集合。holdout 不参与提示词调优。

它不会使用普通用户会话，而是在独立 workspace/session 下执行；但会调用配置中的
模型服务，批量运行前应确认 API 配额和费用。
# 真实评测口径

报告中的 `pass_rate` 是 Rubric hard gate，不等同于模型质量或任务完成率。真实评测同时查看：

- `execution_completed_cases`：AgentLoop 正常终止的场景数；
- `quality_passed_cases / quality_assessed_cases`：Judge 达到质量阈值的场景数；
- `rubric_acceptance_cases`：执行已完成且所有确定性与 Judge 条件均满足的场景数。

故障恢复场景使用 `recovery_v2_judge.jsonl`。备用夹具必须绑定 `status=success|empty|unavailable`，分别表示有可交付数据、明确空结果和数据不可用，禁止用占位字符串伪造成功。

## V2 多轮验收与稳定性评测

```powershell
python -m eval.live --dataset eval/datasets/acceptance_v2.jsonl `
  --split dev --limit 0 --repeats 3 --judge --case-timeout 240 `
  --workspace data/eval-acceptance-v2 `
  --report eval/reports/acceptance-v2.json --markdown eval/reports/acceptance-v2.md
```

- 数据集包含 9 个场景：6 个 dev、3 个 holdout；7 个多轮场景、2 个受控恢复场景。
- `metadata.turns` 描述真实用户轮次，`session` 相同延续会话、不同切换会话；不注入标准答案。
- 每个 case、每次 repeat 新建独立 runtime/workspace；同一场景内部共享记忆，最后统一清理会话与测试记忆。隔离目录和生成文件保留供复核，不触碰生产数据。
- `expected.evidence` 支持真实状态、工具参数/返回体、JSON 类型与数值、记忆条数、指定轮次证据。文件 fixture 只读预先声明路径，记录实际 JSON、字节数和 SHA256，不替 Agent 生成结果。
- `metadata.acceptance.objective_checks` 指定完成目标所需证据；`blocked_checks` 验证受阻原因，`expected_outcome=blocked` 可表示应诚实报告受阻的负向案例。负向案例验收合格依然不是目标完成。
- `delivery_checks` 只能绑定可信收件回执/HTTP 下载证据；本批次没有实际发送，不宣称 delivery 已覆盖。
- 默认 `--gate acceptance` 要求所有 Rubric 条件和声明的验收契约满足，Judge 缺失/降级、空评分均不通过；`--gate hard` 仅兼容旧诊断口径。
- `.rN.json/.md` 保留每轮，主报告汇总全部 trial，显示独立场景数、所有重复都通过的场景数、波动场景、延迟 P50/P95。Wilson 区间仅作描述；重复采样相关，不代表真实用户总体通过率。
- Manifest 记录数据集、代码和配置指纹、模型、Judge、切片、重复次数与超时；不输出凭据。比较时数据集/fixture、Judge 或执行范围变化会标记不可直接比较，不把不同口径称为 A/B 提升。
- 评分失败保存到本地结果库；质量失败即使综合 reward 高，也进入失败热点。逐 case 写入部分报告，避免中断后整批结果丢失。

### Judge 校准

```powershell
python -m eval.calibration --report eval/reports/judge-calibration.json
```

`judge_calibration_v1.jsonl` 的 6 条样例包含“恢复说明误扣分”“不可用误判为空”和
数值错误等正反例。当前标签是 `ai_draft`，只能检查评分一致性；人工复核后才可改为
`human_reviewed`。报告单独统计人工样本混淆矩阵和一致率，未标注时显示 null，不编造校准精度。

### 覆盖边界与后续验收

| 层级 | 当前证据 | 不可外推为 |
|---|---|---|
| 离线契约 | 评分器、聚合、隔离、异常和门禁测试 | 用户任务成功率 |
| 真实内核 | 多轮记忆应用、文件修改/批准/取消、受控恢复 | 网页端到端、真实第三方可用性 |
| 待补产品闭环 | 网页发问→流式收尾→附件下载；停止/进程退出→重启恢复；主动取消/免打扰→排队发送回执 | 未运行不能标记通过 |

下一阶段应把上述产品闭环各做成独立适配器，复用本套评分与报告。现有 Workflow
WAIT_USER fixture 只测图暂停继续，不等同于真实进程崩溃与副作用去重。Token 成本、
首字延迟、缓存命中需要完整使用量/流事件证据，当前报告不以估算填补缺失值。
