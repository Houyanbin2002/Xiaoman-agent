"""Live-LLM A/B benchmark; isolated SQLite, synthetic documents, no outbound channels.

Run with python -m eval.delegation_live_benchmark. API credentials are loaded by
the normal config loader and never written to reports. No simulated tool delay.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from agent.config import load_config
from agent.core.passive_turn import AgentExecutionKernel
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.ports import LLMConfig, LLMServices
from agent.runtime.delegation import current_scope
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.runtime.reasoning_policy import WORKFLOW_REASONING_CONTEXT_KEY
from agent.subagent import SubAgent
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.workflow import TaskCreateTool, TaskManageTool
from agent.workflows.runtime import WorkflowRuntime
from bootstrap.providers import build_providers
from core.workflow.models import StepExecutor, StepKind, StepSpec
from infra.persistence.workflow_store import WorkflowStore
from prompts.agent import build_agent_behavior_rules_prompt

DOCS = {
    "trip_brief": "出差安排已确认：周四列车19:40抵达杭州东站，车站至酒店约35分钟。请先读取酒店政策文件 hotel_policy 再决定能否赶上入住。用户预算500元，只住一晚。不要改动预订。",
    "hotel_policy": "现有预订为城南酒店，房费420元；前台21:00停止办理入住；有不可退的服务费30元。酒店允许晚餐外卖。用户要求总价含服务费，给出预计到达酒店时间。",
    "venue_linan": "林岸会务报价：可容纳20人，场租6000元，餐饮每人120元，设备费400元；18人活动。有素食，可提供2份；取消需提前48小时。总费用没有其他项目。报价有效7天，停车另计但本次全员公共交通。",
    "venue_yungu": "云谷会务报价：可容纳25人，场租4800元，餐饮每人150元，设备费800元；18人活动。不提供素食，也不允许外带餐；取消需提前7天。没有其他费用。",
    "venue_muguang": "沐光会务报价：最多容纳16人，场租5500元，餐饮每人110元，设备费500元；本次18人，不允许超员。有素食；取消需提前24小时。没有其他费用。",
}
CASES = {
    "simple": '会议12:30开始，持续90分钟。只输出JSON：{"end_time":"HH:MM"}。不需要检索。',
    "dependent": "帮我核对出差入住，先读 trip_brief，再按其中指引读酒店政策。只核对不预订。最后只输出JSON，包含arrival、total、can_check_in三个字段，分别为预计抵达酒店时间HH:MM、总价数字、能否在截止前入住的布尔值。",
    "independent": "帮我比较林岸、云谷、沐光三个会务方案，对应 venue_linan、venue_yungu、venue_muguang 三份独立报价。18人，必须照顾2名素食者，预算9000元，需要可提前48小时取消。请分别核实各方案再汇总。只分析，不预订；最后只输出JSON：totals为三家名称到含餐饮设备总价数字的映射，recommended为唯一满足全部条件的场地名称，reasons为其他方案不适合的原因。",
}


class ReadDocument(Tool):
    name = "read_case_document"
    description = (
        "读取本次个人助手任务的只读资料，document_id 使用用户或已读资料提供的精确编号。"
    )
    parameters = {
        "type": "object",
        "properties": {"document_id": {"type": "string"}},
        "required": ["document_id"],
    }

    def __init__(self, events):
        self.events = events

    async def execute(self, **kwargs):
        key = kwargs.get("document_id", "")
        self.events.append({"document_id": key, "at": time.perf_counter()})
        return DOCS.get(key, "资料编号不存在，请检查已提供的编号。")


class Meter:
    def __init__(self, provider, limits):
        self.provider, self.limits = provider, limits
        self.calls = []

    async def chat(self, **kwargs):
        if (
            self.limits["calls"] >= self.limits["max_calls"]
            or self.limits["tokens"] >= self.limits["max_tokens"]
        ):
            raise RuntimeError("benchmark global API budget reached")
        self.limits["calls"] += 1
        started = time.perf_counter()
        scope = current_scope.get()
        row = {
            "start": started,
            "child": bool(scope and scope.child),
            "model": kwargs["model"],
            "disable_thinking": kwargs.get("disable_thinking"),
            "extra_body": kwargs.get("extra_body"),
            "first_content_s": None,
            "first_delta_s": None,
        }
        self.calls.append(row)
        original = kwargs.get("on_content_delta")

        async def delta(value):
            elapsed = time.perf_counter() - started
            if row["first_delta_s"] is None:
                row["first_delta_s"] = elapsed
            if value.get("content_delta") and row["first_content_s"] is None:
                row["first_content_s"] = elapsed
            if original is not None:
                await original(value)

        if original is not None:
            kwargs["on_content_delta"] = delta
        try:
            response = await self.provider.chat(**kwargs)
            for field in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cache_prompt_tokens",
                "cache_hit_tokens",
                "finish_reason",
            ):
                row[field] = getattr(response, field, None)
            row["tools"] = [
                {"name": c.name, "arguments": c.arguments} for c in response.tool_calls
            ]
            row["content"] = response.content
            row["thinking_chars"] = len(response.thinking or "")
            self.limits["tokens"] += int(
                response.total_tokens
                or (response.input_tokens or 0) + (response.output_tokens or 0)
            )
            return response
        except BaseException as exc:
            row["error"] = type(exc).__name__  # no provider error payload / credentials
            raise
        finally:
            row["seconds"] = time.perf_counter() - started


class PushSink:
    async def execute(self, **kwargs):
        return "accepted by isolated benchmark sink; no real message sent"


def grade(case, output):
    text = output.strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        result = json.loads(text[start:end])
    except (ValueError, TypeError):
        return {"passed": False, "checks": {"valid_json": False}}
    if case == "simple":
        checks = {"end_time": result.get("end_time") == "14:00"}
    elif case == "dependent":
        checks = {
            "arrival": result.get("arrival") == "20:15",
            "total": result.get("total") == 450,
            "can_check_in": result.get("can_check_in") is True,
        }
    else:
        totals = result.get("totals") or {}
        checks = {
            f"total_{key}": totals.get(key) == value
            for key, value in {"林岸": 8560, "云谷": 8300, "沐光": 7980}.items()
        }
        reasons = json.dumps(result.get("reasons", ""), ensure_ascii=False)
        checks.update(
            recommended=result.get("recommended") == "林岸",
            vegetarian="素食" in reasons,
            capacity="16" in reasons or "人数" in reasons or "容量" in reasons,
        )
    return {"passed": all(checks.values()), "checks": checks, "parsed": result}


async def trial(config, provider, root, case, mode, repeat, timeout):
    run_id = f"{case}-{mode}-r{repeat}"
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=False)
    parent_graph = LangGraphRuntime(folder / "parent.db")
    child_graph = LangGraphRuntime(folder / "child.db")
    workflow_graph = LangGraphRuntime(folder / "workflow.db")
    meter = Meter(provider.provider, provider.limits)
    events = []
    read_tool = ReadDocument(events)
    registry = ToolRegistry()
    registry.register(read_tool, always_on=True, risk="read-only")
    system = (
        build_agent_behavior_rules_prompt(workspace=root)
        + "\n本次是隔离的个人助手评测。可用资料只有 read_case_document；不要检索用户记忆、访问真实文件或发送消息。无关工具不可用。以用户要求的JSON作为最终交付。"
    )
    guard = config.execution_guard

    class ChildExecutor:
        async def execute(
            self,
            *,
            task,
            label,
            profile="research",
            execution_id=None,
            reasoning_effort="",
        ):
            agent = SubAgent(
                meter,
                config.model,
                [read_tool],
                system_prompt="你是隔离的只读调研执行器，只完成当前阶段。必须读取指定资料，不猜测事实。请输出简短可复核的结果。",
                max_iterations=config.max_iterations,
                max_tokens=config.max_tokens,
                graph_runtime=child_graph,
                reasoning_config=config.reasoning,
                execution_guard_config=guard,
            )
            return await agent.run(
                task, execution_id=execution_id, reasoning_effort=reasoning_effort
            )

    kernel = None

    class MainExecutor:
        async def process_direct(self, content, **kwargs):
            result = await kernel.run(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                tool_event_session_key=kwargs["session_key"],
                request_text=content,
                disabled_tools=set(kwargs.get("disabled_tools") or []),
                reasoning_effort=kwargs.get("reasoning_effort", ""),
            )
            return result.reply

    rt = WorkflowRuntime(
        store=WorkflowStore(folder / "index.db"),
        graph_runtime=workflow_graph,
        agent_loop_provider=lambda: MainExecutor(),
        push_tool=PushSink(),
        tool_registry=registry,
        subagent_executor=ChildExecutor(),
        max_concurrency=1 if mode.startswith("dag_serial") else 2,
        poll_interval_seconds=0.1,
        delegation_guard=replace(guard, autonomous_delegation=True),
    )
    registry.register(TaskCreateTool(rt), always_on=True, risk="write")
    registry.register(TaskManageTool(rt), always_on=True, risk="external-side-effect")
    kernel = AgentExecutionKernel(
        llm=LLMServices(provider=meter, light_provider=meter),
        llm_config=LLMConfig(
            model=config.model,
            max_iterations=config.max_iterations,
            max_tokens=config.max_tokens,
            reasoning=config.reasoning,
        ),
        tools=registry,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=0,
        graph_runtime=parent_graph,
        execution_guard_config=guard,
    )
    started = time.perf_counter()
    worker = asyncio.create_task(rt.run())
    report = {
        "run_id": run_id,
        "case": case,
        "mode": mode,
        "repeat": repeat,
        "status": "running",
    }

    async def execute():
        if mode.startswith("dag_"):
            steps = [
                StepSpec(
                    id=key,
                    title=key,
                    description=f"读取 {key} 并核算18人的含餐饮设备总价，说明容量、素食和取消政策，不做总推荐。",
                    kind=StepKind.AGENT,
                    executor=StepExecutor.SUBAGENT,
                    max_attempts=1,
                )
                for key in ("venue_linan", "venue_yungu", "venue_muguang")
            ]
            steps.append(
                StepSpec(
                    id="merge",
                    title="汇总",
                    description=(
                        "当前阶段仅汇总已完成的三个调研步骤。前置步骤结果是已核实的资料，直接作为证据使用。"
                        "不要再次读取报价、不要重复调查或调用工具。对18人/2名素食者/9000元预算/可提前48小时取消的条件进行最终取舍。"
                        "只输出JSON：totals为林岸、云谷、沐光三个中文名称到含餐饮设备总价数字的映射；recommended为推荐场地中文名；reasons为其他场地不适合的原因。"
                        if mode.endswith("_contract")
                        else CASES[case]
                    ),
                    depends_on=tuple(s.id for s in steps),
                    kind=StepKind.AGENT,
                    max_attempts=1,
                )
            )
            rt.create_workflow(
                name="会务方案核对",
                goal=CASES[case],
                steps=steps,
                session_key="benchmark",
                channel="benchmark",
                chat_id=run_id,
                context={
                    WORKFLOW_REASONING_CONTEXT_KEY: config.reasoning.default_effort
                },
            )
            reply = ""
        else:

            async def consume(_delta):
                return None

            result = await kernel.run(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": CASES[case]},
                ],
                tool_event_session_key=run_id,
                request_text=CASES[case],
                reasoning_effort=config.reasoning.default_effort,
                autonomous_delegation=mode == "auto_on",
                on_content_delta=consume,
            )
            reply = result.reply
            report["root_metadata"] = result.metadata
            report["parent_reply"] = reply
            report["parent_seconds"] = time.perf_counter() - started
        while True:
            workflows = rt.store.list_workflows()
            if not workflows or all(
                w.status.value
                in {"succeeded", "failed", "blocked", "cancelled", "waiting", "draft"}
                for w in workflows
            ):
                break
            await asyncio.sleep(0.1)
        report["workflows"] = [w.to_dict() for w in workflows]
        if workflows:
            reply = "\n".join(str(w.steps[-1].output or "") for w in workflows)
            report["status"] = (
                "completed"
                if all(w.status.value == "succeeded" for w in workflows)
                else "workflow_incomplete"
            )
        else:
            report["status"] = "completed"
        report["output"] = reply
        report["quality"] = grade(case, reply)

    try:
        await asyncio.wait_for(execute(), timeout=timeout)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        report.update(
            status="timeout" if isinstance(exc, TimeoutError) else "error",
            error=type(exc).__name__,
            quality={"passed": False},
        )
    finally:
        report["seconds"] = time.perf_counter() - started
        await rt.aclose()
        await asyncio.gather(worker, return_exceptions=True)
        await parent_graph.aclose()
        await child_graph.aclose()
    report["calls"] = meter.calls
    report["document_reads"] = events
    report["metrics"] = {
        field: sum(int(c.get(field) or 0) for c in meter.calls)
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_prompt_tokens",
            "cache_hit_tokens",
        )
    }
    report["metrics"].update(
        model_calls=len(meter.calls),
        child_model_calls=sum(c["child"] for c in meter.calls),
        usage_missing=sum(c.get("total_tokens") is None for c in meter.calls),
    )
    (folder / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("run_id", "status", "seconds", "metrics", "quality")
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


async def run(args):
    config = load_config(args.config)
    # Do not export traces or initialize channels/memory/production services.
    config = replace(config, dev_mode=False)
    main, light, agent = build_providers(config)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    limits = {
        "calls": 0,
        "tokens": 0,
        "max_calls": args.max_calls,
        "max_tokens": args.max_tokens,
    }
    provider = Meter(main, limits)
    reports = []
    try:
        for repeat in range(1, args.repeats + 1):
            for case in (CASES if args.suite == "full" else ()):
                modes = (
                    ["auto_off", "auto_on"] if repeat % 2 else ["auto_on", "auto_off"]
                )
                for mode in modes:
                    reports.append(
                        await trial(
                            config, provider, root, case, mode, repeat, args.timeout
                        )
                    )
            modes = (
                ["dag_serial", "dag_parallel"]
                if repeat % 2
                else ["dag_parallel", "dag_serial"]
            )
            if args.suite == "contract":
                modes = [mode + "_contract" for mode in modes]
            for mode in modes:
                reports.append(
                    await trial(
                        config,
                        provider,
                        root,
                        "independent",
                        mode,
                        repeat,
                        args.timeout,
                    )
                )
            if limits["calls"] >= args.max_calls or limits["tokens"] >= args.max_tokens:
                break
    finally:
        for item in (main, light, agent):
            if item is not None:
                await item.aclose()
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": config.model,
            "effort": config.reasoning.default_effort,
            "limits": limits,
            "method": "live model; isolated kernel+Workflow; synthetic deterministic tools without injected latency; no personal memory or channels; alternating pair order; no monetary prices assumed",
            "runs": reports,
        }
        (root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def render_report(root: Path):
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    runs = summary["runs"]
    repetitions = max(
        sum(r["case"] == case and r["mode"] == mode for r in runs)
        for case, mode in {(r["case"], r["mode"]) for r in runs}
    )
    lines = [
        "# 小满多 Agent 成本与延迟实测",
        "",
        f"- 模型：{summary['model']}；思考等级：{summary['effort']}。",
        "- 真实模型 API + 现有 AgentExecutionKernel / WorkflowRuntime / SubAgent；只读资料为固定的模拟个人助手资料，工具没有人为延迟。",
        "- 隔离 SQLite，不接入个人记忆、QQ、微信、生产通知或外部实时检索；不是完整网页端到端性能测试。",
        f"- 每组最多{repetitions}次，有第二次时反转执行顺序；样本少，缓存及模型输出随机性未完全消除，不能外推为稳定提升百分比。",
        "- 成本使用 API 返回的累计 token 用量；缓存输入仍包含在总数中。未核验账户单价，不给人民币金额。",
        "",
        "## 单次原始结果",
        "",
        "| 场景 | 模式 | 轮次 | 总耗时(s) | 输入token | 输出token | 缓存命中输入 | 模型调用 | 峰值并发模型调用 | 关键字段检查 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        events = sorted(
            (
                event
                for c in run["calls"]
                for event in ((c["start"], 1), (c["start"] + c["seconds"], -1))
            )
        )
        active = peak = 0
        for _, change in events:
            active += change
            peak = max(peak, active)
        metric = run["metrics"]
        lines.append(
            f"| {run['case']} | {run['mode']} | {run['repeat']} | {run['seconds']:.2f} | {metric['input_tokens']} | {metric['output_tokens']} | {metric['cache_hit_tokens']} | {metric['model_calls']} | {peak} | {'通过' if run['quality']['passed'] else '未通过'} |"
        )
    lines += [
        "",
        "## 按模式汇总（算术平均）",
        "",
        "| 场景 | 模式 | n | 平均耗时(s) | 平均总token | 平均未命中输入token | 自动创建Workflow次数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case, mode in sorted({(r["case"], r["mode"]) for r in runs}):
        group = [r for r in runs if r["case"] == case and r["mode"] == mode]
        mean_time = statistics.mean(r["seconds"] for r in group)
        mean_tokens = statistics.mean(r["metrics"]["total_tokens"] for r in group)
        uncached = statistics.mean(
            r["metrics"]["input_tokens"] - r["metrics"]["cache_hit_tokens"]
            for r in group
        )
        workflows = sum(len(r.get("workflows", [])) for r in group)
        lines.append(
            f"| {case} | {mode} | {len(group)} | {mean_time:.2f} | {mean_tokens:.0f} | {uncached:.0f} | {workflows if mode.startswith('auto_') else '预置DAG，非自主'} |"
        )
    lines += [
        "",
        "## 如何解释",
        "",
        "- auto_off/auto_on：只改变用户委派开关，不强制模型使用子 Agent。必须结合实际 Workflow 数判断，不能把两组耗时差直接称为多 Agent 加速。",
        "- dag_serial/dag_parallel：预置相同的三路独立分析加汇总 DAG，仅改变调度并发上限 1/2，不含模型规划 DAG 的成本；用于验证调度效果，不代表自主路由端到端收益。",
        "- 耗时为整个任务完成时间（含 Workflow），不是首字时间。本轮 first_content_s 因采集字段错误而为空，不报告首正文延迟；后续复现脚本已修正 content_delta 字段。流式 first_delta_s 可包含思考增量，不能代替首正文时间；非流式子请求也没有 TTFT。",
        "- 关键字段检查涵盖时间、费用、容量、素食、最终推荐；并不等同于所有自然语言理由都正确，须结合人工事实核对。",
        "- 所有实际 API 请求用量和工具轨迹可在各 run 目录 result.json 中复核；不保存模型的思考文本。",
        "",
    ]
    review = root / "REVIEW.md"
    if review.exists():
        lines += ["", review.read_text(encoding="utf-8")]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--suite", choices=["full", "contract"], default="full")
    args = parser.parse_args()
    if args.summarize_only:
        render_report(Path(args.output).resolve())
    else:
        asyncio.run(run(args))
        render_report(Path(args.output).resolve())
