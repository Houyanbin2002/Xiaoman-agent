from __future__ import annotations

"""Execution adapters and the local evaluation harness."""

import asyncio
import inspect
import time
import uuid
from dataclasses import replace
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .models import AgentRun, CaseResult, EvalCase, EvalSummary, ToolCall
from .outcomes import assess
from .scorers import aggregate_scores, score_rubric


class CaseFixtureManager(Protocol):
    async def prepare(
        self,
        case: EvalCase,
        *,
        session_key: str,
        trace_id: str,
    ) -> Any: ...

    async def observe(self, prepared: Any) -> Any: ...

    async def cleanup(self, prepared: Any) -> None: ...

    def disabled_tools(self, prepared: Any) -> list[str]: ...


class AgentExecutor(Protocol):
    def __call__(
        self, case: EvalCase
    ) -> (
        AgentRun
        | Mapping[str, Any]
        | str
        | Awaitable[AgentRun | Mapping[str, Any] | str]
    ): ...


class ReplayExecutor:
    """Run the deterministic ``replay`` payload embedded in a case."""

    async def __call__(self, case: EvalCase) -> AgentRun:
        await asyncio.sleep(0)
        if case.replay.get("raise"):
            raise RuntimeError(str(case.replay["raise"]))
        return AgentRun.from_value(case.replay)


class ProcessDirectExecutor:
    """Bridge the real AgentLoop into the same evaluator contract.

    The adapter intentionally keeps no global state. A unique session and trace
    id per case make repeated local runs idempotent and keep checkpoint/memory
    pollution out of the user's normal conversation.
    """

    def __init__(
        self,
        loop: Any,
        *,
        trace_store: Any | None = None,
        session_prefix: str = "eval",
        settle: Callable[[str], Awaitable[None]] | None = None,
        memory_runtime: Any | None = None,
        fixture_manager: CaseFixtureManager | None = None,
    ) -> None:
        self.loop = loop
        self.trace_store = trace_store
        self.session_prefix = session_prefix
        self.settle = settle
        self.memory_runtime = memory_runtime
        self.fixture_manager = fixture_manager

    async def __call__(self, case: EvalCase) -> AgentRun:
        # All turns share one fixture and memory store; session names can change
        # to test cross-conversation recall. Teardown occurs after the episode.
        turns = case.metadata.get("turns") or [{"input": case.input, "session": "main"}]
        sessions: dict[str, str] = {}
        prepared_items: list[Any] = []
        runs: list[AgentRun] = []
        try:
            for turn in turns:
                alias = str(turn.get("session", "main"))
                trace_id = f"eval-{uuid.uuid4().hex}"
                if alias not in sessions:
                    sessions[alias] = (
                        f"{self.session_prefix}:{case.case_id}:{uuid.uuid4().hex[:8]}"
                    )
                    if self.fixture_manager is not None:
                        fixture_case = (
                            case if not prepared_items else replace(case, metadata={})
                        )
                        prepared_items.append(
                            await self.fixture_manager.prepare(
                                fixture_case,
                                session_key=sessions[alias],
                                trace_id=trace_id,
                            )
                        )
                current = replace(case, input=str(turn["input"]))
                runs.append(
                    await self._run_prepared(
                        current,
                        sessions[alias],
                        trace_id,
                        prepared_items[0] if prepared_items else None,
                    )
                )
                if runs[-1].status != "completed":
                    break
            last = runs[-1]
            return replace(
                last,
                tools=tuple(tool for run in runs for tool in run.tools),
                latency_ms=sum(run.latency_ms or 0 for run in runs),
                metadata={
                    **last.metadata,
                    "coverage_level": case.metadata.get(
                        "coverage_level", "kernel_integration"
                    ),
                    "turn_runs": [run.to_dict() for run in runs],
                    "turn_inputs": [str(turn["input"]) for turn in turns[: len(runs)]],
                    "planned_turns": len(turns),
                },
            )
        finally:
            if self.fixture_manager is not None:
                for prepared in reversed(prepared_items):
                    await self.fixture_manager.cleanup(prepared)

    async def _run_prepared(
        self,
        case: EvalCase,
        session_key: str,
        trace_id: str,
        prepared: Any,
    ) -> AgentRun:
        disabled_tools: list[str] | None = None
        disabled_for = (
            getattr(self.fixture_manager, "disabled_tools", None)
            if self.fixture_manager is not None
            else None
        )
        if prepared is not None and callable(disabled_for):
            disabled_tools = list(disabled_for(prepared))
        started = time.perf_counter()
        outbound_method = getattr(self.loop, "process_direct_outbound", None)
        execute = (
            outbound_method if callable(outbound_method) else self.loop.process_direct
        )
        raw_response = await execute(
            case.input,
            session_key=session_key,
            channel="eval",
            chat_id=session_key,
            trace_id=trace_id,
            trace_flow="eval",
            trace_title=case.title,
            disabled_tools=disabled_tools,
        )
        response = getattr(raw_response, "content", raw_response)
        outbound_metadata = dict(getattr(raw_response, "metadata", {}) or {})
        if self.settle is not None:
            await self.settle(session_key)
        latency_ms = (time.perf_counter() - started) * 1000
        tools: list[ToolCall] = []
        metadata: dict[str, Any] = {}
        if self.trace_store is not None:
            trace = self.trace_store.get_trace(trace_id)
            if trace is not None:
                metadata.update(trace.metadata)
                metadata["trace_status"] = trace.status
            for event in self.trace_store.list_events(trace_id):
                if event.category.lower() not in {
                    "tool",
                    "tool_call",
                    "tool_result",
                    "execution",
                }:
                    continue
                payload = dict(event.payload)
                tools.append(
                    ToolCall.from_dict(
                        {
                            "name": payload.get("tool_name", event.name),
                            **payload,
                            "output": payload.get(
                                "output",
                                payload.get("result", payload.get("result_preview")),
                            ),
                            "error": payload.get("error")
                            or (
                                payload.get("result_preview", "")
                                if event.status in {"error", "failed"}
                                else ""
                            ),
                            "status": event.status,
                        }
                    )
                )
        # The structured outbound is the kernel's result. Trace lifecycle
        # success only means the request returned, not that the task ended
        # normally (guards can return partial text without throwing).
        context_retry = outbound_metadata.get("context_retry", {})
        if isinstance(context_retry, Mapping) and context_retry.get("exit_reason"):
            metadata["exit_reason"] = str(context_retry["exit_reason"])
            metadata["execution_status_source"] = "outbound.context_retry"
        tool_chain = outbound_metadata.get("tool_chain")
        if isinstance(tool_chain, list):
            tools = [
                ToolCall.from_dict(
                    {
                        **call,
                        "status": (
                            "completed"
                            if call.get("status") == "success"
                            else call.get("status", "unknown")
                        ),
                        "error": call.get("error")
                        or (
                            str(call.get("result") or "")
                            if call.get("status") in {"error", "failed"}
                            else ""
                        ),
                    }
                )
                for group in tool_chain
                if isinstance(group, Mapping)
                for call in group.get("calls", [])
                if isinstance(call, Mapping)
            ]
            metadata["tool_evidence_source"] = "outbound.tool_chain"
        elif tools:
            metadata["tool_evidence_source"] = "trace.result_preview"
        memory_events: list[dict[str, Any]] = []
        if self.memory_runtime is not None:
            query = getattr(
                self.memory_runtime,
                "retrieve_personal_memory_async",
                None,
            )
            if callable(query):
                retrieved = await query(case.input, limit=20)
                for hit in getattr(retrieved, "hits", ()):
                    record = getattr(hit, "record", None)
                    if record is None:
                        continue
                    data = dict(getattr(record, "data", {}) or {})
                    kind = str(data.get("kind") or "memory")
                    memory_events.append(
                        {
                            "type": (
                                "user_preference" if kind == "preference" else kind
                            ),
                            "value": data.get("value") or "",
                            "content": data.get("content")
                            or getattr(record, "summary", ""),
                            "summary": getattr(record, "summary", ""),
                            "confidence": getattr(record, "confidence", 0.0),
                            "user_locked": getattr(record, "user_locked", False),
                            "record_key": getattr(record, "record_key", ""),
                        }
                    )
        state: dict[str, Any] = {}
        if self.fixture_manager is not None and prepared is not None:
            observation = await self.fixture_manager.observe(prepared)
            state.update(dict(getattr(observation, "state", {}) or {}))
            metadata.update(dict(getattr(observation, "metadata", {}) or {}))
            memory_events.extend(
                dict(item) for item in (getattr(observation, "memory_events", ()) or ())
            )
        exit_reason = str(metadata.get("exit_reason") or "unknown")
        trace_status = str(metadata.get("trace_status") or "")
        metadata["exit_reason"] = exit_reason
        status = "completed" if exit_reason == "completed" else "incomplete"
        if trace_status in {"error", "failed", "interrupted", "timeout"}:
            status = "error" if trace_status in {"error", "failed"} else "incomplete"
        if exit_reason != "completed":
            status = "error" if exit_reason in {"error", "failed"} else "incomplete"
        if not str(response or "").strip() and status == "completed":
            status = "incomplete"
            metadata["exit_reason"] = "empty_result"
        return AgentRun(
            response=str(response or ""),
            tools=tuple(tools),
            state=state,
            memory_events=tuple(memory_events),
            metadata=metadata,
            status=status,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )


class EvalHarness:
    def __init__(
        self,
        *,
        dataset_name: str = "local",
        version: str = "v1",
        judge: Callable[[EvalCase, AgentRun], Mapping[str, Any]] | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.version = version
        self.judge = judge

    async def run_case(self, case: EvalCase, executor: AgentExecutor) -> CaseResult:
        try:
            raw = executor(case)
            if inspect.isawaitable(raw):
                raw = await raw
            run = AgentRun.from_value(raw)
        except Exception as exc:  # one bad case must not hide the rest of a dataset
            run = AgentRun(status="error")
            return CaseResult(
                case.case_id,
                case.title,
                False,
                0.0,
                (),
                run,
                error=f"{type(exc).__name__}: {exc}",
            )

        judge_error = ""
        try:
            # Rubric is the only public evaluation surface. Its criteria may
            # use deterministic checks or an injected LLM judge internally.
            scores = await asyncio.to_thread(score_rubric, case, run, judge=self.judge)
        except Exception as exc:
            # A flaky/malformed Judge response is an evaluation degradation,
            # not an Agent execution failure. Preserve the real run and score
            # every criterion that has a deterministic fallback.
            judge_error = f"judge_degraded: {type(exc).__name__}: {exc}"
            scores = score_rubric(case, run, judge=None)
        reward, passed = aggregate_scores(scores)
        return CaseResult(
            case.case_id,
            case.title,
            passed,
            reward,
            tuple(scores),
            run,
            error=judge_error,
            assessment=assess(case, run, scores, judge_error),
        )

    async def run(
        self, cases: Iterable[EvalCase], executor: AgentExecutor
    ) -> EvalSummary:
        case_list = list(cases)
        results = tuple([await self.run_case(case, executor) for case in case_list])
        return self.summarize(case_list, results)

    def summarize(
        self, case_list: list[EvalCase], results: tuple[CaseResult, ...]
    ) -> EvalSummary:
        if len({case.case_id for case in case_list}) != len(case_list):
            raise ValueError("duplicate case IDs in summary")
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        mean_reward = sum(result.reward for result in results) / total if total else 0.0
        metric_values: dict[str, list[float]] = {}
        for result in results:
            for score in result.scores:
                metric_values.setdefault(score.name, []).append(score.value)
        metrics = {
            name: sum(values) / len(values)
            for name, values in metric_values.items()
            if values
        }
        quality_results = [
            result.assessment["quality_passed"]
            for result in results
            if result.assessment.get("quality_assessed")
        ]
        metrics.update(
            {
                "execution_completed_cases": float(
                    sum(result.run.status == "completed" for result in results)
                ),
                "quality_assessed_cases": float(len(quality_results)),
                "quality_passed_cases": float(sum(quality_results)),
                "rubric_acceptance_cases": float(
                    sum(bool(result.assessment.get("accepted")) for result in results)
                ),
                "objective_assessed_cases": float(
                    sum(
                        result.assessment.get("objective_outcome", "unassessed")
                        not in {"unassessed", "unknown"}
                        for result in results
                    )
                ),
                "objective_completed_cases": float(
                    sum(
                        result.assessment.get("objective_outcome") == "completed"
                        for result in results
                    )
                ),
                "objective_blocked_cases": float(
                    sum(
                        result.assessment.get("objective_outcome") == "blocked"
                        for result in results
                    )
                ),
                "delivery_assessed_cases": float(
                    sum(
                        result.assessment.get("delivery_status", "not_applicable")
                        not in {"not_applicable", "unknown"}
                        for result in results
                    )
                ),
                "delivery_succeeded_cases": float(
                    sum(
                        result.assessment.get("delivery_status") == "delivered"
                        for result in results
                    )
                ),
                "unscored_cases": float(sum(not result.scores for result in results)),
            }
        )
        result_by_id = {result.case_id: result for result in results}
        slice_rows: dict[str, list[CaseResult]] = {}
        for case in case_list:
            result = result_by_id[case.case_id]
            for tag in case.tags:
                slice_rows.setdefault(tag, []).append(result)
        slices: dict[str, dict[str, float]] = {}
        for tag, slice_results in slice_rows.items():
            slice_total = len(slice_results)
            hard_failures = sum(
                any(score.hard and not score.passed for score in result.scores)
                for result in slice_results
            )
            slices[tag] = {
                "count": float(slice_total),
                "pass_rate": sum(result.passed for result in slice_results)
                / slice_total,
                "mean_reward": sum(result.reward for result in slice_results)
                / slice_total,
                "hard_fail_rate": hard_failures / slice_total,
                "acceptance_rate": sum(
                    bool(r.assessment.get("accepted")) for r in slice_results
                )
                / slice_total,
            }
        return EvalSummary(
            self.dataset_name,
            self.version,
            total,
            passed,
            passed / total if total else 0.0,
            mean_reward,
            results,
            metrics,
            slices,
        )

    def run_sync(
        self, cases: Iterable[EvalCase], executor: AgentExecutor
    ) -> EvalSummary:
        return asyncio.run(self.run(cases, executor))


def write_report(summary: EvalSummary, path: str | Path) -> None:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def render_markdown(summary: EvalSummary) -> str:
    lines = [
        f"# Xiaoman Eval Report: {summary.dataset} {summary.version}",
        "",
        f"- Hard-gate pass rate: **{summary.pass_rate:.1%}** ({summary.passed}/{summary.total}); not a task-quality pass rate",
        f"- Execution completed: {int(summary.metrics.get('execution_completed_cases', 0))}/{summary.total} (kernel termination only)",
        f"- Judge quality passed/assessed: {int(summary.metrics.get('quality_passed_cases', 0))}/{int(summary.metrics.get('quality_assessed_cases', 0))}",
        f"- All Rubric criteria accepted: {int(summary.metrics.get('rubric_acceptance_cases', 0))}/{summary.total} (not a claim of universal product coverage)",
        f"- Objective outcome: {int(summary.metrics.get('objective_completed_cases', 0))}/{int(summary.metrics.get('objective_assessed_cases', 0))} assessed",
        f"- Objective blocked (not completed): {int(summary.metrics.get('objective_blocked_cases', 0))}",
        f"- Delivery outcome: {int(summary.metrics.get('delivery_succeeded_cases', 0))}/{int(summary.metrics.get('delivery_assessed_cases', 0))} assessed",
        f"- Unscored cases: {int(summary.metrics.get('unscored_cases', 0))} (these cannot support a quality claim)",
        f"- Mean reward: **{summary.mean_reward:.3f}**",
        "",
        "## Metrics",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {value:.3f} |" for name, value in sorted(summary.metrics.items())
    )
    if summary.manifest:
        lines.extend(
            [
                "",
                "## Experiment boundaries",
                "",
                f"- Split: {summary.manifest.get('split', 'unknown')}; repetitions: {summary.manifest.get('repeats', 1)}; isolation: {summary.manifest.get('isolation', '')}",
                f"- Model: {summary.manifest.get('model', '')}; Judge: {summary.manifest.get('judge_model', '')}",
                f"- Dataset fingerprint: {summary.manifest.get('dataset_sha256', '')}",
                f"- Code fingerprint: {summary.manifest.get('code_sha256', '')}",
                f"- Coverage: {summary.manifest.get('scope_note', '')}",
                "- Repeats are correlated; the Wilson interval is descriptive, not a product-wide confidence claim.",
            ]
        )
    if summary.slices:
        lines.extend(
            [
                "",
                "## Slices",
                "",
                "| Slice | Cases | Pass rate | Mean reward | Hard fail rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {name} | {int(values['count'])} | {values['pass_rate']:.1%} | {values['mean_reward']:.3f} | {values['hard_fail_rate']:.1%} |"
            for name, values in sorted(summary.slices.items())
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Hard gate | Execution | Objective | Delivery | Accepted | Reward | Error |",
            "|---|---:|---|---|---|---|---:|---|",
        ]
    )
    lines.extend(
        f"| {result.case_id} | {'✅' if result.passed else '❌'} | {result.run.status} / {result.run.metadata.get('exit_reason', '')} | {result.assessment.get('objective_outcome', 'unassessed')} | {result.assessment.get('delivery_status', 'unassessed')} | {result.assessment.get('accepted', False)} | {result.reward:.3f} | {_md_cell(result.error)} |"
        for result in summary.results
    )
    lines.extend(
        [
            "",
            "## Criterion evidence",
            "",
            "| Case | Criterion | Score | Passed | Reason |",
            "|---|---|---:|---|---|",
        ]
    )
    for result in summary.results:
        for score in result.scores:
            if score.source == "rubric_judge" or not score.passed:
                lines.append(
                    f"| {result.case_id} | {score.name} | {score.value:.3f} | {score.passed} | {_md_cell(score.reason)} |"
                )
    return "\n".join(lines) + "\n"


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>").replace("\r", "")
