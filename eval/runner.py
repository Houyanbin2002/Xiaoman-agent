from __future__ import annotations

"""Execution adapters and the local evaluation harness."""

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .models import AgentRun, CaseResult, EvalCase, EvalSummary, ToolCall
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
    def __call__(self, case: EvalCase) -> AgentRun | Mapping[str, Any] | str | Awaitable[AgentRun | Mapping[str, Any] | str]: ...


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
        session_key = f"{self.session_prefix}:{case.case_id}:{uuid.uuid4().hex[:8]}"
        trace_id = f"eval-{uuid.uuid4().hex}"
        prepared: Any = None
        try:
            if self.fixture_manager is not None:
                prepared = await self.fixture_manager.prepare(
                    case,
                    session_key=session_key,
                    trace_id=trace_id,
                )
            disabled_tools: list[str] | None = None
            disabled_for = (
                getattr(self.fixture_manager, "disabled_tools", None)
                if self.fixture_manager is not None
                else None
            )
            if prepared is not None and callable(disabled_for):
                disabled_tools = list(disabled_for(prepared))
            started = time.perf_counter()
            response = await self.loop.process_direct(
                case.input,
                session_key=session_key,
                channel="eval",
                chat_id=session_key,
                trace_id=trace_id,
                trace_flow="eval",
                trace_title=case.title,
                disabled_tools=disabled_tools,
            )
            if self.settle is not None:
                await self.settle(session_key)
            latency_ms = (time.perf_counter() - started) * 1000
            tools: list[ToolCall] = []
            metadata: dict[str, Any] = {}
            if self.trace_store is not None:
                trace = self.trace_store.get_trace(trace_id)
                if trace is not None:
                    metadata.update(trace.metadata)
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
                                "status": event.status,
                            }
                        )
                    )
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
                                    "user_preference"
                                    if kind == "preference"
                                    else kind
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
                    dict(item)
                    for item in (getattr(observation, "memory_events", ()) or ())
                )
            return AgentRun(
                response=str(response or ""),
                tools=tuple(tools),
                state=state,
                memory_events=tuple(memory_events),
                metadata=metadata,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )
        finally:
            if self.fixture_manager is not None and prepared is not None:
                await self.fixture_manager.cleanup(prepared)


class EvalHarness:
    def __init__(self, *, dataset_name: str = "local", version: str = "v1", judge: Callable[[EvalCase, AgentRun], Mapping[str, Any]] | None = None) -> None:
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
            return CaseResult(case.case_id, case.title, False, 0.0, (), run, error=f"{type(exc).__name__}: {exc}")

        judge_error = ""
        try:
            # Rubric is the only public evaluation surface. Its criteria may
            # use deterministic checks or an injected LLM judge internally.
            scores = score_rubric(case, run, judge=self.judge)
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
        )

    async def run(self, cases: Iterable[EvalCase], executor: AgentExecutor) -> EvalSummary:
        case_list = list(cases)
        results = tuple([await self.run_case(case, executor) for case in case_list])
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        mean_reward = sum(result.reward for result in results) / total if total else 0.0
        metric_values: dict[str, list[float]] = {}
        for result in results:
            for score in result.scores:
                metric_values.setdefault(score.name, []).append(score.value)
        metrics = {name: sum(values) / len(values) for name, values in metric_values.items() if values}
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
                "pass_rate": sum(result.passed for result in slice_results) / slice_total,
                "mean_reward": sum(result.reward for result in slice_results) / slice_total,
                "hard_fail_rate": hard_failures / slice_total,
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

    def run_sync(self, cases: Iterable[EvalCase], executor: AgentExecutor) -> EvalSummary:
        return asyncio.run(self.run(cases, executor))


def write_report(summary: EvalSummary, path: str | Path) -> None:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def render_markdown(summary: EvalSummary) -> str:
    lines = [
        f"# Xiaoman Eval Report: {summary.dataset} {summary.version}",
        "",
        f"- Pass rate: **{summary.pass_rate:.1%}** ({summary.passed}/{summary.total})",
        f"- Mean reward: **{summary.mean_reward:.3f}**",
        "",
        "## Metrics",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {value:.3f} |" for name, value in sorted(summary.metrics.items()))
    if summary.slices:
        lines.extend(["", "## Slices", "", "| Slice | Cases | Pass rate | Mean reward | Hard fail rate |", "|---|---:|---:|---:|---:|"])
        lines.extend(
            f"| {name} | {int(values['count'])} | {values['pass_rate']:.1%} | {values['mean_reward']:.3f} | {values['hard_fail_rate']:.1%} |"
            for name, values in sorted(summary.slices.items())
        )
    lines.extend(["", "## Cases", "", "| Case | Pass | Reward | Error |", "|---|---:|---:|---|"])
    lines.extend(f"| {result.case_id} | {'✅' if result.passed else '❌'} | {result.reward:.3f} | {result.error} |" for result in summary.results)
    return "\n".join(lines) + "\n"
