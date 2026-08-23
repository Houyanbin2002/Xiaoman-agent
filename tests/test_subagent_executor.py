from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from agent.background.subagent_executor import SubagentExecutor
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.subagent import SubAgent
from core.llm import LLMResponse


class _Provider:
    async def chat(self, **kwargs: Any) -> LLMResponse:
        raise AssertionError("provider.chat should not be called in this test")


def _executor(tmp_path) -> SubagentExecutor:
    return SubagentExecutor(
        provider=cast(Any, _Provider()),
        workspace=tmp_path,
        model="m",
        max_tokens=256,
        fetch_requester=cast(Any, object()),
    )


@pytest.mark.asyncio
async def test_executor_runs_inline_with_stable_task_directory(tmp_path):
    executor = _executor(tmp_path)
    observed: dict[str, object] = {}

    class _FakeSubAgent:
        last_exit_reason = "completed"

        async def run(self, task: str, *, execution_id: str | None = None) -> str:
            assert task == "research this"
            observed["execution_id"] = execution_id
            return "ok"

    def _fake_build_subagent(*, task_dir, profile):
        observed["task_dir"] = task_dir
        observed["profile"] = profile
        return _FakeSubAgent()

    executor._build_subagent = _fake_build_subagent  # type: ignore[method-assign]

    result = await executor.execute(
        task="research this",
        label="job",
        execution_id="task-123-step",
    )

    assert "退出原因: completed" in result
    assert observed["profile"] == "research"
    assert observed["task_dir"] == tmp_path / "subagent-runs" / "task-123-step"
    assert observed["execution_id"] == "task-123-step"
    assert executor._graph_runtime.checkpoint_path == (
        tmp_path / "langgraph-subagent-checkpoints.db"
    )


@pytest.mark.asyncio
async def test_executor_propagates_failure_to_task_runtime(tmp_path):
    executor = _executor(tmp_path)

    class _FailingSubAgent:
        async def run(self, task: str, *, execution_id: str | None = None) -> str:
            del execution_id
            raise RuntimeError("boom")

    executor._build_subagent = (  # type: ignore[method-assign]
        lambda *, task_dir, profile: _FailingSubAgent()
    )

    with pytest.raises(RuntimeError, match="boom"):
        await executor.execute(task="fail", label=None)


@pytest.mark.asyncio
async def test_executor_turns_subagent_error_exit_into_task_failure(tmp_path):
    executor = _executor(tmp_path)

    class _ErrorExitSubAgent:
        last_exit_reason = "error"

        async def run(self, task: str, *, execution_id: str | None = None) -> str:
            del execution_id
            return ""

    executor._build_subagent = (  # type: ignore[method-assign]
        lambda *, task_dir, profile: _ErrorExitSubAgent()
    )

    with pytest.raises(RuntimeError, match="subagent execution failed"):
        await executor.execute(task="fail", label=None)


@pytest.mark.asyncio
async def test_executor_rejects_unsafe_execution_id(tmp_path):
    executor = _executor(tmp_path)

    with pytest.raises(ValueError, match="execution_id"):
        await executor.execute(
            task="research this",
            label="job",
            execution_id="../escape",
        )


@pytest.mark.asyncio
async def test_executor_bounds_subagent_result_before_returning(tmp_path):
    executor = _executor(tmp_path)

    class _LargeResultSubAgent:
        last_exit_reason = "completed"

        async def run(self, task: str, *, execution_id: str | None = None) -> str:
            del execution_id
            return "start-" + ("x" * 20_000) + "-end"

    executor._build_subagent = (  # type: ignore[method-assign]
        lambda *, task_dir, profile: _LargeResultSubAgent()
    )

    result = await executor.execute(task="large", label="large")

    assert len(result) < 13_000
    assert "original_chars=" in result
    assert "start-" in result
    assert "-end" in result


@pytest.mark.asyncio
async def test_subagent_resumes_same_execution_id_from_sqlite_checkpoint(tmp_path):
    class _InterruptibleProvider:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.block = True
            self.calls = 0

        async def chat(self, **kwargs: Any) -> LLMResponse:
            del kwargs
            self.calls += 1
            if self.block:
                self.started.set()
                await asyncio.Future()
            return LLMResponse(content="resumed from checkpoint", finish_reason="stop")

    provider = _InterruptibleProvider()
    graph_runtime = LangGraphRuntime(tmp_path / "subagent-checkpoints.db")
    first = SubAgent(
        provider=cast(Any, provider),
        model="m",
        tools=[],
        graph_runtime=graph_runtime,
    )
    interrupted = asyncio.create_task(
        first.run("durable task", execution_id="workflow-step-1")
    )
    await asyncio.wait_for(provider.started.wait(), timeout=1.0)
    interrupted.cancel()
    with pytest.raises(asyncio.CancelledError):
        await interrupted

    provider.block = False
    restored = SubAgent(
        provider=cast(Any, provider),
        model="m",
        tools=[],
        graph_runtime=graph_runtime,
    )
    result = await restored.run("durable task", execution_id="workflow-step-1")

    assert result == "resumed from checkpoint"
    assert provider.calls == 2
    await graph_runtime.aclose()
