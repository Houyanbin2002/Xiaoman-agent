from __future__ import annotations

from typing import Any, cast

import pytest

from agent.background.subagent_executor import SubagentExecutor
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

        async def run(self, task: str) -> str:
            assert task == "research this"
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


@pytest.mark.asyncio
async def test_executor_propagates_failure_to_task_runtime(tmp_path):
    executor = _executor(tmp_path)

    class _FailingSubAgent:
        async def run(self, task: str) -> str:
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

        async def run(self, task: str) -> str:
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
