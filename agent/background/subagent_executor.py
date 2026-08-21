from __future__ import annotations

import logging
import uuid
from pathlib import Path

from agent.background.subagent_profiles import (
    PROFILE_RESEARCH,
    SubagentRuntime,
    build_subagent_spec,
)
from agent.subagent import SubAgent
from agent.tool_hooks.base import ToolHook
from core.llm import LLMProvider
from core.net.http import HttpRequester
from prompts.background import build_subagent_prompt

logger = logging.getLogger(__name__)

_RESULT_MAX_CHARS = 100_000
_MAX_ITERATIONS = 10


class SubagentExecutor:
    """Execute one isolated Task Runtime step with a least-privilege profile."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        workspace: Path,
        model: str,
        max_tokens: int,
        fetch_requester: HttpRequester,
        multimodal: bool = True,
    ) -> None:
        self._workspace = workspace
        self._runtime = SubagentRuntime(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
        )
        self._fetch_requester = fetch_requester
        self._multimodal = multimodal

    def add_tool_hooks(self, hooks: list[ToolHook]) -> None:
        object.__setattr__(self._runtime, "tool_hooks", list(hooks))

    def reconfigure(self, *, provider: LLMProvider, model: str) -> None:
        object.__setattr__(self._runtime, "provider", provider)
        object.__setattr__(self._runtime, "model", model)

    async def execute(
        self,
        *,
        task: str,
        label: str | None,
        profile: str = PROFILE_RESEARCH,
        execution_id: str | None = None,
    ) -> str:
        """Run a step inline so Task Runtime owns status, retry, and cancellation."""
        run_id = self._validated_run_id(execution_id)
        display_label = (label or task[:30] or run_id).strip()
        task_dir = self._task_dir(run_id)

        logger.info(
            "subagent step started run_id=%s label=%r profile=%s",
            run_id,
            display_label,
            profile,
        )
        subagent = self._build_subagent(task_dir=task_dir, profile=profile)
        try:
            result = await subagent.run(task)
        except Exception:
            logger.exception("subagent step failed run_id=%s", run_id)
            raise

        exit_reason = getattr(subagent, "last_exit_reason", None) or "completed"
        if exit_reason == "error":
            raise RuntimeError("subagent execution failed")
        result = self._truncate_result(result)
        logger.info(
            "subagent step finished run_id=%s exit_reason=%s result_len=%d",
            run_id,
            exit_reason,
            len(result),
        )
        return (
            f"[子任务「{display_label}」结果]\n" f"退出原因: {exit_reason}\n\n{result}"
        )

    @staticmethod
    def _validated_run_id(execution_id: str | None) -> str:
        run_id = (execution_id or uuid.uuid4().hex[:8]).strip()
        if (
            not run_id
            or len(run_id) > 96
            or any(not (char.isalnum() or char in "-_") for char in run_id)
        ):
            raise ValueError("execution_id 只能包含字母、数字、短横线和下划线")
        return run_id

    def _task_dir(self, run_id: str) -> Path:
        task_dir = self._workspace / "subagent-runs" / run_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _build_subagent(
        self,
        *,
        task_dir: Path,
        profile: str,
    ) -> SubAgent:
        spec = build_subagent_spec(
            workspace=self._workspace,
            task_dir=task_dir,
            fetch_requester=self._fetch_requester,
            system_prompt=build_subagent_prompt(
                self._workspace,
                task_dir,
                profile,
            ),
            max_iterations=_MAX_ITERATIONS,
            profile=profile,
            multimodal=self._multimodal,
        )
        return spec.build(self._runtime)

    @staticmethod
    def _truncate_result(result: str) -> str:
        if len(result) <= _RESULT_MAX_CHARS:
            return result
        original_len = len(result)
        return (
            result[:_RESULT_MAX_CHARS] + f"\n...[结果已截断，原始长度 {original_len}]"
        )
