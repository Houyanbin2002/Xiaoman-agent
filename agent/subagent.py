"""Isolated, bounded SubAgent adapter over the shared execution kernel."""

from __future__ import annotations

import logging

from agent.core.passive_turn import AgentExecutionKernel
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.ports import LLMConfig, LLMServices
from agent.runtime.execution_policy import SUBAGENT_EXECUTION_POLICY
from agent.tool_hooks.base import ToolHook
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from core.llm import LLMProvider

logger = logging.getLogger(__name__)


class SubAgent:
    """一次性子任务适配器。

    每个实例拥有独立的工具注册表、hook 状态和执行策略；它不接入主会话、
    长期记忆或主 EventBus。ReAct 控制流与主 Agent 共用 AgentExecutionKernel。
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        tools: list[Tool],
        *,
        system_prompt: str = "",
        max_iterations: int = 30,
        max_tokens: int = 8192,
    ) -> None:
        self._system_prompt = system_prompt
        self.last_exit_reason = "idle"
        self.iterations_used = 0
        self.tools_called: list[str] = []
        self._run_seq = 0

        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool, always_on=True)
        self._kernel = AgentExecutionKernel(
            llm=LLMServices(provider=provider, light_provider=provider),
            llm_config=LLMConfig(
                model=model,
                light_model=model,
                max_iterations=max_iterations,
                max_tokens=max_tokens,
                tool_search_enabled=False,
            ),
            tools=registry,
            discovery=ToolDiscoveryState(),
            tool_search_enabled=False,
            memory_window=0,
            execution_policy=SUBAGENT_EXECUTION_POLICY,
        )

    def add_tool_hooks(self, hooks: list[ToolHook]) -> None:
        self._kernel.add_tool_hooks(hooks)

    async def run(self, task: str) -> str:
        """Run one isolated task and return its final or bounded summary text."""
        self.last_exit_reason = "running"
        self.iterations_used = 0
        self.tools_called = []
        self._run_seq += 1

        messages: list[dict] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": task})

        try:
            result = await self._kernel.run(
                messages,
                tool_event_session_key=f"subagent:{id(self)}:{self._run_seq}",
                request_text=task,
                permission_mode="full_access",
            )
        except Exception as exc:
            logger.error("[subagent] 执行失败: %s", exc, exc_info=True)
            self.last_exit_reason = "error"
            return ""

        metadata = result.metadata
        react_stats = metadata.get("react_stats") or {}
        self.iterations_used = int(react_stats.get("iteration_count") or 0)
        self.tools_called = list(dict.fromkeys(metadata.get("tools_used") or []))
        kernel_reason = str(metadata.get("exit_reason") or "completed")
        self.last_exit_reason = {
            "max_iterations": "forced_summary",
            "max_iterations_fallback": "forced_summary_fallback",
        }.get(kernel_reason, kernel_reason)
        return result.reply.strip()
