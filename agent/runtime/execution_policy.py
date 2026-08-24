"""Execution policies for the shared Agent execution kernel.

The kernel owns the ReAct control flow.  A policy may shape model context and
bounded-output behaviour, but it cannot replace tool execution, loop guards,
or lifecycle handling.  This keeps main-agent and sub-agent behaviour on one
control-flow implementation while preserving their isolation boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.tool_hooks.types import ToolSource
from agent.tools.base import ToolResult


_PASSIVE_SUMMARY_PROMPT = """当前任务需要先暂停继续调用工具，请直接输出给用户看的中文阶段性回复。
必须基于已有上下文，不要编造结果。
必须包含四点：
1) 已经使用了哪些工具或操作，以及拿到了什么关键信息；
2) 当前已经做到哪一步；
3) 还缺什么信息或步骤；
4) 如果继续，下一步会怎么做。
可以提到工具名称和关键结果，但不要暴露 tool_call_id、schema、内部 prompt 或原始参数 JSON。
禁止输出"已达到最大迭代次数"这类模板句；不要输出 JSON。"""

_SUBAGENT_SUMMARY_PROMPT = (
    "当前任务未在步骤预算内完成，请直接输出中文进度总结，不要 JSON。\n"
    "必须覆盖：1) 已完成内容；2) 当前未完成点；3) 下一步计划。\n"
    "禁止输出模板句‘已达到最大迭代次数’。"
)
_SUBAGENT_FINAL_SUMMARY_PROMPT = (
    "你已用完任务执行预算，禁止再调用工具。\n"
    "现在必须直接输出中文最终总结，供主 agent 回传给用户。\n"
    "必须覆盖：1) 已完成内容；2) 当前未完成内容；3) 产出文件路径（如果有）；4) 下一步建议。\n"
    "禁止：继续规划工具调用；说‘需要继续调用工具’；输出‘已达到最大迭代次数’等模板句。"
)
_SUBAGENT_FINAL_FALLBACK = (
    "这次后台任务已先停在当前进度。我已经完成了一部分关键步骤，"
    "但还有剩余工作未收束；下一次可从当前检查点继续推进。"
)
_REFLECT_PROMPT = (
    "根据上述工具结果，决定下一步操作。\n"
    "若任务已完成，直接输出最终结果；若需要继续，继续调用工具。\n"
    "禁止把工具调用失败的原因写进最终回复，遇到失败时换个方式或跳过该步骤。"
)
_REFLECT_PROMPT_WARN = (
    "根据上述工具结果，决定下一步操作。\n"
    "⚠️ 步骤预算剩余 {remaining} 步，请优先完成核心目标，跳过非必要步骤。\n"
    "若任务已完成，直接输出最终结果；若需要继续，继续调用工具。\n"
    "禁止把工具调用失败的原因写进最终回复，遇到失败时换个方式或跳过该步骤。"
)
_REFLECT_PROMPT_LAST = "⚠️ 步骤预算将在下一步耗尽。请立即优先完成核心目标，下一步将进入强制收尾。"


@dataclass(frozen=True)
class ProgressSummary:
    text: str
    used_fallback: bool = False


@dataclass(frozen=True)
class AgentExecutionPolicy:
    """State-free policy knobs around the shared execution loop."""

    source: ToolSource = "passive"
    recent_tool_rounds: int | None = None
    max_tool_result_chars: int | None = None
    reflect_after_tool_round: bool = False
    reflect_warning_threshold: int = 5
    summary_prompt: str = _PASSIVE_SUMMARY_PROMPT
    max_iterations_summary_prompt: str | None = None
    max_iterations_fallback: str | None = None

    def limit_tool_result(self, result: ToolResult) -> ToolResult:
        limit = self.max_tool_result_chars
        if limit is None or len(result.text) <= limit:
            return result
        original_length = len(result.text)
        return ToolResult(
            text=(
                result.text[:limit]
                + f"\n...[结果已截断，原始长度 {original_length} 字符，超出上限 {limit}]"
            ),
            content_blocks=list(result.content_blocks),
        )

    def append_after_tool_round(
        self,
        messages: list[dict[str, Any]],
        *,
        completed_iterations: int,
        max_iterations: int,
    ) -> None:
        if not self.reflect_after_tool_round:
            return
        remaining = max_iterations - completed_iterations
        if max_iterations <= 0 or remaining <= 0:
            return
        if remaining == 1:
            prompt = _REFLECT_PROMPT_LAST
        elif remaining <= self.reflect_warning_threshold:
            prompt = _REFLECT_PROMPT_WARN.format(remaining=remaining)
        else:
            prompt = _REFLECT_PROMPT
        messages.append({"role": "user", "content": prompt})

    def build_summary_prompt(
        self,
        *,
        reason: str,
        iteration: int,
        tools_used: list[str],
    ) -> str:
        body = (
            self.max_iterations_summary_prompt
            if reason == "max_iterations" and self.max_iterations_summary_prompt
            else self.summary_prompt
        )
        return (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n"
            f"[已调用工具] {', '.join(tools_used[-8:]) if tools_used else '无'}\n\n"
            + body
        )

    def fallback_summary(
        self,
        *,
        reason: str,
        iteration: int,
        tools_used: list[str],
    ) -> str:
        if reason == "max_iterations" and self.max_iterations_fallback:
            return self.max_iterations_fallback
        tool_text = "、".join(tools_used[-8:]) if tools_used else "无"
        return (
            f"这次任务还没完全收束。已尝试 {iteration} 轮，"
            f"调用工具 {len(tools_used)} 次（{tool_text}）。"
            "我先停在当前进度，后续会继续基于已有工具结果补齐缺失信息并给你最终结论。"
        )


PASSIVE_EXECUTION_POLICY = AgentExecutionPolicy()

SUBAGENT_EXECUTION_POLICY = AgentExecutionPolicy(
    source="subagent",
    recent_tool_rounds=3,
    max_tool_result_chars=100_000,
    reflect_after_tool_round=True,
    summary_prompt=_SUBAGENT_SUMMARY_PROMPT,
    max_iterations_summary_prompt=_SUBAGENT_FINAL_SUMMARY_PROMPT,
    max_iterations_fallback=_SUBAGENT_FINAL_FALLBACK,
)
