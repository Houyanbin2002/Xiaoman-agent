"""Test-only adapter for executing the shared reasoner kernel directly."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.core.passive_turn import build_turn_injection_prompt
from agent.looping.core import AgentLoop
from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)


async def run_agent_kernel(
    loop: AgentLoop,
    initial_messages: list[dict[str, Any]],
    request_time: datetime | None = None,
    preloaded_tools: set[str] | None = None,
) -> tuple[str, list[str], list[dict[str, Any]], set[str] | None, str | None]:
    """Run the reasoner with the same per-turn tool hint used by normal turns."""
    visible = preloaded_tools if loop._tool_search_enabled else None
    hint = build_turn_injection_prompt(
        tools=loop.tools,
        tool_search_enabled=loop._tool_search_enabled,
        visible_names=visible,
    )
    if hint:
        hint_message = build_context_frame_message(
            build_context_frame_content(
                [
                    PromptSectionRender(
                        name="turn_injection",
                        content=hint,
                        is_static=False,
                    )
                ]
            )
        )
        if initial_messages and initial_messages[-1].get("role") == "user":
            initial_messages = initial_messages[:-1] + [
                hint_message,
                initial_messages[-1],
            ]
        else:
            initial_messages = [*initial_messages, hint_message]

    result = await loop._reasoner.run(
        initial_messages,
        request_time=request_time,
        preloaded_tools=preloaded_tools,
        preflight_injected=True,
    )
    return (
        result.reply,
        list(result.metadata.get("tools_used") or []),
        list(result.metadata.get("tool_chain") or []),
        result.metadata.get("visible_names"),
        result.thinking,
    )
