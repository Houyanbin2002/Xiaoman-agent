from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ContentSafetyError(Exception):
    """The upstream model provider rejected content under its safety policy."""


class ContextLengthError(Exception):
    """The upstream model provider rejected a request that exceeded its context."""


class ToolArgumentsDecodeError(Exception):
    """A provider returned a tool call whose arguments are not one JSON object."""

    def __init__(
        self,
        *,
        tool_name: str,
        call_id: str,
        raw_arguments: str,
        reason: str,
    ) -> None:
        self.tool_name = tool_name
        self.call_id = call_id
        self.raw_arguments = raw_arguments[:2000]
        self.reason = reason
        super().__init__(
            f"工具 {tool_name or '<unknown>'} 参数不是合法 JSON 对象：{reason}"
        )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    provider_fields: dict[str, Any] = field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_prompt_tokens: int | None = None
    cache_hit_tokens: int | None = None
    finish_reason: str | None = None


StreamDelta = dict[str, str]
