from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ContentSafetyError(Exception):
    """The upstream model provider rejected content under its safety policy."""


class ContextLengthError(Exception):
    """The upstream model provider rejected a request that exceeded its context."""


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


StreamDelta = dict[str, str]
