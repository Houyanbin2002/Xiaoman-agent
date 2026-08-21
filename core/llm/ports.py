from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from core.llm.models import LLMResponse, StreamDelta


class LLMProvider(Protocol):
    """Application-facing contract for a chat-completion provider."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        tool_choice: str | dict[str, Any] = "auto",
        extra_body: dict[str, Any] | None = None,
        disable_thinking: bool = False,
        on_content_delta: Callable[[StreamDelta], Awaitable[None]] | None = None,
    ) -> LLMResponse: ...
