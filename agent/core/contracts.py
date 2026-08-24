from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent.core.types import ContextBundle, ReasonerResult
from agent.lifecycle.types import PromptRenderInput, PromptRenderResult
from bus.events import InboundMessage

if TYPE_CHECKING:
    from agent.core.runtime_support import SessionLike, TurnRunResult
    from agent.tool_hooks.base import ToolHook


class ContextStore(ABC):
    """读取会话、检索记忆并构建本轮上下文的稳定端口。"""

    @abstractmethod
    async def prepare(
        self,
        *,
        msg: InboundMessage,
        session_key: str,
        session: SessionLike,
    ) -> ContextBundle: ...


class Reasoner(ABC):
    """统一执行内核面向主循环暴露的推理端口。"""

    @abstractmethod
    async def run(
        self,
        initial_messages: list[dict[str, Any]],
        *,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preloaded_tool_order: list[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
        request_text: str = "",
        permission_mode: str = "full_access",
        disabled_tools: set[str] | None = None,
        resume_from_checkpoint: bool = False,
    ) -> ReasonerResult: ...

    @abstractmethod
    async def run_turn(
        self,
        *,
        msg: InboundMessage,
        session: SessionLike,
        skill_names: list[str] | None = None,
        base_history: list[dict[str, Any]] | None = None,
        retrieved_memory_block: str = "",
        extra_hints: list[str] | None = None,
    ) -> TurnRunResult: ...

    def add_tool_hooks(self, hooks: list[ToolHook]) -> None:
        pass

    def add_prompt_render_plugin_modules(self, modules: list[object]) -> None:
        pass

    def add_before_step_plugin_modules(self, modules: list[object]) -> None:
        pass

    def add_after_step_plugin_modules(self, modules: list[object]) -> None:
        pass

    async def render_prompt(self, input: PromptRenderInput) -> PromptRenderResult:
        raise NotImplementedError
