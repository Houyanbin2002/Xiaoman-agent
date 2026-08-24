from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent.config_models import ConversationSemanticsConfig
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted, TurnStarted
from agent.conversation_semantics.batcher import ConversationSemanticBatcher
from core.conversation_semantics.analyzer import ConversationSemanticAnalyzer
from core.conversation_semantics.store import ConversationSemanticStore
from core.llm import LLMProvider
from session.store import SessionStore


@dataclass
class ConversationSemanticsRuntime:
    batcher: ConversationSemanticBatcher
    event_bus: EventBus
    _unsubscribers: list[Callable[[], None]] = field(default_factory=list, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._unsubscribers.extend(
            [
                self.event_bus.on(TurnCommitted, self.batcher.on_turn_committed),
                self.event_bus.on(TurnStarted, self.batcher.on_turn_started),
            ]
        )

    async def start(self) -> None:
        await self.batcher.start()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for unsubscribe in reversed(self._unsubscribers):
            unsubscribe()
        self._unsubscribers.clear()
        await self.batcher.aclose()


def build_conversation_semantics_runtime(
    *,
    config: ConversationSemanticsConfig,
    workspace: Path,
    provider: LLMProvider,
    model: str,
    session_store: SessionStore,
    event_bus: EventBus,
) -> ConversationSemanticsRuntime | None:
    if not config.enabled:
        return None
    analyzer = ConversationSemanticAnalyzer(
        provider,
        model,
        analysis_version=config.analysis_version,
    )
    batcher = ConversationSemanticBatcher(
        message_source=session_store,
        store=ConversationSemanticStore(workspace / "sessions.db"),
        analyzer=analyzer,
        event_bus=event_bus,
        idle_seconds=config.idle_seconds,
        max_turns=config.max_turns,
    )
    return ConversationSemanticsRuntime(batcher=batcher, event_bus=event_bus)
