from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted, TurnStarted
from agent.config_models import ConversationSemanticsConfig
from agent.conversation_semantics.runtime import (
    ConversationSemanticsRuntime,
    build_conversation_semantics_runtime,
)
from session.store import SessionStore


@pytest.mark.asyncio
async def test_runtime_binds_starts_and_unbinds_batcher() -> None:
    event_bus = EventBus()
    batcher = Mock()
    batcher.start = AsyncMock()
    batcher.on_turn_committed = AsyncMock()
    batcher.on_turn_started = AsyncMock()
    batcher.aclose = AsyncMock()
    runtime = ConversationSemanticsRuntime(batcher=batcher, event_bus=event_bus)

    await runtime.start()
    await event_bus.fanout(
        TurnCommitted(
            session_key="web:1",
            channel="web",
            chat_id="1",
            input_message="hello",
            persisted_user_message="hello",
            assistant_response="hi",
            tools_used=[],
        )
    )

    batcher.start.assert_awaited_once()
    batcher.on_turn_committed.assert_awaited_once()
    assert TurnCommitted in event_bus._handlers
    assert TurnStarted in event_bus._handlers

    await runtime.aclose()

    batcher.aclose.assert_awaited_once()
    assert TurnCommitted not in event_bus._handlers
    assert TurnStarted not in event_bus._handlers


@pytest.mark.asyncio
async def test_runtime_builder_respects_enabled_flag(tmp_path) -> None:
    event_bus = EventBus()
    session_store = SessionStore(tmp_path / "sessions.db")

    runtime = build_conversation_semantics_runtime(
        config=ConversationSemanticsConfig(enabled=False),
        workspace=tmp_path,
        provider=Mock(),
        model="light",
        session_store=session_store,
        event_bus=event_bus,
    )

    assert runtime is None
    session_store.close()
