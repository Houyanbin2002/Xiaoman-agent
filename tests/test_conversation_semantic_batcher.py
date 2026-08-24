from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from agent.conversation_semantics.batcher import ConversationSemanticBatcher
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import SemanticBatchPayload
from core.conversation_semantics.store import ConversationSemanticStore


class _MessageSource:
    def __init__(self) -> None:
        self.messages: dict[str, list[dict[str, object]]] = {}

    def append_turn(self, session_key: str, user: str) -> None:
        rows = self.messages.setdefault(session_key, [])
        seq = len(rows)
        rows.extend(
            [
                {
                    "id": f"{session_key}:{seq}",
                    "session_key": session_key,
                    "seq": seq,
                    "role": "user",
                    "content": user,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "id": f"{session_key}:{seq + 1}",
                    "session_key": session_key,
                    "seq": seq + 1,
                    "role": "assistant",
                    "content": "好的",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ]
        )

    def fetch_session_messages(self, session_key: str) -> list[dict[str, object]]:
        return list(self.messages.get(session_key, []))

    def list_sessions(self) -> list[dict[str, object]]:
        return [{"key": key} for key in self.messages]


class _Analyzer:
    ANALYSIS_VERSION = "conversation-v1"

    def __init__(self) -> None:
        self.call_count = 0

    async def analyze(self, messages: list[dict[str, object]]) -> SemanticBatchPayload:
        self.call_count += 1
        assert messages
        return SemanticBatchPayload.empty()


class _EventBus:
    def __init__(self) -> None:
        self.published: list[object] = []

    async def emit(self, event: object) -> object:
        self.published.append(event)
        return event


def _turn(session_key: str, text: str = "测试") -> TurnCommitted:
    channel, chat_id = session_key.split(":", 1)
    return TurnCommitted(
        session_key=session_key,
        channel=channel,
        chat_id=chat_id,
        input_message=text,
        persisted_user_message=text,
        assistant_response="好的",
        tools_used=[],
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_single_turn_only_arms_idle_timer_without_model_call(tmp_path) -> None:
    source = _MessageSource()
    analyzer = _Analyzer()
    bus = _EventBus()
    batcher = ConversationSemanticBatcher(
        message_source=source,
        store=ConversationSemanticStore(tmp_path / "sessions.db"),
        analyzer=analyzer,
        event_bus=bus,
        idle_seconds=3600,
        max_turns=2,
    )
    source.append_turn("web:1", "第一轮")

    await batcher.on_turn_committed(_turn("web:1", "第一轮"))
    await batcher.drain()

    assert analyzer.call_count == 0
    assert bus.published == []
    await batcher.aclose()


@pytest.mark.asyncio
async def test_explicit_preference_flushes_without_waiting_for_idle_timer(tmp_path) -> None:
    source = _MessageSource()
    analyzer = _Analyzer()
    bus = _EventBus()
    batcher = ConversationSemanticBatcher(
        message_source=source,
        store=ConversationSemanticStore(tmp_path / "sessions.db"),
        analyzer=analyzer,
        event_bus=bus,
        idle_seconds=3600,
        max_turns=8,
    )
    source.append_turn("web:explicit", "我喜欢先给结论，再给简短步骤。")

    await batcher.on_turn_committed(_turn("web:explicit", "我喜欢先给结论，再给简短步骤。"))
    await batcher.drain()

    assert analyzer.call_count == 1
    assert len(bus.published) == 1
    await batcher.aclose()


@pytest.mark.asyncio
async def test_threshold_flush_publishes_one_durable_batch(tmp_path) -> None:
    source = _MessageSource()
    analyzer = _Analyzer()
    bus = _EventBus()
    store = ConversationSemanticStore(tmp_path / "sessions.db")
    batcher = ConversationSemanticBatcher(
        message_source=source,
        store=store,
        analyzer=analyzer,
        event_bus=bus,
        idle_seconds=3600,
        max_turns=2,
    )
    for text in ("第一轮", "第二轮"):
        source.append_turn("web:1", text)
        await batcher.on_turn_committed(_turn("web:1", text))

    await batcher.drain()

    assert analyzer.call_count == 1
    assert len(bus.published) == 1
    event = bus.published[0]
    assert len(event.message_ids) == 4
    # Semantic extraction no longer owns the model-context cursor.
    assert event.context_consolidate_through == -1
    assert store.pending_cursor("web:1") == 3
    await batcher.aclose()


@pytest.mark.asyncio
async def test_start_replays_prepared_batch_without_reanalysis(tmp_path) -> None:
    source = _MessageSource()
    analyzer = _Analyzer()
    bus = _EventBus()
    store = ConversationSemanticStore(tmp_path / "sessions.db")
    prepared = store.prepare(
        session_key="qq:7",
        channel="qq",
        chat_id="7",
        analysis_version="conversation-v1",
        message_ids=["qq:7:0", "qq:7:1"],
        end_seq=1,
        context_consolidate_through=0,
        payload=SemanticBatchPayload.empty(),
    )
    batcher = ConversationSemanticBatcher(
        message_source=source,
        store=store,
        analyzer=analyzer,
        event_bus=bus,
        idle_seconds=3600,
        max_turns=2,
    )

    await batcher.start()

    assert analyzer.call_count == 0
    assert bus.published[0].batch_id == prepared.batch_id
    assert store.pending_cursor("qq:7") == 1
    await batcher.aclose()


@pytest.mark.asyncio
async def test_start_recovers_unprepared_pending_messages(tmp_path) -> None:
    source = _MessageSource()
    source.append_turn("web:recovered", "恢复关闭前尚未分析的回合")
    analyzer = _Analyzer()
    bus = _EventBus()
    batcher = ConversationSemanticBatcher(
        message_source=source,
        store=ConversationSemanticStore(tmp_path / "sessions.db"),
        analyzer=analyzer,
        event_bus=bus,
        idle_seconds=3600,
        max_turns=8,
    )

    await batcher.start()
    await batcher.drain()

    assert analyzer.call_count == 1
    assert len(bus.published) == 1
    await batcher.aclose()


@pytest.mark.asyncio
async def test_shutdown_flushes_below_threshold_batch(tmp_path) -> None:
    source = _MessageSource()
    source.append_turn("web:shutdown", "退出前保留这轮")
    analyzer = _Analyzer()
    bus = _EventBus()
    batcher = ConversationSemanticBatcher(
        message_source=source,
        store=ConversationSemanticStore(tmp_path / "sessions.db"),
        analyzer=analyzer,
        event_bus=bus,
        idle_seconds=3600,
        max_turns=8,
    )
    await batcher.on_turn_committed(_turn("web:shutdown", "退出前保留这轮"))

    await batcher.aclose()

    assert analyzer.call_count == 1
    assert len(bus.published) == 1


@pytest.mark.asyncio
async def test_durable_delivery_retries_only_failed_consumer(tmp_path) -> None:
    source = _MessageSource()
    source.append_turn("web:durable", "测试分消费者回执")
    analyzer = _Analyzer()
    event_bus = EventBus()
    calls = {"healthy": 0, "flaky": 0}

    async def healthy(_event: object) -> None:
        calls["healthy"] += 1

    async def flaky(_event: object) -> None:
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise RuntimeError("temporary failure")

    event_bus.on(ConversationSemanticBatchCommitted, healthy)
    event_bus.on(ConversationSemanticBatchCommitted, flaky)
    store = ConversationSemanticStore(tmp_path / "sessions.db")
    batcher = ConversationSemanticBatcher(
        message_source=source,
        store=store,
        analyzer=analyzer,
        event_bus=event_bus,
        idle_seconds=3600,
        max_turns=8,
    )

    with pytest.raises(RuntimeError, match="durable event delivery failed"):
        await batcher.flush("web:durable", reason="test")
    assert calls == {"healthy": 1, "flaky": 1}
    assert len(store.list_undelivered()) == 1

    await batcher.flush("web:durable", reason="retry")

    assert calls == {"healthy": 1, "flaky": 2}
    assert store.pending_cursor("web:durable") == 1
    await batcher.aclose()


@pytest.mark.asyncio
async def test_shutdown_keeps_failed_delivery_for_startup_retry(tmp_path) -> None:
    source = _MessageSource()
    source.append_turn("web:retry-on-start", "退出时消费者暂时失败")
    analyzer = _Analyzer()
    event_bus = EventBus()

    async def unavailable(_event: object) -> None:
        raise RuntimeError("service unavailable")

    event_bus.on(ConversationSemanticBatchCommitted, unavailable)
    database = tmp_path / "sessions.db"
    batcher = ConversationSemanticBatcher(
        message_source=source,
        store=ConversationSemanticStore(database),
        analyzer=analyzer,
        event_bus=event_bus,
        idle_seconds=3600,
        max_turns=8,
    )
    await batcher.on_turn_committed(_turn("web:retry-on-start", "退出时消费者暂时失败"))

    await batcher.aclose()

    reopened = ConversationSemanticStore(database)
    assert len(reopened.list_undelivered()) == 1
    assert reopened.pending_cursor("web:retry-on-start") == -1
    reopened.close()
