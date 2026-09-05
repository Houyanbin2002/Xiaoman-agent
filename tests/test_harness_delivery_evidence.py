"""Isolated integration checks for the September memory/delivery audit."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.tools.message_push import MessagePushTool
from agent.turns.outbound import OutboundDispatch
from bus.queue import ChatLane
from bus.events_lifecycle import TurnCommitted
from core.attention.source import PersonalAttentionSource
from core.attention.policies import PolicyEngine
from core.memory.execution import execution_memory_uses, build_execution_state
from core.personal.governance import MemoryConflictAction
from core.personal.models import MemoryData, MemoryKind, RecordSource
from memory2.store import MemoryStore2
from plugins.default_memory.engine import DefaultMemoryEngine
from proactive_v2.outbound import ProactiveOutboundPort
from proactive_v2.state import ProactiveStateStore
from tests.test_memory_governance import _governance, _close


def test_background_quote_cannot_authorize_replacement(tmp_path):
    service = _governance(tmp_path)
    try:
        first = service.propose(memory=MemoryData(MemoryKind.PREFERENCE, "喜欢茶"), summary="喜欢茶", record_key="drink", source=RecordSource("user", "u1"), actor="user")
        result = service.propose(memory=MemoryData(MemoryKind.PREFERENCE, "喜欢咖啡"), summary="喜欢咖啡", record_key="drink", source=RecordSource("conversation_semantic_batch", "u2"), actor="user", user_confirmed=True, confidence=1.0, evidence_quote="我现在喜欢咖啡", review_reason="extracted_correction_requires_confirmation")
        assert result.status == "conflict_pending"
        assert service.personal_data.get(first.record.id).status.value == "active"
        accepted = service.resolve(result.conflict.id, action=MemoryConflictAction.ACCEPT_CANDIDATE)
        assert accepted.record.user_locked is True
        assert accepted.record.allow_auto_update is False
    finally:
        _close(service)


@pytest.mark.parametrize("markers,expected", [
    ('<used-execution-memory id="m" call_ids="c1"/>', {"m": ["c1"]}),
    ('<used-execution-memory id="m"/>', {}),
    ('<used-execution-memory id="m" call_ids="fake"/>', {}),
    ('<used-execution-memory id="fake" call_ids="c1"/>', {}),
    ('<used-execution-memory id="m" call_ids="c1"/><used-execution-memory id="n" call_ids="c1"/>', {}),
])
def test_use_requires_unique_observed_call(markers, expected):
    assert execution_memory_uses(markers, ["m", "n"], [{"calls": [{"call_id": "c1"}]}]) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status,result,counts", [
    ("success", '{"exit_code":0}', (1, 0)),
    ("success", '{"status":"running"}', (0, 0)),
    ("error", "network timeout", (0, 0)),
    ("error", '{"error_code":"invalid_arguments"}', (0, 1)),
])
async def test_feedback_terminal_evidence_and_idempotency(tmp_path, status, result, counts):
    store = MemoryStore2(tmp_path / "execution.db", vec_dim=4)
    try:
        item_id = store.upsert_item(memory_type="procedure", summary="读取项目环境后调用 shell", embedding=None, source_ref="fixture").split(":", 1)[1]
        store.execution.upsert(build_execution_state(item_id=item_id, metadata={"tool_requirement": "shell", "required_tools": ["shell"]}, source_ref="fixture", verified=False))
        engine = object.__new__(DefaultMemoryEngine)
        engine._v2_store = store
        event = TurnCommitted(session_key="test:1", channel="test", chat_id="1", input_message="执行", persisted_user_message="执行", assistant_response="完成", tools_used=["shell"], timestamp=datetime.now(timezone.utc), tool_chain_raw=[{"calls": [{"call_id": "c1", "name": "shell", "status": status, "result": result}]}], extra={"memory_retrieval": {"execution_memory_ids": [item_id], "execution_memory_uses": {item_id: ["c1"]}}})
        await engine._on_execution_feedback(event)
        await engine._on_execution_feedback(event)
        state = store.execution.get(item_id)
        assert (state.success_count, state.failure_count) == counts
    finally:
        store.close()


def _outbound():
    return OutboundDispatch(channel="test", chat_id="1", content="测试提醒", metadata={"session_key": "test:1", "delivery_key": "event-1"})


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_kind", ["policy", "stop"])
async def test_cancel_after_queue_wait_prevents_send(tmp_path, cancel_kind):
    state = ProactiveStateStore(tmp_path / "proactive.db")
    lane = ChatLane()
    push = MessagePushTool(chat_lane=lane)
    sender = AsyncMock()
    push.register_channel("test", text=sender)
    allowed = True
    port = ProactiveOutboundPort(push, state=state, allowed=lambda _: allowed, dedupe_hours=24)
    try:
        await lane.mark_passive_pending("test", "1")
        task = asyncio.create_task(port.dispatch(_outbound()))
        await asyncio.sleep(0)
        assert not task.done()
        if cancel_kind == "policy":
            allowed = False
        else:
            port.cancel_pending()
        await lane.mark_passive_done("test", "1")
        assert await asyncio.wait_for(task, 2) is False
        sender.assert_not_awaited()
    finally:
        state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_persisted_send_prevents_duplicate_after_reopen(tmp_path, failure):
    path = tmp_path / "proactive.db"
    state = ProactiveStateStore(path)
    sender = AsyncMock(side_effect=RuntimeError("remote receipt unknown") if failure else None)
    push = MessagePushTool()
    push.register_channel("test", text=sender)
    port = ProactiveOutboundPort(push, state=state, allowed=lambda _: True, dedupe_hours=24)
    assert await port.dispatch(_outbound()) is (not failure)
    state.close()
    state = ProactiveStateStore(path)
    try:
        port = ProactiveOutboundPort(push, state=state, allowed=lambda _: True, dedupe_hours=24)
        assert await port.dispatch(_outbound()) is False
        assert sender.await_count == 1
        row = state._db.execute("SELECT status FROM delivery_attempts").fetchone()
        assert row[0] == ("unknown" if failure else "accepted")
    finally:
        state.close()


@pytest.mark.parametrize("dnd,focus,expected", [(False, False, True), (True, False, False), (False, True, False)])
def test_live_policy_for_context_only_message(dnd, focus, expected):
    source = PersonalAttentionSource(personal_data=None, rhythm=SimpleNamespace(snapshot=lambda **_: SimpleNamespace(scene=SimpleNamespace(value="neutral"), focus_active=focus, do_not_disturb=dnd, allow_high_priority=True)), engine=SimpleNamespace(repository=SimpleNamespace(list_policies=lambda: []), policy_engine=PolicyEngine()))
    assert source.delivery_allowed([], channel="dashboard") is expected
