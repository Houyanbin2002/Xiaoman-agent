"""Real factory/pipeline/outbound wiring, isolated model and channel adapters."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from agent.tools.message_push import MessagePushTool
from bootstrap.dashboard_api.proactive_reader import ProactiveDashboardReader
from bus.queue import ChatLane
from core.attention.policies import PolicyEngine
from core.attention.source import PersonalAttentionSource
from proactive_v2.agent_tick_factory import AgentTickDeps, AgentTickFactory
from proactive_v2.config import ProactiveConfig
from proactive_v2.context import AgentTickContext
from proactive_v2.loop import ProactiveLoop
from proactive_v2.modules_resolve import build_delivery_key
from proactive_v2.outbound import ProactiveOutboundPort
from proactive_v2.state import ProactiveStateStore
from tests.proactive_v2.conftest import (
    FakeLLM,
    FakeRng,
    _FakeSession,
    run_proactive_pipeline,
)
from tests.proactive_v2.test_drift import _write_skill


@pytest.fixture
def harness(tmp_path, monkeypatch):
    _write_skill(tmp_path / "drift")
    cfg = ProactiveConfig(
        enabled=True,
        drift_enabled=True,
        default_channel="test",
        default_chat_id="1",
        drift_min_interval_hours=0,
        agent_tick_delivery_cooldown_hours=0,
    )
    state = ProactiveStateStore(tmp_path / "proactive.db")
    session = _FakeSession("test:1")
    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    deduper = NS(is_duplicate=AsyncMock(return_value=(False, "")))
    flags = NS(dnd=False)
    source = PersonalAttentionSource(
        personal_data=None,
        rhythm=NS(
            snapshot=lambda **_: NS(
                scene=NS(value="neutral"),
                focus_active=False,
                do_not_disturb=flags.dnd,
                allow_high_priority=True,
            )
        ),
        engine=NS(
            repository=NS(list_policies=lambda: []), policy_engine=PolicyEngine()
        ),
    )
    loop = object.__new__(ProactiveLoop)
    loop._cfg = cfg
    loop._delivery_stopped = False
    loop._passive_busy_fn = None
    loop._personal_source = source
    queued = asyncio.Event()

    class Lane(ChatLane):
        async def run_non_passive(self, channel, chat_id, send):
            queued.set()
            return await super().run_non_passive(channel, chat_id, send)

    lane = Lane()
    push = MessagePushTool(chat_lane=lane)
    sender = AsyncMock()
    push.register_channel("test", text=sender, image=sender)
    port = ProactiveOutboundPort(
        push, state=state, allowed=loop._delivery_allowed, dedupe_hours=24
    )
    loop._delivery_port = port
    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=NS(
                session_manager=NS(
                    get_or_create=lambda _: session, append_messages=AsyncMock()
                ),
                presence=NS(record_proactive_sent=MagicMock()),
            ),
            outbound=port,
        )
    )
    deps = AgentTickDeps(
        cfg=cfg,
        sense=NS(
            target_session_key=lambda: "test:1",
            collect_recent=lambda: [],
            collect_recent_proactive=lambda _: session.messages,
        ),
        presence=None,
        provider=None,
        model="isolated",
        max_tokens=128,
        memory=None,
        state_store=state,
        any_action_gate=gate,
        passive_busy_fn=None,
        deduper=deduper,
        rng=FakeRng(1.0),
        workspace_context_fn=lambda: "",
        turn_orchestrator=orchestrator,
    )

    def build(*, silent=False, unfinished=False, media=False):
        calls = [("select_skill", {"skill_name": "explore-curiosity"})]
        if not silent:
            args = {"message": "A useful exploration result"}
            if media:
                args = {"image": "https://example.invalid/isolated-image.png"}
            calls.append(("message_push", args))
        if not unfinished:
            calls.append(
                (
                    "finish_drift",
                    {
                        "skill_used": "explore-curiosity",
                        "status": "completed",
                        "briefing": "Exploration completed; delivery not yet attempted",
                        "message_result": "silent" if silent else "proposed",
                    },
                )
            )
        monkeypatch.setattr(AgentTickFactory, "_build_llm_fn", lambda _: FakeLLM(calls))
        return AgentTickFactory(deps).build()

    yield NS(
        build=build,
        state=state,
        sender=sender,
        gate=gate,
        flags=flags,
        lane=lane,
        queued=queued,
        loop=loop,
        session=session,
        deduper=deduper,
    )
    state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("media", [False, True])
async def test_drift_candidate_uses_real_delivery_identity_and_receipt(harness, media):
    tick = harness.build(media=media)

    async def accept(*_):
        # The transport is called only AFTER exploration finishes, BEFORE final logging/quota.
        assert tick.last_ctx.drift_finished
        harness.gate.record_action.assert_not_called()
        row = harness.state._db.execute("SELECT finished_at FROM tick_log").fetchone()
        assert row[0] is None

    harness.sender.side_effect = accept
    await run_proactive_pipeline(tick, session_key="test:1")
    harness.sender.assert_awaited_once()
    harness.deduper.is_duplicate.assert_awaited_once()
    harness.gate.record_action.assert_called_once()
    assert tick.last_ctx.delivery_status == "accepted"
    assert len(harness.session.messages) == 1
    assert (
        harness.state._db.execute("SELECT status FROM delivery_attempts").fetchone()[0]
        == "accepted"
    )
    reader = ProactiveDashboardReader(harness.state.db_path)
    try:
        assert (
            reader.get_tick_log(tick.last_ctx.tick_id)["delivery_status"] == "accepted"
        )
        assert reader.list_tick_logs()[0][0]["delivery_status"] == "accepted"
        assert reader.get_overview()["recent_tick"]["delivery_status"] == "accepted"
    finally:
        reader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block", ["dnd", "stop", "duplicate", "send_error", "silent", "unfinished"]
)
async def test_unified_guards_never_claim_unsent_candidate_was_delivered(
    harness, block
):
    tick = harness.build(silent=block == "silent", unfinished=block == "unfinished")
    if block in {"dnd", "stop"}:
        await harness.lane.mark_passive_pending("test", "1")
        task = asyncio.create_task(run_proactive_pipeline(tick, session_key="test:1"))
        try:
            await asyncio.wait_for(harness.queued.wait(), 3)
            assert not task.done()
            if block == "dnd":
                harness.flags.dnd = True
            else:
                harness.loop.stop()
            await harness.lane.mark_passive_done("test", "1")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    else:
        if block == "duplicate":
            harness.deduper.is_duplicate.return_value = (True, "same information")
        if block == "send_error":
            harness.sender.side_effect = RuntimeError("receipt unknown")
        await run_proactive_pipeline(tick, session_key="test:1")
    assert harness.sender.await_count == (1 if block == "send_error" else 0)
    harness.gate.record_action.assert_not_called()
    assert harness.session.messages == []
    assert tick.last_ctx.delivery_status != "accepted"
    assert harness.state.count_deliveries_in_window("test:1", 24) == 0
    row = harness.state._db.execute("SELECT delivery_status FROM tick_log").fetchone()
    assert row[0] == tick.last_ctx.delivery_status


@pytest.mark.asyncio
async def test_drift_repeated_candidate_is_deduped_by_shared_store(harness):
    await run_proactive_pipeline(harness.build(), session_key="test:1")
    second = harness.build()
    await run_proactive_pipeline(second, session_key="test:1")
    assert harness.sender.await_count == 1
    assert second.last_ctx.delivery_status == "not_requested"
    assert harness.gate.record_action.call_count == 1


def test_media_identity_is_part_of_shared_delivery_key():
    first = AgentTickContext(final_media=["/one.png"])
    second = AgentTickContext(final_media=["/two.png"])
    assert build_delivery_key(first) != build_delivery_key(second)


def test_old_tick_database_is_migrated_without_fabricating_receipts(tmp_path):
    path = tmp_path / "old.db"
    state = ProactiveStateStore(path)
    state._db.execute("ALTER TABLE tick_log DROP COLUMN delivery_status")
    state._db.commit()
    state.close()
    reader = ProactiveDashboardReader(path)
    try:
        assert reader.list_tick_logs()[0] == []
    finally:
        reader.close()
    state = ProactiveStateStore(path)
    try:
        columns = {
            row["name"] for row in state._db.execute("PRAGMA table_info(tick_log)")
        }
        assert "delivery_status" in columns
    finally:
        state.close()
