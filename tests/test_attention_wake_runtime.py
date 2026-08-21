from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.attention.events import (
    CanonicalEntity,
    CanonicalEvent,
    DeliverySemantics,
    EntityState,
    EventStatus,
    WakePlan,
    WakeStatus,
)
from core.attention.events.runtime import AttentionWakeRuntime
from core.attention.events.service import EventDrivenAttentionService
from infra.persistence.attention_engine_store import AttentionEngineStore


async def test_wake_runtime_only_ticks_for_due_events(tmp_path) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    _seed_due_event(store, now)
    tick = AsyncMock(return_value=0.72)
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        now_fn=lambda: now,
    )
    runtime = AttentionWakeRuntime(
        repository=store,
        events=service,
        tick=tick,
        now_fn=lambda: now,
    )

    assert await runtime.run_due_once(now=now) == 1
    tick.assert_awaited_once()
    assert store.get_wake("wake:due").status is WakeStatus.COMPLETED
    assert await runtime.run_due_once(now=now) == 0
    store.close()


async def test_wake_runtime_defers_a_failed_decision_with_a_limit(tmp_path) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    _seed_due_event(store, now, max_attempts=1)
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        now_fn=lambda: now,
    )
    runtime = AttentionWakeRuntime(
        repository=store,
        events=service,
        tick=AsyncMock(return_value=None),
        now_fn=lambda: now,
    )

    assert await runtime.run_due_once(now=now) == 1

    assert store.get_wake("wake:due").status is WakeStatus.DEAD
    store.close()


def _seed_due_event(
    store: AttentionEngineStore,
    now: datetime,
    *,
    max_attempts: int = 3,
) -> None:
    entity = CanonicalEntity(
        id="entity:due",
        source_id="test",
        external_id="due",
        kind="task",
        title="提交报告",
        state=EntityState.OPEN,
        source_version="1",
        payload_ref="test",
        updated_at=now.isoformat(),
    )
    event = CanonicalEvent(
        id="event:due",
        entity_id=entity.id,
        source_id="test",
        kind="deadline",
        occurred_at=now.isoformat(),
        due_at=(now + timedelta(hours=2)).isoformat(),
        active_from="",
        expires_at=(now + timedelta(hours=3)).isoformat(),
        urgency=0.8,
        confidence=0.9,
        delivery_semantics=DeliverySemantics.BEFORE_DEADLINE,
        dedupe_key="test:due",
        source_version="1",
        payload_ref="test",
        status=EventStatus.ACTIVE,
    )
    wake = WakePlan(
        id="wake:due",
        event_id=event.id,
        wake_at=(now - timedelta(seconds=1)).isoformat(),
        reason="review",
        attempt=0,
        max_attempts=max_attempts,
        status=WakeStatus.PENDING,
        last_decision="",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )
    store.upsert_entity(entity)
    store.upsert_event(event)
    store.upsert_wake(wake)
