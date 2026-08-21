from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.attention.events.models import (
    CanonicalEntity,
    CanonicalEvent,
    DeliverySemantics,
    EntityState,
    EventStatus,
    WakePlan,
    WakeStatus,
)
from infra.persistence.attention_engine_store import AttentionEngineStore


def test_event_store_is_idempotent_and_closing_event_cancels_wakes(tmp_path) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    now = datetime.now(timezone.utc)
    entity = CanonicalEntity(
        id="entity:conversation:task-1",
        source_id="conversation",
        external_id="task-1",
        kind="task",
        title="周五前提交报告",
        state=EntityState.OPEN,
        source_version="batch-1",
        payload_ref="batch-1",
        updated_at=now.isoformat(),
    )
    event = CanonicalEvent(
        id="event:conversation:task-1",
        entity_id=entity.id,
        source_id="conversation",
        kind="deadline",
        occurred_at=now.isoformat(),
        due_at=(now + timedelta(days=2)).isoformat(),
        active_from="",
        expires_at=(now + timedelta(days=2)).isoformat(),
        urgency=0.7,
        confidence=0.95,
        delivery_semantics=DeliverySemantics.BEFORE_DEADLINE,
        dedupe_key="conversation:task-1",
        source_version="batch-1",
        payload_ref="batch-1",
        status=EventStatus.ACTIVE,
    )
    wake = WakePlan(
        id="wake:event-1:0",
        event_id=event.id,
        wake_at=(now + timedelta(hours=2)).isoformat(),
        reason="deadline_review",
        attempt=0,
        max_attempts=3,
        status=WakeStatus.PENDING,
        last_decision="",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    assert store.upsert_entity(entity) == entity
    assert store.upsert_event(event) == event
    assert store.upsert_event(event).id == event.id
    assert store.upsert_wake(wake) == wake
    assert store.next_wake_at() == wake.wake_at

    closed = store.close_event(event.id, EventStatus.COMPLETED)

    assert closed.status is EventStatus.COMPLETED
    assert store.get_wake(wake.id).status is WakeStatus.CANCELLED
    assert store.list_pending_wakes() == []
    store.close()


def test_event_store_claims_due_wake_once_and_recovers_processing(tmp_path) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    now = datetime.now(timezone.utc)
    wake = WakePlan(
        id="wake:due",
        event_id="event:due",
        wake_at=(now - timedelta(seconds=1)).isoformat(),
        reason="deadline_review",
        attempt=0,
        max_attempts=2,
        status=WakeStatus.PENDING,
        last_decision="",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )
    store.upsert_wake(wake)

    claimed = store.claim_due_wakes(now=now, limit=10)

    assert [item.id for item in claimed] == [wake.id]
    assert claimed[0].status is WakeStatus.PROCESSING
    assert claimed[0].attempt == 1
    assert store.claim_due_wakes(now=now, limit=10) == []
    assert store.recover_processing_wakes() == 1
    assert store.get_wake(wake.id).status is WakeStatus.PENDING
    store.close()
