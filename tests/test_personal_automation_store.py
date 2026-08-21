from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.personal.events import EventStatus, OperationStatus
from infra.persistence.personal_automation_store import PersonalAutomationStore
from infra.persistence.personal_store import PersonalStore


def _store(tmp_path: Path) -> PersonalAutomationStore:
    PersonalStore(tmp_path / "personal.db").close()
    return PersonalAutomationStore(tmp_path / "personal.db")


def test_event_dedupe_claim_and_complete(tmp_path: Path):
    store = _store(tmp_path)
    first, created = store.enqueue_event(
        event_type="health.observation.received",
        source="xiaomi_health",
        source_ref="sample-1",
        dedupe_key="xiaomi:sample-1",
        payload={"metric": "steps", "value": 6000},
    )
    duplicate, duplicate_created = store.enqueue_event(
        event_type="health.observation.received",
        source="xiaomi_health",
        dedupe_key="xiaomi:sample-1",
        payload={"metric": "steps", "value": 9999},
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate.id == first.id
    claimed = store.claim_events(worker_id="worker-a")
    assert [item.id for item in claimed] == [first.id]
    assert claimed[0].attempts == 1
    completed = store.complete_event(first.id, worker_id="worker-a")
    assert completed.status == EventStatus.SUCCEEDED
    store.close()


def test_event_retry_becomes_dead_after_attempt_budget(tmp_path: Path):
    store = _store(tmp_path)
    event, _ = store.enqueue_event(
        event_type="notion.sync",
        source="notion",
        payload={},
        max_attempts=2,
    )
    store.claim_events(worker_id="worker-a")
    retried = store.fail_event(
        event.id, worker_id="worker-a", error="temporary", retry_delay_seconds=0
    )
    assert retried.status == EventStatus.PENDING
    store.claim_events(worker_id="worker-a")
    dead = store.fail_event(
        event.id, worker_id="worker-a", error="permanent", retry_delay_seconds=0
    )
    assert dead.status == EventStatus.DEAD
    assert dead.attempts == 2
    store.close()


def test_expired_event_lease_can_be_recovered(tmp_path: Path):
    store = _store(tmp_path)
    event, _ = store.enqueue_event(
        event_type="calendar.changed", source="calendar", payload={}
    )
    store.claim_events(worker_id="worker-a", lease_seconds=1)
    with store._transaction() as db:
        db.execute(
            "UPDATE personal_events SET lease_until = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), event.id),
        )
    recovered = store.claim_events(worker_id="worker-b")
    assert recovered[0].id == event.id
    assert recovered[0].lease_owner == "worker-b"
    assert recovered[0].attempts == 2
    store.close()


def test_idempotent_operation_waits_for_approval_and_keeps_audit(tmp_path: Path):
    store = _store(tmp_path)
    operation, created = store.create_operation(
        idempotency_key="notion:create:commitment-42",
        action="notion.page.create",
        target="notion:tasks",
        request={"title": "Finish report"},
        requires_approval=True,
    )
    same, same_created = store.create_operation(
        idempotency_key="notion:create:commitment-42",
        action="notion.page.create",
        target="notion:tasks",
        request={"title": "Duplicate"},
        requires_approval=True,
    )

    assert created is True
    assert same_created is False
    assert same.id == operation.id
    assert operation.status == OperationStatus.AWAITING_APPROVAL
    approved = store.approve_operation(operation.id, actor="user", note="Proceed")
    assert approved.status == OperationStatus.READY
    running = store.start_operation(operation.id)
    assert running.attempt_count == 1
    completed = store.complete_operation(
        operation.id, result={"external_id": "page-1"}
    )
    assert completed.status == OperationStatus.SUCCEEDED
    assert completed.result["external_id"] == "page-1"
    assert [entry.action for entry in store.list_operation_audit(operation.id)] == [
        "created",
        "ready",
        "started",
        "succeeded",
    ]
    store.close()


def test_unapproved_operation_cannot_start(tmp_path: Path):
    store = _store(tmp_path)
    operation, _ = store.create_operation(
        idempotency_key="message:send:1",
        action="message.send",
        target="telegram:1",
        request={"text": "hello"},
        requires_approval=True,
    )

    with pytest.raises(ValueError, match="cannot start"):
        store.start_operation(operation.id)
    rejected = store.reject_operation(operation.id, actor="user", note="Not now")
    assert rejected.status == OperationStatus.REJECTED
    store.close()
