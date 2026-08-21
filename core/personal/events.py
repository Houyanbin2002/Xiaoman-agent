from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.personal.models import PersonalRecord


class EventStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    DEAD = "dead"


class OperationStatus(StrEnum):
    AWAITING_APPROVAL = "awaiting_approval"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PersonalEvent:
    id: str
    event_type: str
    source: str
    source_ref: str
    payload: dict[str, Any]
    dedupe_key: str
    status: EventStatus
    attempts: int
    max_attempts: int
    available_at: str
    lease_owner: str
    lease_until: str | None
    last_error: str
    created_at: str
    updated_at: str
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "source": self.source,
            "source_ref": self.source_ref,
            "payload": self.payload,
            "dedupe_key": self.dedupe_key,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "available_at": self.available_at,
            "lease_owner": self.lease_owner,
            "lease_until": self.lease_until,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class OperationReceipt:
    id: str
    idempotency_key: str
    action: str
    target: str
    request: dict[str, Any]
    result: dict[str, Any]
    status: OperationStatus
    requires_approval: bool
    approval_actor: str
    approval_note: str
    approved_at: str | None
    attempt_count: int
    error: str
    created_at: str
    updated_at: str
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "idempotency_key": self.idempotency_key,
            "action": self.action,
            "target": self.target,
            "request": self.request,
            "result": self.result,
            "status": self.status.value,
            "requires_approval": self.requires_approval,
            "approval_actor": self.approval_actor,
            "approval_note": self.approval_note,
            "approved_at": self.approved_at,
            "attempt_count": self.attempt_count,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class OperationAuditEntry:
    id: int
    operation_id: str
    action: str
    actor: str
    details: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class PersonalRecordChanged:
    record: PersonalRecord
    change: str
    actor: str
