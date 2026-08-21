from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class EntityState(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class EventStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class DeliverySemantics(StrEnum):
    EXACT = "exact"
    BEFORE_DEADLINE = "before_deadline"
    OPPORTUNISTIC = "opportunistic"
    SILENT = "silent"


class WakeStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEAD = "dead"


def exact_reminder_job_id(event_id: str) -> str:
    digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:20]
    return f"attention-exact:{digest}"


@dataclass(frozen=True)
class CanonicalEntity:
    id: str
    source_id: str
    external_id: str
    kind: str
    title: str
    state: EntityState
    source_version: str
    payload_ref: str
    updated_at: str
    start_at: str = ""
    due_at: str = ""
    local_override: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalEntity:
        return cls(
            id=str(value.get("id") or ""),
            source_id=str(value.get("source_id") or ""),
            external_id=str(value.get("external_id") or ""),
            kind=str(value.get("kind") or ""),
            title=str(value.get("title") or ""),
            state=EntityState(str(value.get("state") or EntityState.OPEN.value)),
            source_version=str(value.get("source_version") or ""),
            payload_ref=str(value.get("payload_ref") or ""),
            updated_at=str(value.get("updated_at") or ""),
            start_at=str(value.get("start_at") or ""),
            due_at=str(value.get("due_at") or ""),
            local_override=dict(value.get("local_override") or {}),
        )


@dataclass(frozen=True)
class CanonicalEvent:
    id: str
    entity_id: str
    source_id: str
    kind: str
    occurred_at: str
    due_at: str
    active_from: str
    expires_at: str
    urgency: float
    confidence: float
    delivery_semantics: DeliverySemantics
    dedupe_key: str
    source_version: str
    payload_ref: str
    status: EventStatus

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["delivery_semantics"] = self.delivery_semantics.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalEvent:
        return cls(
            id=str(value.get("id") or ""),
            entity_id=str(value.get("entity_id") or ""),
            source_id=str(value.get("source_id") or ""),
            kind=str(value.get("kind") or ""),
            occurred_at=str(value.get("occurred_at") or ""),
            due_at=str(value.get("due_at") or ""),
            active_from=str(value.get("active_from") or ""),
            expires_at=str(value.get("expires_at") or ""),
            urgency=max(0.0, min(1.0, float(value.get("urgency") or 0.0))),
            confidence=max(0.0, min(1.0, float(value.get("confidence") or 0.0))),
            delivery_semantics=DeliverySemantics(
                str(value.get("delivery_semantics") or DeliverySemantics.SILENT.value)
            ),
            dedupe_key=str(value.get("dedupe_key") or ""),
            source_version=str(value.get("source_version") or ""),
            payload_ref=str(value.get("payload_ref") or ""),
            status=EventStatus(str(value.get("status") or EventStatus.ACTIVE.value)),
        )


@dataclass(frozen=True)
class WakePlan:
    id: str
    event_id: str
    wake_at: str
    reason: str
    attempt: int
    max_attempts: int
    status: WakeStatus
    last_decision: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WakePlan:
        return cls(
            id=str(value.get("id") or ""),
            event_id=str(value.get("event_id") or ""),
            wake_at=str(value.get("wake_at") or ""),
            reason=str(value.get("reason") or ""),
            attempt=max(0, int(value.get("attempt") or 0)),
            max_attempts=max(1, int(value.get("max_attempts") or 1)),
            status=WakeStatus(str(value.get("status") or WakeStatus.PENDING.value)),
            last_decision=str(value.get("last_decision") or ""),
            created_at=str(value.get("created_at") or ""),
            updated_at=str(value.get("updated_at") or ""),
        )
