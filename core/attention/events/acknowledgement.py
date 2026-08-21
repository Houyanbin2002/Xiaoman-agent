from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from core.attention.events import (
    CanonicalEvent,
    EntityState,
    EventStatus,
    exact_reminder_job_id,
)
from core.attention.feedback import FeedbackKind
from core.attention.feedback.service import FeedbackService
from core.attention.ports import AttentionRepository
from core.personal.service import PersonalDataService


class EventAcknowledgementService:
    """Apply user-authoritative completion to the event and its source projection."""

    def __init__(
        self,
        *,
        repository: AttentionRepository,
        personal_data: PersonalDataService,
        feedback: FeedbackService,
        scheduler: object | None = None,
    ) -> None:
        self._repository = repository
        self._personal_data = personal_data
        self._feedback = feedback
        self._scheduler = scheduler

    def complete(self, event_id: str, *, actor: str = "user") -> dict[str, object]:
        event = self._repository.get_event(event_id)
        if event is None:
            raise ValueError(f"主动事件不存在: {event_id}")
        entity = self._repository.get_entity(event.entity_id)
        if entity is None:
            raise ValueError(f"主动实体不存在: {event.entity_id}")
        now = datetime.now(timezone.utc).isoformat()
        override = {
            **entity.local_override,
            "state": EntityState.COMPLETED.value,
            "authority": actor,
            "source_sync": "pending",
            "updated_at": now,
        }
        self._repository.upsert_entity(
            replace(
                entity,
                state=EntityState.COMPLETED,
                local_override=override,
                updated_at=now,
            )
        )
        self._repository.close_event(event.id, EventStatus.COMPLETED)
        self._cancel_scheduled_delivery(event)
        record_id = self._personal_record_id(entity.id)
        source_updated = False
        if record_id:
            record = self._personal_data.get(record_id)
            if record is not None:
                self._personal_data.update(
                    record.id,
                    {
                        "data": {
                            "state": "completed",
                            "completed_at": now,
                            "source_sync": "pending",
                        }
                    },
                    actor=actor,
                    reason="user acknowledged proactive event completion",
                )
                source_updated = True
        plan_ids = self._record_plan_feedback(event.id)
        return {
            "event_id": event.id,
            "entity_id": entity.id,
            "status": EventStatus.COMPLETED.value,
            "source_projection_updated": source_updated,
            "feedback_plan_ids": plan_ids,
        }

    def cancel(self, event_id: str, *, actor: str = "user") -> dict[str, object]:
        event = self._repository.get_event(event_id)
        if event is None:
            raise ValueError(f"主动事件不存在: {event_id}")
        entity = self._repository.get_entity(event.entity_id)
        if entity is not None:
            self._repository.upsert_entity(
                replace(
                    entity,
                    state=EntityState.CANCELLED,
                    local_override={
                        **entity.local_override,
                        "state": EntityState.CANCELLED.value,
                        "authority": actor,
                        "source_sync": "pending",
                    },
                )
            )
        self._repository.close_event(event.id, EventStatus.CANCELLED)
        self._cancel_scheduled_delivery(event)
        return {"event_id": event.id, "status": EventStatus.CANCELLED.value}

    def _cancel_scheduled_delivery(self, event: CanonicalEvent) -> None:
        cancel = getattr(self._scheduler, "cancel_job", None)
        if callable(cancel):
            job_id = (
                event.payload_ref
                if event.source_id == "scheduler"
                else exact_reminder_job_id(event.id)
            )
            cancel(job_id)

    def _record_plan_feedback(self, event_id: str) -> list[str]:
        plan_ids: list[str] = []
        for plan in self._repository.list_plans(limit=1000):
            if not any(
                (signal := self._repository.get_signal(signal_id)) is not None
                and str(signal.metadata.get("event_id") or "") == event_id
                for signal_id in plan.signal_ids
            ):
                continue
            self._feedback.record(
                plan_id=plan.id,
                kind=FeedbackKind.COMPLETED,
                note="用户确认事项已完成",
                metadata={"event_id": event_id},
            )
            plan_ids.append(plan.id)
        return plan_ids

    @staticmethod
    def _personal_record_id(entity_id: str) -> str:
        prefix = "entity:personal:"
        return entity_id[len(prefix) :] if entity_id.startswith(prefix) else ""


__all__ = ["EventAcknowledgementService"]
