from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from core.attention._shared import parse_datetime
from core.attention.events.models import (
    CanonicalEntity,
    CanonicalEvent,
    DeliverySemantics,
    EntityState,
    EventStatus,
    WakePlan,
    WakeStatus,
    exact_reminder_job_id,
)
from core.attention.ports import AttentionRepository
from core.attention.signals import AttentionSignal, SignalSource
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import TaskEventCandidate
from core.personal.events import PersonalRecordChanged
from core.personal.models import PersonalEntityType, PersonalRecord, RecordStatus
from core.scheduling import ScheduledJob, ScheduledJobChanged

_ATTENTION_ENTITY_TYPES = frozenset(
    {
        PersonalEntityType.COMMITMENT,
        PersonalEntityType.CALENDAR_EVENT,
        PersonalEntityType.IMPORTANT_DATE,
        PersonalEntityType.FINANCIAL_OBLIGATION,
        PersonalEntityType.TRIP,
        PersonalEntityType.GOAL,
        PersonalEntityType.PROACTIVE_INTENT,
    }
)

logger = logging.getLogger(__name__)
_MIN_CONVERSATION_TASK_CONFIDENCE = 0.75


def _record_has_attention_semantics(record: PersonalRecord) -> bool:
    data = record.data
    semantics = str(data.get("delivery_semantics") or "").strip()
    if semantics in {item.value for item in DeliverySemantics}:
        return True
    if isinstance(data.get("attention_signal"), dict):
        return True
    if any(data.get(key) for key in ("due_at", "active_from", "start_at")):
        return True
    return record.entity_type in _ATTENTION_ENTITY_TYPES


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EventDrivenAttentionService:
    """Normalize durable facts into events without replacing AttentionEngine."""

    def __init__(
        self,
        *,
        repository: AttentionRepository,
        attention_engine: Any,
        scheduler: Any | None = None,
        now_fn: Callable[[], datetime] = _utc_now,
        default_channel: str = "",
        default_chat_id: str = "",
    ) -> None:
        self._repository = repository
        self._attention_engine = attention_engine
        self._scheduler = scheduler
        self._now = now_fn
        self._default_channel = str(default_channel or "").strip()
        self._default_chat_id = str(default_chat_id or "").strip()
        self._wake_notifier: Callable[[], None] | None = None

    def bind_wake_notifier(self, notifier: Callable[[], None] | None) -> None:
        self._wake_notifier = notifier

    async def handle_semantic_batch(
        self,
        batch: ConversationSemanticBatchCommitted,
    ) -> None:
        user_message_ids = set(batch.user_message_ids) & set(batch.message_ids)
        for task in batch.payload.task_events:
            if (
                task.source_message_id not in user_message_ids
                or task.confidence < _MIN_CONVERSATION_TASK_CONFIDENCE
            ):
                logger.info(
                    "task candidate rejected: missing user evidence or low confidence"
                )
                continue
            due = parse_datetime(task.due_at)
            if (
                task.operation == "upsert"
                and task.delivery_semantics
                in {
                    DeliverySemantics.EXACT.value,
                    DeliverySemantics.BEFORE_DEADLINE.value,
                }
                and due is None
            ):
                logger.info(
                    "task candidate rejected: temporal semantics without due_at"
                )
                continue
            if (
                task.operation == "upsert"
                and task.delivery_semantics == DeliverySemantics.EXACT.value
                and due is not None
                and due <= self._now().astimezone(timezone.utc)
            ):
                logger.info(
                    "task candidate rejected: exact reminder is not in the future"
                )
                continue
            if task.operation in {"complete", "cancel"}:
                self._close_conversation_task(batch, task)
            else:
                self._ingest_conversation_task(batch, task)

    def _close_conversation_task(
        self,
        batch: ConversationSemanticBatchCommitted,
        task: TaskEventCandidate,
    ) -> CanonicalEvent | None:
        query = task.related_summary or task.summary
        event = (
            self._repository.get_event(task.related_event_id)
            if task.related_event_id
            else self._resolve_conversation_event(batch.session_key, query)
        )
        if event is None:
            return None
        entity = self._repository.get_entity(event.entity_id)
        if (
            entity is None
            or entity.source_id != "conversation"
            or not entity.external_id.startswith(f"{batch.session_key}:")
        ):
            return None
        status = (
            EventStatus.COMPLETED
            if task.operation == "complete"
            else EventStatus.CANCELLED
        )
        self._repository.upsert_entity(
            replace(
                entity,
                state=(
                    EntityState.COMPLETED
                    if status is EventStatus.COMPLETED
                    else EntityState.CANCELLED
                ),
                updated_at=self._now().astimezone(timezone.utc).isoformat(),
            )
        )
        closed = self._repository.close_event(event.id, status)
        self._notify_wake_change()
        return closed

    def _resolve_conversation_event(
        self,
        session_key: str,
        query: str,
    ) -> CanonicalEvent | None:
        normalized = self._normalized_text(query)
        candidates: list[tuple[CanonicalEvent, str]] = []
        for event in self._repository.list_active_events():
            entity = self._repository.get_entity(event.entity_id)
            if (
                entity is None
                or entity.source_id != "conversation"
                or not entity.external_id.startswith(f"{session_key}:")
            ):
                continue
            candidates.append((event, self._normalized_text(entity.title)))
        exact = [
            event for event, title in candidates if normalized and title == normalized
        ]
        if len(exact) == 1:
            return exact[0]
        contained = [
            event
            for event, title in candidates
            if normalized and title and (normalized in title or title in normalized)
        ]
        if len(contained) == 1:
            return contained[0]
        related = [
            event
            for event, title in candidates
            if self._text_overlap(normalized, title) >= 0.5
        ]
        return related[0] if len(related) == 1 else None

    def _ingest_conversation_task(
        self,
        batch: ConversationSemanticBatchCommitted,
        task: TaskEventCandidate,
    ) -> CanonicalEvent:
        now = self._now().astimezone(timezone.utc)
        external_id = task.source_message_id or self._digest(task.summary)
        identity = f"{batch.session_key}:{external_id}"
        entity_id = f"entity:conversation:{identity}"
        event_id = f"event:conversation:{identity}"
        semantics = DeliverySemantics(task.delivery_semantics)
        due = parse_datetime(task.due_at)
        active_from = parse_datetime(task.active_from)
        expires = parse_datetime(task.expires_at) or due
        entity = CanonicalEntity(
            id=entity_id,
            source_id="conversation",
            external_id=identity,
            kind="task",
            title=task.summary,
            state=EntityState.OPEN,
            source_version=batch.batch_id,
            payload_ref=batch.batch_id,
            updated_at=now.isoformat(),
            due_at=due.isoformat() if due else "",
        )
        event = CanonicalEvent(
            id=event_id,
            entity_id=entity.id,
            source_id="conversation",
            kind=(
                "deadline"
                if semantics
                in {DeliverySemantics.EXACT, DeliverySemantics.BEFORE_DEADLINE}
                else "opportunity"
            ),
            occurred_at=now.isoformat(),
            due_at=due.isoformat() if due else "",
            active_from=active_from.isoformat() if active_from else "",
            expires_at=expires.isoformat() if expires else "",
            urgency=self._urgency(now=now, due=due),
            confidence=task.confidence,
            delivery_semantics=semantics,
            dedupe_key=f"conversation:{identity}",
            source_version=batch.batch_id,
            payload_ref=batch.batch_id,
            status=EventStatus.ACTIVE,
        )
        self._repository.upsert_entity(entity)
        stored = self._repository.upsert_event(event)
        self._route_event(
            stored,
            entity,
            channel=batch.channel,
            chat_id=batch.chat_id,
            now=now,
            due=due,
            active_from=active_from,
        )
        return stored

    async def handle_personal_record_changed(
        self,
        change: PersonalRecordChanged,
    ) -> None:
        record = change.record
        now = self._now().astimezone(timezone.utc)
        entity_id = f"entity:personal:{record.id}"
        previous_entity = self._repository.get_entity(entity_id)
        if not _record_has_attention_semantics(record):
            if previous_entity is not None:
                self._repository.upsert_entity(
                    replace(
                        previous_entity,
                        state=EntityState.CANCELLED,
                        updated_at=record.updated_at or now.isoformat(),
                    )
                )
                self._cancel_exact_events_for_entity(entity_id)
                self._repository.close_events_for_entity(
                    entity_id,
                    EventStatus.CANCELLED,
                )
                self._notify_wake_change()
            return
        raw_state = str(record.data.get("state") or record.data.get("status") or "")
        terminal_change = change.change in {"forgotten", "superseded", "deleted"}
        completed = raw_state.lower() in {"done", "completed", "closed"}
        cancelled = raw_state.lower() in {"cancelled", "canceled", "inactive"}
        if terminal_change or record.status is not RecordStatus.ACTIVE or cancelled:
            entity_state = EntityState.CANCELLED
        elif completed:
            entity_state = EntityState.COMPLETED
        else:
            entity_state = EntityState.OPEN
        source_id = record.source.source or "personal"
        due = parse_datetime(record.data.get("due_at"))
        active_from = parse_datetime(record.data.get("active_from")) or parse_datetime(
            record.data.get("start_at")
        )
        explicit_semantics = str(record.data.get("delivery_semantics") or "").strip()
        if explicit_semantics in {item.value for item in DeliverySemantics}:
            semantics = DeliverySemantics(explicit_semantics)
        elif isinstance(record.data.get("attention_signal"), dict):
            semantics = DeliverySemantics.OPPORTUNISTIC
        elif due is not None or active_from is not None:
            semantics = DeliverySemantics.BEFORE_DEADLINE
        else:
            semantics = DeliverySemantics.SILENT
        local_override = (
            dict(previous_entity.local_override) if previous_entity is not None else {}
        )
        override_state = str(local_override.get("state") or "")
        if local_override.get("source_sync") == "pending" and override_state:
            if change.actor == "external-sync" and entity_state.value == override_state:
                local_override = {}
            else:
                entity_state = EntityState(override_state)
        entity = CanonicalEntity(
            id=entity_id,
            source_id=source_id,
            external_id=record.record_key or record.id,
            kind=record.entity_type.value,
            title=record.title,
            state=entity_state,
            source_version=str(record.revision),
            payload_ref=record.source.source_ref or record.id,
            updated_at=record.updated_at,
            start_at=active_from.isoformat() if active_from else "",
            due_at=due.isoformat() if due else "",
            local_override=local_override,
        )
        self._repository.upsert_entity(entity)
        if entity_state is not EntityState.OPEN:
            status = (
                EventStatus.COMPLETED
                if entity_state is EntityState.COMPLETED
                else EventStatus.CANCELLED
            )
            self._cancel_exact_events_for_entity(entity.id)
            self._repository.close_events_for_entity(entity.id, status)
            self._notify_wake_change()
            return
        event = CanonicalEvent(
            id=f"event:personal:{record.id}",
            entity_id=entity.id,
            source_id=source_id,
            kind="deadline" if due is not None else "opportunity",
            occurred_at=record.updated_at or now.isoformat(),
            due_at=due.isoformat() if due else "",
            active_from=active_from.isoformat() if active_from else "",
            expires_at=str(record.expires_at or (due.isoformat() if due else "")),
            urgency=self._urgency(now=now, due=due),
            confidence=record.confidence,
            delivery_semantics=semantics,
            dedupe_key=f"personal:{record.id}",
            source_version=str(record.revision),
            payload_ref=record.source.source_ref or record.id,
            status=EventStatus.ACTIVE,
        )
        stored = self._repository.upsert_event(event)
        self._route_event(
            stored,
            entity,
            channel=str(record.data.get("channel") or ""),
            chat_id=str(record.data.get("chat_id") or ""),
            now=now,
            due=due,
            active_from=active_from,
        )

    async def handle_scheduled_job_changed(
        self,
        change: ScheduledJobChanged,
    ) -> None:
        job = dict(change.job)
        metadata = job.get("metadata")
        linked_event_id = str(
            (metadata.get("canonical_event_id") or "")
            if isinstance(metadata, dict)
            else ""
        ).strip()
        if linked_event_id:
            if change.action in {"cancelled", "expired", "fired"}:
                self._close_linked_scheduled_event(
                    linked_event_id,
                    status={
                        "fired": EventStatus.COMPLETED,
                        "cancelled": EventStatus.CANCELLED,
                        "expired": EventStatus.EXPIRED,
                    }[change.action],
                )
            return
        now = self._now().astimezone(timezone.utc)
        due = parse_datetime(job.get("fire_at"))
        title = str(
            job.get("name") or job.get("message") or job.get("prompt") or "定时任务"
        ).strip()
        entity_id = f"entity:scheduler:{change.job_id}"
        entity = CanonicalEntity(
            id=entity_id,
            source_id="scheduler",
            external_id=change.job_id,
            kind="schedule",
            title=title,
            state=EntityState.OPEN,
            source_version=f"{job.get('fire_at')}:{job.get('run_count', 0)}",
            payload_ref=change.job_id,
            updated_at=now.isoformat(),
            due_at=due.isoformat() if due else "",
        )
        event = CanonicalEvent(
            id=f"event:scheduler:{change.job_id}",
            entity_id=entity.id,
            source_id="scheduler",
            kind="scheduled_action",
            occurred_at=str(job.get("created_at") or now.isoformat()),
            due_at=due.isoformat() if due else "",
            active_from=due.isoformat() if due else "",
            expires_at=(
                ""
                if str(job.get("trigger") or "") == "every"
                else (due.isoformat() if due else "")
            ),
            urgency=self._urgency(now=now, due=due),
            confidence=1.0,
            delivery_semantics=DeliverySemantics.EXACT,
            dedupe_key=f"scheduler:{change.job_id}",
            source_version=entity.source_version,
            payload_ref=change.job_id,
            status=EventStatus.ACTIVE,
        )
        self._repository.upsert_entity(entity)
        self._repository.upsert_event(event)
        if change.action not in {"cancelled", "expired", "fired"}:
            return
        status = {
            "fired": EventStatus.COMPLETED,
            "cancelled": EventStatus.CANCELLED,
            "expired": EventStatus.EXPIRED,
        }[change.action]
        self._repository.upsert_entity(
            replace(
                entity,
                state=(
                    EntityState.COMPLETED
                    if status is EventStatus.COMPLETED
                    else EntityState.CANCELLED
                ),
            )
        )
        self._repository.close_event(event.id, status)
        self._notify_wake_change()

    def _close_linked_scheduled_event(
        self,
        event_id: str,
        *,
        status: EventStatus,
    ) -> None:
        event = self._repository.get_event(event_id)
        if event is None or event.status is not EventStatus.ACTIVE:
            return
        entity = self._repository.get_entity(event.entity_id)
        if entity is not None:
            self._repository.upsert_entity(
                replace(
                    entity,
                    state=(
                        EntityState.COMPLETED
                        if status is EventStatus.COMPLETED
                        else EntityState.CANCELLED
                    ),
                    updated_at=self._now().astimezone(timezone.utc).isoformat(),
                )
            )
        self._repository.close_event(event.id, status)
        self._notify_wake_change()

    def activate_wake(
        self,
        wake: WakePlan,
        *,
        now: datetime | None = None,
    ) -> AttentionSignal | None:
        event = self._repository.get_event(wake.event_id)
        if event is None or event.status is not EventStatus.ACTIVE:
            return None
        current = (now or self._now()).astimezone(timezone.utc)
        expires = parse_datetime(event.expires_at) or parse_datetime(event.due_at)
        if expires is not None and expires < current:
            self._repository.close_event(event.id, EventStatus.EXPIRED)
            return None
        signal_expires = expires or current + timedelta(hours=6)
        signal = AttentionSignal.create(
            signal_id=f"signal:{event.id}:{wake.attempt}",
            kind=f"event.{event.kind}",
            domain="personal",
            summary=self._event_title(event),
            source=SignalSource(
                type="canonical_event",
                name=event.source_id,
                reference=event.id,
            ),
            occurred_at=current,
            expires_at=signal_expires,
            severity=max(0.4, event.urgency),
            urgency=event.urgency,
            actionability=0.85,
            confidence=event.confidence,
            estimated_attention_minutes=5,
            suggested_capabilities=("message.notify",),
            evidence=({"event_id": event.id, "payload_ref": event.payload_ref},),
            metadata={
                "event_id": event.id,
                "entity_id": event.entity_id,
                "content": self._event_title(event),
                "delivery_semantics": event.delivery_semantics.value,
                "opportunity": {
                    "scene": "neutral",
                    "starts_at": current.isoformat(),
                    "ends_at": signal_expires.isoformat(),
                    "available_minutes": 5,
                },
            },
        )
        self._attention_engine.ingest_signal(signal)
        return signal

    def _schedule_exact(
        self,
        event: CanonicalEvent,
        entity: CanonicalEntity,
        channel: str,
        chat_id: str,
    ) -> None:
        due = parse_datetime(event.due_at)
        if self._scheduler is None or due is None:
            return
        job_id = exact_reminder_job_id(event.id)
        self._scheduler.add_job(
            ScheduledJob(
                id=job_id,
                trigger="at",
                tier="instant",
                fire_at=due,
                channel=channel,
                chat_id=chat_id,
                message=entity.title,
                name=f"小满提醒：{entity.title[:40]}",
                timezone="UTC",
                metadata={"canonical_event_id": event.id},
            )
        )

    def _cancel_exact_events_for_entity(self, entity_id: str) -> None:
        cancel = getattr(self._scheduler, "cancel_job", None)
        if not callable(cancel):
            return
        for event in self._repository.list_active_events():
            if (
                event.entity_id == entity_id
                and event.delivery_semantics is DeliverySemantics.EXACT
            ):
                cancel(exact_reminder_job_id(event.id))

    def _route_event(
        self,
        event: CanonicalEvent,
        entity: CanonicalEntity,
        *,
        channel: str,
        chat_id: str,
        now: datetime,
        due: datetime | None,
        active_from: datetime | None,
    ) -> None:
        channel = str(channel or self._default_channel).strip()
        chat_id = str(chat_id or self._default_chat_id).strip()
        if event.delivery_semantics is DeliverySemantics.EXACT:
            if channel and chat_id:
                self._schedule_exact(event, entity, channel, chat_id)
            return
        if event.delivery_semantics is DeliverySemantics.SILENT:
            return
        wake_at = self._plan_wake_at(
            semantics=event.delivery_semantics,
            now=now,
            due=due,
            active_from=active_from,
        )
        self._repository.upsert_wake(
            WakePlan(
                id=f"wake:{event.id}:0",
                event_id=event.id,
                wake_at=wake_at.isoformat(),
                reason="event_attention_review",
                attempt=0,
                max_attempts=3,
                status=WakeStatus.PENDING,
                last_decision="",
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
            )
        )
        self._notify_wake_change()

    def _notify_wake_change(self) -> None:
        if self._wake_notifier is not None:
            self._wake_notifier()

    def _event_title(self, event: CanonicalEvent) -> str:
        entity = self._repository.get_entity(event.entity_id)
        return entity.title if entity is not None else event.kind

    @staticmethod
    def _plan_wake_at(
        *,
        semantics: DeliverySemantics,
        now: datetime,
        due: datetime | None,
        active_from: datetime | None,
    ) -> datetime:
        if active_from is not None and active_from > now:
            return active_from
        if semantics is DeliverySemantics.BEFORE_DEADLINE and due is not None:
            remaining = max(timedelta(), due - now)
            lead_seconds = min(
                timedelta(hours=24).total_seconds(),
                max(
                    timedelta(minutes=15).total_seconds(),
                    remaining.total_seconds() * 0.25,
                ),
            )
            return max(now, due - timedelta(seconds=lead_seconds))
        return now

    @staticmethod
    def _urgency(*, now: datetime, due: datetime | None) -> float:
        if due is None or due <= now:
            return 0.8 if due is not None else 0.5
        hours = (due - now).total_seconds() / 3600
        return max(0.2, min(0.95, 1.0 - hours / (7 * 24)))

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _normalized_text(value: str) -> str:
        return "".join(character.lower() for character in value if character.isalnum())

    @staticmethod
    def _text_overlap(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        left_chars = set(left)
        right_chars = set(right)
        return len(left_chars & right_chars) / max(
            1, min(len(left_chars), len(right_chars))
        )


__all__ = ["EventDrivenAttentionService"]
