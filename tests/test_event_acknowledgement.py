from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from core.attention.events.acknowledgement import EventAcknowledgementService
from core.attention.events.service import EventDrivenAttentionService
from agent.scheduler import ScheduledJobChanged
from core.attention.feedback.service import FeedbackService
from core.personal.events import PersonalRecordChanged
from core.personal.models import PersonalEntityType, RecordSource
from core.personal.service import PersonalDataService
from infra.persistence.attention_engine_store import AttentionEngineStore
from infra.persistence.personal_store import PersonalStore


async def test_user_completion_closes_event_wakes_and_updates_projection(tmp_path) -> None:
    database = tmp_path / "personal.db"
    store = AttentionEngineStore(database)
    personal = PersonalDataService(PersonalStore(database))
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    events = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        now_fn=lambda: now,
    )
    record = personal.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="提交组会汇报",
        summary="下午前完成",
        data={
            "title": "提交组会汇报",
            "due_at": (now + timedelta(hours=4)).isoformat(),
        },
        source=RecordSource("notion", "page-42"),
        actor="external-sync",
    )
    await events.handle_personal_record_changed(
        PersonalRecordChanged(record=record, change="created", actor="external-sync")
    )
    event = store.list_active_events()[0]
    acknowledgements = EventAcknowledgementService(
        repository=store,
        personal_data=personal,
        feedback=FeedbackService(store),
        scheduler=SimpleNamespace(cancel_job=lambda job_id: True),
    )

    result = acknowledgements.complete(event.id)

    assert result["status"] == "completed"
    assert store.list_active_events() == []
    assert store.list_pending_wakes() == []
    updated = personal.get(record.id)
    assert updated is not None
    assert updated.data["state"] == "completed"
    assert updated.data["source_sync"] == "pending"
    entity = store.get_entity(event.entity_id)
    assert entity is not None
    assert entity.local_override["authority"] == "user"

    await events.handle_personal_record_changed(
        PersonalRecordChanged(record=updated, change="updated", actor="user")
    )
    entity = store.get_entity(event.entity_id)
    assert entity is not None
    assert entity.local_override["source_sync"] == "pending"

    stale = personal.update(
        record.id,
        {"data": {"state": "open"}},
        actor="external-sync",
        reason="stale external projection",
        automatic=True,
    )
    await events.handle_personal_record_changed(
        PersonalRecordChanged(record=stale, change="updated", actor="external-sync")
    )
    entity = store.get_entity(event.entity_id)
    assert entity is not None
    assert entity.state.value == "completed"
    assert entity.local_override["source_sync"] == "pending"

    confirmed = personal.update(
        record.id,
        {"data": {"state": "completed"}},
        actor="external-sync",
        reason="external source confirmed completion",
        automatic=True,
    )
    await events.handle_personal_record_changed(
        PersonalRecordChanged(
            record=confirmed,
            change="updated",
            actor="external-sync",
        )
    )
    entity = store.get_entity(event.entity_id)
    assert entity is not None
    assert entity.local_override == {}
    personal.close()
    store.close()


async def test_completing_scheduler_event_cancels_original_job(tmp_path) -> None:
    database = tmp_path / "personal.db"
    store = AttentionEngineStore(database)
    personal = PersonalDataService(PersonalStore(database))
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    events = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        now_fn=lambda: now,
    )
    await events.handle_scheduled_job_changed(
        ScheduledJobChanged(
            action="upserted",
            job_id="job-42",
            job={
                "name": "提醒汇报",
                "trigger": "at",
                "fire_at": (now + timedelta(hours=4)).isoformat(),
                "created_at": now.isoformat(),
                "run_count": 0,
                "metadata": {},
            },
        )
    )
    cancelled: list[str] = []
    acknowledgements = EventAcknowledgementService(
        repository=store,
        personal_data=personal,
        feedback=FeedbackService(store),
        scheduler=SimpleNamespace(
            cancel_job=lambda job_id: cancelled.append(job_id) or True
        ),
    )

    acknowledgements.complete("event:scheduler:job-42")

    assert cancelled == ["job-42"]
    assert store.list_active_events() == []
    personal.close()
    store.close()
