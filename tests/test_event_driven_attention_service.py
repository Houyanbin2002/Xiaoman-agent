from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent.scheduler import ScheduledJobChanged
from core.attention.events.service import EventDrivenAttentionService
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import SemanticBatchPayload
from core.personal.events import PersonalRecordChanged
from core.personal.models import PersonalEntityType, RecordSource
from core.personal.service import PersonalDataService
from infra.persistence.attention_engine_store import AttentionEngineStore
from infra.persistence.personal_store import PersonalStore


class _Scheduler:
    def __init__(self) -> None:
        self.jobs = []
        self.cancelled: list[str] = []

    def add_job(self, job) -> None:
        self.jobs = [item for item in self.jobs if item.id != job.id]
        self.jobs.append(job)

    def cancel_job(self, job_id: str) -> bool:
        before = len(self.jobs)
        self.jobs = [item for item in self.jobs if item.id != job_id]
        self.cancelled.append(job_id)
        return len(self.jobs) != before


async def test_semantic_tasks_use_scheduler_for_exact_and_wake_for_soft(
    tmp_path,
) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    scheduler = _Scheduler()
    ingested = []
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=ingested.append),
        scheduler=scheduler,
        now_fn=lambda: now,
    )
    event = ConversationSemanticBatchCommitted(
        batch_id="batch-1",
        session_key="qq:42",
        channel="qq",
        chat_id="42",
        analysis_version="conversation-v1",
        message_ids=("m1", "m2"),
        user_message_ids=("m1", "m2"),
        end_seq=1,
        context_consolidate_through=-1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "task_events": [
                    {
                        "summary": "下午三点提醒我汇报",
                        "delivery_semantics": "exact",
                        "confidence": 0.99,
                        "due_at": (now + timedelta(hours=6)).isoformat(),
                        "source_message_id": "m1",
                    },
                    {
                        "summary": "周五前提交报告",
                        "delivery_semantics": "before_deadline",
                        "confidence": 0.92,
                        "due_at": (now + timedelta(days=2)).isoformat(),
                        "source_message_id": "m2",
                    },
                ]
            }
        ),
    )

    await service.handle_semantic_batch(event)

    assert len(scheduler.jobs) == 1
    assert scheduler.jobs[0].tier == "instant"
    assert scheduler.jobs[0].message == "下午三点提醒我汇报"
    assert len(store.list_active_events()) == 2
    wakes = store.list_pending_wakes()
    assert len(wakes) == 1
    assert wakes[0].event_id.endswith("m2")

    signal = service.activate_wake(wakes[0], now=now + timedelta(days=1))

    assert signal is not None
    assert signal.metadata["event_id"] == wakes[0].event_id
    assert ingested == [signal]
    store.close()


async def test_personal_exact_event_uses_configured_default_target(tmp_path) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    scheduler = _Scheduler()
    personal = PersonalDataService(PersonalStore(tmp_path / "personal.db"))
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        scheduler=scheduler,
        now_fn=lambda: now,
        default_channel="qqbot",
        default_chat_id="c2c:owner",
    )
    record = personal.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="下午三点汇报",
        summary="准时提醒",
        data={
            "title": "下午三点汇报",
            "due_at": (now + timedelta(hours=6)).isoformat(),
            "delivery_semantics": "exact",
        },
        source=RecordSource("notion", "page-exact"),
        actor="external-sync",
    )

    await service.handle_personal_record_changed(
        PersonalRecordChanged(record=record, change="created", actor="external-sync")
    )

    assert len(scheduler.jobs) == 1
    assert scheduler.jobs[0].channel == "qqbot"
    assert scheduler.jobs[0].chat_id == "c2c:owner"

    completed = personal.update(
        record.id,
        {"data": {"state": "completed"}},
        actor="user",
        reason="done",
    )
    await service.handle_personal_record_changed(
        PersonalRecordChanged(record=completed, change="updated", actor="user")
    )
    assert scheduler.jobs == []
    assert len(scheduler.cancelled) == 1
    personal.close()
    store.close()


async def test_ordinary_personal_memory_is_not_projected_to_attention(tmp_path) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    personal = PersonalDataService(PersonalStore(tmp_path / "personal.db"))
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
    )
    record = personal.create(
        entity_type=PersonalEntityType.MEMORY,
        title="回答工具数量时给出精确数字",
        summary="Agent 执行经验，不是待提醒事项",
        data={"kind": "preference"},
        source=RecordSource("conversation_semantic_batch", "batch-memory"),
        actor="memory",
    )

    await service.handle_personal_record_changed(
        PersonalRecordChanged(record=record, change="created", actor="memory")
    )

    assert store.list_active_events() == []
    assert store.list_entities() == []
    personal.close()
    store.close()


async def test_replaying_semantic_batch_does_not_duplicate_jobs_or_wakes(
    tmp_path,
) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    scheduler = _Scheduler()
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        scheduler=scheduler,
        now_fn=lambda: now,
    )
    event = ConversationSemanticBatchCommitted(
        batch_id="batch-same",
        session_key="web:1",
        channel="web",
        chat_id="1",
        analysis_version="conversation-v1",
        message_ids=("m1",),
        user_message_ids=("m1",),
        end_seq=0,
        context_consolidate_through=-1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "task_events": [
                    {
                        "summary": "明天前完成总结",
                        "delivery_semantics": "before_deadline",
                        "confidence": 0.9,
                        "due_at": (now + timedelta(days=1)).isoformat(),
                        "source_message_id": "m1",
                    }
                ]
            }
        ),
    )

    await service.handle_semantic_batch(event)
    await service.handle_semantic_batch(event)

    assert len(store.list_active_events()) == 1
    assert len(store.list_pending_wakes()) == 1
    assert scheduler.jobs == []
    store.close()


async def test_completed_personal_record_closes_event_and_cancels_wake(
    tmp_path,
) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    personal = PersonalDataService(PersonalStore(tmp_path / "personal.db"))
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        now_fn=lambda: now,
    )
    record = personal.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="提交周报",
        summary="今天提交",
        data={
            "title": "提交周报",
            "due_at": (now + timedelta(hours=8)).isoformat(),
        },
        source=RecordSource("notion", "page-1"),
        actor="external-sync",
    )
    await service.handle_personal_record_changed(
        PersonalRecordChanged(record=record, change="created", actor="external-sync")
    )
    assert len(store.list_pending_wakes()) == 1

    completed = personal.update(
        record.id,
        {"data": {"state": "completed"}},
        actor="user",
        reason="done",
    )
    await service.handle_personal_record_changed(
        PersonalRecordChanged(record=completed, change="updated", actor="user")
    )

    assert store.list_active_events() == []
    assert store.list_pending_wakes() == []
    personal.close()
    store.close()


async def test_semantic_completion_closes_matching_conversation_event(tmp_path) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        now_fn=lambda: now,
    )
    created = ConversationSemanticBatchCommitted(
        batch_id="batch-create",
        session_key="qq:42",
        channel="qq",
        chat_id="42",
        analysis_version="conversation-v1",
        message_ids=("m1",),
        user_message_ids=("m1",),
        end_seq=0,
        context_consolidate_through=-1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "task_events": [
                    {
                        "summary": "给组长提交周报",
                        "operation": "upsert",
                        "delivery_semantics": "before_deadline",
                        "confidence": 0.95,
                        "due_at": (now + timedelta(hours=8)).isoformat(),
                        "source_message_id": "m1",
                    }
                ]
            }
        ),
    )
    await service.handle_semantic_batch(created)
    assert len(store.list_active_events()) == 1
    assert len(store.list_pending_wakes()) == 1

    completed = ConversationSemanticBatchCommitted(
        batch_id="batch-complete",
        session_key="qq:42",
        channel="qq",
        chat_id="42",
        analysis_version="conversation-v1",
        message_ids=("m2",),
        user_message_ids=("m2",),
        end_seq=2,
        context_consolidate_through=-1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "task_events": [
                    {
                        "summary": "周报已经提交完成",
                        "related_summary": "给组长提交周报",
                        "operation": "complete",
                        "delivery_semantics": "silent",
                        "confidence": 0.99,
                        "source_message_id": "m2",
                    }
                ]
            }
        ),
    )
    await service.handle_semantic_batch(completed)

    assert store.list_active_events() == []
    assert store.list_pending_wakes() == []
    store.close()


async def test_scheduler_lifecycle_is_projected_without_duplicate_linked_event(
    tmp_path,
) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    now = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
    service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        now_fn=lambda: now,
    )
    job = {
        "name": "下午三点汇报",
        "trigger": "at",
        "tier": "instant",
        "fire_at": (now + timedelta(hours=6)).isoformat(),
        "created_at": now.isoformat(),
        "run_count": 0,
        "metadata": {},
    }
    await service.handle_scheduled_job_changed(
        ScheduledJobChanged(action="upserted", job_id="job-1", job=job)
    )
    assert [event.id for event in store.list_active_events()] == [
        "event:scheduler:job-1"
    ]

    await service.handle_scheduled_job_changed(
        ScheduledJobChanged(action="fired", job_id="job-1", job=job)
    )
    assert store.list_active_events() == []

    semantic = ConversationSemanticBatchCommitted(
        batch_id="batch-linked",
        session_key="qq:42",
        channel="qq",
        chat_id="42",
        analysis_version="conversation-v1",
        message_ids=("m-exact",),
        user_message_ids=("m-exact",),
        end_seq=3,
        context_consolidate_through=-1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "task_events": [
                    {
                        "summary": "晚上九点提醒休息",
                        "delivery_semantics": "exact",
                        "confidence": 1.0,
                        "due_at": (now + timedelta(hours=12)).isoformat(),
                        "source_message_id": "m-exact",
                    }
                ]
            }
        ),
    )
    linked_service = EventDrivenAttentionService(
        repository=store,
        attention_engine=SimpleNamespace(ingest_signal=lambda signal: signal),
        scheduler=_Scheduler(),
        now_fn=lambda: now,
    )
    await linked_service.handle_semantic_batch(semantic)
    linked = store.list_active_events()[0]
    linked_job = {
        **job,
        "metadata": {"canonical_event_id": linked.id},
    }
    await service.handle_scheduled_job_changed(
        ScheduledJobChanged(
            action="fired",
            job_id="attention-exact:linked",
            job=linked_job,
        )
    )
    assert store.list_active_events() == []
    assert all(
        entity.id != "entity:scheduler:attention-exact:linked"
        for entity in store.list_entities()
    )
    store.close()
