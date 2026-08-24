from __future__ import annotations

from core.personal.events import PersonalRecordChanged
from core.personal.models import PersonalEntityType, RecordSource
from core.personal.service import PersonalDataService
from infra.persistence.personal_store import PersonalStore


def test_personal_data_service_publishes_structured_record_changes(tmp_path) -> None:
    published: list[PersonalRecordChanged] = []
    service = PersonalDataService(
        PersonalStore(tmp_path / "personal.db"),
        event_publisher=published.append,
    )

    created = service.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="提交报告",
        summary="周五前提交",
        data={"title": "提交报告", "due_at": "2026-07-18T09:00:00+00:00"},
        source=RecordSource("chat", "message-1"),
        actor="user",
    )
    service.update(
        created.id,
        {"data": {"state": "completed"}},
        actor="user",
        reason="done",
    )

    assert [event.change for event in published] == ["created", "updated"]
    assert published[0].record.id == created.id
    assert published[1].record.data["state"] == "completed"
    service.close()
