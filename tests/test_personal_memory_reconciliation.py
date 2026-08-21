from __future__ import annotations

from pathlib import Path

import pytest

from core.personal.governance import (
    MemoryConflictAction,
    MemoryConflictStatus,
    MemoryGovernanceService,
)
from core.personal.memory_reconciliation import MemorySemanticRelation
from core.personal.models import (
    MemoryData,
    MemoryKind,
    PersonalEntityType,
    RecordSource,
    RecordStatus,
)
from core.personal.service import PersonalDataService
from infra.persistence.memory_governance_store import MemoryGovernanceStore
from infra.persistence.personal_store import PersonalStore


def _service(tmp_path: Path) -> MemoryGovernanceService:
    db_path = tmp_path / "personal.db"
    return MemoryGovernanceService(
        personal_data=PersonalDataService(PersonalStore(db_path)),
        conflict_store=MemoryGovernanceStore(db_path),
    )


def _close(service: MemoryGovernanceService) -> None:
    service.close()
    service.personal_data.close()


def _workout_fact(value: str, content: str, **attributes: object) -> MemoryData:
    return MemoryData(
        kind=MemoryKind.PREFERENCE,
        content=content,
        subject="用户",
        predicate="偏好锻炼时间",
        value=value,
        scope="日常",
        attributes=dict(attributes),
    )


def test_same_structured_fact_deduplicates_paraphrases_and_keeps_evidence(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    first = service.propose(
        memory=_workout_fact("上午", "用户偏好上午锻炼"),
        summary="用户偏好上午锻炼",
        source=RecordSource("user", "chat:1"),
        actor="user",
    )
    repeated = service.propose(
        memory=_workout_fact("上午", "锻炼最好安排在早上"),
        summary="锻炼最好安排在早上",
        source=RecordSource("conversation_semantic_batch", "batch:2"),
        confidence=0.93,
    )
    replayed = service.propose(
        memory=_workout_fact("上午", "锻炼最好安排在早上"),
        summary="锻炼最好安排在早上",
        source=RecordSource("conversation_semantic_batch", "batch:2"),
        confidence=0.93,
    )

    assert repeated.status == "unchanged"
    assert repeated.relation == MemorySemanticRelation.SAME
    assert repeated.record is not None
    assert repeated.record.id == first.record.id  # type: ignore[union-attr]
    assert replayed.status == "unchanged"
    assert len(service.list_memories()) == 1
    evidence = service.personal_data.memory_evidence(repeated.record.id)
    assert {item.source.source_ref for item in evidence} == {"chat:1", "batch:2"}
    assert len(evidence) == 2
    _close(service)


def test_structured_contradiction_is_quarantined_until_confirmed(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    original = service.propose(
        memory=_workout_fact("上午", "用户偏好上午锻炼"),
        summary="用户偏好上午锻炼",
        source=RecordSource("user", "chat:1"),
        actor="user",
    ).record
    proposal = service.propose(
        memory=_workout_fact("晚上", "用户可能更喜欢晚上锻炼"),
        summary="用户可能更喜欢晚上锻炼",
        source=RecordSource("assistant", "inference:2"),
        confidence=0.96,
    )

    assert proposal.status == "conflict_pending"
    assert proposal.relation == MemorySemanticRelation.CONTRADICT
    assert proposal.conflict is not None
    assert proposal.conflict.reason == "semantic_contradiction"
    current = service.list_memories()
    assert [item.id for item in current] == [original.id]  # type: ignore[union-attr]
    _close(service)


def test_user_correction_creates_temporal_version_edge(tmp_path: Path) -> None:
    service = _service(tmp_path)
    original = service.propose(
        memory=_workout_fact("上午", "用户偏好上午锻炼"),
        summary="用户偏好上午锻炼",
        source=RecordSource("user", "chat:1"),
        actor="user",
        valid_from="2026-01-01T00:00:00+08:00",
    ).record
    corrected = service.propose(
        memory=_workout_fact("晚上", "用户现在偏好晚上锻炼"),
        summary="用户现在偏好晚上锻炼",
        source=RecordSource("user", "chat:2"),
        actor="user",
        user_confirmed=True,
        valid_from="2026-07-20T12:00:00+08:00",
    )

    assert corrected.status == "superseded"
    assert corrected.record is not None
    assert corrected.record.supersedes_id == original.id  # type: ignore[union-attr]
    old = service.personal_data.get(original.id)  # type: ignore[union-attr]
    assert old is not None
    assert old.status == RecordStatus.SUPERSEDED
    assert old.valid_to == corrected.record.valid_from
    lineage = service.personal_data.lineage(corrected.record.id)
    assert [item.id for item in lineage] == [old.id, corrected.record.id]
    assert len(service.list_memories()) == 1

    graph = service.export_bundle()["graph"]
    assert {
        "from": corrected.record.id,
        "type": "supersedes",
        "to": old.id,
    } in graph["edges"]
    _close(service)


def test_unstructured_apple_facts_are_not_fuzzy_merged(tmp_path: Path) -> None:
    service = _service(tmp_path)
    phone = service.propose(
        memory=MemoryData(MemoryKind.PREFERENCE, "用户喜欢苹果手机"),
        summary="用户喜欢苹果手机",
        source=RecordSource("assistant", "batch:1"),
        confidence=0.9,
    )
    fruit = service.propose(
        memory=MemoryData(MemoryKind.PREFERENCE, "用户喜欢吃苹果"),
        summary="用户喜欢吃苹果",
        source=RecordSource("assistant", "batch:2"),
        confidence=0.9,
    )

    assert phone.status == fruit.status == "created"
    assert len(service.list_memories()) == 2
    assert service.conflict_store.list_conflicts() == []
    _close(service)


def test_low_confidence_candidate_never_becomes_active_memory(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    proposal = service.propose(
        memory=MemoryData(MemoryKind.FACT, "用户可能正在准备考试"),
        summary="用户可能正在准备考试",
        source=RecordSource("assistant", "batch:uncertain"),
        confidence=0.42,
    )

    assert proposal.status == "conflict_pending"
    assert proposal.conflict is not None
    assert proposal.conflict.reason == "low_confidence_candidate"
    assert service.list_memories() == []
    _close(service)


def test_invalid_model_validity_window_is_quarantined_instead_of_crashing(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    proposal = service.propose(
        memory=MemoryData(MemoryKind.TEMPORARY_STATE, "用户最近睡眠不足"),
        summary="用户最近睡眠不足",
        source=RecordSource("conversation_semantic_batch", "batch:bad-time"),
        confidence=0.9,
        expires_at="明天晚上",
    )

    assert proposal.status == "conflict_pending"
    assert proposal.conflict is not None
    assert proposal.conflict.reason == "invalid_expires_at"
    assert service.list_memories() == []
    _close(service)


def test_forget_purges_provenance_evidence(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.propose(
        memory=_workout_fact("上午", "用户偏好上午锻炼"),
        summary="用户偏好上午锻炼",
        source=RecordSource("user", "chat:sensitive"),
        actor="user",
    ).record
    assert record is not None
    assert service.personal_data.memory_evidence(record.id)

    service.forget(record.id, reason="user requested deletion")

    assert service.personal_data.memory_evidence(record.id) == []
    _close(service)


def test_forget_cancels_pending_conflicts_and_prevents_resurrection(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    record = service.propose(
        memory=_workout_fact("上午", "用户偏好上午锻炼"),
        summary="用户偏好上午锻炼",
        source=RecordSource("user", "chat:1"),
        actor="user",
    ).record
    assert record is not None
    conflict = service.propose(
        memory=_workout_fact("晚上", "用户可能偏好晚上锻炼"),
        summary="用户可能偏好晚上锻炼",
        source=RecordSource("assistant", "batch:2"),
        confidence=0.95,
    ).conflict
    assert conflict is not None

    service.forget(record.id, reason="user requested deletion")

    cancelled = service.conflict_store.get_conflict(conflict.id)
    assert cancelled is not None
    assert cancelled.status == MemoryConflictStatus.CANCELLED
    assert cancelled.candidate == {}
    with pytest.raises(ValueError, match="already resolved"):
        service.resolve(
            conflict.id,
            action=MemoryConflictAction.ACCEPT_CANDIDATE,
        )
    assert service.list_memories() == []
    _close(service)


def test_stale_conflict_cannot_replace_an_inactive_target(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.propose(
        memory=_workout_fact("上午", "用户偏好上午锻炼"),
        summary="用户偏好上午锻炼",
        source=RecordSource("user", "chat:1"),
        actor="user",
    ).record
    assert record is not None
    conflict = service.propose(
        memory=_workout_fact("晚上", "用户可能偏好晚上锻炼"),
        summary="用户可能偏好晚上锻炼",
        source=RecordSource("assistant", "batch:2"),
        confidence=0.95,
    ).conflict
    assert conflict is not None
    service.personal_data.forget(record.id, reason="simulated interrupted cleanup")

    with pytest.raises(ValueError, match="target is no longer active"):
        service.resolve(
            conflict.id,
            action=MemoryConflictAction.ACCEPT_CANDIDATE,
        )

    cancelled = service.conflict_store.get_conflict(conflict.id)
    assert cancelled is not None
    assert cancelled.status == MemoryConflictStatus.CANCELLED
    assert cancelled.candidate == {}
    assert service.list_memories() == []
    _close(service)


def test_supersede_transaction_rolls_back_when_new_version_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    original = service.propose(
        memory=_workout_fact("上午", "用户偏好上午锻炼"),
        summary="用户偏好上午锻炼",
        source=RecordSource("user", "chat:1"),
        actor="user",
    ).record
    store = service.personal_data.store

    def fail_insert(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated insert failure")

    monkeypatch.setattr(store, "_insert_record", fail_insert)
    with pytest.raises(RuntimeError, match="simulated insert failure"):
        service.propose(
            memory=_workout_fact("晚上", "用户偏好晚上锻炼"),
            summary="用户偏好晚上锻炼",
            source=RecordSource("user", "chat:2"),
            actor="user",
        )

    current = store.find_active_by_key(
        PersonalEntityType.MEMORY, original.record_key  # type: ignore[union-attr]
    )
    assert current is not None
    assert current.id == original.id  # type: ignore[union-attr]
    assert current.valid_to is None
    _close(service)
