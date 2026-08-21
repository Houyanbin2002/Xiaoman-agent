from __future__ import annotations

from pathlib import Path

import pytest

from core.personal.governance import (
    MemoryConflictAction,
    MemoryConflictStatus,
    MemoryGovernanceService,
)
from core.personal.models import (
    AccessPolicy,
    DataCategory,
    MemoryData,
    MemoryKind,
    PersonalEntityType,
    RecordSource,
    RecordStatus,
    SensitivityLevel,
)
from core.personal.service import PersonalDataService
from infra.persistence.memory_governance_store import MemoryGovernanceStore
from infra.persistence.personal_store import PersonalStore


def _governance(tmp_path: Path) -> MemoryGovernanceService:
    db_path = tmp_path / "personal.db"
    data = PersonalDataService(PersonalStore(db_path))
    return MemoryGovernanceService(
        personal_data=data,
        conflict_store=MemoryGovernanceStore(db_path),
    )


def _close(service: MemoryGovernanceService) -> None:
    service.close()
    service.personal_data.close()


def test_memory_kinds_and_sensitive_defaults_are_explicit(tmp_path: Path):
    service = _governance(tmp_path)
    result = service.propose(
        memory=MemoryData(
            MemoryKind.TEMPORARY_STATE,
            "Sleep deprived today",
            category=DataCategory.HEALTH,
        ),
        summary="Temporary sleep state",
        record_key="health:sleep-state",
        source=RecordSource("user", "chat:1"),
        actor="user",
    )

    assert result.record is not None
    assert result.record.data["kind"] == "temporary_state"
    assert result.record.data_category == DataCategory.HEALTH
    assert result.record.sensitivity == SensitivityLevel.SENSITIVE
    assert result.record.access_policy == AccessPolicy.CONFIRM_WRITE
    _close(service)


def test_assistant_conflict_never_overwrites_active_memory(tmp_path: Path):
    service = _governance(tmp_path)
    first = service.propose(
        memory=MemoryData(MemoryKind.PREFERENCE, "Prefers morning workouts"),
        summary="Workout time preference",
        record_key="preference:workout-time",
        source=RecordSource("user", "chat:1"),
        actor="user",
    )
    proposed = service.propose(
        memory=MemoryData(MemoryKind.PREFERENCE, "Prefers evening workouts"),
        summary="Workout time preference",
        record_key="preference:workout-time",
        source=RecordSource("assistant", "chat:2"),
    )

    assert proposed.status == "conflict_pending"
    assert proposed.conflict is not None
    assert proposed.conflict.reason == "semantic_contradiction"
    current = service.personal_data.store.find_active_by_key(
        PersonalEntityType.MEMORY, "preference:workout-time"
    )
    assert current is not None
    assert current.id == first.record.id  # type: ignore[union-attr]
    assert current.data["content"] == "Prefers morning workouts"
    _close(service)


def test_sensitive_assistant_memory_waits_for_authorization(tmp_path: Path):
    service = _governance(tmp_path)
    proposal = service.propose(
        memory=MemoryData(
            MemoryKind.FACT,
            "Account recovery answer",
            category=DataCategory.ACCOUNT,
        ),
        summary="Account recovery detail",
        record_key="account:recovery",
        source=RecordSource("assistant", "chat:secure"),
    )

    assert proposal.status == "conflict_pending"
    assert proposal.conflict is not None
    assert proposal.conflict.reason == "authorization_required"
    assert service.list_memories() == []
    accepted = service.resolve(
        proposal.conflict.id,
        action=MemoryConflictAction.ACCEPT_CANDIDATE,
        note="User explicitly approved",
    )
    assert accepted.record is not None
    assert accepted.record.access_policy == AccessPolicy.OWNER_ONLY
    assert accepted.conflict.candidate == {}  # type: ignore[union-attr]
    _close(service)


def test_conflict_can_keep_existing_or_merge(tmp_path: Path):
    service = _governance(tmp_path)
    original = service.propose(
        memory=MemoryData(MemoryKind.FACT, "Uses metric units"),
        summary="Preferred measurement system",
        record_key="preference:units",
        source=RecordSource("user", "settings"),
        actor="user",
    )
    conflict = service.propose(
        memory=MemoryData(MemoryKind.FACT, "Sometimes uses imperial units"),
        summary="Preferred measurement system",
        record_key="preference:units",
        source=RecordSource("assistant", "chat:3"),
    ).conflict
    assert conflict is not None
    kept = service.resolve(
        conflict.id, action=MemoryConflictAction.KEEP_EXISTING
    )
    assert kept.status == "kept_existing"
    assert kept.record.id == original.record.id  # type: ignore[union-attr]
    assert kept.conflict.status == MemoryConflictStatus.KEPT_EXISTING  # type: ignore[union-attr]

    second = service.propose(
        memory=MemoryData(MemoryKind.FACT, "Uses metric, except body height"),
        summary="Preferred measurement system",
        record_key="preference:units",
        source=RecordSource("assistant", "chat:4"),
    ).conflict
    assert second is not None
    merged = service.resolve(
        second.id,
        action=MemoryConflictAction.MERGE,
        merged={"content": "Uses metric units, but records body height in feet"},
    )
    assert merged.record is not None
    assert merged.record.data["content"].endswith("in feet")
    assert service.personal_data.store.get_record(original.record.id).status == RecordStatus.SUPERSEDED  # type: ignore[union-attr]
    _close(service)


def test_export_and_hard_delete_remove_content_and_history(tmp_path: Path):
    service = _governance(tmp_path)
    record = service.propose(
        memory=MemoryData(MemoryKind.FACT, "Private fact"),
        summary="Private fact",
        record_key="fact:private",
        source=RecordSource("user", "chat:5"),
        actor="user",
    ).record
    assert record is not None
    service.personal_data.update(
        record.id, {"user_locked": True}, actor="user", reason="lock"
    )
    bundle = service.export_bundle()
    assert bundle["format"] == "xiaoman-memory-governance-v1"
    assert bundle["records"][0]["source_ref"] == "chat:5"
    assert len(bundle["revisions"][record.id]) == 2

    assert service.hard_delete(record.id) is True
    assert service.personal_data.store.get_record(record.id) is None
    with pytest.raises(ValueError, match="not found"):
        service.personal_data.history(record.id)
    _close(service)
