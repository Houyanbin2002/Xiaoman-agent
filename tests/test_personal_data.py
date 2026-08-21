from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from core.personal.models import (
    CommitmentData,
    MemoryData,
    MemoryKind,
    PersonalEntityType,
    PersonalProfileData,
    RecordSource,
    RecordStatus,
    SensitivityLevel,
)
from core.personal.service import PersonalDataService
from infra.persistence.personal_store import PersonalStore


def _service(tmp_path: Path) -> PersonalDataService:
    return PersonalDataService(PersonalStore(tmp_path / "personal.db"))


def test_existing_personal_database_migrates_before_governance_index(tmp_path: Path):
    db_path = tmp_path / "personal.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE personal_records (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                record_key TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                data_json TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL,
                source_ref TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                sensitivity TEXT NOT NULL DEFAULT 'personal',
                status TEXT NOT NULL DEFAULT 'active',
                valid_from TEXT,
                expires_at TEXT,
                last_confirmed_at TEXT,
                user_locked INTEGER NOT NULL DEFAULT 0,
                allow_auto_update INTEGER NOT NULL DEFAULT 1,
                supersedes_id TEXT,
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                forgotten_at TEXT
            )
            """
        )

    store = PersonalStore(db_path)
    columns = {
        str(row["name"])
        for row in store._db.execute("PRAGMA table_info(personal_records)").fetchall()
    }
    indexes = {
        str(row["name"])
        for row in store._db.execute("PRAGMA index_list(personal_records)").fetchall()
    }
    assert {"data_category", "access_policy", "valid_to"}.issubset(columns)
    assert "idx_personal_records_governance" in indexes
    store.close()


def test_profile_and_typed_personal_records_share_governed_envelope(tmp_path: Path):
    service = _service(tmp_path)
    profile = service.upsert_owner_profile(
        PersonalProfileData(display_name="Xiaoman Owner", preferences={"tea": "green"}),
        source=RecordSource("dashboard", "profile-form"),
    )
    commitment = service.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="Finish architecture review",
        summary="Review the personal assistant boundaries",
        data=CommitmentData(title="Finish architecture review", priority="high"),
        source=RecordSource("chat", "dashboard:test:42"),
        confidence=0.9,
    )

    assert profile.record_key == "owner"
    assert profile.user_locked is True
    assert profile.allow_auto_update is False
    assert commitment.data["priority"] == "high"
    assert commitment.source.source_ref == "dashboard:test:42"
    assert {item.entity_type for item in service.list(limit=10)} == {
        PersonalEntityType.PROFILE,
        PersonalEntityType.COMMITMENT,
    }
    service.close()


def test_commitment_writes_share_one_canonical_task_shape(tmp_path: Path):
    service = _service(tmp_path)
    record = service.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="整理三项最重要的工作",
        summary="",
        data={
            "action": "整理三项最重要的工作",
            "deadline": "2026-07-15 上午",
            "estimated_minutes": "15",
        },
        source=RecordSource("dashboard", "workflow:test"),
    )

    assert record.summary == "整理三项最重要的工作"
    assert record.data == {
        "action": "整理三项最重要的工作",
        "deadline": "2026-07-15 上午",
        "estimated_minutes": 15,
        "state": "open",
        "priority": "normal",
        "energy": "medium",
        "progress": 0.0,
        "contexts": ["any"],
        "next_action": "整理三项最重要的工作",
        "due_text": "2026-07-15 上午",
        "due_date": "2026-07-15",
        "due_period": "morning",
    }
    service.close()


def test_model_task_aliases_are_normalized_for_rhythm_consumers(tmp_path: Path):
    service = _service(tmp_path)
    record = service.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="明天上午整理工作",
        summary="明天（2026-07-15）上午整理三项工作，预计 15 分钟",
        data={
            "state": "pending",
            "next_action": "整理三项工作",
            "priority": "medium",
            "estimated_minutes": 15,
            "energy": "normal",
            "contexts": ["planning"],
            "progress": "pending",
        },
        source=RecordSource("assistant", "workflow:test"),
    )

    assert record.data["state"] == "open"
    assert record.data["priority"] == "normal"
    assert record.data["energy"] == "medium"
    assert record.data["contexts"] == ["any"]
    assert record.data["progress"] == 0.0
    assert record.data["due_date"] == "2026-07-15"
    assert record.data["due_period"] == "morning"
    assert record.data["due_text"] == "2026-07-15 上午"
    service.close()


def test_partial_task_update_preserves_existing_task_context(tmp_path: Path):
    service = _service(tmp_path)
    record = service.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="Finish report",
        summary="Finish the weekly report",
        data={
            "state": "open",
            "estimated_minutes": 20,
            "next_action": "Write the conclusion",
            "contexts": ["home"],
        },
        source=RecordSource("assistant", "chat:test"),
    )

    completed = service.update(
        record.id,
        {"data": {"state": "completed", "progress": 1}},
        actor="assistant",
        automatic=True,
    )

    assert completed.data["state"] == "completed"
    assert completed.data["progress"] == 1.0
    assert completed.data["estimated_minutes"] == 20
    assert completed.data["next_action"] == "Write the conclusion"
    assert completed.data["contexts"] == ["home"]
    service.close()


def test_automatic_update_respects_user_lock_and_confirmation(tmp_path: Path):
    service = _service(tmp_path)
    record = service.remember(
        MemoryData(MemoryKind.PREFERENCE, "Prefers morning workouts"),
        summary="Morning workout preference",
        source=RecordSource("assistant", "chat:1"),
        confidence=0.6,
    )
    locked = service.update(
        record.id,
        {"user_locked": True},
        actor="user",
        reason="Keep this preference stable",
    )

    with pytest.raises(PermissionError, match="explicit user update"):
        service.update(
            locked.id,
            {"summary": "Changed automatically"},
            actor="assistant",
            automatic=True,
        )

    confirmed = service.confirm(locked.id)
    assert confirmed.confidence == 1.0
    assert confirmed.last_confirmed_at is not None
    assert [revision.action for revision in service.history(record.id)] == [
        "created",
        "updated",
        "confirmed",
    ]
    service.close()


def test_supersede_preserves_lineage_and_forget_redacts_history(tmp_path: Path):
    service = _service(tmp_path)
    original = service.remember(
        MemoryData(MemoryKind.FACT, "Old home address"),
        summary="Home address",
        source=RecordSource("user", "chat:address"),
        sensitivity=SensitivityLevel.RESTRICTED,
    )
    replacement = service.supersede(
        original.id,
        replacement={
            "data": MemoryData(MemoryKind.FACT, "New home address"),
            "source": "user",
            "source_ref": "chat:new-address",
        },
        actor="user",
        reason="Moved home",
    )

    old = service.store.get_record(original.id)
    assert old.status == RecordStatus.SUPERSEDED  # type: ignore[union-attr]
    assert old.valid_to is not None  # type: ignore[union-attr]
    assert old.valid_to == replacement.valid_from  # type: ignore[union-attr]
    assert replacement.supersedes_id == original.id
    forgotten = service.forget(replacement.id, reason="Remove address")
    assert forgotten.status == RecordStatus.FORGOTTEN
    assert forgotten.valid_to is not None
    assert forgotten.data == {}
    assert all(
        revision.snapshot == {"redacted": True}
        or revision.action == "forgotten"
        for revision in service.history(replacement.id)
    )
    service.close()


def test_expire_due_only_transitions_elapsed_records(tmp_path: Path):
    service = _service(tmp_path)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    expired = service.remember(
        MemoryData(MemoryKind.EPISODE, "Temporary fatigue"),
        summary="Temporary fatigue",
        source=RecordSource("check_in", "today"),
        expires_at=past,
    )
    active = service.remember(
        MemoryData(MemoryKind.PREFERENCE, "Likes concise reminders"),
        summary="Reminder style",
        source=RecordSource("user", "settings"),
        expires_at=future,
    )

    assert service.expire_due() == [expired.id]
    expired_record = service.store.get_record(expired.id)
    assert expired_record.status == RecordStatus.EXPIRED  # type: ignore[union-attr]
    assert expired_record.valid_to is not None  # type: ignore[union-attr]
    assert service.expire_due() == []
    assert service.store.get_record(active.id).status == RecordStatus.ACTIVE  # type: ignore[union-attr]
    service.close()


def test_expiry_is_normalized_to_utc_before_comparison(tmp_path: Path):
    service = _service(tmp_path)
    record = service.remember(
        MemoryData(MemoryKind.EPISODE, "Short-lived context"),
        summary="Short-lived context",
        source=RecordSource("check_in", "timezone-test"),
        expires_at="2020-01-01T08:00:00+08:00",
    )

    assert record.expires_at == "2020-01-01T00:00:00+00:00"
    assert service.expire_due() == [record.id]
    service.close()
