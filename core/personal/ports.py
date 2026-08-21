from __future__ import annotations

from typing import Any, Protocol, Sequence

from core.personal.models import (
    AccessPolicy,
    DataCategory,
    MemoryEvidence,
    PersonalEntityType,
    PersonalRecord,
    RecordRevision,
    RecordSource,
    RecordStatus,
    SensitivityLevel,
)


class PersonalRecordStore(Protocol):
    def create_record(
        self,
        *,
        entity_type: PersonalEntityType,
        title: str,
        summary: str,
        data: dict[str, Any],
        source: RecordSource,
        record_key: str = "",
        confidence: float = 1.0,
        sensitivity: SensitivityLevel = SensitivityLevel.PERSONAL,
        data_category: DataCategory = DataCategory.GENERAL,
        access_policy: AccessPolicy = AccessPolicy.STANDARD,
        valid_from: str | None = None,
        expires_at: str | None = None,
        user_locked: bool = False,
        allow_auto_update: bool = True,
        supersedes_id: str | None = None,
        actor: str = "user",
    ) -> PersonalRecord: ...

    def get_record(self, record_id: str) -> PersonalRecord | None: ...

    def find_active_by_key(
        self, entity_type: PersonalEntityType, record_key: str
    ) -> PersonalRecord | None: ...

    def list_records(
        self,
        *,
        entity_type: PersonalEntityType | None = None,
        statuses: Sequence[RecordStatus] | None = None,
        limit: int = 100,
    ) -> list[PersonalRecord]: ...

    def update_record(
        self,
        record_id: str,
        *,
        changes: dict[str, Any],
        actor: str,
        reason: str = "",
        automatic: bool = False,
    ) -> PersonalRecord: ...

    def confirm_record(
        self, record_id: str, *, actor: str = "user"
    ) -> PersonalRecord: ...

    def supersede_record(
        self,
        record_id: str,
        *,
        replacement: dict[str, Any],
        actor: str,
        reason: str = "",
    ) -> PersonalRecord: ...

    def forget_record(
        self,
        record_id: str,
        *,
        actor: str = "user",
        reason: str = "",
        purge_content: bool = True,
    ) -> PersonalRecord: ...

    def hard_delete_record(self, record_id: str) -> bool: ...

    def expire_due(
        self, *, actor: str = "system", now: str | None = None
    ) -> list[str]: ...

    def list_revisions(self, record_id: str) -> list[RecordRevision]: ...

    def list_lineage(self, record_id: str, *, limit: int = 100) -> list[PersonalRecord]: ...

    def add_memory_evidence(
        self,
        record_id: str,
        *,
        source: RecordSource,
        statement: str,
        confidence: float,
        observed_at: str | None = None,
    ) -> tuple[MemoryEvidence, bool]: ...

    def list_memory_evidence(
        self, record_id: str, *, limit: int = 100
    ) -> list[MemoryEvidence]: ...

    def close(self) -> None: ...
