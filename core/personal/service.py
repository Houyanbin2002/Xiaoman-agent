from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from core.personal.models import (
    AccessPolicy,
    DataCategory,
    MemoryData,
    MemoryEvidence,
    PersonalEntityType,
    PersonalProfileData,
    PersonalRecord,
    RecordRevision,
    RecordSource,
    RecordStatus,
    SensitivityLevel,
    normalize_payload,
)
from core.personal.events import PersonalRecordChanged
from core.personal.normalization import normalize_personal_payload
from core.personal.ports import PersonalRecordStore


@dataclass
class PersonalDataService:
    store: PersonalRecordStore
    event_publisher: Callable[[PersonalRecordChanged], None] | None = None

    def create(
        self,
        *,
        entity_type: PersonalEntityType,
        title: str,
        summary: str,
        data: Any,
        source: RecordSource,
        record_key: str = "",
        confidence: float = 1.0,
        sensitivity: SensitivityLevel | None = None,
        data_category: DataCategory | None = None,
        access_policy: AccessPolicy | None = None,
        valid_from: str | None = None,
        expires_at: str | None = None,
        user_locked: bool = False,
        allow_auto_update: bool = True,
        actor: str = "user",
    ) -> PersonalRecord:
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("title must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        category = data_category or self._default_category(entity_type, data)
        resolved_sensitivity, resolved_policy = self.governance_defaults(
            category,
            sensitivity=sensitivity,
            access_policy=access_policy,
        )
        normalized_data = normalize_personal_payload(
            entity_type,
            data,
            title=normalized_title,
            summary=summary,
        )
        normalized_summary = summary.strip()
        if not normalized_summary and isinstance(normalized_data, dict):
            normalized_summary = str(
                normalized_data.get("content")
                or normalized_data.get("next_action")
                or normalized_title
            ).strip()
        record = self.store.create_record(
            entity_type=entity_type,
            record_key=record_key.strip(),
            title=normalized_title,
            summary=normalized_summary,
            data=normalized_data,
            source=source,
            confidence=confidence,
            sensitivity=resolved_sensitivity,
            data_category=category,
            access_policy=resolved_policy,
            valid_from=valid_from,
            expires_at=expires_at,
            user_locked=user_locked,
            allow_auto_update=allow_auto_update,
            actor=actor,
        )
        self._publish(record, change="created", actor=actor)
        return record

    def upsert_owner_profile(
        self,
        profile: PersonalProfileData,
        *,
        source: RecordSource,
        actor: str = "user",
    ) -> PersonalRecord:
        existing = self.store.find_active_by_key(
            PersonalEntityType.PROFILE, "owner"
        )
        data = normalize_payload(profile)
        if existing is None:
            return self.create(
                entity_type=PersonalEntityType.PROFILE,
                record_key="owner",
                title=profile.display_name or "Owner profile",
                summary="Primary personal assistant profile",
                data=data,
                source=source,
                user_locked=True,
                allow_auto_update=False,
                actor=actor,
            )
        record = self.store.update_record(
            existing.id,
            changes={
                "title": profile.display_name or existing.title,
                "data": data,
                "source": source.source,
                "source_ref": source.source_ref,
            },
            actor=actor,
            reason="owner profile updated",
        )
        self._publish(record, change="updated", actor=actor)
        return record

    def upsert_external(
        self,
        *,
        entity_type: PersonalEntityType,
        record_key: str,
        title: str,
        summary: str,
        data: Any,
        source: RecordSource,
        confidence: float = 0.95,
    ) -> tuple[PersonalRecord, bool]:
        """Create or automatically refresh one canonical external fact."""

        key = record_key.strip()
        if not key:
            raise ValueError("external record_key must not be empty")
        existing = self.store.find_active_by_key(entity_type, key)
        if existing is None:
            return (
                self.create(
                    entity_type=entity_type,
                    record_key=key,
                    title=title,
                    summary=summary,
                    data=data,
                    source=source,
                    confidence=confidence,
                    actor="external-sync",
                ),
                True,
            )
        updated = self.update(
            existing.id,
            {
                "title": title,
                "summary": summary,
                "data": data,
                "source": source.source,
                "source_ref": source.source_ref,
                "confidence": confidence,
            },
            actor="external-sync",
            reason="external source changed",
            automatic=True,
        )
        return updated, False

    def remember(
        self,
        memory: MemoryData,
        *,
        summary: str,
        source: RecordSource,
        confidence: float = 0.7,
        sensitivity: SensitivityLevel | None = None,
        data_category: DataCategory | None = None,
        access_policy: AccessPolicy | None = None,
        record_key: str = "",
        expires_at: str | None = None,
        actor: str = "assistant",
    ) -> PersonalRecord:
        return self.create(
            entity_type=PersonalEntityType.MEMORY,
            record_key=record_key,
            title=summary,
            summary=summary,
            data=memory,
            source=source,
            confidence=confidence,
            sensitivity=sensitivity,
            data_category=data_category or memory.category,
            access_policy=access_policy,
            expires_at=expires_at,
            actor=actor,
        )

    def update(
        self,
        record_id: str,
        changes: dict[str, Any],
        *,
        actor: str,
        reason: str = "",
        automatic: bool = False,
    ) -> PersonalRecord:
        normalized = dict(changes)
        if "data" in normalized:
            current = self.get(record_id)
            if current is None:
                raise ValueError(f"personal record not found: {record_id}")
            incoming_data = normalize_payload(normalized["data"])
            if isinstance(current.data, Mapping) and isinstance(
                incoming_data, Mapping
            ):
                incoming_data = {**current.data, **incoming_data}
            normalized["data"] = normalize_personal_payload(
                current.entity_type,
                incoming_data,
                title=str(normalized.get("title") or current.title),
                summary=str(normalized.get("summary") or current.summary),
            )
        record = self.store.update_record(
            record_id,
            changes=normalized,
            actor=actor,
            reason=reason,
            automatic=automatic,
        )
        self._publish(record, change="updated", actor=actor)
        return record

    def confirm(self, record_id: str, *, actor: str = "user") -> PersonalRecord:
        record = self.store.confirm_record(record_id, actor=actor)
        self._publish(record, change="confirmed", actor=actor)
        return record

    def supersede(
        self,
        record_id: str,
        *,
        replacement: dict[str, Any],
        actor: str,
        reason: str = "",
    ) -> PersonalRecord:
        normalized = dict(replacement)
        if "data" in normalized:
            normalized["data"] = normalize_payload(normalized["data"])
        record = self.store.supersede_record(
            record_id,
            replacement=normalized,
            actor=actor,
            reason=reason,
        )
        self._publish(record, change="superseded", actor=actor)
        return record

    def forget(
        self,
        record_id: str,
        *,
        actor: str = "user",
        reason: str = "",
        purge_content: bool = True,
    ) -> PersonalRecord:
        record = self.store.forget_record(
            record_id,
            actor=actor,
            reason=reason,
            purge_content=purge_content,
        )
        self._publish(record, change="forgotten", actor=actor)
        return record

    def hard_delete(self, record_id: str) -> bool:
        existing = self.get(record_id)
        deleted = self.store.hard_delete_record(record_id)
        if deleted and existing is not None:
            self._publish(existing, change="deleted", actor="user")
        return deleted

    def expire_due(self, *, now: str | None = None) -> list[str]:
        return self.store.expire_due(now=now)

    def list(
        self,
        *,
        entity_type: PersonalEntityType | None = None,
        statuses: Sequence[RecordStatus] | None = None,
        limit: int = 100,
    ) -> list[PersonalRecord]:
        return self.store.list_records(
            entity_type=entity_type,
            statuses=statuses,
            limit=limit,
        )

    def get(self, record_id: str) -> PersonalRecord | None:
        return self.store.get_record(record_id)

    def find_active_by_key(
        self, entity_type: PersonalEntityType, record_key: str
    ) -> PersonalRecord | None:
        return self.store.find_active_by_key(entity_type, record_key)

    def history(self, record_id: str) -> list[RecordRevision]:
        return self.store.list_revisions(record_id)

    def lineage(self, record_id: str, *, limit: int = 100) -> list[PersonalRecord]:
        return self.store.list_lineage(record_id, limit=limit)

    def add_memory_evidence(
        self,
        record_id: str,
        *,
        source: RecordSource,
        statement: str,
        confidence: float,
        observed_at: str | None = None,
    ) -> tuple[MemoryEvidence, bool]:
        return self.store.add_memory_evidence(
            record_id,
            source=source,
            statement=statement,
            confidence=confidence,
            observed_at=observed_at,
        )

    def memory_evidence(
        self, record_id: str, *, limit: int = 100
    ) -> list[MemoryEvidence]:
        return self.store.list_memory_evidence(record_id, limit=limit)

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()

    def _publish(self, record: PersonalRecord, *, change: str, actor: str) -> None:
        if self.event_publisher is not None:
            self.event_publisher(
                PersonalRecordChanged(record=record, change=change, actor=actor)
            )

    @staticmethod
    def _default_category(entity_type: PersonalEntityType, data: Any) -> DataCategory:
        if entity_type == PersonalEntityType.HEALTH_OBSERVATION:
            return DataCategory.HEALTH
        if entity_type == PersonalEntityType.CHECK_IN:
            return DataCategory.EMOTIONAL
        if entity_type in {
            PersonalEntityType.RELATIONSHIP,
            PersonalEntityType.IMPORTANT_DATE,
        }:
            return DataCategory.RELATIONSHIP
        if entity_type == PersonalEntityType.FINANCIAL_OBLIGATION:
            return DataCategory.FINANCIAL
        if entity_type == PersonalEntityType.TRIP:
            return DataCategory.LOCATION
        if isinstance(data, MemoryData):
            return data.category
        return DataCategory.GENERAL

    @staticmethod
    def governance_defaults(
        category: DataCategory,
        *,
        sensitivity: SensitivityLevel | None,
        access_policy: AccessPolicy | None,
    ) -> tuple[SensitivityLevel, AccessPolicy]:
        if category in {DataCategory.ACCOUNT, DataCategory.FINANCIAL, DataCategory.LOCATION}:
            return (
                sensitivity or SensitivityLevel.RESTRICTED,
                access_policy or AccessPolicy.OWNER_ONLY,
            )
        if category in {DataCategory.HEALTH, DataCategory.EMOTIONAL}:
            return (
                sensitivity or SensitivityLevel.SENSITIVE,
                access_policy or AccessPolicy.CONFIRM_WRITE,
            )
        if category == DataCategory.RELATIONSHIP:
            return (
                sensitivity or SensitivityLevel.SENSITIVE,
                access_policy or AccessPolicy.CONFIRM_WRITE,
            )
        return (
            sensitivity or SensitivityLevel.PERSONAL,
            access_policy or AccessPolicy.STANDARD,
        )
