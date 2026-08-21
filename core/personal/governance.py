from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol, Sequence

from core.personal.models import (
    AccessPolicy,
    DataCategory,
    MemoryData,
    MemoryKind,
    PersonalEntityType,
    PersonalRecord,
    RecordSource,
    RecordStatus,
    SensitivityLevel,
    normalize_payload,
)
from core.personal.memory_reconciliation import (
    MemoryReconciler,
    MemorySemanticRelation,
    normalized_memory_kind,
)
from core.personal.service import PersonalDataService


class MemoryConflictStatus(StrEnum):
    PENDING = "pending"
    KEPT_EXISTING = "kept_existing"
    ACCEPTED_CANDIDATE = "accepted_candidate"
    MERGED = "merged"
    CANCELLED = "cancelled"


class MemoryConflictAction(StrEnum):
    KEEP_EXISTING = "keep_existing"
    ACCEPT_CANDIDATE = "accept_candidate"
    MERGE = "merge"


@dataclass(frozen=True)
class MemoryConflict:
    id: str
    record_key: str
    existing_record_id: str | None
    candidate: dict[str, Any]
    candidate_hash: str
    reason: str
    status: MemoryConflictStatus
    resolved_record_id: str | None
    resolution_note: str
    created_at: str
    resolved_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "record_key": self.record_key,
            "existing_record_id": self.existing_record_id,
            "candidate": self.candidate,
            "candidate_hash": self.candidate_hash,
            "reason": self.reason,
            "status": self.status.value,
            "resolved_record_id": self.resolved_record_id,
            "resolution_note": self.resolution_note,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


@dataclass(frozen=True)
class MemoryProposalResult:
    status: str
    record: PersonalRecord | None = None
    conflict: MemoryConflict | None = None
    relation: MemorySemanticRelation = MemorySemanticRelation.INDEPENDENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "record": self.record.to_dict() if self.record is not None else None,
            "conflict": self.conflict.to_dict() if self.conflict is not None else None,
            "relation": self.relation.value,
        }


class MemoryConflictStore(Protocol):
    def create_conflict(
        self,
        *,
        record_key: str,
        existing_record_id: str | None,
        candidate: dict[str, Any],
        reason: str,
    ) -> tuple[MemoryConflict, bool]: ...

    def get_conflict(self, conflict_id: str) -> MemoryConflict | None: ...

    def list_conflicts(
        self,
        *,
        statuses: Sequence[MemoryConflictStatus] | None = None,
        limit: int = 200,
    ) -> list[MemoryConflict]: ...

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        status: MemoryConflictStatus,
        resolved_record_id: str | None,
        note: str,
    ) -> MemoryConflict: ...

    def purge_record_references(self, record_id: str) -> None: ...

    def cancel_pending_for_record_key(self, record_key: str, *, note: str) -> int: ...

    def close(self) -> None: ...


class MemoryGovernanceService:
    def __init__(
        self,
        *,
        personal_data: PersonalDataService,
        conflict_store: MemoryConflictStore,
    ) -> None:
        self.personal_data = personal_data
        self.conflict_store = conflict_store
        self._reconciler = MemoryReconciler()

    def propose(
        self,
        *,
        memory: MemoryData,
        summary: str,
        source: RecordSource,
        record_key: str = "",
        confidence: float = 0.7,
        sensitivity: SensitivityLevel | None = None,
        data_category: DataCategory | None = None,
        access_policy: AccessPolicy | None = None,
        valid_from: str | None = None,
        expires_at: str | None = None,
        actor: str = "assistant",
        user_confirmed: bool = False,
        explicit_replaces: bool = False,
    ) -> MemoryProposalResult:
        kind = normalized_memory_kind(memory.kind)
        category = data_category or memory.category
        resolved_sensitivity, resolved_policy = self.personal_data.governance_defaults(
            category,
            sensitivity=sensitivity,
            access_policy=access_policy,
        )
        identity = self._reconciler.identity(
            MemoryData(
                kind=kind,
                content=memory.content,
                tags=list(memory.tags),
                category=category,
                subject=memory.subject,
                predicate=memory.predicate,
                value=memory.value,
                scope=memory.scope,
                attributes=dict(memory.attributes),
                identity_quality=memory.identity_quality,
            ),
            summary=summary,
            supplied_key=record_key,
        )
        normalized_memory = MemoryData(
            kind=kind,
            content=memory.content.strip(),
            tags=list(memory.tags),
            category=category,
            subject=memory.subject.strip(),
            predicate=memory.predicate.strip(),
            value=memory.value.strip(),
            scope=memory.scope.strip(),
            attributes=dict(memory.attributes),
            identity_quality=identity.quality,
        )
        actual_confidence = max(0.0, min(1.0, float(confidence)))
        candidate = {
            "entity_type": PersonalEntityType.MEMORY.value,
            "record_key": identity.record_key,
            "title": summary.strip(),
            "summary": summary.strip(),
            "data": normalize_payload(normalized_memory),
            "source": source.source,
            "source_ref": source.source_ref,
            "confidence": actual_confidence,
            "sensitivity": resolved_sensitivity.value,
            "data_category": category.value,
            "access_policy": resolved_policy.value,
            "valid_from": valid_from or datetime.now(timezone.utc).isoformat(),
            "expires_at": expires_at,
            "user_locked": False,
            "allow_auto_update": resolved_policy == AccessPolicy.STANDARD,
        }
        existing = self.personal_data.store.find_active_by_key(
            PersonalEntityType.MEMORY, identity.record_key
        )
        reconciliation = self._reconciler.classify(
            existing,
            candidate=normalized_memory,
            identity=identity,
            explicit_replaces=explicit_replaces,
        )
        validity_error = self._validity_error(
            str(candidate["valid_from"]),
            str(candidate.get("expires_at") or ""),
        )
        if validity_error:
            conflict, _ = self.conflict_store.create_conflict(
                record_key=identity.record_key,
                existing_record_id=existing.id if existing is not None else None,
                candidate=candidate,
                reason=validity_error,
            )
            return MemoryProposalResult(
                status="conflict_pending",
                conflict=conflict,
                relation=reconciliation.relation,
            )
        governance_unchanged = bool(
            existing is not None
            and existing.data_category == category
            and existing.access_policy == resolved_policy
            and existing.sensitivity == resolved_sensitivity
        )
        if (
            existing is not None
            and reconciliation.relation == MemorySemanticRelation.SAME
            and governance_unchanged
        ):
            self._add_evidence(existing, candidate)
            return MemoryProposalResult(
                status="unchanged",
                record=existing,
                relation=reconciliation.relation,
            )

        authorized = actor == "user" or user_confirmed
        if authorized:
            candidate["user_locked"] = True
            candidate["allow_auto_update"] = False
        if existing is not None and authorized:
            replacement = self.personal_data.supersede(
                existing.id,
                replacement=candidate,
                actor="user",
                reason="user confirmed memory replacement",
            )
            self._add_evidence(replacement, candidate)
            return MemoryProposalResult(
                status="superseded",
                record=replacement,
                relation=reconciliation.relation,
            )
        if existing is None and authorized:
            record = self._create_candidate(candidate, actor="user")
            return MemoryProposalResult(
                status="created",
                record=record,
                relation=reconciliation.relation,
            )

        if (
            existing is None
            and resolved_policy == AccessPolicy.STANDARD
            and actual_confidence >= 0.75
            and not explicit_replaces
        ):
            record = self._create_candidate(candidate, actor=actor)
            return MemoryProposalResult(
                status="created",
                record=record,
                relation=reconciliation.relation,
            )

        if existing is not None:
            reason = (
                "governance_metadata_change"
                if reconciliation.relation == MemorySemanticRelation.SAME
                else {
                    MemorySemanticRelation.REFINE: "refinement_requires_confirmation",
                    MemorySemanticRelation.CONTRADICT: "semantic_contradiction",
                    MemorySemanticRelation.INDEPENDENT: "ambiguous_fact_identity",
                }.get(reconciliation.relation, "conflicting_active_memory")
            )
        elif resolved_policy != AccessPolicy.STANDARD:
            reason = "authorization_required"
        elif actual_confidence < 0.75:
            reason = "low_confidence_candidate"
        elif explicit_replaces:
            reason = "replacement_target_not_found"
        else:
            reason = "authorization_required"
        conflict, _ = self.conflict_store.create_conflict(
            record_key=identity.record_key,
            existing_record_id=existing.id if existing is not None else None,
            candidate=candidate,
            reason=reason,
        )
        return MemoryProposalResult(
            status="conflict_pending",
            conflict=conflict,
            relation=reconciliation.relation,
        )

    def resolve(
        self,
        conflict_id: str,
        *,
        action: MemoryConflictAction,
        note: str = "",
        merged: dict[str, Any] | None = None,
    ) -> MemoryProposalResult:
        conflict = self.conflict_store.get_conflict(conflict_id)
        if conflict is None:
            raise ValueError(f"memory conflict not found: {conflict_id}")
        if conflict.status != MemoryConflictStatus.PENDING:
            raise ValueError("memory conflict is already resolved")
        existing = (
            self.personal_data.store.get_record(conflict.existing_record_id)
            if conflict.existing_record_id
            else None
        )
        if conflict.existing_record_id and (
            existing is None or existing.status != RecordStatus.ACTIVE
        ):
            self.conflict_store.resolve_conflict(
                conflict.id,
                status=MemoryConflictStatus.CANCELLED,
                resolved_record_id=None,
                note="conflict target is no longer active",
            )
            raise ValueError("memory conflict target is no longer active")
        if action == MemoryConflictAction.KEEP_EXISTING:
            resolved = self.conflict_store.resolve_conflict(
                conflict.id,
                status=MemoryConflictStatus.KEPT_EXISTING,
                resolved_record_id=existing.id if existing is not None else None,
                note=note,
            )
            return MemoryProposalResult(
                status="kept_existing", record=existing, conflict=resolved
            )

        candidate = dict(conflict.candidate)
        if not candidate:
            raise ValueError("memory conflict candidate has been redacted")
        if action == MemoryConflictAction.MERGE:
            if existing is None:
                raise ValueError("merge requires an existing active memory")
            if not merged:
                raise ValueError("merged content is required")
            candidate.update(
                {key: value for key, value in merged.items() if value is not None}
            )
            if "content" in merged:
                data = dict(candidate.get("data") or {})
                data["content"] = str(merged["content"])
                candidate["data"] = data
            record = self.personal_data.supersede(
                existing.id,
                replacement=candidate,
                actor="user",
                reason="user merged conflicting memory",
            )
            self._add_evidence(record, candidate)
            status = MemoryConflictStatus.MERGED
            result_status = "merged"
        else:
            if existing is not None and existing.status == RecordStatus.ACTIVE:
                record = self.personal_data.supersede(
                    existing.id,
                    replacement=candidate,
                    actor="user",
                    reason="user accepted conflicting memory",
                )
                self._add_evidence(record, candidate)
            else:
                record = self._create_candidate(candidate, actor="user")
            status = MemoryConflictStatus.ACCEPTED_CANDIDATE
            result_status = "accepted_candidate"
        resolved = self.conflict_store.resolve_conflict(
            conflict.id,
            status=status,
            resolved_record_id=record.id,
            note=note,
        )
        return MemoryProposalResult(
            status=result_status, record=record, conflict=resolved
        )

    def list_memories(
        self,
        *,
        include_inactive: bool = False,
        kind: str = "",
        category: str = "",
        query: str = "",
        limit: int = 1000,
    ) -> list[PersonalRecord]:
        statuses = list(RecordStatus) if include_inactive else None
        rows = self.personal_data.list(
            entity_type=PersonalEntityType.MEMORY,
            statuses=statuses,
            limit=limit,
        )
        normalized_query = query.strip().casefold()
        return [
            row
            for row in rows
            if (not kind or str(row.data.get("kind") or "") == kind)
            and (not category or row.data_category.value == category)
            and (
                not normalized_query
                or normalized_query
                in json.dumps(row.to_dict(), ensure_ascii=False).casefold()
            )
        ]

    def knowledge_graph(self) -> dict[str, Any]:
        """Build a user-facing semantic graph from active governed memories."""

        records = self.list_memories(include_inactive=False, limit=2000)
        nodes: dict[str, dict[str, Any]] = {
            "person:self": {
                "id": "person:self",
                "label": "我",
                "kind": "self",
                "memory_ids": [],
                "confidence": 1.0,
            }
        }
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in records:
            data = record.data
            kind = str(data.get("kind") or MemoryKind.FACT.value)
            subject = str(data.get("subject") or "我").strip() or "我"
            predicate = str(data.get("predicate") or "").strip()
            value = str(data.get("value") or "").strip()
            if not predicate:
                predicate = self._knowledge_relation_label(kind)
            if not value:
                value = str(
                    record.title or record.summary or data.get("content") or ""
                ).strip()
            if not value:
                continue

            source_id = self._knowledge_entity_id(subject)
            target_id = self._knowledge_entity_id(value)
            self._merge_knowledge_node(
                nodes,
                node_id=source_id,
                label="我" if source_id == "person:self" else subject,
                kind="self" if source_id == "person:self" else "entity",
                record_id=record.id,
                confidence=record.confidence,
            )
            self._merge_knowledge_node(
                nodes,
                node_id=target_id,
                label=value,
                kind=kind,
                record_id=record.id,
                confidence=record.confidence,
            )
            edge_key = (source_id, predicate, target_id)
            edge = edges.setdefault(
                edge_key,
                {
                    "id": f"edge:{len(edges) + 1}",
                    "source": source_id,
                    "target": target_id,
                    "label": predicate,
                    "kind": kind,
                    "memory_ids": [],
                },
            )
            if record.id not in edge["memory_ids"]:
                edge["memory_ids"].append(record.id)
        return {
            "center_id": "person:self",
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
        }

    @staticmethod
    def _knowledge_entity_id(label: str) -> str:
        normalized = " ".join(label.strip().split())[:160]
        if normalized.casefold() in {"我", "用户", "本人", "自己", "user", "self"}:
            return "person:self"
        return f"entity:{normalized.casefold()}"

    @staticmethod
    def _knowledge_relation_label(kind: str) -> str:
        return {
            MemoryKind.REQUESTED.value: "明确记住",
            MemoryKind.FACT.value: "事实",
            MemoryKind.PREFERENCE.value: "偏好",
            MemoryKind.TEMPORARY_STATE.value: "当前状态",
            MemoryKind.HISTORICAL_EVENT.value: "经历",
            MemoryKind.EPISODE.value: "经历",
            MemoryKind.RELATIONSHIP.value: "关系",
            MemoryKind.PROCEDURE.value: "助手操作上下文",
        }.get(kind, "关联")

    @staticmethod
    def _merge_knowledge_node(
        nodes: dict[str, dict[str, Any]],
        *,
        node_id: str,
        label: str,
        kind: str,
        record_id: str,
        confidence: float,
    ) -> None:
        node = nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "label": label[:160],
                "kind": kind,
                "memory_ids": [],
                "confidence": 0.0,
            },
        )
        if record_id not in node["memory_ids"]:
            node["memory_ids"].append(record_id)
        node["confidence"] = max(float(node["confidence"]), float(confidence))

    def export_bundle(self) -> dict[str, Any]:
        records = self.list_memories(include_inactive=True, limit=10000)
        return {
            "format": "xiaoman-memory-governance-v1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "records": [record.to_dict() for record in records],
            "revisions": {
                record.id: [
                    {
                        "revision": item.revision,
                        "action": item.action,
                        "actor": item.actor,
                        "reason": item.reason,
                        "snapshot": item.snapshot,
                        "created_at": item.created_at,
                    }
                    for item in self.personal_data.history(record.id)
                ]
                for record in records
            },
            "conflicts": [
                conflict.to_dict()
                for conflict in self.conflict_store.list_conflicts(limit=10000)
            ],
            "graph": self._graph_snapshot(records),
        }

    def hard_delete(self, record_id: str) -> bool:
        record = self.personal_data.store.get_record(record_id)
        if record is None:
            return False
        if record.entity_type != PersonalEntityType.MEMORY:
            raise ValueError(
                "hard delete through memory governance only accepts memory records"
            )
        self.conflict_store.cancel_pending_for_record_key(
            record.record_key,
            note="memory lineage deleted",
        )
        self.conflict_store.purge_record_references(record_id)
        deleted = self.personal_data.hard_delete(record_id)
        return deleted

    def update_memory(
        self,
        record_id: str,
        changes: dict[str, Any],
        *,
        actor: str,
        reason: str = "",
        automatic: bool = False,
    ) -> PersonalRecord:
        record = self.personal_data.get(record_id)
        if record is None or record.entity_type != PersonalEntityType.MEMORY:
            raise ValueError("memory record not found")
        updated = self.personal_data.update(
            record_id,
            changes,
            actor=actor,
            reason=reason,
            automatic=automatic,
        )
        return updated

    def confirm_memory(self, record_id: str, *, actor: str = "user") -> PersonalRecord:
        record = self.personal_data.get(record_id)
        if record is None or record.entity_type != PersonalEntityType.MEMORY:
            raise ValueError("memory record not found")
        confirmed = self.personal_data.confirm(record_id, actor=actor)
        return confirmed

    def forget(
        self, record_id: str, *, actor: str = "user", reason: str = ""
    ) -> PersonalRecord:
        record = self.personal_data.get(record_id)
        if record is None or record.entity_type != PersonalEntityType.MEMORY:
            raise ValueError("memory record not found")
        forgotten = self.personal_data.forget(record_id, actor=actor, reason=reason)
        self.conflict_store.cancel_pending_for_record_key(
            record.record_key,
            note="memory lineage forgotten",
        )
        return forgotten

    def close(self) -> None:
        self.conflict_store.close()

    def _create_candidate(
        self, candidate: dict[str, Any], *, actor: str
    ) -> PersonalRecord:
        record = self.personal_data.create(
            entity_type=PersonalEntityType.MEMORY,
            record_key=str(candidate["record_key"]),
            title=str(candidate["title"]),
            summary=str(candidate["summary"]),
            data=dict(candidate["data"]),
            source=RecordSource(
                str(candidate["source"]), str(candidate.get("source_ref") or "")
            ),
            confidence=float(candidate["confidence"]),
            sensitivity=SensitivityLevel(candidate["sensitivity"]),
            data_category=DataCategory(candidate["data_category"]),
            access_policy=AccessPolicy(candidate["access_policy"]),
            valid_from=candidate.get("valid_from"),
            expires_at=candidate.get("expires_at"),
            user_locked=bool(candidate.get("user_locked", False)),
            allow_auto_update=bool(candidate.get("allow_auto_update", True)),
            actor=actor,
        )
        self._add_evidence(record, candidate)
        return record

    def _add_evidence(self, record: PersonalRecord, candidate: dict[str, Any]) -> None:
        raw_data = candidate.get("data")
        candidate_data = raw_data if isinstance(raw_data, dict) else {}
        self.personal_data.add_memory_evidence(
            record.id,
            source=RecordSource(
                str(candidate.get("source") or record.source.source),
                str(candidate.get("source_ref") or record.source.source_ref),
            ),
            statement=str(
                candidate_data.get("content")
                or candidate.get("summary")
                or record.summary
            ),
            confidence=float(candidate.get("confidence", record.confidence)),
        )

    def _graph_snapshot(
        self, records: Sequence[PersonalRecord]
    ) -> dict[str, list[dict[str, Any]]]:
        conflicts = self.conflict_store.list_conflicts(limit=10000)
        fact_keys = sorted(
            {record.record_key for record in records}
            | {conflict.record_key for conflict in conflicts}
        )
        nodes: list[dict[str, Any]] = [
            {"id": f"fact:{key}", "type": "fact_slot", "record_key": key}
            for key in fact_keys
        ]
        edges: list[dict[str, Any]] = []
        for record in records:
            nodes.append(
                {
                    "id": record.id,
                    "type": "memory_version",
                    "status": record.status.value,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                }
            )
            edges.append(
                {
                    "from": record.id,
                    "type": "version_of",
                    "to": f"fact:{record.record_key}",
                }
            )
            if record.supersedes_id:
                edges.append(
                    {
                        "from": record.id,
                        "type": "supersedes",
                        "to": record.supersedes_id,
                    }
                )
            for evidence in self.personal_data.memory_evidence(record.id, limit=1000):
                nodes.append(
                    {
                        "id": evidence.id,
                        "type": "evidence",
                        "source": evidence.source.source,
                        "source_ref": evidence.source.source_ref,
                        "confidence": evidence.confidence,
                        "observed_at": evidence.observed_at,
                    }
                )
                edges.append(
                    {
                        "from": evidence.id,
                        "type": "supports",
                        "to": record.id,
                    }
                )
        for conflict in conflicts:
            nodes.append(
                {
                    "id": conflict.id,
                    "type": "memory_conflict",
                    "status": conflict.status.value,
                    "reason": conflict.reason,
                }
            )
            edges.append(
                {
                    "from": conflict.id,
                    "type": "proposes_version_of",
                    "to": f"fact:{conflict.record_key}",
                }
            )
            if conflict.existing_record_id:
                edges.append(
                    {
                        "from": conflict.id,
                        "type": "conflicts_with",
                        "to": conflict.existing_record_id,
                    }
                )
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _validity_error(valid_from: str, expires_at: str) -> str:
        def parse(value: str) -> datetime | None:
            raw = value.strip()
            if not raw:
                return None
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                return None
            return parsed if parsed.tzinfo is not None else None

        start = parse(valid_from)
        if valid_from.strip() and start is None:
            return "invalid_valid_from"
        end = parse(expires_at)
        if expires_at.strip() and end is None:
            return "invalid_expires_at"
        if start is not None and end is not None and end <= start:
            return "invalid_validity_window"
        return ""
