from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from core.personal.governance import MemoryConflictAction, MemoryConflictStatus
from core.personal.models import (
    AccessPolicy,
    DataCategory,
    MemoryData,
    MemoryKind,
    RecordSource,
    SensitivityLevel,
)

from ..contracts import DashboardRuntimeServices
from ..schemas import (
    GovernedMemoryCreatePayload,
    MemoryConflictResolvePayload,
)
from ..support import require_memory_governance


def register_memory_routes(
    app: FastAPI,
    services: DashboardRuntimeServices,
) -> None:
    @app.get("/api/dashboard/control/memory-governance/overview")
    def memory_governance_overview() -> dict[str, Any]:
        governance = require_memory_governance(services)
        records = governance.list_memories(include_inactive=False, limit=10000)
        conflicts = governance.conflict_store.list_conflicts(
            statuses=[MemoryConflictStatus.PENDING], limit=10000
        )
        by_kind = {item.value: 0 for item in MemoryKind}
        by_category = {item.value: 0 for item in DataCategory}
        by_policy = {item.value: 0 for item in AccessPolicy}
        for record in records:
            kind = str(record.data.get("kind") or MemoryKind.FACT.value)
            by_kind[kind] = by_kind.get(kind, 0) + 1
            by_category[record.data_category.value] += 1
            by_policy[record.access_policy.value] += 1
        return {
            "total_active": len(records),
            "pending_conflicts": len(conflicts),
            "locked": sum(record.user_locked for record in records),
            "expiring": sum(bool(record.expires_at) for record in records),
            "by_kind": by_kind,
            "by_category": by_category,
            "by_policy": by_policy,
        }

    @app.get("/api/dashboard/control/memory-governance/graph")
    def memory_knowledge_graph() -> dict[str, Any]:
        governance = require_memory_governance(services)
        governance.personal_data.expire_due()
        return governance.knowledge_graph()

    @app.get("/api/dashboard/control/memory-governance/memories")
    def list_governed_memories(
        q: str = "",
        kind: str = "",
        category: str = "",
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        governance = require_memory_governance(services)
        try:
            if kind:
                MemoryKind(kind)
            if category:
                DataCategory(category)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="未知记忆分类") from exc
        governance.personal_data.expire_due()
        return [
            record.to_dict()
            for record in governance.list_memories(
                include_inactive=include_inactive,
                kind=kind,
                category=category,
                query=q,
            )
        ]

    @app.post("/api/dashboard/control/memory-governance/memories")
    def propose_governed_memory(
        payload: GovernedMemoryCreatePayload,
    ) -> dict[str, Any]:
        governance = require_memory_governance(services)
        try:
            result = governance.propose(
                memory=MemoryData(
                    kind=MemoryKind(payload.kind),
                    content=payload.content,
                    tags=payload.tags,
                    category=DataCategory(payload.data_category),
                    subject=payload.subject,
                    predicate=payload.predicate,
                    value=payload.value,
                    scope=payload.scope,
                    attributes=payload.attributes,
                ),
                summary=payload.summary,
                record_key=payload.record_key,
                source=RecordSource("dashboard", "memory-governance"),
                confidence=payload.confidence,
                sensitivity=(
                    SensitivityLevel(payload.sensitivity)
                    if payload.sensitivity
                    else None
                ),
                access_policy=(
                    AccessPolicy(payload.access_policy)
                    if payload.access_policy
                    else None
                ),
                expires_at=payload.expires_at,
                actor="dashboard",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    @app.delete("/api/dashboard/control/memory-governance/memories/{record_id}")
    def delete_governed_memory(
        record_id: str,
        hard: bool = False,
    ) -> dict[str, Any]:
        governance = require_memory_governance(services)
        try:
            if hard:
                deleted = governance.hard_delete(record_id)
                if not deleted:
                    raise HTTPException(status_code=404, detail="记忆不存在")
                return {"deleted": True, "hard": True}
            record = governance.forget(
                record_id,
                actor="user",
                reason="Dashboard memory governance",
            )
            return {"deleted": True, "hard": False, "record": record.to_dict()}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dashboard/control/memory-governance/conflicts")
    def list_memory_conflicts(
        pending_only: bool = True,
    ) -> list[dict[str, Any]]:
        governance = require_memory_governance(services)
        statuses = [MemoryConflictStatus.PENDING] if pending_only else None
        rows = governance.conflict_store.list_conflicts(statuses=statuses, limit=1000)
        result: list[dict[str, Any]] = []
        for conflict in rows:
            existing = (
                governance.personal_data.store.get_record(conflict.existing_record_id)
                if conflict.existing_record_id
                else None
            )
            result.append(
                {
                    **conflict.to_dict(),
                    "existing": existing.to_dict() if existing is not None else None,
                }
            )
        return result

    @app.post(
        "/api/dashboard/control/memory-governance/conflicts/{conflict_id}/resolve"
    )
    def resolve_memory_conflict(
        conflict_id: str,
        payload: MemoryConflictResolvePayload,
    ) -> dict[str, Any]:
        governance = require_memory_governance(services)
        try:
            result = governance.resolve(
                conflict_id,
                action=MemoryConflictAction(payload.action),
                note=payload.note,
                merged=payload.merged,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/api/dashboard/control/memory-governance/export")
    def export_governed_memories() -> JSONResponse:
        governance = require_memory_governance(services)
        return JSONResponse(
            content=governance.export_bundle(),
            headers={
                "Content-Disposition": 'attachment; filename="xiaoman-memory-export.json"'
            },
        )
