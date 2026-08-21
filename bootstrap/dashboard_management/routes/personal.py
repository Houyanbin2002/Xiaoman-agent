from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException

from agent.workflows.personal import RoutineKind
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

from ..contracts import DashboardRuntimeServices
from ..schemas import (
    PersonalActionPayload,
    PersonalRecordCreatePayload,
    PersonalRecordUpdatePayload,
    PersonalRoutinePayload,
)
from ..support import require_personal_data


def register_personal_routes(
    app: FastAPI,
    services: DashboardRuntimeServices,
) -> None:
    @app.get("/api/dashboard/control/personal/overview")
    def personal_overview() -> dict[str, Any]:
        data = require_personal_data(services)
        active = data.list(limit=1000)
        counts = {entity.value: 0 for entity in PersonalEntityType}
        for record in active:
            counts[record.entity_type.value] += 1
        return {
            "counts": counts,
            "total_active": len(active),
            "profile_configured": counts[PersonalEntityType.PROFILE.value] > 0,
            "routines_available": services.personal_routines is not None,
        }

    @app.get("/api/dashboard/control/personal/today")
    def personal_today(
        local_date: str,
        timezone: str = "Asia/Shanghai",
    ) -> dict[str, Any]:
        if services.personal_today is None:
            raise HTTPException(status_code=503, detail="今日数据视图未启用")
        try:
            return services.personal_today.get(
                local_date=local_date,
                timezone_name=timezone,
            ).to_dict()
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dashboard/control/personal/records")
    def list_personal_records(
        entity_type: str = "",
        include_inactive: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        data = require_personal_data(services)
        try:
            entity = PersonalEntityType(entity_type) if entity_type else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="未知个人数据类型") from exc
        statuses = list(RecordStatus) if include_inactive else None
        data.expire_due()
        return [
            record.to_dict()
            for record in data.list(
                entity_type=entity,
                statuses=statuses,
                limit=limit,
            )
        ]

    @app.post("/api/dashboard/control/personal/records")
    def create_personal_record(
        payload: PersonalRecordCreatePayload,
    ) -> dict[str, Any]:
        data = require_personal_data(services)
        try:
            entity_type = PersonalEntityType(payload.entity_type)
            if entity_type == PersonalEntityType.MEMORY and services.memory_governance:
                content = str(payload.data.get("content") or payload.summary or payload.title)
                category = DataCategory(
                    payload.data_category
                    or payload.data.get("category")
                    or DataCategory.GENERAL.value
                )
                result = services.memory_governance.propose(
                    memory=MemoryData(
                        kind=MemoryKind(str(payload.data.get("kind") or MemoryKind.FACT.value)),
                        content=content,
                        tags=[str(item) for item in payload.data.get("tags", [])],
                        category=category,
                    ),
                    summary=payload.summary or payload.title,
                    source=RecordSource("dashboard", "personal-workbench"),
                    record_key=payload.record_key,
                    confidence=payload.confidence,
                    sensitivity=(
                        SensitivityLevel(payload.sensitivity)
                        if payload.sensitivity
                        else None
                    ),
                    data_category=category,
                    access_policy=(
                        AccessPolicy(payload.access_policy)
                        if payload.access_policy
                        else None
                    ),
                    expires_at=payload.expires_at,
                    actor="user",
                    user_confirmed=True,
                )
                if result.record is None:
                    raise ValueError("记忆需要先处理冲突")
                return result.record.to_dict()
            record = data.create(
                entity_type=entity_type,
                record_key=payload.record_key,
                title=payload.title,
                summary=payload.summary,
                data=payload.data,
                source=RecordSource("dashboard", "personal-workbench"),
                confidence=payload.confidence,
                sensitivity=(
                    SensitivityLevel(payload.sensitivity)
                    if payload.sensitivity
                    else None
                ),
                data_category=(
                    DataCategory(payload.data_category)
                    if payload.data_category
                    else None
                ),
                access_policy=(
                    AccessPolicy(payload.access_policy)
                    if payload.access_policy
                    else None
                ),
                expires_at=payload.expires_at,
                actor="user",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.to_dict()

    @app.patch("/api/dashboard/control/personal/records/{record_id}")
    def update_personal_record(
        record_id: str, payload: PersonalRecordUpdatePayload
    ) -> dict[str, Any]:
        data = require_personal_data(services)
        changes = payload.model_dump(
            exclude_unset=True,
            exclude={"reason"},
        )
        changes = {key: value for key, value in changes.items() if value is not None}
        try:
            existing = data.get(record_id)
            if (
                existing is not None
                and existing.entity_type == PersonalEntityType.MEMORY
                and services.memory_governance is not None
            ):
                record = services.memory_governance.update_memory(
                    record_id,
                    changes,
                    actor="user",
                    reason=payload.reason,
                )
            else:
                record = data.update(
                    record_id,
                    changes,
                    actor="user",
                    reason=payload.reason,
                )
        except (PermissionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.to_dict()

    @app.post("/api/dashboard/control/personal/records/{record_id}/confirm")
    def confirm_personal_record(record_id: str) -> dict[str, Any]:
        data = require_personal_data(services)
        try:
            existing = data.get(record_id)
            if (
                existing is not None
                and existing.entity_type == PersonalEntityType.MEMORY
                and services.memory_governance is not None
            ):
                record = services.memory_governance.confirm_memory(record_id)
            else:
                record = data.confirm(record_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.to_dict()

    @app.delete("/api/dashboard/control/personal/records/{record_id}")
    def forget_personal_record(
        record_id: str, payload: PersonalActionPayload | None = None
    ) -> dict[str, Any]:
        data = require_personal_data(services)
        try:
            reason = payload.reason if payload is not None else "Dashboard user request"
            existing = data.get(record_id)
            if (
                existing is not None
                and existing.entity_type == PersonalEntityType.MEMORY
                and services.memory_governance is not None
            ):
                record = services.memory_governance.forget(record_id, reason=reason)
            else:
                record = data.forget(record_id, reason=reason)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.to_dict()

    @app.get("/api/dashboard/control/personal/records/{record_id}/history")
    def personal_record_history(record_id: str) -> list[dict[str, Any]]:
        data = require_personal_data(services)
        try:
            revisions = data.history(record_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [asdict(revision) for revision in revisions]

    @app.post("/api/dashboard/control/personal/routines")
    def create_personal_routine(payload: PersonalRoutinePayload) -> dict[str, Any]:
        if services.personal_routines is None:
            raise HTTPException(status_code=503, detail="个人例程运行时未启用")
        try:
            workflow, created = services.personal_routines.create(
                RoutineKind(payload.routine),
                session_key=f"dashboard:{payload.chat_id}",
                channel="dashboard",
                chat_id=payload.chat_id,
                local_date=payload.local_date,
                timezone_name=payload.timezone,
                candidate=payload.candidate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "created": created,
            "task": workflow.to_dict(include_steps=False),
        }
