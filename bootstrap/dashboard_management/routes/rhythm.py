from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from core.personal.models import (
    FollowUpTrigger,
    PersonalEntityType,
    RecordSource,
    SceneMode,
)

from ..contracts import DashboardRuntimeServices
from ..schemas import (
    RhythmFocusPayload,
    RhythmFollowUpPayload,
    RhythmRecordPayload,
    RhythmReportPayload,
    RhythmScenePayload,
)
from ..support import require_personal_rhythm


def register_rhythm_routes(
    app: FastAPI,
    services: DashboardRuntimeServices,
) -> None:
    @app.get("/api/dashboard/control/rhythm/overview")
    def rhythm_overview() -> dict[str, Any]:
        return require_personal_rhythm(services).overview()

    @app.get("/api/dashboard/control/rhythm/context")
    def rhythm_context() -> dict[str, Any]:
        return require_personal_rhythm(services).snapshot().to_dict()

    @app.post("/api/dashboard/control/rhythm/scene")
    def set_rhythm_scene(payload: RhythmScenePayload) -> dict[str, Any]:
        rhythm = require_personal_rhythm(services)
        try:
            record = rhythm.set_scene(
                SceneMode(payload.scene),
                duration_minutes=payload.duration_minutes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"record": record.to_dict(), "context": rhythm.snapshot().to_dict()}

    @app.post("/api/dashboard/control/rhythm/focus")
    def start_rhythm_focus(payload: RhythmFocusPayload) -> dict[str, Any]:
        rhythm = require_personal_rhythm(services)
        record = rhythm.start_focus(
            minutes=payload.minutes,
            label=payload.label,
            allow_high_priority=payload.allow_high_priority,
        )
        return {"record": record.to_dict(), "context": rhythm.snapshot().to_dict()}

    @app.delete("/api/dashboard/control/rhythm/focus")
    def stop_rhythm_focus() -> dict[str, Any]:
        rhythm = require_personal_rhythm(services)
        return {
            "stopped": rhythm.stop_focus(),
            "context": rhythm.snapshot().to_dict(),
        }

    @app.get("/api/dashboard/control/rhythm/recommendations")
    def rhythm_recommendations(minutes: int = 30, limit: int = 5) -> dict[str, Any]:
        try:
            return require_personal_rhythm(services).recommend(
                minutes=minutes,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dashboard/control/rhythm/records/{entity_type}")
    def create_rhythm_record(
        entity_type: str,
        payload: RhythmRecordPayload,
    ) -> dict[str, Any]:
        rhythm = require_personal_rhythm(services)
        try:
            record = rhythm.create_domain_record(
                entity_type=PersonalEntityType(entity_type),
                title=payload.title,
                summary=payload.summary,
                data=payload.data,
                record_key=payload.record_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.to_dict()

    @app.post("/api/dashboard/control/rhythm/follow-ups")
    def create_rhythm_follow_up(
        payload: RhythmFollowUpPayload,
    ) -> dict[str, Any]:
        rhythm = require_personal_rhythm(services)
        try:
            record = rhythm.create_follow_up(
                title=payload.title,
                message=payload.message,
                reason=payload.reason,
                trigger_type=FollowUpTrigger(payload.trigger_type),
                next_trigger_at=payload.next_trigger_at,
                interval_minutes=payload.interval_minutes,
                target_entity_type=payload.target_entity_type,
                target_record_key=payload.target_record_key,
                inactivity_days=payload.inactivity_days,
                condition=payload.condition,
                cooldown_minutes=payload.cooldown_minutes,
                user_confirmed=payload.enabled,
                source=RecordSource("dashboard", "rhythm:follow-up"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.to_dict()

    @app.post("/api/dashboard/control/rhythm/reports")
    def generate_rhythm_report(payload: RhythmReportPayload) -> dict[str, Any]:
        try:
            return (
                require_personal_rhythm(services)
                .generate_report(
                    period=payload.period,
                    persist=payload.persist,
                )
                .to_dict()
            )
        except (PermissionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
