from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from core.personal.models import PersonalEntityType

from ..contracts import DashboardRuntimeServices
from ..schemas import ExternalSourceCreatePayload, ExternalSourceUpdatePayload


def _service(services: DashboardRuntimeServices) -> Any:
    if services.external_sources is None:
        raise HTTPException(status_code=503, detail="外部数据订阅服务未启用")
    return services.external_sources


def register_source_routes(app: FastAPI, services: DashboardRuntimeServices) -> None:
    @app.get("/api/dashboard/control/sources")
    def list_sources() -> list[dict[str, Any]]:
        return [item.to_dict() for item in _service(services).store.list_subscriptions()]

    @app.post("/api/dashboard/control/sources")
    def create_source(payload: ExternalSourceCreatePayload) -> dict[str, Any]:
        try:
            subscription = _service(services).store.create_subscription(
                provider=payload.provider,
                server_name=payload.server_name,
                name=payload.name,
                resource_url=payload.resource_url,
                entity_type=PersonalEntityType(payload.entity_type),
                mapping=payload.mapping,
                poll_interval_minutes=payload.poll_interval_minutes,
                enabled=payload.enabled,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return subscription.to_dict()

    @app.patch("/api/dashboard/control/sources/{subscription_id}")
    def update_source(
        subscription_id: str,
        payload: ExternalSourceUpdatePayload,
    ) -> dict[str, Any]:
        changes = {
            key: value
            for key, value in payload.model_dump(exclude_unset=True).items()
            if value is not None
        }
        try:
            return _service(services).store.update_subscription(
                subscription_id,
                changes=changes,
            ).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/dashboard/control/sources/{subscription_id}")
    def delete_source(subscription_id: str) -> dict[str, Any]:
        deleted = _service(services).store.delete_subscription(subscription_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="外部数据订阅不存在")
        return {"deleted": True, "id": subscription_id}

    @app.post("/api/dashboard/control/sources/{subscription_id}/sync")
    async def sync_source(subscription_id: str) -> dict[str, Any]:
        try:
            result = await _service(services).sync_one(subscription_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return result.to_dict()


__all__ = ["register_source_routes"]
