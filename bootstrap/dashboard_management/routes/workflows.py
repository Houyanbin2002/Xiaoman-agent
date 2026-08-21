from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from ..contracts import DashboardRuntimeServices
from ..schemas import (
    WorkflowActionPayload,
    WorkflowApprovalPayload,
    WorkflowResponsePayload,
)
from ..support import require_workflow_runtime, workflow_rows


def register_workflow_routes(
    app: FastAPI,
    services: DashboardRuntimeServices,
) -> None:
    @app.get("/api/dashboard/control/tasks")
    def list_workflows() -> list[dict[str, Any]]:
        return workflow_rows(services)

    @app.get("/api/dashboard/control/tasks/{workflow_id}")
    def get_workflow(workflow_id: str) -> dict[str, Any]:
        runtime = require_workflow_runtime(services)
        try:
            workflow = runtime.store.require_workflow(workflow_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return workflow.to_dict()

    @app.delete("/api/dashboard/control/tasks/{workflow_id}")
    def delete_workflow(workflow_id: str) -> dict[str, Any]:
        runtime = require_workflow_runtime(services)
        try:
            deleted = runtime.store.delete_workflow(workflow_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="任务不存在")
        return {"deleted": True, "id": workflow_id}

    @app.post("/api/dashboard/control/tasks/{workflow_id}/cancel")
    async def cancel_workflow(
        workflow_id: str, payload: WorkflowActionPayload
    ) -> dict[str, Any]:
        runtime = require_workflow_runtime(services)
        try:
            workflow = await runtime.cancel_workflow(workflow_id, reason=payload.note)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return workflow.to_dict()

    @app.post("/api/dashboard/control/tasks/{workflow_id}/retry")
    def retry_workflow_step(
        workflow_id: str, payload: WorkflowActionPayload
    ) -> dict[str, Any]:
        runtime = require_workflow_runtime(services)
        if not payload.step_id:
            raise HTTPException(status_code=400, detail="step_id 必填")
        try:
            workflow = runtime.store.retry_step(workflow_id, payload.step_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime.wake()
        return workflow.to_dict()

    @app.post("/api/dashboard/control/tasks/{workflow_id}/approval")
    def approve_workflow_step(
        workflow_id: str, payload: WorkflowApprovalPayload
    ) -> dict[str, Any]:
        runtime = require_workflow_runtime(services)
        try:
            workflow = runtime.store.approve_step(
                workflow_id,
                payload.step_id,
                approved=payload.approved,
                note=payload.note,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if payload.approved:
            runtime.wake()
        return workflow.to_dict()

    @app.post("/api/dashboard/control/tasks/{workflow_id}/respond")
    def respond_to_workflow_step(
        workflow_id: str, payload: WorkflowResponsePayload
    ) -> dict[str, Any]:
        runtime = require_workflow_runtime(services)
        try:
            workflow = runtime.store.respond_to_step(
                workflow_id,
                payload.step_id,
                response=payload.response,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime.wake()
        return workflow.to_dict()
