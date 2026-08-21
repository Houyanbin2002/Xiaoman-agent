from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query

from infra.persistence.trace_store import TraceStore


def register_trace_routes(app: FastAPI, *, store: TraceStore) -> None:
    router = APIRouter()

    @router.get("/api/dashboard/traces")
    def list_traces(
        session_key: str = "",
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        items = store.list_traces(
            session_key=session_key.strip() or None,
            limit=limit,
        )
        return {"items": [item.to_dict() for item in items], "total": len(items)}

    @router.get("/api/dashboard/traces/{trace_id}")
    def get_trace(trace_id: str) -> dict[str, Any]:
        trace = store.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="任务航迹不存在")
        return {
            "trace": trace.to_dict(),
            "events": [item.to_dict() for item in store.list_events(trace_id)],
        }

    app.include_router(router)
