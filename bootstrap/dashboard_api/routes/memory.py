from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from bootstrap.dashboard_api.contracts import (
    ManualMemoryOptimizer,
    MemoryBatchDeletePayload,
    MemoryUpdatePayload,
)
from core.memory.engine import ExecutionMemoryAdminApi, MemoryAdminApi
from core.memory.execution import ExecutionMemoryState
from proactive_v2.memory_optimizer import MemoryOptimizerBusy

logger = logging.getLogger(__name__)


def register_memory_routes(
    app: FastAPI,
    *,
    memory_admin: MemoryAdminApi,
    manual_memory_optimizer: ManualMemoryOptimizer | None,
) -> None:
    router = APIRouter()
    optimizer_task: asyncio.Task[None] | None = None
    optimizer_last_status = "idle"
    optimizer_last_error: str | None = None

    async def run_memory_optimizer() -> None:
        nonlocal optimizer_last_error, optimizer_last_status
        assert manual_memory_optimizer is not None
        optimizer_last_status = "running"
        optimizer_last_error = None
        try:
            await manual_memory_optimizer.optimize()
            optimizer_last_status = "succeeded"
        except MemoryOptimizerBusy:
            optimizer_last_status = "skipped"
            logger.info("manual memory optimizer skipped because it is already running")
        except asyncio.CancelledError:
            optimizer_last_status = "failed"
            optimizer_last_error = "memory optimizer 已取消"
            raise
        except Exception as exc:
            optimizer_last_status = "failed"
            optimizer_last_error = str(exc)
            logger.exception("manual memory optimizer failed: %s", exc)

    @router.get("/api/dashboard/memory/engine-info")
    def get_memory_engine_info() -> dict[str, Any]:
        description = memory_admin.describe()
        return {
            "name": description.name,
            "profile": description.profile.value,
            "capabilities": sorted(item.value for item in description.capabilities),
        }

    @router.get("/api/dashboard/memory/optimizer")
    async def get_memory_optimizer_status() -> dict[str, Any]:
        running = bool(
            manual_memory_optimizer is not None
            and (
                (optimizer_task is not None and not optimizer_task.done())
                or manual_memory_optimizer.is_running
            )
        )
        return {
            "enabled": manual_memory_optimizer is not None,
            "running": running,
            "last_status": "running" if running else optimizer_last_status,
            "last_error": optimizer_last_error,
        }

    @router.post("/api/dashboard/memory/optimize", status_code=202)
    async def trigger_memory_optimizer() -> dict[str, Any]:
        nonlocal optimizer_last_error, optimizer_last_status, optimizer_task
        if manual_memory_optimizer is None:
            raise HTTPException(status_code=503, detail="memory optimizer 未启用")
        if (
            optimizer_task is not None and not optimizer_task.done()
        ) or manual_memory_optimizer.is_running:
            raise HTTPException(status_code=409, detail="memory optimizer 正在运行")
        logger.info("Manual memory optimizer triggered via dashboard")
        optimizer_last_status = "running"
        optimizer_last_error = None
        optimizer_task = asyncio.create_task(
            run_memory_optimizer(),
            name="manual_memory_optimizer",
        )
        return {"status": "started", "message": "Memory optimizer started"}

    @router.get("/api/dashboard/memories")
    def list_memories(
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = memory_admin.list_items_for_dashboard(
            q=q,
            memory_type=memory_type,
            status=status,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            has_embedding=has_embedding,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
            "vec_enabled": True,
            "vec_dim": 0,
        }

    @router.get("/api/dashboard/memory/execution")
    def list_execution_memory(
        include_inactive: bool = False,
        limit: int = 500,
    ) -> dict[str, Any]:
        if not isinstance(memory_admin, ExecutionMemoryAdminApi):
            return {"items": [], "total": 0}
        rows = memory_admin.list_execution_memories(
            include_inactive=include_inactive,
            limit=max(1, min(limit, 5000)),
        )
        items: list[dict[str, object]] = []
        for row in rows:
            state = row.get("execution")
            items.append(
                {
                    **row,
                    "execution": (
                        state.to_dict()
                        if isinstance(state, ExecutionMemoryState)
                        else state
                    ),
                }
            )
        return {"items": items, "total": len(items)}

    @router.get("/api/dashboard/memories/{memory_id:path}/similar")
    def list_similar_memories(
        memory_id: str,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        try:
            items = memory_admin.find_similar_items_for_dashboard(
                memory_id,
                top_k=top_k,
                memory_type=memory_type,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="memory 不存在") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "items": items,
            "total": len(items),
            "source_id": memory_id,
        }

    @router.get("/api/dashboard/memories/{memory_id:path}")
    def get_memory(
        memory_id: str,
        include_embedding: bool = False,
    ) -> dict[str, Any]:
        item = memory_admin.get_item_for_dashboard(
            memory_id,
            include_embedding=include_embedding,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return item

    @router.patch("/api/dashboard/memories/{memory_id:path}")
    def update_memory(
        memory_id: str,
        payload: MemoryUpdatePayload,
    ) -> dict[str, Any]:
        try:
            item = memory_admin.update_item_for_dashboard(
                memory_id,
                status=payload.status,
                extra_json=payload.extra_json,
                source_ref=payload.source_ref,
                happened_at=payload.happened_at,
                emotional_weight=payload.emotional_weight,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return item

    @router.delete("/api/dashboard/memories/{memory_id:path}")
    def delete_memory(memory_id: str) -> dict[str, Any]:
        deleted = memory_admin.delete_item(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return {"deleted": True, "id": memory_id}

    @router.post("/api/dashboard/memories/batch-delete")
    def delete_memories_batch(payload: MemoryBatchDeletePayload) -> dict[str, Any]:
        deleted_count = memory_admin.delete_items_batch(payload.ids)
        return {"deleted_count": deleted_count}

    app.include_router(router)
