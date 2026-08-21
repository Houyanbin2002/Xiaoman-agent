from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query

from bootstrap.dashboard_api.proactive_reader import ProactiveDashboardReader


def register_proactive_routes(
    app: FastAPI,
    *,
    get_reader: Callable[[], ProactiveDashboardReader],
) -> None:
    router = APIRouter()

    @router.get("/api/dashboard/proactive/overview")
    def get_proactive_overview() -> dict[str, Any]:
        return get_reader().get_overview()

    @router.get("/api/dashboard/proactive/deliveries")
    def list_proactive_deliveries(
        session_key: str = "",
        sent_from: str = "",
        sent_to: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        items, total = get_reader().list_deliveries(
            session_key=session_key,
            sent_from=sent_from,
            sent_to=sent_to,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @router.get("/api/dashboard/proactive/tick_logs")
    def list_proactive_tick_logs(
        session_key: str = "",
        terminal_action: str = "",
        gate_exit: str = "",
        flow: str = Query(default="", pattern="^(|drift|proactive)$"),
        started_from: str = "",
        started_to: str = "",
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "started_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = get_reader().list_tick_logs(
            session_key=session_key,
            terminal_action=terminal_action,
            gate_exit=gate_exit,
            flow=flow,
            started_from=started_from,
            started_to=started_to,
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
        }

    @router.get("/api/dashboard/proactive/tick_logs/{tick_id}")
    def get_proactive_tick_log(tick_id: str) -> dict[str, Any]:
        item = get_reader().get_tick_log(tick_id)
        if item is None:
            raise HTTPException(status_code=404, detail="tick 不存在")
        return item

    @router.get("/api/dashboard/proactive/tick_logs/{tick_id}/steps")
    def list_proactive_tick_steps(tick_id: str) -> dict[str, Any]:
        item = get_reader().get_tick_log(tick_id)
        if item is None:
            raise HTTPException(status_code=404, detail="tick 不存在")
        steps = get_reader().list_tick_steps(tick_id)
        return {
            "items": steps,
            "total": len(steps),
            "tick_id": tick_id,
        }

    app.include_router(router)
