from __future__ import annotations

import asyncio
import ipaddress
from contextlib import suppress
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from ..attachments import (
    MAX_ATTACHMENT_BYTES,
    AttachmentError,
    DashboardAttachmentStore,
)
from ..broker import DashboardChatBroker
from ..contracts import DashboardRuntimeServices
from ..document_parser import DashboardDocumentParser


def _is_loopback_name(value: str | None) -> bool:
    name = str(value or "").strip().lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _trusted_websocket_origin(websocket: WebSocket) -> bool:
    """Block browser cross-site WebSocket requests and DNS rebinding."""
    origin = websocket.headers.get("origin")
    if not origin:
        return True  # Non-browser clients do not normally send Origin.
    parsed_origin = urlsplit(origin)
    parsed_host = urlsplit(f"//{websocket.headers.get('host', '')}")
    if parsed_origin.scheme not in {"http", "https"}:
        return False
    if not _is_loopback_name(parsed_origin.hostname):
        return False
    if not _is_loopback_name(parsed_host.hostname):
        return False
    origin_port = parsed_origin.port or (443 if parsed_origin.scheme == "https" else 80)
    host_port = parsed_host.port or origin_port
    return origin_port == host_port


async def _forward_events(
    websocket: WebSocket,
    queue: asyncio.Queue[dict[str, object]],
) -> None:
    while True:
        await websocket.send_json(await queue.get())


def register_chat_routes(
    app: FastAPI,
    services: DashboardRuntimeServices,
    broker: DashboardChatBroker,
    attachments: DashboardAttachmentStore,
    document_parser: DashboardDocumentParser,
) -> None:
    @app.post("/api/dashboard/chat/{chat_id}/attachments")
    async def dashboard_chat_upload(
        request: Request,
        chat_id: str,
        filename: str = Query(min_length=1, max_length=255),
    ) -> dict[str, object]:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_ATTACHMENT_BYTES:
                    raise HTTPException(status_code=413, detail="文件超过 128 MB 上限")
            except ValueError:
                raise HTTPException(status_code=400, detail="文件大小信息无效") from None
        try:
            record = await attachments.save_stream(
                chat_id=chat_id,
                filename=filename,
                mime_type=request.headers.get("content-type", ""),
                chunks=request.stream(),
            )
        except AttachmentError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        try:
            content_path = await document_parser.prepare(record)
            record = attachments.set_content_path(chat_id, record.id, content_path)
        except AttachmentError as exc:
            attachments.remove(chat_id, record.id)
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return record.public()

    @app.delete("/api/dashboard/chat/{chat_id}/attachments/{attachment_id}")
    def dashboard_chat_remove_attachment(
        chat_id: str,
        attachment_id: str,
    ) -> dict[str, object]:
        return {
            "deleted": attachments.remove(chat_id, attachment_id),
            "id": attachment_id,
        }

    @app.get("/api/dashboard/chat/runs")
    def dashboard_chat_runs() -> dict[str, object]:
        return {"items": broker.list_active()}

    @app.get("/api/dashboard/chat/gateway-runs")
    def dashboard_gateway_runs() -> dict[str, object]:
        items: list[dict[str, object]] = []
        for session_key, state in services.agent_loop.active_turn_states.items():
            if session_key.startswith("dashboard:"):
                continue
            channel, separator, chat_id = session_key.partition(":")
            items.append(
                {
                    "session_key": session_key,
                    "channel": channel if separator else "gateway",
                    "chat_id": chat_id if separator else session_key,
                    "prompt": state.original_user_message,
                    "status": "running",
                }
            )
        return {"items": items}

    @app.post("/api/dashboard/chat/runs/stop")
    async def dashboard_stop_run(
        session_key: str = Query(min_length=3, max_length=500),
    ) -> dict[str, object]:
        if session_key.startswith("dashboard:"):
            stopped = await broker.cancel(
                agent_loop=services.agent_loop,
                session_key=session_key,
            )
            if stopped:
                return {"stopped": True, "status": "interrupted"}

        controller = getattr(services.agent_loop, "request_interrupt", None)
        result = (
            controller(session_key, sender="dashboard", command="stop")
            if callable(controller)
            else None
        )
        status = str(getattr(result, "status", "idle"))
        if status != "interrupted":
            raise HTTPException(status_code=409, detail="该任务已经结束或无法停止")
        return {"stopped": True, "status": status}

    @app.websocket("/api/dashboard/chat/{chat_id}")
    async def dashboard_chat(websocket: WebSocket, chat_id: str) -> None:
        if not _trusted_websocket_origin(websocket):
            await websocket.close(
                code=1008, reason="Dashboard WebSocket origin rejected"
            )
            return
        await websocket.accept()
        session_key = f"dashboard:{chat_id}"
        queue = broker.open(session_key)
        permission_service = services.permission_service
        permission_queue = (
            permission_service.open(session_key)
            if permission_service is not None
            else None
        )
        senders: list[asyncio.Task[None]] = []
        try:
            await websocket.send_json({"type": "ready", "session_key": session_key})
            snapshot = broker.snapshot(session_key)
            if snapshot is not None:
                await websocket.send_json(snapshot)
            if permission_service is not None:
                for approval in permission_service.snapshots(session_key):
                    await websocket.send_json(approval)
            senders.append(asyncio.create_task(_forward_events(websocket, queue)))
            if permission_queue is not None:
                senders.append(
                    asyncio.create_task(_forward_events(websocket, permission_queue))
                )
            while True:
                payload = await websocket.receive_json()
                message_type = str(payload.get("type") or "message")
                if message_type == "approval_response":
                    decision = str(payload.get("decision") or "").strip().lower()
                    approval_id = str(payload.get("approval_id") or "").strip()
                    resolved = bool(
                        permission_service is not None
                        and decision in {"approve", "deny"}
                        and approval_id
                        and permission_service.resolve(
                            session_key=session_key,
                            approval_id=approval_id,
                            approved=decision == "approve",
                        )
                    )
                    if not resolved:
                        await websocket.send_json(
                            {"type": "error", "message": "审批请求已失效或不存在"}
                        )
                    continue
                if message_type == "stop":
                    if not await broker.cancel(
                        agent_loop=services.agent_loop,
                        session_key=session_key,
                    ):
                        await websocket.send_json(
                            {"type": "idle", "message": "当前没有正在执行的任务"}
                        )
                    continue

                content = str(payload.get("content") or "").strip()
                raw_attachment_ids = payload.get("attachment_ids")
                attachment_ids = (
                    [str(value) for value in raw_attachment_ids]
                    if isinstance(raw_attachment_ids, list)
                    else []
                )
                try:
                    message_attachments = attachments.resolve_many(
                        chat_id,
                        attachment_ids,
                    )
                except AttachmentError as exc:
                    await websocket.send_json(
                        {"type": "error", "message": str(exc)}
                    )
                    continue
                if not content and not message_attachments:
                    await websocket.send_json(
                        {"type": "error", "message": "消息不能为空"}
                    )
                    continue
                if not content:
                    content = "请阅读并分析这些附件。"
                started, current = await broker.start(
                    agent_loop=services.agent_loop,
                    session_key=session_key,
                    chat_id=chat_id,
                    content=content,
                    media=[
                        str(record.content_path or record.path)
                        for record in message_attachments
                    ],
                    attachments=[record.public() for record in message_attachments],
                    run_id=str(payload.get("request_id") or ""),
                    permission_mode=str(payload.get("permission_mode") or ""),
                )
                if not started:
                    await websocket.send_json(
                        {
                            **current,
                            "type": "busy",
                            "message": "当前回复仍在生成，请先停止后再发送",
                        }
                    )
        except WebSocketDisconnect:
            pass
        finally:
            for sender in senders:
                sender.cancel()
            for sender in senders:
                with suppress(asyncio.CancelledError):
                    await sender
            broker.close(session_key, queue)
            if permission_service is not None and permission_queue is not None:
                permission_service.close(session_key, permission_queue)
