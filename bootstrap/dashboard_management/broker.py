from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from agent.permissions import (
    DEFAULT_DASHBOARD_PERMISSION_MODE,
    PermissionMode,
    normalize_permission_mode,
)
from bus.events_lifecycle import StreamDeltaReady


def _chat_title(content: str) -> str:
    title = " ".join(content.split()).strip()
    return f"{title[:48].rstrip()}…" if len(title) > 48 else title or "新对话"


@dataclass
class DashboardChatRun:
    """A dashboard turn whose lifetime is independent from one websocket."""

    run_id: str
    session_key: str
    chat_id: str
    prompt: str
    permission_mode: PermissionMode = DEFAULT_DASHBOARD_PERMISSION_MODE
    started_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat()
    )
    status: str = "thinking"
    content: str = ""
    thinking: str = ""
    media: list[str] = field(default_factory=list, repr=False)
    attachments: list[dict[str, object]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "status",
            "status": self.status,
            "run_id": self.run_id,
            "session_key": self.session_key,
            "chat_id": self.chat_id,
            "prompt": self.prompt,
            "permission_mode": self.permission_mode,
            "title": _chat_title(self.prompt),
            "started_at": self.started_at,
            "content": self.content,
            "thinking": self.thinking,
            "attachments": list(self.attachments),
        }


class DashboardChatBroker:
    """Own dashboard turns and fan their events out to reconnectable clients."""

    def __init__(self, event_bus: Any, agent_loop: Any | None = None) -> None:
        self._subscribers: dict[
            str, set[asyncio.Queue[dict[str, Any]]]
        ] = {}
        self._runs: dict[str, DashboardChatRun] = {}
        self._agent_loop = agent_loop
        event_bus.on(StreamDeltaReady, self._on_delta)

    def open(self, session_key: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault(session_key, set()).add(queue)
        return queue

    def close(
        self,
        session_key: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        subscribers = self._subscribers.get(session_key)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(session_key, None)

    def snapshot(self, session_key: str) -> dict[str, Any] | None:
        run = self._runs.get(session_key)
        return run.snapshot() if run is not None else None

    def list_active(self) -> list[dict[str, Any]]:
        return sorted(
            (run.snapshot() for run in self._runs.values()),
            key=lambda item: item["started_at"],
            reverse=True,
        )

    async def start(
        self,
        *,
        agent_loop: Any,
        session_key: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        attachments: list[dict[str, object]] | None = None,
        run_id: str = "",
        permission_mode: str = DEFAULT_DASHBOARD_PERMISSION_MODE,
    ) -> tuple[bool, dict[str, Any]]:
        current = self._runs.get(session_key)
        if current is not None:
            return False, current.snapshot()

        await self._ensure_session(agent_loop, session_key, content)
        run = DashboardChatRun(
            run_id=run_id.strip() or uuid4().hex,
            session_key=session_key,
            chat_id=chat_id,
            prompt=content,
            permission_mode=normalize_permission_mode(
                permission_mode,
                fallback=DEFAULT_DASHBOARD_PERMISSION_MODE,
            ),
            media=list(media or ()),
            attachments=list(attachments or ()),
        )
        self._runs[session_key] = run
        self._broadcast(session_key, run.snapshot())
        run.task = asyncio.create_task(self._execute(agent_loop, run))
        return True, run.snapshot()

    async def cancel(self, *, agent_loop: Any, session_key: str) -> bool:
        run = self._runs.get(session_key)
        if run is None or run.task is None or run.task.done():
            return False
        run.status = "stopping"
        self._broadcast(session_key, run.snapshot())
        controller = getattr(agent_loop, "request_interrupt", None)
        result = controller(session_key, sender="dashboard", command="stop") if callable(controller) else None
        if getattr(result, "status", "idle") != "interrupted":
            run.task.cancel()
        return True

    async def _execute(self, agent_loop: Any, run: DashboardChatRun) -> None:
        try:
            result = await agent_loop.process_direct(
                run.prompt,
                session_key=run.session_key,
                busy_session_key=run.session_key,
                channel="dashboard",
                chat_id=run.chat_id,
                stream_events=True,
                permission_mode=run.permission_mode,
                media=run.media,
            )
        except asyncio.CancelledError:
            self._broadcast(
                run.session_key,
                {
                    "type": "cancelled",
                    "run_id": run.run_id,
                    "content": run.content,
                    "thinking": run.thinking,
                    "message": "已停止生成",
                },
            )
        except Exception as exc:
            self._broadcast(
                run.session_key,
                {
                    "type": "error",
                    "run_id": run.run_id,
                    "message": str(exc),
                },
            )
        else:
            run.content = result
            self._broadcast(
                run.session_key,
                {
                    "type": "final",
                    "run_id": run.run_id,
                    "content": result,
                    "thinking": run.thinking,
                },
            )
        finally:
            if self._runs.get(run.session_key) is run:
                self._runs.pop(run.session_key, None)

    async def _ensure_session(
        self,
        agent_loop: Any,
        session_key: str,
        content: str,
    ) -> None:
        manager = getattr(agent_loop, "session_manager", None)
        if manager is None:
            return
        session = manager.get_or_create(session_key)
        metadata = getattr(session, "metadata", None)
        if isinstance(metadata, dict) and not str(metadata.get("title") or "").strip():
            metadata["title"] = _chat_title(content)
        save_async = getattr(manager, "save_async", None)
        if callable(save_async):
            await save_async(session)

    def _on_delta(self, event: StreamDeltaReady) -> None:
        run = self._runs.get(event.session_key)
        if run is None:
            return
        if event.content_delta:
            run.content += event.content_delta
            self._broadcast(
                event.session_key,
                {
                    "type": "content_delta",
                    "run_id": run.run_id,
                    "delta": event.content_delta,
                },
            )
        if event.thinking_delta:
            run.thinking += event.thinking_delta
            self._broadcast(
                event.session_key,
                {
                    "type": "thinking_delta",
                    "run_id": run.run_id,
                    "delta": event.thinking_delta,
                },
            )

    def _broadcast(self, session_key: str, payload: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers.get(session_key, ())):
            queue.put_nowait(payload)

    async def push_text(self, chat_id: str, message: str) -> None:
        session_key = f"dashboard:{chat_id}"
        persisted = await self._persist_push(session_key, message)
        connected = bool(self._subscribers.get(session_key))
        if connected:
            self._broadcast(session_key, {"type": "push", "content": message})
        if not persisted and not connected:
            raise RuntimeError("Dashboard 会话当前未连接且无法保存消息")

    async def _persist_push(self, session_key: str, message: str) -> bool:
        manager = getattr(self._agent_loop, "session_manager", None)
        if manager is None:
            return False
        session = manager.get_or_create(session_key)
        metadata = getattr(session, "metadata", None)
        if isinstance(metadata, dict) and not str(metadata.get("title") or "").strip():
            metadata["title"] = "提醒与通知"
        session.add_message(
            "assistant",
            message,
            delivery="dashboard_push",
        )
        save_async = getattr(manager, "save_async", None)
        if callable(save_async):
            await save_async(session)
        else:
            manager.save(session)
        return True
