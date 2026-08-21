from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agent.permissions.models import ApprovalRequest


@dataclass
class _PendingApproval:
    request: ApprovalRequest
    result: asyncio.Future[bool]


class PermissionService:
    """Own pending approvals independently from any one WebSocket connection."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingApproval] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

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

    def snapshots(self, session_key: str) -> list[dict[str, str]]:
        return [
            pending.request.as_event()
            for pending in self._pending.values()
            if pending.request.session_key == session_key
        ]

    async def request(self, request: ApprovalRequest) -> bool:
        pending = _PendingApproval(
            request=request,
            result=asyncio.get_running_loop().create_future(),
        )
        self._pending[request.id] = pending
        self._broadcast(request.session_key, request.as_event())
        status = "cancelled"
        try:
            approved = await pending.result
            status = "approved" if approved else "denied"
            return approved
        finally:
            self._pending.pop(request.id, None)
            self._broadcast(
                request.session_key,
                {
                    "type": "approval_resolved",
                    "approval_id": request.id,
                    "status": status,
                },
            )

    def resolve(
        self,
        *,
        session_key: str,
        approval_id: str,
        approved: bool,
    ) -> bool:
        pending = self._pending.get(approval_id)
        if (
            pending is None
            or pending.request.session_key != session_key
            or pending.result.done()
        ):
            return False
        pending.result.set_result(approved)
        return True

    def create_request(self, **kwargs: Any) -> ApprovalRequest:
        return ApprovalRequest.create(approval_id=uuid4().hex, **kwargs)

    def _broadcast(self, session_key: str, payload: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers.get(session_key, ())):
            queue.put_nowait(payload)
