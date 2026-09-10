from __future__ import annotations

import inspect
import mimetypes
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bus.events import OutboundMessage
from agent.tools.message_push import PushDeliveryResult


@dataclass
class OutboundDispatch:
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    media: list[str] = field(default_factory=list)


class OutboundPort(Protocol):
    async def dispatch(self, outbound: OutboundDispatch) -> bool: ...


class BusOutboundPort:
    def __init__(self, bus: Any) -> None:
        self._bus = bus

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        maybe = self._bus.publish_outbound(
            OutboundMessage(
                channel=outbound.channel,
                chat_id=outbound.chat_id,
                content=outbound.content,
                thinking=outbound.thinking,
                metadata=dict(outbound.metadata or {}),
                media=list(outbound.media or []),
            )
        )
        if inspect.isawaitable(maybe):
            await maybe
        return True


class PushToolOutboundPort:
    def __init__(self, push_tool: Any, *, before_send: Callable[[OutboundDispatch], bool] | None = None) -> None:
        self._push = push_tool
        self._before_send = before_send

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        return (await self.dispatch_result(outbound)).success

    async def dispatch_result(self, outbound: OutboundDispatch) -> PushDeliveryResult:
        message = str(outbound.content or "").strip()
        channel = str(outbound.channel or "").strip()
        chat_id = str(outbound.chat_id or "").strip()
        media = [str(item).strip() for item in outbound.media if str(item).strip()]
        if (not message and not media) or not channel or not chat_id:
            return PushDeliveryResult(False, "empty or invalid outbound")
        parts: list[dict[str, str]] = [{"message": message}] if message else []
        for item in media:
            mime_type = mimetypes.guess_type(Path(item).name)[0] or ""
            parts.append({"image" if mime_type.startswith("image/") else "file": item})
        sent: list[str] = []
        for part in parts:
            try:
                result = await self._push.send(channel=channel, chat_id=chat_id, before_send=(lambda: self._before_send(outbound)) if self._before_send else None, **part)
            except Exception:
                return PushDeliveryResult(False, "sender outcome unknown", tuple(sent), tuple(part))
            sent.extend(result.sent_parts)
            if not result.success:
                return PushDeliveryResult(False, result.message, tuple(sent), result.unknown_parts)
        return PushDeliveryResult(True, "sender accepted", tuple(sent))
