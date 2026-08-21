from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, cast

from bus.events import InboundMessage, OutboundMessage
from infra.channels.base import MessageDeduper
from infra.channels.contract import ChannelContext
from infra.channels.session_identity import remember_channel_session

from .config import QQBotConfig, QQBotGroupRule
from .message_format import build_message_body

logger = logging.getLogger(__name__)

_CHANNEL = "qqbot"
_TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
_API_URL = "https://api.sgroup.qq.com"
_SANDBOX_API_URL = "https://sandbox.api.sgroup.qq.com"
_GROUP_AND_C2C_INTENT = 1 << 25
_RECONNECT_MAX_SECONDS = 30.0


class QQBotChannel:
    """QQ Open Platform channel using its official Gateway and OpenAPI."""

    name = _CHANNEL

    def __init__(self, config: QQBotConfig) -> None:
        self._config = config
        self._ctx: ChannelContext | None = None
        self._runner: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._connected = False
        self._last_error = ""
        self._token = ""
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._sequence: int | None = None
        self._session_id = ""
        self._last_message_id: dict[str, str] = {}
        self._message_deduper = MessageDeduper(1000)
        self._outbound_bound = False

    def status_snapshot(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "detail": "QQ 开放平台长连接" if self._connected else self._last_error,
        }

    async def start(self, ctx: ChannelContext) -> None:
        self._ctx = ctx
        if not self._outbound_bound:
            ctx.bus.subscribe_outbound(self.name, self._on_response)
            self._outbound_bound = True
        ctx.push_tool.register_channel(self.name, text=self.send)
        self._stopping.clear()
        self._runner = asyncio.create_task(self._run(), name="channel:qqbot")

    async def stop(self) -> None:
        self._stopping.set()
        self._connected = False
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None

    async def _run(self) -> None:
        delay = 1.0
        while not self._stopping.is_set():
            try:
                await self._connect_once()
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._connected = False
                self._last_error = _safe_error(exc)
                logger.warning("[qqbot] 长连接中断，%.0f 秒后重连: %s", delay, exc)
            if self._stopping.is_set():
                return
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(_RECONNECT_MAX_SECONDS, delay * 2)

    async def _connect_once(self) -> None:
        from websockets.asyncio.client import connect

        token = await self._access_token()
        gateway_url = await self._gateway_url(token)
        async with connect(gateway_url, open_timeout=10, close_timeout=5) as socket:
            hello = json.loads(await socket.recv())
            if int(hello.get("op", -1)) != 10:
                raise RuntimeError("QQ Gateway 未返回 Hello 帧")
            heartbeat_ms = int(cast(dict[str, Any], hello.get("d") or {}).get("heartbeat_interval") or 45000)
            if self._session_id and self._sequence is not None:
                await socket.send(json.dumps({
                    "op": 6,
                    "d": {
                        "token": f"QQBot {token}",
                        "session_id": self._session_id,
                        "seq": self._sequence,
                    },
                }))
            else:
                await socket.send(json.dumps({
                    "op": 2,
                    "d": {
                        "token": f"QQBot {token}",
                        "intents": _GROUP_AND_C2C_INTENT,
                        "shard": [0, 1],
                        "properties": {
                            "$os": "windows",
                            "$browser": "xiaoman",
                            "$device": "xiaoman",
                        },
                    },
                }))
            heartbeat = asyncio.create_task(
                self._heartbeat(socket, heartbeat_ms / 1000),
                name="channel:qqbot:heartbeat",
            )
            try:
                async for raw in socket:
                    payload = json.loads(raw)
                    if payload.get("s") is not None:
                        self._sequence = int(payload["s"])
                    op = int(payload.get("op", -1))
                    if op == 0:
                        event_type = str(payload.get("t") or "")
                        if event_type == "READY":
                            self._session_id = str(cast(dict[str, Any], payload.get("d") or {}).get("session_id") or "")
                            self._connected = True
                            self._last_error = ""
                        elif event_type == "RESUMED":
                            self._connected = True
                            self._last_error = ""
                        await self._handle_dispatch(event_type, cast(dict[str, Any], payload.get("d") or {}))
                    elif op == 7:
                        return
                    elif op == 9:
                        self._session_id = ""
                        self._sequence = None
                        raise RuntimeError("QQ Gateway 会话已失效")
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
                self._connected = False

    async def _heartbeat(self, socket: Any, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(max(1.0, interval_seconds))
            await socket.send(json.dumps({"op": 1, "d": self._sequence}))

    async def _access_token(self, *, force: bool = False) -> str:
        ctx = self._require_ctx()
        async with self._token_lock:
            if not force and self._token and time.monotonic() < self._token_expires_at:
                return self._token
            response = await ctx.http_resources.external_default.post(
                _TOKEN_URL,
                json={
                    "appId": self._config.app_id,
                    "clientSecret": self._config.client_secret,
                },
                timeout_s=10,
            )
            response.raise_for_status()
            payload = response.json()
            token = str(payload.get("access_token") or "")
            if not token:
                raise RuntimeError(str(payload.get("message") or "QQ AppID/AppSecret 无效"))
            expires_in = max(120, int(payload.get("expires_in") or 7200))
            self._token = token
            self._token_expires_at = time.monotonic() + expires_in - 60
            return token

    async def _gateway_url(self, token: str) -> str:
        response = await self._require_ctx().http_resources.external_default.get(
            f"{self._api_url}/gateway",
            headers={"Authorization": f"QQBot {token}"},
            timeout_s=10,
        )
        response.raise_for_status()
        url = str(response.json().get("url") or "")
        if not url:
            raise RuntimeError("QQ Gateway 地址为空")
        return url

    @property
    def _api_url(self) -> str:
        return _SANDBOX_API_URL if self._config.sandbox else _API_URL

    async def _handle_dispatch(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type not in {"C2C_MESSAGE_CREATE", "GROUP_AT_MESSAGE_CREATE"}:
            return
        message_id = str(data.get("id") or "")
        if not message_id or self._message_deduper.seen(message_id):
            return
        author = cast(dict[str, Any], data.get("author") or {})
        is_group = event_type == "GROUP_AT_MESSAGE_CREATE"
        sender_id = str(
            author.get("member_openid")
            or author.get("user_openid")
            or author.get("id")
            or ""
        )
        group_id = str(data.get("group_openid") or "") if is_group else ""
        chat_id = f"group:{group_id}" if is_group else f"c2c:{sender_id}"
        group_rule = self._group_rule(group_id) if is_group else None
        if not self._is_allowed(sender_id, group_rule=group_rule, is_group=is_group):
            logger.warning("[qqbot] 拒绝未授权消息 sender=%s chat=%s", sender_id, chat_id)
            return
        content = str(data.get("content") or "").strip()
        if content.lower() == "/stop":
            await self._interrupt(chat_id, sender_id, message_id)
            return
        media = await self._download_attachments(cast(list[Any], data.get("attachments") or []))
        if not content and not media:
            return
        self._last_message_id[chat_id] = message_id
        ctx = self._require_ctx()
        title = f"QQ 群聊 · {group_id[-8:]}" if is_group else f"QQ · {sender_id[-8:]}"
        await remember_channel_session(
            ctx.session_manager,
            session_key=f"{self.name}:{chat_id}",
            channel=self.name,
            chat_id=chat_id,
            sender_id=sender_id,
            title=title,
            chat_type="group" if is_group else "single",
            metadata={"group_openid": group_id} if is_group else {},
        )
        await ctx.bus.publish_inbound(InboundMessage(
            channel=self.name,
            sender=sender_id,
            chat_id=chat_id,
            content=content,
            media=media,
            metadata={
                "message_id": message_id,
                "chat_type": "group" if is_group else "single",
                "group_openid": group_id,
            },
        ))

    async def _interrupt(self, chat_id: str, sender_id: str, message_id: str) -> None:
        controller = self._require_ctx().interrupt_controller
        text = "当前没有运行中的回复。"
        if controller is not None:
            result = controller.request_interrupt(f"{self.name}:{chat_id}", sender_id, "/stop")
            text = result.message or ("已停止当前回复。" if result.status == "interrupted" else text)
        self._last_message_id[chat_id] = message_id
        await self.send(chat_id, text)

    def _group_rule(self, group_id: str) -> QQBotGroupRule | None:
        return next((rule for rule in self._config.groups if rule.group_openid == group_id), None)

    def _is_allowed(
        self,
        sender_id: str,
        *,
        group_rule: QQBotGroupRule | None,
        is_group: bool,
    ) -> bool:
        if is_group and self._config.groups and group_rule is None:
            return False
        allowed = group_rule.allow_from if group_rule and group_rule.allow_from else self._config.allow_from
        return not allowed or sender_id in allowed

    async def _download_attachments(self, attachments: list[Any]) -> list[str]:
        ctx = self._require_ctx()
        paths: list[str] = []
        for raw in attachments[:10]:
            if not isinstance(raw, dict):
                continue
            url = str(raw.get("url") or "")
            if not url.startswith(("http://", "https://")):
                continue
            try:
                response = await ctx.http_resources.external_default.get(url, timeout_s=20)
                response.raise_for_status()
                content_type = str(response.headers.get("content-type") or "").split(";", 1)[0]
                suffix = mimetypes.guess_extension(content_type) or Path(url).suffix or ".bin"
                path = ctx.attachment_store.write_bytes(
                    response.content,
                    prefix="qqbot_",
                    suffix=suffix[:12],
                )
                paths.append(str(path))
            except Exception as exc:
                logger.warning("[qqbot] 附件下载失败: %s", exc)
        return paths

    async def send(self, chat_id: str, message: str) -> None:
        token = await self._access_token()
        path = _message_path(chat_id)
        reply_to = self._last_message_id.get(chat_id)
        if chat_id.startswith("group:") and not reply_to:
            group_id = chat_id.partition(":")[2]
            rule = self._group_rule(group_id)
            if rule is None or not rule.allow_proactive:
                raise PermissionError("该 QQ 群未允许主动推送")
        for sequence, part in enumerate(_split_message(message), start=1):
            body = build_message_body(
                part,
                markdown=self._config.markdown_support,
                sequence=sequence,
                reply_to=reply_to,
            )
            response, token = await self._post_message(path, body, token)
            if self._config.markdown_support and response.status_code in {400, 403, 415, 422}:
                logger.warning(
                    "[qqbot] Markdown 消息被 QQ 拒绝（HTTP %s），降级为纯文本",
                    response.status_code,
                )
                fallback = build_message_body(
                    part,
                    markdown=False,
                    sequence=sequence,
                    reply_to=reply_to,
                )
                response, token = await self._post_message(path, fallback, token)
            response.raise_for_status()
        if reply_to:
            self._last_message_id.pop(chat_id, None)

    async def _post_message(
        self,
        path: str,
        body: dict[str, Any],
        token: str,
    ) -> tuple[Any, str]:
        response = await self._require_ctx().http_resources.external_default.post(
            f"{self._api_url}{path}",
            headers={"Authorization": f"QQBot {token}"},
            json=body,
            timeout_s=10,
        )
        if response.status_code == 401:
            token = await self._access_token(force=True)
            response = await self._require_ctx().http_resources.external_default.post(
                f"{self._api_url}{path}",
                headers={"Authorization": f"QQBot {token}"},
                json=body,
                timeout_s=10,
            )
        return response, token

    async def _on_response(self, message: OutboundMessage) -> None:
        await self.send(message.chat_id, message.content)

    def _require_ctx(self) -> ChannelContext:
        if self._ctx is None:
            raise RuntimeError("QQBot 渠道尚未启动")
        return self._ctx


def _message_path(chat_id: str) -> str:
    kind, separator, target = chat_id.partition(":")
    if not separator or not target:
        raise ValueError("QQBot chat_id 必须是 c2c:OPENID 或 group:GROUP_OPENID")
    if kind == "c2c":
        return f"/v2/users/{target}/messages"
    if kind == "group":
        return f"/v2/groups/{target}/messages"
    raise ValueError("QQBot chat_id 仅支持 c2c 或 group")


def _split_message(text: str, limit: int = 1800) -> list[str]:
    clean = str(text or "").strip()
    if not clean:
        return ["（空回复）"]
    return [clean[index:index + limit] for index in range(0, len(clean), limit)]


def _safe_error(exc: Exception) -> str:
    text = " ".join(str(exc).split()).strip()
    return text[:160] or type(exc).__name__
