from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import struct
import uuid
from pathlib import Path
from typing import Any, cast

from bus.events import InboundMessage, OutboundMessage
from core.net.http import RequestBudget
from infra.channels.base import MessageDeduper
from infra.channels.contract import ChannelContext
from infra.channels.session_identity import remember_channel_session

from .config import WeixinConfig

logger = logging.getLogger(__name__)

_CHANNEL = "weixin"
_APP_ID = "bot"
_CLIENT_VERSION = str((2 << 16) | (2 << 8))
_BASE_INFO = {"channel_version": "2.2.0"}
_GET_UPDATES = "ilink/bot/getupdates"
_SEND_MESSAGE = "ilink/bot/sendmessage"
_SESSION_EXPIRED = -14


class WeixinChannel:
    """Tencent iLink Bot channel for personal WeChat direct messages."""

    name = _CHANNEL

    def __init__(
        self,
        config: WeixinConfig,
        *,
        account_id: str,
        token: str,
        base_url: str,
        data_dir: Path,
    ) -> None:
        self._config = config
        self._account_id = account_id
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._data_dir = data_dir
        self._ctx: ChannelContext | None = None
        self._runner: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._connected = False
        self._last_error = ""
        self._deduper = MessageDeduper(1000)
        self._context_tokens: dict[str, str] = {}
        self._outbound_bound = False

    def status_snapshot(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "detail": "微信 iLink 长轮询" if self._connected else self._last_error,
        }

    async def start(self, ctx: ChannelContext) -> None:
        self._ctx = ctx
        self._restore_context_tokens()
        if not self._outbound_bound:
            ctx.bus.subscribe_outbound(self.name, self._on_response)
            self._outbound_bound = True
        ctx.push_tool.register_channel(self.name, text=self.send)
        self._stopping.clear()
        self._runner = asyncio.create_task(self._poll_loop(), name="channel:weixin")

    async def stop(self) -> None:
        self._stopping.set()
        self._connected = False
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None

    async def _poll_loop(self) -> None:
        sync_buf = self._load_sync_buf()
        delay = 1.0
        while not self._stopping.is_set():
            try:
                payload = await self._post(
                    _GET_UPDATES,
                    {"get_updates_buf": sync_buf},
                    timeout_s=40,
                    budget_s=44,
                )
                ret = payload.get("ret", 0)
                errcode = payload.get("errcode", 0)
                if ret not in {0, None} or errcode not in {0, None}:
                    if ret == _SESSION_EXPIRED or errcode == _SESSION_EXPIRED:
                        self._connected = False
                        self._last_error = "微信登录已过期，请重新扫码"
                        await asyncio.sleep(30)
                        continue
                    raise RuntimeError(str(payload.get("errmsg") or f"iLink 错误 {ret}/{errcode}"))
                self._connected = True
                self._last_error = ""
                delay = 1.0
                new_sync = str(payload.get("get_updates_buf") or "")
                if new_sync:
                    sync_buf = new_sync
                    self._save_sync_buf(sync_buf)
                for raw in cast(list[Any], payload.get("msgs") or []):
                    if isinstance(raw, dict):
                        await self._handle_message(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._connected = False
                self._last_error = _safe_error(exc)
                logger.warning("[weixin] 轮询失败，%.0f 秒后重试: %s", delay, exc)
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(30.0, delay * 2)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        sender_id = str(message.get("from_user_id") or "").strip()
        message_id = str(message.get("message_id") or "").strip()
        if not sender_id or sender_id == self._account_id:
            return
        if message_id and self._deduper.seen(message_id):
            return
        room_id = str(message.get("room_id") or message.get("chat_room_id") or "").strip()
        is_group = bool(room_id)
        target_id = room_id or sender_id
        if is_group:
            if not self._config.groups_enabled:
                return
            if self._config.group_allow_from and target_id not in self._config.group_allow_from:
                return
        elif self._config.allow_from and sender_id not in self._config.allow_from:
            return
        text = _extract_text(cast(list[Any], message.get("item_list") or []))
        if not text:
            return
        chat_id = f"group:{target_id}" if is_group else f"dm:{sender_id}"
        context_token = str(message.get("context_token") or "").strip()
        if context_token:
            self._context_tokens[chat_id] = context_token
            self._persist_context_tokens()
        if text.lower() == "/stop":
            await self._interrupt(chat_id, sender_id)
            return
        ctx = self._require_ctx()
        await remember_channel_session(
            ctx.session_manager,
            session_key=f"{self.name}:{chat_id}",
            channel=self.name,
            chat_id=chat_id,
            sender_id=sender_id,
            title=f"微信{'群聊' if is_group else ''} · {target_id[-8:]}",
            chat_type="group" if is_group else "single",
            metadata={"weixin_account_id": self._account_id},
        )
        await ctx.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender=sender_id,
                chat_id=chat_id,
                content=text,
                metadata={"message_id": message_id, "chat_type": "group" if is_group else "single"},
            )
        )

    async def _interrupt(self, chat_id: str, sender_id: str) -> None:
        controller = self._require_ctx().interrupt_controller
        text = "当前没有运行中的回复。"
        if controller is not None:
            result = controller.request_interrupt(f"{self.name}:{chat_id}", sender_id, "/stop")
            text = result.message or text
        await self.send(chat_id, text)

    async def send(self, chat_id: str, message: str) -> None:
        _, separator, target_id = chat_id.partition(":")
        if not separator or not target_id:
            raise ValueError("微信 chat_id 必须是 dm:USERID 或 group:ROOMID")
        context_token = self._context_tokens.get(chat_id)
        for part in _split_message(message):
            body: dict[str, Any] = {
                "from_user_id": "",
                "to_user_id": target_id,
                "client_id": f"xiaoman-weixin-{uuid.uuid4().hex}",
                "message_type": 2,
                "message_state": 2,
                "item_list": [{"type": 1, "text_item": {"text": part}}],
            }
            if context_token:
                body["context_token"] = context_token
            response = await self._post(_SEND_MESSAGE, {"msg": body}, timeout_s=15, budget_s=20)
            ret = response.get("ret", 0)
            errcode = response.get("errcode", 0)
            if (ret == _SESSION_EXPIRED or errcode == _SESSION_EXPIRED) and context_token:
                context_token = None
                body.pop("context_token", None)
                response = await self._post(_SEND_MESSAGE, {"msg": body}, timeout_s=15, budget_s=20)
                ret = response.get("ret", 0)
                errcode = response.get("errcode", 0)
            if ret not in {0, None} or errcode not in {0, None}:
                raise RuntimeError(str(response.get("errmsg") or f"iLink 发送失败 {ret}/{errcode}"))
            await asyncio.sleep(0.3)

    async def _on_response(self, message: OutboundMessage) -> None:
        await self.send(message.chat_id, message.content)

    async def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout_s: float,
        budget_s: float,
    ) -> dict[str, Any]:
        body = json.dumps({**payload, "base_info": _BASE_INFO}, ensure_ascii=False, separators=(",", ":"))
        response = await self._require_ctx().http_resources.external_default.post(
            f"{self._base_url}/{endpoint}",
            content=body,
            headers=_headers(self._token, body),
            timeout_s=timeout_s,
            budget=RequestBudget(total_timeout_s=budget_s),
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def _state_path(self, name: str) -> Path:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        return self._data_dir / name

    def _load_sync_buf(self) -> str:
        try:
            return str(json.loads(self._state_path("sync.json").read_text(encoding="utf-8")).get("get_updates_buf") or "")
        except (OSError, ValueError, TypeError):
            return ""

    def _save_sync_buf(self, value: str) -> None:
        self._state_path("sync.json").write_text(json.dumps({"get_updates_buf": value}), encoding="utf-8")

    def _restore_context_tokens(self) -> None:
        try:
            raw = json.loads(self._state_path("context-tokens.json").read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._context_tokens = {str(key): str(value) for key, value in raw.items() if value}
        except (OSError, ValueError, TypeError):
            self._context_tokens = {}

    def _persist_context_tokens(self) -> None:
        self._state_path("context-tokens.json").write_text(
            json.dumps(self._context_tokens, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _require_ctx(self) -> ChannelContext:
        if self._ctx is None:
            raise RuntimeError("微信渠道尚未启动")
        return self._ctx


def _headers(token: str, body: str) -> dict[str, str]:
    uin = struct.unpack(">I", secrets.token_bytes(4))[0]
    headers = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode("utf-8"))),
        "X-WECHAT-UIN": base64.b64encode(str(uin).encode()).decode(),
        "iLink-App-Id": _APP_ID,
        "iLink-App-ClientVersion": _CLIENT_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _extract_text(items: list[Any]) -> str:
    for item in items:
        if isinstance(item, dict) and item.get("type") == 1:
            return str(cast(dict[str, Any], item.get("text_item") or {}).get("text") or "").strip()
    for item in items:
        if isinstance(item, dict) and item.get("type") == 4:
            text = str(cast(dict[str, Any], item.get("voice_item") or {}).get("text") or "").strip()
            if text:
                return text
    return ""


def _split_message(text: str, limit: int = 4000) -> list[str]:
    value = str(text or "").strip() or "（空回复）"
    return [value[index:index + limit] for index in range(0, len(value), limit)]


def _safe_error(exc: Exception) -> str:
    value = " ".join(str(exc).split()).strip()
    return value[:160] or type(exc).__name__
