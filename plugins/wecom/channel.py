from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote
from uuid import uuid4

from bus.events import InboundMessage, OutboundMessage
from infra.channels.base import MessageDeduper
from infra.channels.contract import ChannelContext
from infra.channels.session_identity import remember_channel_session

from .config import WeComConfig

logger = logging.getLogger(__name__)

_CHANNEL = "wecom"
_RECONNECT_MAX_SECONDS = 30.0


class WeComChannel:
    """Official WeCom intelligent-bot WebSocket channel."""

    name = _CHANNEL

    def __init__(self, config: WeComConfig) -> None:
        self._config = config
        self._ctx: ChannelContext | None = None
        self._runner: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._socket: Any = None
        self._send_lock = asyncio.Lock()
        self._connected = False
        self._last_error = ""
        self._message_deduper = MessageDeduper(1000)
        self._reply_req_ids: dict[str, str] = {}
        self._outbound_bound = False

    def status_snapshot(self) -> dict[str, object]:
        return {
            "connected": self._connected,
            "detail": "企业微信智能机器人长连接" if self._connected else self._last_error,
        }

    async def start(self, ctx: ChannelContext) -> None:
        self._ctx = ctx
        if not self._outbound_bound:
            ctx.bus.subscribe_outbound(self.name, self._on_response)
            self._outbound_bound = True
        ctx.push_tool.register_channel(self.name, text=self.send)
        self._stopping.clear()
        self._runner = asyncio.create_task(self._run(), name="channel:wecom")

    async def stop(self) -> None:
        self._stopping.set()
        self._connected = False
        if self._socket is not None:
            await self._socket.close()
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
                logger.warning("[wecom] 长连接中断，%.0f 秒后重连: %s", delay, exc)
            if self._stopping.is_set():
                return
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            delay = min(_RECONNECT_MAX_SECONDS, delay * 2)

    async def _connect_once(self) -> None:
        from websockets.asyncio.client import connect

        async with connect(self._config.websocket_url, open_timeout=10, close_timeout=5) as socket:
            self._socket = socket
            auth_id = _request_id("aibot_subscribe")
            await self._send_frame({
                "cmd": "aibot_subscribe",
                "headers": {"req_id": auth_id},
                "body": {"bot_id": self._config.bot_id, "secret": self._config.secret},
            })
            heartbeat = asyncio.create_task(self._heartbeat(), name="channel:wecom:heartbeat")
            try:
                async for raw in socket:
                    frame = cast(dict[str, Any], json.loads(raw))
                    req_id = str(cast(dict[str, Any], frame.get("headers") or {}).get("req_id") or "")
                    if req_id == auth_id:
                        if int(frame.get("errcode") or 0) != 0:
                            raise RuntimeError(str(frame.get("errmsg") or "企业微信凭据无效"))
                        self._connected = True
                        self._last_error = ""
                        continue
                    if frame.get("cmd") == "aibot_msg_callback":
                        await self._handle_message(frame)
                    elif frame.get("cmd") == "aibot_event_callback":
                        event = cast(dict[str, Any], cast(dict[str, Any], frame.get("body") or {}).get("event") or {})
                        if event.get("eventtype") == "disconnected_event":
                            raise RuntimeError("机器人已在另一个客户端建立连接")
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
                self._connected = False
                self._socket = None

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(30)
            await self._send_frame({
                "cmd": "ping",
                "headers": {"req_id": _request_id("ping")},
            })

    async def _handle_message(self, frame: dict[str, Any]) -> None:
        body = cast(dict[str, Any], frame.get("body") or {})
        message_id = str(body.get("msgid") or "")
        if not message_id or self._message_deduper.seen(message_id):
            return
        sender_id = str(cast(dict[str, Any], body.get("from") or {}).get("userid") or "")
        if self._config.allow_from and sender_id not in self._config.allow_from:
            logger.warning("[wecom] 拒绝未授权消息 sender=%s", sender_id)
            return
        is_group = str(body.get("chattype") or "single") == "group"
        raw_chat_id = str(body.get("chatid") or sender_id)
        chat_id = f"group:{raw_chat_id}" if is_group else f"single:{sender_id}"
        content, media = await self._extract_content(body)
        if content.lower() == "/stop":
            await self._interrupt(chat_id, sender_id, frame)
            return
        if not content and not media:
            return
        req_id = str(cast(dict[str, Any], frame.get("headers") or {}).get("req_id") or "")
        if req_id:
            self._reply_req_ids[chat_id] = req_id
        ctx = self._require_ctx()
        title = f"企业微信群聊 · {raw_chat_id[-8:]}" if is_group else f"企业微信 · {sender_id}"
        await remember_channel_session(
            ctx.session_manager,
            session_key=f"{self.name}:{chat_id}",
            channel=self.name,
            chat_id=chat_id,
            sender_id=sender_id,
            title=title,
            chat_type="group" if is_group else "single",
            metadata={"wecom_chat_id": raw_chat_id},
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
                "wecom_chat_id": raw_chat_id,
            },
        ))

    async def _extract_content(self, body: dict[str, Any]) -> tuple[str, list[str]]:
        msg_type = str(body.get("msgtype") or "")
        if msg_type == "text":
            content = str(cast(dict[str, Any], body.get("text") or {}).get("content") or "")
            return _with_quote(content, body), []
        if msg_type == "voice":
            content = str(cast(dict[str, Any], body.get("voice") or {}).get("content") or "")
            return _with_quote(content, body), []
        if msg_type == "mixed":
            text_parts: list[str] = []
            media: list[str] = []
            mixed = cast(dict[str, Any], body.get("mixed") or {})
            for item in cast(list[Any], mixed.get("msg_item") or []):
                if not isinstance(item, dict):
                    continue
                if item.get("msgtype") == "text":
                    text_parts.append(str(cast(dict[str, Any], item.get("text") or {}).get("content") or ""))
                elif item.get("msgtype") == "image":
                    path = await self._download_media(cast(dict[str, Any], item.get("image") or {}), "image")
                    if path:
                        media.append(path)
            return _with_quote("\n".join(text_parts), body), media
        if msg_type in {"image", "file", "video"}:
            path = await self._download_media(cast(dict[str, Any], body.get(msg_type) or {}), msg_type)
            return _with_quote(f"[{_media_label(msg_type)}]", body), [path] if path else []
        return "", []

    async def _download_media(self, media: dict[str, Any], media_type: str) -> str:
        url = str(media.get("url") or "")
        if not url.startswith(("http://", "https://")):
            return ""
        try:
            response = await self._require_ctx().http_resources.external_default.get(url, timeout_s=20)
            response.raise_for_status()
            data = response.content
            aes_key = str(media.get("aeskey") or "")
            if aes_key:
                data = _decrypt_media(data, aes_key)
            filename = _filename_from_headers(response.headers) or Path(url).name
            suffix = Path(filename).suffix
            if not suffix:
                content_type = str(response.headers.get("content-type") or "").split(";", 1)[0]
                suffix = mimetypes.guess_extension(content_type) or _media_suffix(media_type)
            path = self._require_ctx().attachment_store.write_bytes(
                data,
                prefix="wecom_",
                suffix=suffix[:16],
            )
            return str(path)
        except Exception as exc:
            logger.warning("[wecom] 媒体下载或解密失败: %s", exc)
            return ""

    async def _interrupt(self, chat_id: str, sender_id: str, frame: dict[str, Any]) -> None:
        req_id = str(cast(dict[str, Any], frame.get("headers") or {}).get("req_id") or "")
        if req_id:
            self._reply_req_ids[chat_id] = req_id
        controller = self._require_ctx().interrupt_controller
        text = "当前没有运行中的回复。"
        if controller is not None:
            result = controller.request_interrupt(f"{self.name}:{chat_id}", sender_id, "/stop")
            text = result.message or ("已停止当前回复。" if result.status == "interrupted" else text)
        await self.send(chat_id, text)

    async def send(self, chat_id: str, message: str) -> None:
        req_id = self._reply_req_ids.pop(chat_id, "")
        if req_id:
            frame = {
                "cmd": "aibot_respond_msg",
                "headers": {"req_id": req_id},
                "body": {
                    "msgtype": "stream",
                    "stream": {
                        "id": _request_id("stream"),
                        "content": _utf8_clip(message, 20000),
                        "finish": True,
                    },
                },
            }
        else:
            target = chat_id.partition(":")[2]
            if not target:
                raise ValueError("企业微信 chat_id 必须是 single:USERID 或 group:CHATID")
            frame = {
                "cmd": "aibot_send_msg",
                "headers": {"req_id": _request_id("aibot_send_msg")},
                "body": {
                    "chatid": target,
                    "msgtype": "markdown",
                    "markdown": {"content": _utf8_clip(message, 20000)},
                },
            }
        await self._send_frame(frame)

    async def _on_response(self, message: OutboundMessage) -> None:
        await self.send(message.chat_id, message.content)

    async def _send_frame(self, frame: dict[str, Any]) -> None:
        if self._socket is None:
            raise RuntimeError("企业微信长连接尚未就绪")
        async with self._send_lock:
            await self._socket.send(json.dumps(frame, ensure_ascii=False))

    def _require_ctx(self) -> ChannelContext:
        if self._ctx is None:
            raise RuntimeError("企业微信渠道尚未启动")
        return self._ctx


def _request_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _with_quote(content: str, body: dict[str, Any]) -> str:
    text = str(content or "").strip()
    quote = cast(dict[str, Any], body.get("quote") or {})
    quote_type = str(quote.get("msgtype") or "")
    quoted = ""
    if quote_type == "text":
        quoted = str(cast(dict[str, Any], quote.get("text") or {}).get("content") or "")
    elif quote_type:
        quoted = f"[{_media_label(quote_type)}]"
    if not quoted:
        return text
    return f"【引用消息】\n{quoted}\n\n【当前消息】\n{text}".strip()


def _decrypt_media(encrypted: bytes, aes_key: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    key = base64.b64decode(aes_key)
    if len(key) != 32:
        raise ValueError("企业微信媒体 AES Key 长度无效")
    decryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).decryptor()
    decrypted = decryptor.update(encrypted) + decryptor.finalize()
    if not decrypted:
        raise ValueError("企业微信媒体解密结果为空")
    padding = decrypted[-1]
    if padding < 1 or padding > 32 or decrypted[-padding:] != bytes([padding]) * padding:
        raise ValueError("企业微信媒体 PKCS#7 填充无效")
    return decrypted[:-padding]


def _filename_from_headers(headers: Any) -> str:
    disposition = str(headers.get("content-disposition") or "")
    for marker in ("filename*=UTF-8''", "filename="):
        if marker not in disposition:
            continue
        raw = disposition.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')
        return unquote(raw)
    return ""


def _media_suffix(media_type: str) -> str:
    return {"image": ".jpg", "video": ".mp4", "file": ".bin"}.get(media_type, ".bin")


def _media_label(media_type: str) -> str:
    return {"image": "图片", "video": "视频", "file": "文件", "voice": "语音"}.get(media_type, media_type)


def _utf8_clip(text: str, limit: int) -> str:
    raw = str(text or "").encode("utf-8")
    if len(raw) <= limit:
        return str(text or "")
    return raw[:limit].decode("utf-8", errors="ignore")


def _safe_error(exc: Exception) -> str:
    text = " ".join(str(exc).split()).strip()
    return text[:160] or type(exc).__name__
