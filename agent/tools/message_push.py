"""
统一消息推送工具，agent 通过 channel + chat_id 向任意已注册渠道发送消息、文件或图片。
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent.tools.base import Tool
from bus.queue import ChatLane

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PushDeliveryResult:
    success: bool
    message: str


class MessagePushTool(Tool):
    name = "message_push"
    description = (
        "向指定渠道的用户主动发送消息、文件或图片。"
        "需要提供渠道名（如 telegram、qqbot、weixin、wecom）和目标 chat_id。"
        "message/file/image 三者至少提供一个。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "目标渠道名，如 telegram、qqbot、weixin、wecom",
            },
            "chat_id": {
                "type": "string",
                "description": "目标会话 ID",
            },
            "message": {
                "type": "string",
                "description": "要发送的文本内容（可与 file/image 同时提供）",
            },
            "file": {
                "type": "string",
                "description": "要发送的文件本地路径，例如 /tmp/report.pdf",
            },
            "image": {
                "type": "string",
                "description": "要发送的图片本地路径或 URL",
            },
        },
        "required": ["channel", "chat_id"],
    }

    def __init__(self, chat_lane: ChatLane | None = None) -> None:
        # channel -> {type: sender_fn}
        self._senders: dict[str, dict[str, Callable[..., Awaitable[None]]]] = {}
        self._chat_lane = chat_lane

    def register_channel(
        self,
        channel: str,
        text: Callable[[str, str], Awaitable[None]] | None = None,
        stream_text: Callable[[str, str], Awaitable[None]] | None = None,
        file: Callable[[str, str, str | None], Awaitable[None]] | None = None,
        image: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        """注册渠道的各类 sender。
        - text(chat_id, message)
        - stream_text(chat_id, message)
        - file(chat_id, file_path, name=None)
        - image(chat_id, image_path_or_url)
        """
        self._senders[channel] = {}
        if text:
            self._senders[channel]["text"] = text
        if stream_text:
            self._senders[channel]["stream_text"] = stream_text
        if file:
            self._senders[channel]["file"] = file
        if image:
            self._senders[channel]["image"] = image
        logger.debug(
            f"message_push: 注册渠道 {channel!r}  支持: {list(self._senders[channel])}"
        )

    async def execute(self, **kwargs: Any) -> str:
        result = await self.send(
            channel=str(kwargs["channel"]),
            chat_id=str(kwargs["chat_id"]),
            message=kwargs.get("message"),
            file=kwargs.get("file"),
            image=kwargs.get("image"),
            commit_role=str(kwargs.get("_commit_role") or "").strip(),
        )
        return result.message

    async def send(
        self,
        *,
        channel: str,
        chat_id: str,
        message: str | None = None,
        file: str | None = None,
        image: str | None = None,
        commit_role: str = "",
    ) -> PushDeliveryResult:
        """Send content and return a typed result for deterministic callers."""

        if not message and not file and not image:
            return PushDeliveryResult(False, "错误：message、file、image 至少提供一个")

        senders = self._senders.get(channel)
        if senders is None:
            return PushDeliveryResult(
                False,
                f"渠道 {channel!r} 未注册，可用渠道：{list(self._senders) or ['（无）']}",
            )

        async def _send() -> PushDeliveryResult:
            return await self._send_now(
                channel=channel,
                chat_id=chat_id,
                message=message,
                file=file,
                image=image,
                senders=senders,
            )

        if self._chat_lane is not None and commit_role != "passive":
            return await self._chat_lane.run_non_passive(channel, chat_id, _send)
        return await _send()

    async def _send_now(
        self,
        *,
        channel: str,
        chat_id: str,
        message: str | None,
        file: str | None,
        image: str | None,
        senders: dict[str, Callable[..., Awaitable[None]]],
    ) -> PushDeliveryResult:

        results: list[str] = []
        errors: list[str] = []
        try:
            if message:
                sender_name = "stream_text" if "stream_text" in senders else "text"
                if sender_name not in senders:
                    errors.append(f"渠道 {channel!r} 不支持发送文本")
                else:
                    await senders[sender_name](chat_id, message)
                    preview = message[:60] + "..." if len(message) > 60 else message
                    logger.info(f"[message_push] {channel}:{chat_id} ← text: {preview!r}")
                    results.append("文本已发送")

            if file:
                if "file" not in senders:
                    errors.append(f"渠道 {channel!r} 不支持发送文件")
                else:
                    import os

                    name = os.path.basename(file)
                    await senders["file"](chat_id, file, name)
                    logger.info(f"[message_push] {channel}:{chat_id} ← file: {file!r}")
                    results.append(f"文件 {name!r} 已发送")

            if image:
                if "image" not in senders:
                    errors.append(f"渠道 {channel!r} 不支持发送图片")
                else:
                    await senders["image"](chat_id, image)
                    logger.info(
                        f"[message_push] {channel}:{chat_id} ← image: {image!r}"
                    )
                    results.append("图片已发送")

        except Exception as e:
            log = logger.info if "未连接" in str(e) else logger.error
            log(f"[message_push] 发送失败 {channel}:{chat_id}: {e}")
            return PushDeliveryResult(False, f"发送失败：{e}")

        if errors:
            detail = "；".join([*results, *errors])
            return PushDeliveryResult(False, detail)
        if results:
            return PushDeliveryResult(True, "；".join(results))
        return PushDeliveryResult(False, f"渠道 {channel!r} 没有可用的 sender")
