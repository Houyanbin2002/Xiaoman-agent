from __future__ import annotations

import base64
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from plugins.qqbot.channel import QQBotChannel, _message_path, _split_message as split_qq
from plugins.qqbot.config import QQBotConfig, QQBotGroupRule
from plugins.qqbot.message_format import build_message_body, strip_markdown
from plugins.wecom.channel import _decrypt_media
from plugins.weixin.channel import _extract_text, _headers, _split_message as split_weixin


def test_qq_official_routes_and_chunking() -> None:
    assert _message_path("c2c:user-openid") == "/v2/users/user-openid/messages"
    assert _message_path("group:group-openid") == "/v2/groups/group-openid/messages"
    assert split_qq("a" * 1801) == ["a" * 1800, "a"]
    with pytest.raises(ValueError, match="c2c"):
        _message_path("legacy:123")


def test_qq_official_markdown_payload_and_plain_text_fallback() -> None:
    markdown = build_message_body(
        "**加粗** 和 `代码`",
        markdown=True,
        sequence=1,
        reply_to="message-id",
    )
    assert markdown == {
        "markdown": {"content": "**加粗** 和 `代码`"},
        "msg_type": 2,
        "msg_seq": 1,
        "msg_id": "message-id",
    }
    assert strip_markdown("# 标题\n**加粗** 和 `代码`") == "标题\n加粗 和 代码"


def test_qq_official_allowlist_and_group_rules() -> None:
    rule = QQBotGroupRule(group_openid="g1", allow_from=["u1"])
    channel = QQBotChannel(
        QQBotConfig(
            app_id="app",
            client_secret="secret",
            allow_from=["owner"],
            groups=[rule],
        )
    )
    assert channel._is_allowed("u1", group_rule=rule, is_group=True) is True
    assert channel._is_allowed("u2", group_rule=rule, is_group=True) is False
    assert channel._is_allowed("owner", group_rule=None, is_group=False) is True
    assert channel._is_allowed("other", group_rule=None, is_group=False) is False


@pytest.mark.asyncio
async def test_qq_generated_file_uses_official_chunked_media_flow(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "答辩报告.pptx"
    file_path.write_bytes(b"generated-pptx")

    class _Response:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.status_code = 200
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class _Requester:
        def __init__(self) -> None:
            self.posts: list[tuple[str, dict[str, Any]]] = []
            self.puts: list[tuple[str, bytes]] = []

        async def post(self, url: str, **kwargs: Any) -> _Response:
            body = kwargs.get("json") or {}
            self.posts.append((url, body))
            if url.endswith("/upload_prepare"):
                return _Response(
                    {
                        "upload_id": "upload-1",
                        "block_size": "1048576",
                        "parts": [
                            {
                                "index": 0,
                                "block_size": "1048576",
                                "presigned_url": "https://upload.example/part-0",
                            }
                        ],
                    }
                )
            if url.endswith("/files"):
                return _Response({"file_info": "opaque-file-info"})
            return _Response({})

        async def request(
            self,
            method: str,
            url: str,
            **kwargs: Any,
        ) -> _Response:
            assert method == "PUT"
            self.puts.append((url, kwargs["content"]))
            return _Response({})

    requester = _Requester()
    channel = QQBotChannel(QQBotConfig(app_id="app", client_secret="secret"))
    channel._ctx = SimpleNamespace(
        http_resources=SimpleNamespace(external_default=requester)
    )
    channel._token = "token"
    channel._token_expires_at = time.monotonic() + 3600
    channel._last_message_id["c2c:user"] = "incoming-message"
    channel._last_message_at["c2c:user"] = time.monotonic()

    await channel.send_file("c2c:user", str(file_path))

    assert requester.puts == [("https://upload.example/part-0", b"generated-pptx")]
    urls = [url for url, _body in requester.posts]
    assert urls[-4:] == [
        "https://api.sgroup.qq.com/v2/users/user/upload_prepare",
        "https://api.sgroup.qq.com/v2/users/user/upload_part_finish",
        "https://api.sgroup.qq.com/v2/users/user/files",
        "https://api.sgroup.qq.com/v2/users/user/messages",
    ]
    message_body = requester.posts[-1][1]
    assert message_body == {
        "msg_type": 7,
        "media": {"file_info": "opaque-file-info"},
        "msg_seq": 1,
        "msg_id": "incoming-message",
    }


def test_wecom_media_uses_aes256_cbc_and_pkcs7_32() -> None:
    key = bytes(range(32))
    plaintext = b"xiaoman-wecom-media"
    padding = 32 - len(plaintext) % 32
    padded = plaintext + bytes([padding]) * padding
    encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    assert _decrypt_media(encrypted, base64.b64encode(key).decode()) == plaintext


def test_weixin_protocol_headers_text_and_chunking() -> None:
    body = "{}"
    headers = _headers("token", body)
    assert headers["Authorization"] == "Bearer token"
    assert headers["AuthorizationType"] == "ilink_bot_token"
    assert base64.b64decode(headers["X-WECHAT-UIN"]).decode().isdigit()
    assert _extract_text([{"type": 1, "text_item": {"text": "你好"}}]) == "你好"
    assert _extract_text([{"type": 4, "voice_item": {"text": "语音文字"}}]) == "语音文字"
    assert split_weixin("你" * 4001) == ["你" * 4000, "你"]
