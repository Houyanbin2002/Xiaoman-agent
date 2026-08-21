from __future__ import annotations

import base64

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
