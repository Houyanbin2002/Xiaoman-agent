from __future__ import annotations

import re
from typing import Any


MARKDOWN_MESSAGE_TYPE = 2
TEXT_MESSAGE_TYPE = 0


def build_message_body(
    content: str,
    *,
    markdown: bool,
    sequence: int,
    reply_to: str | None = None,
) -> dict[str, Any]:
    """Build the QQ OpenAPI payload for a text-like message."""
    if markdown:
        body: dict[str, Any] = {
            "markdown": {"content": content},
            "msg_type": MARKDOWN_MESSAGE_TYPE,
            "msg_seq": sequence,
        }
    else:
        body = {
            "content": strip_markdown(content),
            "msg_type": TEXT_MESSAGE_TYPE,
            "msg_seq": sequence,
        }
    if reply_to:
        body["msg_id"] = reply_to
    return body


def strip_markdown(content: str) -> str:
    """Keep a readable QQ reply when the account cannot send Markdown."""
    text = re.sub(r"```[^\n]*\n?", "", content)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1（\2）", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"~~(.+?)~~", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    return text.strip()
