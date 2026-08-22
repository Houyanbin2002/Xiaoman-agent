from __future__ import annotations

"""Conservative fallback candidates for explicit user directives.

The fallback is part of the background semantic batch.  It does not persist
anything itself; every candidate still passes through the canonical memory
governance service.
"""

import re
from collections.abc import Mapping, Sequence

from core.conversation_semantics.evidence import sanitize_text


_DIRECTIVE_RE = re.compile(
    r"(?:以后|默认|优先|我喜欢|我的默认|不要再|不要打扰|不要主动提醒|不要发到|规则|作废|取消|改为|改成|"
    r"从.+开始|按.+处理|已经结束|当前主要关注)",
    re.IGNORECASE,
)


def extract_explicit_candidates(
    messages: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    memory: list[dict[str, object]] = []
    for message in messages:
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        message_id = str(message.get("id") or "").strip()
        text = sanitize_text(message.get("content"), limit=2000)
        if not message_id or not text or not _DIRECTIVE_RE.search(text):
            continue
        correction = _correction_candidate(text, message_id)
        if correction is not None:
            memory.append(correction)
            continue
        candidate = _preference_candidate(text, message_id)
        if candidate is not None:
            memory.append(candidate)
    return {"memory_candidates": memory}


def _base(
    text: str,
    message_id: str,
    *,
    key: str,
    tag: str = "preference",
) -> dict[str, object]:
    return {
        "tag": tag,
        "content": text,
        "confidence": 0.98,
        "origin": "explicit_user" if tag == "preference" else "user_correction",
        "evidence_refs": [message_id],
        "source_message_id": message_id,
        "subject": "用户",
        "attributes": {"preference_key": key},
    }


def _preference_candidate(text: str, message_id: str) -> dict[str, object] | None:
    lowered = text.casefold()
    if "python" in lowered or "代码示例" in text:
        value = "Python" if "python" in lowered else text
        item = _base(text, message_id, key="code_language")
        item.update({"predicate": "代码示例语言", "value": value})
        return item
    if "asia/shanghai" in lowered or "时区" in text:
        value = "Asia/Shanghai" if "asia/shanghai" in lowered else text
        item = _base(text, message_id, key="timezone")
        item.update({"predicate": "默认时区", "value": value})
        return item
    if "先给结论" in text or "简短步骤" in text:
        item = _base(text, message_id, key="response_style")
        item.update({"predicate": "回复风格", "value": "先给结论，再给简短步骤"})
        return item
    if "markdown" in lowered or "表格" in text:
        item = _base(text, message_id, key="document_format")
        item.update(
            {
                "predicate": "技术方案格式",
                "value": "Markdown 表格" if "表格" in text else "Markdown",
            }
        )
        return item
    if ("提醒" in text or "打扰" in text) and ("不要" in text or "免打扰" in text):
        item = _base(text, message_id, key="notification_quiet_hours")
        item.update({"predicate": "主动提醒限制", "value": text})
        return item
    if ("当前对话" in text or "当前会话" in text) and ("群" in text or "外部" in text):
        item = _base(text, message_id, key="communication_channel")
        item.update(
            {
                "predicate": "工作任务回复渠道",
                "value": "当前对话" if "当前对话" in text else "当前会话",
            }
        )
        return item
    return None


def _correction_candidate(text: str, message_id: str) -> dict[str, object] | None:
    pairs = (
        (r"之前记的\s*(.+?)\s*偏好作废.*?改为\s*(.+?)[。.!！]?$", "code_language"),
        (r"不要再默认\s*(.+?)了.*?(?:用|改为)\s*(.+?)[。.!！]?$", "document_format"),
        (r"之前允许发(.+?)的规则取消.*?只在(.+?)处理", "communication_channel"),
        (r"免打扰时间从(.+?)改成(.+?)(?:开始)?[。.!！]?$", "notification_quiet_hours"),
        (r"(.+?)已经结束.*?当前主要关注\s*(.+?)[。.!！]?$", "active_project"),
        (r"回复不要再写得\s*(.+?)，默认控制在\s*(.+?)[。.!！]?$", "response_length"),
    )
    for pattern, key in pairs:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        old, new = (part.strip(" ，,。.!！\"") for part in match.groups())
        if not old or not new or old == new:
            continue
        item = _base(text, message_id, key=key, tag="correction")
        item.update({"value": new, "replaces": old, "predicate": key})
        return item
    return None


__all__ = ["extract_explicit_candidates"]
