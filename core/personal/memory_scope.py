"""Shared, conservative memory boundaries. Never infer a scene from prose."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping


def _normalize(value: object) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).casefold().split()
    ).rstrip("。.!！")


def memory_subject(value: object) -> str:
    text = _normalize(value)
    return "用户" if text in {"", "用户", "我", "本人", "user", "self"} else text


def memory_scope(value: object) -> str:
    text = _normalize(value)
    aliases = {
        "": "",
        "global": "",
        "general": "",
        "通用": "",
        "全局": "",
        "默认": "",
        "通用默认": "",
        "work": "work",
        "工作": "work",
        "工作时": "work",
        "工作场景": "work",
        "casual": "casual",
        "闲聊": "casual",
        "闲聊时": "casual",
        "日常闲聊": "casual",
    }
    # Free-form project/scene scopes are preserved, not merged by similarity.
    return aliases.get(text, text)


def memory_boundary(subject: object, scope: object) -> tuple[str, str]:
    return memory_subject(subject), memory_scope(scope)


def preference_slot(raw: Mapping[str, object], attributes: Mapping[str, object]) -> str:
    """Shared candidate/write identity; normalization never creates a candidate."""
    supplied = str(attributes.get("preference_key") or "").strip().lower()
    if supplied and re.fullmatch(r"[a-z][a-z0-9_]{1,63}", supplied):
        return supplied
    if str(raw.get("tag") or "").strip().lower() not in {"preference", "correction"}:
        return ""
    text = " ".join(
        str(raw.get(name) or "")
        for name in ("content", "predicate", "value", "replaces")
    ).casefold()
    aliases = (
        ("code_language", ("python", "javascript", "代码示例", "编程语言")),
        ("timezone", ("asia/shanghai", "时区")),
        ("response_style", ("先给结论", "简短步骤", "回复风格")),
        ("response_length", ("三段以内", "回复长度", "写得很长")),
        ("document_format", ("markdown", "表格", "分点说明", "方案格式")),
        (
            "notification_quiet_hours",
            ("免打扰", "不要主动提醒", "提醒限制", "晚上九点", "晚上十点"),
        ),
        ("communication_channel", ("当前对话", "当前会话", "外部群", "发群")),
        ("active_project", ("当前主要关注", "旧项目", "xiaoman 项目")),
    )
    return next(
        (key for key, needles in aliases if any(needle in text for needle in needles)),
        "",
    )


def preference_record_key(slot: str, subject: object, scope: object) -> str:
    boundary = memory_boundary(subject, scope)
    base = f"memory:preference:{slot}"
    if boundary == ("用户", ""):
        return base  # Preserve existing global user preference identity.
    digest = hashlib.sha256(
        json.dumps(boundary, ensure_ascii=False).encode()
    ).hexdigest()[:24]
    return f"{base}:boundary:{digest}"


def render_memory_content(data: Mapping[str, object], fallback: str = "") -> str:
    content = str(data.get("content") or fallback).strip()
    if not content:
        return ""
    subject, scope = memory_boundary(data.get("subject"), data.get("scope"))
    labels = []
    if subject != "用户":
        labels.append(f"主体：{subject}")
    if scope:
        scope_label = {"work": "工作", "casual": "闲聊"}.get(scope, scope)
        labels.append(f"仅适用：{scope_label}")
    elif data.get("kind") == "preference":
        labels.append("通用默认")
    return f"[{'；'.join(labels)}] {content}" if labels else content
