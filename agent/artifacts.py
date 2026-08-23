from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


MAX_OUTBOUND_ARTIFACTS = 8
MAX_OUTBOUND_ARTIFACT_BYTES = 200 * 1024 * 1024

_DELIVERY_REQUEST_RE = re.compile(
    r"(?:发(?:送)?给我|传给我|给我(?:下载|文件|附件)|作为附件|附件发|发送(?:文件|附件)|"
    r"下载(?:链接|文件)|send\s+(?:me\s+)?(?:the\s+)?(?:file|attachment))",
    re.IGNORECASE,
)
_MARKDOWN_PATH_RE = re.compile(r"!?\[[^\]\r\n]*\]\(([^)\r\n]+)\)")
_BACKTICK_PATH_RE = re.compile(r"`([^`\r\n]+)`")
_WINDOWS_PATH_RE = re.compile(
    r"(?<![\w])([A-Za-z]:[\\/][^\r\n<>|?*\"\u3002\uff0c\uff1b\uff09\]\}]+)"
)
_POSIX_PATH_RE = re.compile(r"(?<![\w])(/[^\r\n<>|?*\"\u3002\uff0c\uff1b\uff09\]\}]+)")
_TRAILING_TEXT_RE = re.compile(
    r"(?:\s{2,}|\s+(?:这次|现在|内容|关于|你可以|请|已(?:经)?|文件|大小|共)\b).*$",
    re.IGNORECASE,
)


def requests_file_delivery(text: str) -> bool:
    """Return whether the current user explicitly asks to receive an artifact."""

    return bool(_DELIVERY_REQUEST_RE.search(str(text or "")))


def discover_outbound_artifacts(
    *,
    user_request: str,
    reply: str,
    tool_chain: Iterable[dict[str, Any]] = (),
    workspace: Path,
) -> list[str]:
    """Find generated files that should be attached to the current reply.

    Discovery is deliberately gated by an explicit delivery request and by the
    configured workspace boundary.  This keeps ordinary mentions of local files
    from becoming accidental outbound attachments.
    """

    if not requests_file_delivery(user_request):
        return []

    root = workspace.expanduser().resolve()
    candidates: list[str] = []
    candidates.extend(_extract_path_candidates(reply))
    delivered = _delivered_message_push_paths(tool_chain)
    for step in tool_chain:
        if not isinstance(step, dict):
            continue
        for call in step.get("calls") or ():
            if not isinstance(call, dict):
                continue
            if str(call.get("status") or "success").lower() in {
                "blocked",
                "error",
                "failed",
            }:
                continue
            arguments = call.get("arguments")
            if isinstance(arguments, dict):
                for key in ("file", "path", "output", "output_path"):
                    value = arguments.get(key)
                    if isinstance(value, str):
                        candidates.append(value)
            result = call.get("result")
            if isinstance(result, str):
                candidates.extend(_extract_path_candidates(result))

    artifacts: list[str] = []
    seen: set[str] = set()
    delivered_resolved = {
        str(path)
        for value in delivered
        if (path := _validated_artifact(value, root)) is not None
    }
    for value in candidates:
        path = _validated_artifact(value, root)
        if path is None:
            continue
        normalized = str(path)
        if normalized in seen or normalized in delivered_resolved:
            continue
        seen.add(normalized)
        artifacts.append(normalized)
        if len(artifacts) >= MAX_OUTBOUND_ARTIFACTS:
            break
    return artifacts


def _delivered_message_push_paths(
    tool_chain: Iterable[dict[str, Any]],
) -> set[str]:
    delivered: set[str] = set()
    for step in tool_chain:
        if not isinstance(step, dict):
            continue
        for call in step.get("calls") or ():
            if not isinstance(call, dict) or call.get("name") != "message_push":
                continue
            arguments = call.get("arguments")
            result = str(call.get("result") or "")
            if not isinstance(arguments, dict) or "已发送" not in result:
                continue
            value = arguments.get("file")
            if isinstance(value, str) and value.strip():
                delivered.add(value)
    return delivered


def _extract_path_candidates(text: str) -> list[str]:
    value = str(text or "")
    candidates = [match.group(1) for match in _MARKDOWN_PATH_RE.finditer(value)]
    candidates.extend(match.group(1) for match in _BACKTICK_PATH_RE.finditer(value))
    candidates.extend(match.group(1) for match in _WINDOWS_PATH_RE.finditer(value))
    candidates.extend(match.group(1) for match in _POSIX_PATH_RE.finditer(value))
    return candidates


def _validated_artifact(value: str, root: Path) -> Path | None:
    raw = str(value or "").strip().strip("'\"")
    raw = _TRAILING_TEXT_RE.sub("", raw).strip().rstrip(":;,.，。；）)]}")
    if raw.lower().startswith("file://"):
        raw = raw[7:]
    if not raw:
        return None
    try:
        path = Path(raw).expanduser().resolve()
        path.relative_to(root)
        if (
            not path.is_file()
            or path.stat().st_size <= 0
            or path.stat().st_size > MAX_OUTBOUND_ARTIFACT_BYTES
        ):
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return path
