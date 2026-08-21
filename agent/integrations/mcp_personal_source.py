from __future__ import annotations

import hashlib
import json
from typing import Any

from agent.tools.base import ToolResult
from core.personal.sources.models import ExternalSourceItem, ExternalSourceSubscription


class McpPersonalSourceAdapter:
    """Poll one explicitly selected MCP tool and map its result into personal facts.

    The adapter is intentionally provider-neutral.  A subscription declares the
    MCP tool, its read-only arguments and the response mapping; installing an MCP
    server alone never creates a subscription or starts polling it.
    """

    def __init__(self, tools: Any) -> None:
        self.tools = tools

    async def fetch(
        self,
        subscription: ExternalSourceSubscription,
    ) -> list[ExternalSourceItem]:
        mapping = subscription.mapping
        tool_name = str(mapping.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError("MCP 信号源缺少 tool_name")
        document = self.tools.get_document(tool_name)
        if (
            document is None
            or document.source_type != "mcp"
            or document.source_name != subscription.server_name
        ):
            raise ValueError(
                f"工具 {tool_name!r} 不属于 MCP server {subscription.server_name!r}"
            )
        arguments = mapping.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("MCP 信号源 arguments 必须是对象")
        raw = await self.tools.execute(tool_name, dict(arguments))
        text = raw.text if isinstance(raw, ToolResult) else str(raw)
        if text.startswith(("MCP error", "工具执行出错", "工具 '")):
            raise ValueError(text[:1000])
        payload = _decode_json(text)
        rows = _read_path(payload, str(mapping.get("items_path") or ""))
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            raise ValueError("items_path 没有指向对象列表")

        fields = mapping.get("fields") or {}
        data_mapping = mapping.get("data") or {}
        if not isinstance(fields, dict) or not isinstance(data_mapping, dict):
            raise ValueError("MCP 信号源 fields 和 data 必须是对象")
        result: list[ExternalSourceItem] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            title = _string_value(_resolve(row, fields.get("title", "title")))
            if not title:
                continue
            external_id = _string_value(_resolve(row, fields.get("id", "id")))
            if not external_id:
                external_id = _fallback_id(row, index)
            summary = _string_value(_resolve(row, fields.get("summary", "summary")))
            source_ref = (
                _string_value(_resolve(row, fields.get("source_ref", "source_ref")))
                or subscription.resource_url
            )
            data = {
                str(key): value
                for key, spec in data_mapping.items()
                if (value := _resolve(row, spec)) is not None
            }
            data.update(
                {
                    "external_server": subscription.server_name,
                    "external_tool": tool_name,
                }
            )
            result.append(
                ExternalSourceItem.build(
                    external_id=external_id,
                    title=title,
                    summary=summary or title,
                    data=data,
                    source_ref=source_ref,
                )
            )
        return result


def _decode_json(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    decoded: list[Any] = []
    offset = 0
    while offset < len(candidate):
        starts = [
            position
            for position in (candidate.find("{", offset), candidate.find("[", offset))
            if position >= 0
        ]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(candidate, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        decoded.append(value)
        offset = end
    if decoded:
        return decoded[-1]
    raise ValueError("MCP 工具没有返回可映射的 JSON 数据")


def _read_path(value: Any, path: str) -> Any:
    path = path.strip()
    if not path:
        return value
    if path.startswith("/"):
        parts = [
            item.replace("~1", "/").replace("~0", "~") for item in path[1:].split("/")
        ]
    else:
        parts = [item for item in path.split(".") if item]
    current = value
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _resolve(row: dict[str, Any], spec: Any) -> Any:
    if isinstance(spec, str):
        return _read_path(row, spec)
    if not isinstance(spec, dict):
        return spec
    if "const" in spec:
        return spec["const"]
    value = _read_path(row, str(spec.get("path") or ""))
    return spec.get("default") if value is None else value


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def _fallback_id(row: dict[str, Any], index: int) -> str:
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return f"row-{index}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


__all__ = ["McpPersonalSourceAdapter"]
