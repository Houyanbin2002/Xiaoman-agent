from __future__ import annotations

import json
from typing import Any

from agent.tools.base import ToolResult
from core.personal.sources.models import ExternalSourceItem, ExternalSourceSubscription


class NotionPersonalSourceAdapter:
    """Read one Notion data source through the already-connected MCP registry."""

    def __init__(self, tools: Any) -> None:
        self.tools = tools

    async def fetch(
        self,
        subscription: ExternalSourceSubscription,
    ) -> list[ExternalSourceItem]:
        if not subscription.resource_url.startswith("collection://"):
            raise ValueError("Notion 数据源地址必须以 collection:// 开头")
        tool_name = f"mcp_{subscription.server_name}__notion-query-data-sources"
        resource = subscription.resource_url.replace('"', '""')
        raw = await self.tools.execute(
            tool_name,
            {
                "data": {
                    "mode": "sql",
                    "data_source_urls": [subscription.resource_url],
                    "query": f'SELECT * FROM "{resource}" LIMIT 500',
                }
            },
        )
        payload = self._payload(raw)
        rows = payload.get("results")
        if not isinstance(rows, list):
            raise ValueError("Notion MCP 没有返回可识别的 results 列表")
        result: list[ExternalSourceItem] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = self._map_row(row, subscription.mapping)
            if item is not None:
                result.append(item)
        return result

    @staticmethod
    def _payload(raw: Any) -> dict[str, Any]:
        text = raw.text if isinstance(raw, ToolResult) else str(raw)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Notion MCP 返回的内容不是 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Notion MCP 返回的内容不是对象")
        return payload

    @staticmethod
    def _map_row(
        row: dict[str, Any],
        mapping: dict[str, Any],
    ) -> ExternalSourceItem | None:
        def value(key: str, default: str = "") -> Any:
            column = str(mapping.get(key) or default)
            return row.get(column) if column else None

        archived_values = {str(item) for item in mapping.get("archived_values", ["__YES__", "true", "1"])}
        if str(value("archive", "存档")) in archived_values:
            return None
        external_id = str(row.get("id") or "").strip()
        title = str(value("title", "任务") or "").strip()
        if not external_id or not title:
            return None
        status = str(value("status", "状态") or "").strip()
        completed = {str(item) for item in mapping.get("completed_values", ["已完成", "completed", "done"])}
        priority_raw = str(value("priority", "优先级别") or "").strip()
        priority = {
            "高优先级": "high",
            "high": "high",
            "urgent": "urgent",
            "低优先级": "low",
            "最低优先级": "low",
            "low": "low",
        }.get(priority_raw, "normal")
        summary = str(value("summary", "备注") or "").strip()
        due_at = value("due", "date:到期:start")
        category = str(value("category", "类别") or "").strip()
        data: dict[str, Any] = {
            "state": "completed" if status in completed else "open",
            "priority": priority,
            "next_action": summary or title,
            "external_status": status,
            "external_category": category,
        }
        if due_at:
            data["due_at"] = str(due_at)
        return ExternalSourceItem.build(
            external_id=external_id,
            title=title,
            summary=summary or title,
            data=data,
            source_ref=str(row.get("url") or ""),
        )
