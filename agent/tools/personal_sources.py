from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote, urlparse

from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from core.personal.models import PersonalEntityType
from core.personal.sources.service import ExternalSourceSyncService


class PersonalSourceTool(Tool):
    """Manage user-selected external subscriptions separately from installation."""

    def __init__(
        self,
        service: ExternalSourceSyncService,
        tools: ToolRegistry,
    ) -> None:
        self._service = service
        self._tools = tools

    @property
    def name(self) -> str:
        return "personal_source"

    @property
    def description(self) -> str:
        return (
            "管理主动协助使用的外部信号源。安装 MCP 不会自动读取数据；"
            "只有用户明确要求持续关注某个 MCP 资源或 RSS 地址时才创建订阅。"
            "可列出、创建、启停、立即同步或删除订阅。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "enable", "disable", "sync", "delete"],
                },
                "subscription_id": {"type": "string"},
                "provider": {
                    "type": "string",
                    "enum": ["mcp", "rss"],
                    "description": "信号源类型；普通 RSS/Atom 或 X 的 RSS 地址使用 rss",
                },
                "server_name": {"type": "string"},
                "name": {"type": "string"},
                "resource_url": {
                    "type": "string",
                    "description": "RSS/Atom 的完整 http/https 地址",
                },
                "resource_key": {
                    "type": "string",
                    "description": "该订阅的稳定标识，如 important-unread 或 daily-notes",
                },
                "entity_type": {
                    "type": "string",
                    "enum": [item.value for item in PersonalEntityType],
                },
                "tool_name": {
                    "type": "string",
                    "description": "已连接 MCP 的完整工具名",
                },
                "arguments": {"type": "object"},
                "items_path": {
                    "type": "string",
                    "description": "结果中列表的点路径或 JSON Pointer；根列表留空",
                },
                "fields": {
                    "type": "object",
                    "description": "id/title/summary/source_ref 到结果字段路径的映射",
                },
                "data": {
                    "type": "object",
                    "description": "个人事实 data 字段映射；支持路径或 {path/default/const}",
                },
                "mapping": {
                    "type": "object",
                    "description": "RSS 行为配置，如 domain、notify_initial、valid_for_minutes",
                },
                "poll_interval_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1440,
                },
                "sync_now": {"type": "boolean"},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        subscription_id: str = "",
        provider: str = "mcp",
        server_name: str = "",
        name: str = "",
        resource_url: str = "",
        resource_key: str = "",
        entity_type: str = PersonalEntityType.COMMITMENT.value,
        tool_name: str = "",
        arguments: dict[str, Any] | None = None,
        items_path: str = "",
        fields: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        mapping: dict[str, Any] | None = None,
        poll_interval_minutes: int = 15,
        sync_now: bool = True,
        **_: Any,
    ) -> str:
        if action == "list":
            return json.dumps(
                [item.to_dict() for item in self._service.store.list_subscriptions()],
                ensure_ascii=False,
            )
        if action == "create":
            return await self._create(
                provider=provider,
                server_name=server_name,
                name=name,
                resource_url=resource_url,
                resource_key=resource_key,
                entity_type=entity_type,
                tool_name=tool_name,
                arguments=arguments or {},
                items_path=items_path,
                fields=fields or {},
                data=data or {},
                mapping=mapping or {},
                poll_interval_minutes=poll_interval_minutes,
                sync_now=sync_now,
            )
        if not subscription_id:
            return "错误：该操作需要 subscription_id"
        if action in {"enable", "disable"}:
            item = self._service.store.update_subscription(
                subscription_id,
                changes={"enabled": action == "enable"},
            )
            return json.dumps(item.to_dict(), ensure_ascii=False)
        if action == "sync":
            return json.dumps(
                (await self._service.sync_one(subscription_id)).to_dict(),
                ensure_ascii=False,
            )
        if action == "delete":
            deleted = self._service.store.delete_subscription(subscription_id)
            return "已删除信号源" if deleted else "错误：信号源不存在"
        return f"错误：不支持的 action {action!r}"

    async def _create(
        self,
        *,
        provider: str,
        server_name: str,
        name: str,
        resource_url: str,
        resource_key: str,
        entity_type: str,
        tool_name: str,
        arguments: dict[str, Any],
        items_path: str,
        fields: dict[str, Any],
        data: dict[str, Any],
        mapping: dict[str, Any],
        poll_interval_minutes: int,
        sync_now: bool,
    ) -> str:
        resolved_provider = provider.strip().lower() or "mcp"
        if resolved_provider == "rss":
            return await self._create_rss(
                name=name,
                resource_url=resource_url or resource_key,
                mapping=mapping,
                poll_interval_minutes=poll_interval_minutes,
                sync_now=sync_now,
            )
        if resolved_provider != "mcp":
            return f"错误：不支持的信号源类型 {provider!r}"

        document = self._tools.get_document(tool_name)
        if (
            document is None
            or document.source_type != "mcp"
            or document.source_name != server_name
        ):
            return f"错误：工具 {tool_name!r} 不属于 MCP server {server_name!r}"
        if not name.strip() or not resource_key.strip():
            return "错误：name 和 resource_key 不能为空"
        try:
            record_type = PersonalEntityType(entity_type)
        except ValueError:
            return f"错误：不支持的个人事实类型 {entity_type!r}"
        subscription = self._service.store.create_subscription(
            provider="mcp",
            server_name=server_name,
            name=name,
            resource_url=(
                f"mcp://{quote(server_name, safe='')}/{quote(resource_key, safe='')}"
            ),
            entity_type=record_type,
            mapping={
                "tool_name": tool_name,
                "arguments": arguments,
                "items_path": items_path,
                "fields": fields,
                "data": data,
            },
            poll_interval_minutes=poll_interval_minutes,
            enabled=True,
        )
        payload: dict[str, Any] = {"subscription": subscription.to_dict()}
        if sync_now:
            payload["sync"] = (await self._service.sync_one(subscription.id)).to_dict()
        return json.dumps(payload, ensure_ascii=False)

    async def _create_rss(
        self,
        *,
        name: str,
        resource_url: str,
        mapping: dict[str, Any],
        poll_interval_minutes: int,
        sync_now: bool,
    ) -> str:
        parsed = urlparse(resource_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "错误：RSS 地址必须是有效的 http/https URL"
        if not name.strip():
            return "错误：name 不能为空"
        rss_mapping = {
            "domain": str(mapping.get("domain") or "interest"),
            "notify_initial": bool(mapping.get("notify_initial", False)),
            "valid_for_minutes": mapping.get("valid_for_minutes", 1440),
            "max_items": mapping.get("max_items", 50),
        }
        subscription = self._service.store.create_subscription(
            provider="rss",
            server_name="rss",
            name=name.strip(),
            resource_url=resource_url.strip(),
            entity_type=PersonalEntityType.MONITOR_OBSERVATION,
            mapping=rss_mapping,
            poll_interval_minutes=poll_interval_minutes,
            enabled=True,
        )
        payload: dict[str, Any] = {"subscription": subscription.to_dict()}
        if sync_now:
            payload["sync"] = (await self._service.sync_one(subscription.id)).to_dict()
        return json.dumps(payload, ensure_ascii=False)


__all__ = ["PersonalSourceTool"]
