"""Agent-facing MCP discovery and connection management tools."""

import json
from typing import Any

from agent.mcp.catalog import list_catalog
from agent.mcp.registry import McpServerRegistry
from agent.tools.base import Tool


class McpAddTool(Tool):
    """连接并注册本地或远程 MCP server。"""

    def __init__(self, registry: McpServerRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "mcp_add"

    @property
    def description(self) -> str:
        return (
            "安装配置一个 MCP 连接，但不会自动把它变成主动信号源。"
            "本地服务使用 stdio + command；远程服务使用 streamable_http/SSE + URL，"
            "可声明 OAuth。OAuth 需要用户随后在 MCP 页面完成浏览器授权。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "给这个 MCP server 起一个唯一短名称，如 'calendar'",
                },
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "启动命令列表，如 ['python', '/home/user/.xiaoman/mcp/calendar-mcp/run_server.py']",
                },
                "transport": {
                    "type": "string",
                    "enum": ["stdio", "streamable_http", "sse"],
                    "description": "连接方式，默认 stdio",
                },
                "url": {
                    "type": "string",
                    "description": "远程 MCP 的 HTTPS 地址",
                },
                "auth_type": {
                    "type": "string",
                    "enum": ["none", "oauth"],
                    "description": "模型侧可安全配置无需认证或 OAuth；密钥应在界面填写",
                },
                "scopes": {
                    "type": "string",
                    "description": "OAuth scope，留空则由服务发现",
                },
                "cwd": {
                    "type": "string",
                    "description": "本地 MCP 可选工作目录",
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "可选的额外环境变量，如 {'GOOGLE_CLIENT_ID': 'xxx'}",
                },
            },
            "required": ["name"],
        }

    async def execute(
        self,
        name: str,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        transport: str = "stdio",
        url: str = "",
        auth_type: str = "none",
        scopes: str = "",
        cwd: str | None = None,
        **_: Any,
    ) -> str:
        if transport == "stdio":
            if not command:
                return "错误：stdio MCP 需要 command"
            return await self._registry.add(name, command, env, cwd)
        if not url:
            return "错误：远程 MCP 需要 url"
        return await self._registry.add_remote(
            name,
            url=url,
            transport=transport,
            auth_type=auth_type,
            scopes=scopes,
        )


class McpRemoveTool(Tool):
    """注销并断开一个已注册的 MCP server。"""

    def __init__(self, registry: McpServerRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "mcp_remove"

    @property
    def description(self) -> str:
        return "注销并断开一个已注册的 MCP server，同时移除其所有工具。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要注销的 MCP server 名称",
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str, **_: Any) -> str:
        return await self._registry.remove(name)


class McpListTool(Tool):
    """列出当前所有已注册的 MCP server 及其工具。"""

    def __init__(self, registry: McpServerRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "mcp_list"

    @property
    def description(self) -> str:
        return "列出当前所有已注册的 MCP server 及其工具名称。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **_: Any) -> str:
        return self._registry.list_servers()


class McpCatalogTool(Tool):
    """Let the agent discover maintained MCP presets before configuring one."""

    def __init__(self, registry: McpServerRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "mcp_catalog"

    @property
    def description(self) -> str:
        return (
            "查询小满维护的标准 MCP 推荐目录及其真实配置要求。"
            "先用它确认远程地址、启动命令和是否需要用户授权，再决定如何接入。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(
        self,
        **_: Any,
    ) -> str:
        installed = {str(item["name"]) for item in self._registry.snapshot()}
        return json.dumps(list_catalog(installed), ensure_ascii=False)
