"""Unified MCP registry for local stdio and remote HTTP transports."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from agent.mcp.client import McpClient, McpClientProtocol
from agent.mcp.config import McpServerConfig
from agent.mcp.remote_client import OAuthInteractionRequired, RemoteMcpClient
from agent.mcp.secrets import McpSecretStore, redact_exception
from agent.mcp.tool import McpToolWrapper
from agent.plugins.manager import ActivePluginInfo
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class McpServerRegistry:
    """Own MCP configuration, credentials, connections and registered tools."""

    def __init__(
        self,
        config_path: Path,
        tool_registry: ToolRegistry,
        *,
        secret_store: McpSecretStore | None = None,
    ) -> None:
        self._config_path = config_path
        self._tool_registry = tool_registry
        self._secrets = secret_store or McpSecretStore(config_path)
        self._manual_configs: dict[str, McpServerConfig] = {}
        self._plugin_configs: dict[str, McpServerConfig] = {}
        self._clients: dict[str, McpClientProtocol] = {}
        self._pending_clients: dict[str, RemoteMcpClient] = {}
        self._pending_tasks: dict[str, asyncio.Task[None]] = {}
        self._server_tools: dict[str, list[str]] = {}
        self._states: dict[str, str] = {}
        self._errors: dict[str, str] = {}
        self._connect_task: asyncio.Task[None] | None = None
        self._oauth_callback_base_url = "http://127.0.0.1:2236"

    def set_oauth_callback_base_url(self, base_url: str) -> None:
        self._oauth_callback_base_url = base_url.rstrip("/")

    async def load_and_connect_all(self) -> None:
        migrated = False
        loaded: dict[str, McpServerConfig] = {}
        for name, raw in self._load_raw_configs().items():
            if not isinstance(raw, dict):
                continue
            try:
                config = McpServerConfig.from_raw(raw)
                config, changed = await self._secure_inline_secrets(name, config, raw)
                migrated = migrated or changed
                loaded[name] = config
            except Exception as exc:
                self._states[name] = "error"
                self._errors[name] = redact_exception(exc)
                logger.warning("[mcp] 配置无效 (%s): %s", name, exc)
        self._manual_configs = loaded
        if migrated:
            self._save()
        await asyncio.gather(
            *(self._connect_available(name, config) for name, config in loaded.items())
        )

    def start_connect_all_background(self) -> None:
        if self._connect_task is None or self._connect_task.done():
            self._connect_task = asyncio.create_task(
                self.load_and_connect_all(),
                name="mcp_connect_all",
            )

    async def shutdown(self) -> None:
        if self._connect_task is not None and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
        pending = list(self._pending_tasks.values())
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        clients = list(self._clients.values()) + list(self._pending_clients.values())
        self._clients.clear()
        self._pending_clients.clear()
        self._pending_tasks.clear()
        self._server_tools.clear()
        await asyncio.gather(
            *(client.disconnect() for client in clients),
            return_exceptions=True,
        )

    async def sync_plugin_servers(
        self,
        active_plugins: list[ActivePluginInfo],
    ) -> None:
        desired: dict[str, McpServerConfig] = {}
        for plugin in active_plugins:
            for server_name, raw in plugin.mcp_servers.items():
                if server_name in desired:
                    logger.warning(
                        "[mcp] 插件 MCP server 名称冲突，保留第一项: %s", server_name
                    )
                    continue
                if server_name in self._manual_configs:
                    logger.warning(
                        "[mcp] 插件 MCP server 与手工连接冲突，保留手工连接: %s",
                        server_name,
                    )
                    continue
                try:
                    desired[server_name] = McpServerConfig.from_raw(raw)
                except ValueError as exc:
                    self._states[server_name] = "error"
                    self._errors[server_name] = redact_exception(exc)
                    logger.warning("[mcp] 插件 MCP 配置无效 (%s): %s", server_name, exc)

        for name in sorted(set(self._plugin_configs) - set(desired)):
            await self._disconnect_server(name)
            self._plugin_configs.pop(name, None)
            self._states.pop(name, None)
            self._errors.pop(name, None)

        for name, config in desired.items():
            if self._plugin_configs.get(name) == config and self._states.get(name) in {
                "connected",
                "authorization_required",
                "authorizing",
            }:
                continue
            if name in self._plugin_configs:
                await self._disconnect_server(name)
            self._plugin_configs[name] = config
            await self._connect_available(name, config)

    async def add(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        return await self.add_config(
            name,
            McpServerConfig.from_raw(
                {
                    "transport": "stdio",
                    "command": command,
                    "env_keys": sorted((env or {}).keys()),
                    "cwd": cwd,
                }
            ),
            env=env or {},
        )

    async def add_remote(
        self,
        name: str,
        *,
        url: str,
        transport: str = "streamable_http",
        auth_type: str = "none",
        scopes: str = "",
        bearer_token: str = "",
        headers: dict[str, str] | None = None,
        oauth_client_id: str = "",
        oauth_client_secret: str = "",
    ) -> str:
        header_values = headers or {}
        config = McpServerConfig.from_raw(
            {
                "transport": transport,
                "url": url,
                "auth": {"type": auth_type, "scopes": scopes},
                "header_names": sorted(header_values),
            }
        )
        if name in self._manual_configs or name in self._plugin_configs:
            return f"MCP server {name!r} 已存在。如需更新，请先移除再重新添加。"
        if oauth_client_id:
            await self._secrets.set_oauth_client(
                name,
                client_id=oauth_client_id,
                client_secret=oauth_client_secret,
                redirect_uri=self._callback_url(name),
                scope=scopes,
            )
        return await self.add_config(
            name,
            config,
            headers=header_values,
            bearer_token=bearer_token,
        )

    async def add_config(
        self,
        name: str,
        config: McpServerConfig,
        *,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        bearer_token: str = "",
    ) -> str:
        if name in self._manual_configs or name in self._plugin_configs:
            return f"MCP server {name!r} 已存在。如需更新，请先移除再重新添加。"
        if env:
            await self._secrets.set_bundle(name, "env", env)
        if headers:
            await self._secrets.set_bundle(name, "headers", headers)
        if bearer_token:
            await self._secrets.set_text(name, "bearer", bearer_token)
        self._manual_configs[name] = config
        self._save()
        if config.auth_type == "oauth":
            self._states[name] = "authorization_required"
            return f"已添加 MCP server {name!r}，需要完成 OAuth 授权后才能使用。"
        try:
            await self._connect_available(name, config)
        except Exception as exc:
            return f"连接 MCP server {name!r} 失败：{redact_exception(exc)}"
        if self._states.get(name) != "connected":
            return (
                f"连接 MCP server {name!r} 失败：{self._errors.get(name, '未知错误')}"
            )
        tool_names = self._server_tools.get(name, [])
        return (
            f"已连接 MCP server {name!r}，注册了 {len(tool_names)} 个工具：\n"
            + "\n".join(f"- {item}" for item in tool_names)
        )

    async def begin_oauth(self, name: str) -> str:
        config = self._config_for(name)
        if config is None:
            raise ValueError(f"MCP server {name!r} 不存在")
        if config.auth_type != "oauth" or not config.is_remote:
            raise ValueError("该 MCP server 未配置 OAuth")
        await self._disconnect_server(name)
        await self._secrets.delete(name, "oauth_tokens")
        if not await self._secrets.has_static_oauth_client(name):
            await self._secrets.delete(name, "oauth_client")
        client = RemoteMcpClient(
            name=name,
            config=config,
            secrets=self._secrets,
            callback_url=self._callback_url(name),
            interactive_oauth=True,
        )
        self._pending_clients[name] = client
        self._states[name] = "authorizing"
        self._errors.pop(name, None)
        task = asyncio.create_task(
            self._finish_connect(name, client),
            name=f"mcp_oauth_connect:{name}",
        )
        self._pending_tasks[name] = task
        try:
            authorization_url = await client.wait_for_authorization_url()
            self._states[name] = "authorizing"
            return authorization_url
        except Exception:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    def complete_oauth_callback(
        self,
        name: str,
        *,
        code: str,
        state: str | None,
    ) -> None:
        client = self._pending_clients.get(name)
        if client is None:
            raise ValueError("当前没有等待完成的 OAuth 授权")
        client.submit_oauth_callback(code, state)

    async def forget_credentials(self, names: list[str]) -> None:
        await asyncio.gather(*(self._secrets.delete_server(name) for name in names))

    async def remove_plugin_servers(self, names: list[str]) -> None:
        """Detach plugin-owned connections immediately during plugin uninstall."""
        for name in names:
            if name not in self._plugin_configs:
                continue
            await self._disconnect_server(name)
            self._plugin_configs.pop(name, None)
            self._states.pop(name, None)
            self._errors.pop(name, None)
        await self.forget_credentials(names)

    async def remove(self, name: str) -> str:
        if name in self._plugin_configs:
            return "该连接由系统组件提供，不能从 MCP 页面单独移除。"
        if name not in self._manual_configs:
            known = sorted(set(self._manual_configs) | set(self._plugin_configs))
            return f"MCP server {name!r} 不存在，当前已注册：{known or '无'}"
        await self._disconnect_server(name)
        self._manual_configs.pop(name, None)
        self._states.pop(name, None)
        self._errors.pop(name, None)
        await self._secrets.delete_server(name)
        self._save()
        return f"已注销 MCP server {name!r}，并删除其本地凭据。"

    def list_servers(self) -> str:
        names = sorted(set(self._manual_configs) | set(self._plugin_configs))
        if not names:
            return "当前没有已注册的 MCP server。"
        lines: list[str] = []
        for name in names:
            tools = self._server_tools.get(name, [])
            state = self._states.get(name, "disconnected")
            lines.append(
                f"- {name}（{len(tools)} 个工具）[{state}]：{', '.join(tools) or '无'}"
            )
        return "\n".join(lines)

    def snapshot(self) -> list[dict[str, Any]]:
        names = sorted(set(self._manual_configs) | set(self._plugin_configs))
        result: list[dict[str, Any]] = []
        for name in names:
            config = self._config_for(name)
            if config is None:
                continue
            state = self._states.get(name, "disconnected")
            result.append(
                {
                    "name": name,
                    "connected": state == "connected",
                    "status": state,
                    "error": self._errors.get(name, ""),
                    "authorization_required": state == "authorization_required",
                    "system_managed": name in self._plugin_configs,
                    "tool_names": list(self._server_tools.get(name, [])),
                    "transport": config.transport,
                    "url": config.url,
                    "auth_type": config.auth_type,
                    "command": list(config.command),
                    "cwd": config.cwd or "",
                    "env_keys": list(config.env_keys),
                    "header_names": list(config.header_names),
                }
            )
        return result

    async def _connect_available(
        self,
        name: str,
        config: McpServerConfig,
    ) -> None:
        if config.auth_type == "oauth" and not await self._secrets.has_oauth_tokens(
            name
        ):
            self._states[name] = "authorization_required"
            self._errors.pop(name, None)
            return
        if config.transport == "stdio" and config.env_keys and not config.env:
            config = replace(
                config,
                env=await self._secrets.get_bundle(name, "env"),
            )
        client = self._build_client(name, config, interactive_oauth=False)
        await self._finish_connect(name, client)

    async def _finish_connect(
        self,
        name: str,
        client: McpClientProtocol,
    ) -> None:
        self._states[name] = "connecting"
        self._errors.pop(name, None)
        try:
            infos = await client.connect()  # type: ignore[attr-defined]
            tool_names: list[str] = []
            for info in infos:
                wrapper = McpToolWrapper(client, info)
                self._tool_registry.register(
                    wrapper,
                    risk="external-side-effect",
                    source_type="mcp",
                    source_name=name,
                )
                tool_names.append(wrapper.name)
            self._clients[name] = client
            self._server_tools[name] = tool_names
            self._states[name] = "connected"
        except OAuthInteractionRequired:
            self._states[name] = "authorization_required"
            self._errors.pop(name, None)
            await self._secrets.delete(name, "oauth_tokens")
            await client.disconnect()  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            # The MCP OAuth transport may surface the non-interactive redirect
            # as a cancellation from its internal task group instead of the
            # OAuthInteractionRequired raised by our redirect handler.  That is
            # an unavailable optional integration, not a gateway cancellation.
            if bool(getattr(client, "authorization_required", False)):
                self._states[name] = "authorization_required"
                self._errors.pop(name, None)
                try:
                    await client.disconnect()  # type: ignore[attr-defined]
                except asyncio.CancelledError:
                    pass
                await self._secrets.delete(name, "oauth_tokens")
                return
            await client.disconnect()  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            self._states[name] = "error"
            self._errors[name] = redact_exception(exc)
            await client.disconnect()  # type: ignore[attr-defined]
            logger.warning("[mcp] 连接失败 (%s): %s", name, exc)
        finally:
            self._pending_clients.pop(name, None)
            self._pending_tasks.pop(name, None)

    def _build_client(
        self,
        name: str,
        config: McpServerConfig,
        *,
        interactive_oauth: bool,
    ) -> McpClientProtocol:
        if config.transport == "stdio":
            return McpClient(
                name=name,
                command=list(config.command),
                env=config.env,
                cwd=config.cwd,
            )
        return RemoteMcpClient(
            name=name,
            config=config,
            secrets=self._secrets,
            callback_url=self._callback_url(name),
            interactive_oauth=interactive_oauth,
        )

    async def _secure_inline_secrets(
        self,
        name: str,
        config: McpServerConfig,
        raw: dict[str, Any],
    ) -> tuple[McpServerConfig, bool]:
        changed = False
        if config.env:
            await self._secrets.set_bundle(name, "env", config.env)
            config = replace(config, env={}, env_keys=tuple(sorted(config.env)))
            changed = True
        if config.headers:
            await self._secrets.set_bundle(name, "headers", config.headers)
            config = replace(
                config,
                headers={},
                header_names=tuple(sorted(config.headers)),
            )
            changed = True
        bearer = str(raw.get("bearer_token") or "")
        if bearer:
            await self._secrets.set_text(name, "bearer", bearer)
            changed = True
        if config.transport == "stdio" and config.env_keys:
            env = await self._secrets.get_bundle(name, "env")
            config = replace(config, env=env)
        return config, changed

    def _load_raw_configs(self) -> dict[str, Any]:
        if not self._config_path.exists():
            return {}
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            servers = data.get("servers", {})
            return servers if isinstance(servers, dict) else {}
        except Exception as exc:
            logger.warning("[mcp] 读取配置失败 %s: %s", self._config_path, exc)
            return {}

    def _save(self) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "servers": {
                            name: config.persisted()
                            for name, config in sorted(self._manual_configs.items())
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("[mcp] 保存配置失败: %s", exc)

    async def _disconnect_server(self, name: str) -> None:
        task = self._pending_tasks.pop(name, None)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for tool_name in self._server_tools.pop(name, []):
            self._tool_registry.unregister(tool_name)
        client = self._clients.pop(name, None)
        pending = self._pending_clients.pop(name, None)
        if client is not None:
            await client.disconnect()  # type: ignore[attr-defined]
        if pending is not None and pending is not client:
            await pending.disconnect()

    def _config_for(self, name: str) -> McpServerConfig | None:
        return self._manual_configs.get(name) or self._plugin_configs.get(name)

    def _callback_url(self, name: str) -> str:
        return (
            f"{self._oauth_callback_base_url}/api/dashboard/control/mcp/"
            f"oauth/callback/{quote(name, safe='')}"
        )
