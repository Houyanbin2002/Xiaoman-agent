from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from agent.mcp.client import McpToolInfo
from agent.mcp.config import McpServerConfig
from agent.mcp.secrets import KeyringOAuthTokenStorage, McpSecretStore

logger = logging.getLogger(__name__)


class OAuthInteractionRequired(RuntimeError):
    pass


class XiaomanOAuthClientProvider(OAuthClientProvider):
    """Avoid duplicate client authentication rejected by strict OAuth servers."""

    async def _initialize(self) -> None:
        await super()._initialize()
        tokens = self.context.current_tokens
        if tokens is None:
            return
        expiry_loader = getattr(self.context.storage, "get_token_expiry_time", None)
        expiry_time = await expiry_loader() if expiry_loader is not None else None
        if expiry_time is not None:
            self.context.token_expiry_time = expiry_time
        elif tokens.refresh_token and tokens.expires_in is not None:
            # Older Xiaoman versions did not persist the absolute expiry. Treat
            # those refreshable tokens as expired once so startup refreshes them
            # instead of sending a stale access token and opening OAuth again.
            self.context.token_expiry_time = 1.0
        else:
            self.context.update_token_expiry(tokens)

    async def _handle_refresh_response(self, response: httpx.Response) -> bool:
        previous_refresh_token = (
            self.context.current_tokens.refresh_token
            if self.context.current_tokens is not None
            else None
        )
        refreshed = await super()._handle_refresh_response(response)
        tokens = self.context.current_tokens
        if (
            refreshed
            and tokens is not None
            and not tokens.refresh_token
            and previous_refresh_token
        ):
            tokens = tokens.model_copy(
                update={"refresh_token": previous_refresh_token},
            )
            self.context.current_tokens = tokens
            await self.context.storage.set_tokens(tokens)
        return refreshed

    async def _exchange_token_authorization_code(
        self,
        auth_code: str,
        code_verifier: str,
        *,
        token_data: dict[str, Any] | None = None,
    ) -> httpx.Request:
        if self.context.client_metadata.redirect_uris is None:
            raise RuntimeError("OAuth redirect URI missing")
        info = self.context.client_info
        if info is None:
            raise RuntimeError("OAuth client registration missing")
        data = dict(token_data or {})
        data.update(
            {
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": str(self.context.client_metadata.redirect_uris[0]),
                "client_id": info.client_id,
                "code_verifier": code_verifier,
            }
        )
        if self.context.should_include_resource_param(self.context.protocol_version):
            data["resource"] = self.context.get_resource_url()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data, headers = self.context.prepare_token_auth(data, headers)
        if info.token_endpoint_auth_method == "client_secret_basic":
            data.pop("client_id", None)
        return httpx.Request(
            "POST",
            self._get_token_endpoint(),
            content=urlencode(data),
            headers=headers,
        )

    async def _refresh_token(self) -> httpx.Request:
        tokens = self.context.current_tokens
        info = self.context.client_info
        if not tokens or not tokens.refresh_token:
            raise RuntimeError("OAuth refresh token missing")
        if not info or not info.client_id:
            raise RuntimeError("OAuth client registration missing")
        data: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": tokens.refresh_token,
            "client_id": info.client_id,
        }
        if self.context.should_include_resource_param(self.context.protocol_version):
            data["resource"] = self.context.get_resource_url()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data, headers = self.context.prepare_token_auth(data, headers)
        if info.token_endpoint_auth_method == "client_secret_basic":
            data.pop("client_id", None)
        return httpx.Request(
            "POST",
            self._get_token_endpoint(),
            content=urlencode(data),
            headers=headers,
        )


class RemoteMcpClient:
    """Long-lived MCP client for Streamable HTTP and legacy SSE transports."""

    def __init__(
        self,
        name: str,
        config: McpServerConfig,
        secrets: McpSecretStore,
        callback_url: str,
        *,
        interactive_oauth: bool = False,
    ) -> None:
        if not config.is_remote:
            raise ValueError("RemoteMcpClient requires a remote transport")
        self.name = name
        self.config = config
        self.command: list[str] = []
        self.env: dict[str, str] = {}
        self.cwd: str | None = None
        self._secrets = secrets
        self._callback_url = callback_url
        self._interactive_oauth = interactive_oauth
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._tool_infos: list[McpToolInfo] = []
        self._authorization_url = ""
        self._authorization_ready = asyncio.Event()
        self._callback_future: asyncio.Future[tuple[str, str | None]] | None = None

    @property
    def authorization_required(self) -> bool:
        """Whether a non-interactive connection reached an OAuth redirect."""

        return bool(self._authorization_url and not self._interactive_oauth)

    async def connect(self) -> list[McpToolInfo]:
        headers = dict(self.config.headers)
        headers.update(await self._secrets.get_bundle(self.name, "headers"))
        if self.config.auth_type == "bearer":
            token = await self._secrets.get_text(self.name, "bearer")
            if not token:
                raise ValueError("Bearer token 尚未配置")
            headers["Authorization"] = f"Bearer {token}"

        auth: httpx.Auth | None = None
        if self.config.auth_type == "oauth":
            self._callback_future = asyncio.get_running_loop().create_future()
            auth = XiaomanOAuthClientProvider(
                server_url=self.config.url,
                client_metadata=OAuthClientMetadata(
                    client_name="Xiaoman Agent",
                    redirect_uris=[AnyUrl(self._callback_url)],
                    grant_types=["authorization_code", "refresh_token"],
                    response_types=["code"],
                    scope=self.config.scopes or None,
                ),
                storage=KeyringOAuthTokenStorage(self._secrets, self.name),
                redirect_handler=self._handle_redirect,
                callback_handler=self._wait_for_callback,
                timeout=300.0,
            )

        if self.config.transport == "streamable_http":
            http_client = await self._stack.enter_async_context(
                httpx.AsyncClient(
                    headers=headers,
                    auth=auth,
                    follow_redirects=True,
                    timeout=httpx.Timeout(30.0, read=300.0),
                )
            )
            read, write, _ = await self._stack.enter_async_context(
                streamable_http_client(
                    self.config.url,
                    http_client=http_client,
                )
            )
        else:
            read, write = await self._stack.enter_async_context(
                sse_client(
                    self.config.url,
                    headers=headers,
                    auth=auth,
                    timeout=30.0,
                    sse_read_timeout=300.0,
                )
            )

        self._session = await self._stack.enter_async_context(
            ClientSession(
                read,
                write,
                # The first request may include browser-based OAuth.  Keep the
                # session deadline aligned with the provider's five-minute
                # human authorization window instead of timing out at 60s.
                read_timeout_seconds=timedelta(seconds=300),
            )
        )
        await self._session.initialize()
        result = await self._session.list_tools()
        self._tool_infos = [
            McpToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema or {}),
            )
            for tool in result.tools
        ]
        return self._tool_infos

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> str:
        if self._session is None:
            raise ConnectionError(f"MCP server {self.name!r} 尚未连接")
        result = await self._session.call_tool(
            tool_name,
            arguments,
            read_timeout_seconds=(
                timedelta(seconds=timeout) if timeout is not None else None
            ),
        )
        blocks: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                blocks.append(str(text))
            else:
                blocks.append(
                    json.dumps(
                        block.model_dump(mode="json", by_alias=True),
                        ensure_ascii=False,
                    )
                )
        if result.structuredContent is not None:
            blocks.append(json.dumps(result.structuredContent, ensure_ascii=False))
        rendered = "\n".join(blocks)
        return f"MCP error ({self.name}/{tool_name}): {rendered}" if result.isError else rendered

    async def wait_for_authorization_url(self, timeout: float = 30.0) -> str:
        await asyncio.wait_for(self._authorization_ready.wait(), timeout=timeout)
        return self._authorization_url

    def submit_oauth_callback(self, code: str, state: str | None) -> None:
        future = self._callback_future
        if future is None or future.done():
            raise RuntimeError("当前没有等待完成的 OAuth 授权")
        future.set_result((code, state))

    async def disconnect(self) -> None:
        future = self._callback_future
        if future is not None and not future.done():
            future.cancel()
        self._session = None
        self._tool_infos = []
        try:
            await self._stack.aclose()
        except Exception as exc:
            logger.debug("[mcp] remote cleanup failed (%s): %s", self.name, exc)
        self._stack = AsyncExitStack()

    async def _handle_redirect(self, authorization_url: str) -> None:
        self._authorization_url = authorization_url
        self._authorization_ready.set()
        if not self._interactive_oauth:
            raise OAuthInteractionRequired("需要用户完成 OAuth 授权")

    async def _wait_for_callback(self) -> tuple[str, str | None]:
        if self._callback_future is None:
            raise RuntimeError("OAuth callback future 尚未初始化")
        return await self._callback_future
