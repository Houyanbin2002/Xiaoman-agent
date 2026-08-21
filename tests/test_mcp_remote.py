from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from agent.mcp.catalog import CATALOG
from agent.mcp.config import McpServerConfig
from agent.mcp.remote_client import XiaomanOAuthClientProvider
from agent.mcp.registry import McpServerRegistry
from agent.mcp.secrets import KeyringOAuthTokenStorage
from agent.plugins.manager import ActivePluginInfo


class _Secrets:
    def __init__(self) -> None:
        self.bundles: dict[tuple[str, str], dict[str, str]] = {}
        self.text: dict[tuple[str, str], str] = {}

    async def get_bundle(self, name: str, kind: str) -> dict[str, str]:
        return dict(self.bundles.get((name, kind), {}))

    async def set_bundle(
        self,
        name: str,
        kind: str,
        values: dict[str, str],
    ) -> None:
        self.bundles[(name, kind)] = dict(values)

    async def get_text(self, name: str, kind: str) -> str | None:
        return self.text.get((name, kind))

    async def set_text(self, name: str, kind: str, value: str) -> None:
        self.text[(name, kind)] = value

    async def delete(self, name: str, kind: str) -> None:
        self.text.pop((name, kind), None)
        self.bundles.pop((name, kind), None)

    async def delete_server(self, name: str) -> None:
        for key in list(self.text):
            if key[0] == name:
                self.text.pop(key)
        for key in list(self.bundles):
            if key[0] == name:
                self.bundles.pop(key)

    async def has_oauth_tokens(self, name: str) -> bool:
        return bool(self.text.get((name, "oauth_tokens")))

    async def set_oauth_client(
        self,
        name: str,
        *,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scope: str = "",
    ) -> None:
        self.text[(name, "oauth_client")] = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=[AnyUrl(redirect_uri)],
            token_endpoint_auth_method="client_secret_post",
            scope=scope or None,
        ).model_dump_json()
        self.text[(name, "oauth_client_static")] = "1"

    async def has_static_oauth_client(self, name: str) -> bool:
        return bool(self.text.get((name, "oauth_client_static")))


def _oauth_provider(storage: KeyringOAuthTokenStorage) -> XiaomanOAuthClientProvider:
    return XiaomanOAuthClientProvider(
        server_url="https://mcp.notion.com/mcp",
        client_metadata=OAuthClientMetadata(
            client_name="Xiaoman Agent",
            redirect_uris=[AnyUrl("http://127.0.0.1:2236/oauth/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=storage,
    )


@pytest.mark.asyncio
async def test_oauth_restart_restores_absolute_token_expiry() -> None:
    secrets = _Secrets()
    storage = KeyringOAuthTokenStorage(  # type: ignore[arg-type]
        secrets,
        "notion",
    )
    before = time.time()
    await storage.set_tokens(
        OAuthToken(
            access_token="access-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="refresh-token",
        )
    )

    provider = _oauth_provider(storage)
    await provider._initialize()

    assert provider.context.token_expiry_time is not None
    assert before + 3590 <= provider.context.token_expiry_time <= time.time() + 3610
    assert provider.context.is_token_valid() is True


@pytest.mark.asyncio
async def test_oauth_refresh_keeps_rotatable_refresh_token() -> None:
    secrets = _Secrets()
    storage = KeyringOAuthTokenStorage(  # type: ignore[arg-type]
        secrets,
        "notion",
    )
    provider = _oauth_provider(storage)
    provider.context.current_tokens = OAuthToken(
        access_token="old-access-token",
        token_type="Bearer",
        expires_in=1,
        refresh_token="persistent-refresh-token",
    )
    response = httpx.Response(
        200,
        content=OAuthToken(
            access_token="new-access-token",
            token_type="Bearer",
            expires_in=3600,
        ).model_dump_json(),
    )

    assert await provider._handle_refresh_response(response) is True
    stored = await storage.get_tokens()
    assert stored is not None
    assert stored.access_token == "new-access-token"
    assert stored.refresh_token == "persistent-refresh-token"


def test_remote_mcp_config_validates_transport_and_safe_url() -> None:
    config = McpServerConfig.from_raw(
        {
            "transport": "streamable_http",
            "url": "https://mcp.notion.com/mcp",
            "auth": {"type": "oauth"},
        }
    )
    assert config.is_remote is True
    assert config.auth_type == "oauth"
    assert config.persisted()["url"] == "https://mcp.notion.com/mcp"

    with pytest.raises(ValueError, match="HTTPS"):
        McpServerConfig.from_raw(
            {
                "transport": "streamable_http",
                "url": "http://remote.example/mcp",
            }
        )


@pytest.mark.asyncio
async def test_basic_oauth_token_request_uses_only_http_basic() -> None:
    provider = XiaomanOAuthClientProvider(
        server_url="https://mcp.notion.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:2236/callback")],
            client_name="Xiaoman Agent",
        ),
        storage=SimpleNamespace(),  # type: ignore[arg-type]
    )
    provider.context.client_info = OAuthClientInformationFull(
        client_id="client-id",
        client_secret="client-secret",
        token_endpoint_auth_method="client_secret_basic",
        redirect_uris=[AnyUrl("http://127.0.0.1:2236/callback")],
    )
    provider.context.current_tokens = OAuthToken(
        access_token="access",
        refresh_token="refresh",
    )

    request = await provider._exchange_token_authorization_code("code", "verifier")
    form = parse_qs(request.content.decode())
    refresh = await provider._refresh_token()
    refresh_form = parse_qs(refresh.content.decode())

    assert request.headers["Authorization"].startswith("Basic ")
    assert "client_id" not in form
    assert "client_secret" not in form
    assert "client_id" not in refresh_form


@pytest.mark.asyncio
async def test_manual_mcp_secrets_are_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _RejectedRemote:
        def __init__(self, name: str, *_args: Any, **_kwargs: Any) -> None:
            self.name = name

        async def connect(self):
            raise ConnectionError("offline")

        async def disconnect(self) -> None:
            return None

    monkeypatch.setattr("agent.mcp.registry.RemoteMcpClient", _RejectedRemote)
    secrets = _Secrets()
    registry = McpServerRegistry(
        tmp_path / "mcp_servers.json",
        SimpleNamespace(register=lambda *_args, **_kwargs: None),  # type: ignore[arg-type]
        secret_store=secrets,  # type: ignore[arg-type]
    )

    result = await registry.add_remote(
        "private",
        url="https://mcp.example.com/mcp",
        auth_type="bearer",
        bearer_token="top-secret-token",
    )

    assert "失败" in result
    persisted = (tmp_path / "mcp_servers.json").read_text(encoding="utf-8")
    assert "top-secret-token" not in persisted
    assert secrets.text[("private", "bearer")] == "top-secret-token"


@pytest.mark.asyncio
async def test_pre_registered_oauth_client_survives_authorization_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Remote:
        def __init__(self, name: str, *_args: Any, **_kwargs: Any) -> None:
            self.name = name
            self.waiting = asyncio.Event()

        async def connect(self):
            await self.waiting.wait()
            return []

        async def wait_for_authorization_url(self, timeout: float = 30.0) -> str:
            return "https://accounts.google.com/o/oauth2/auth"

        async def disconnect(self) -> None:
            return None

    monkeypatch.setattr("agent.mcp.registry.RemoteMcpClient", _Remote)
    secrets = _Secrets()
    registry = McpServerRegistry(
        tmp_path / "mcp_servers.json",
        SimpleNamespace(register=lambda *_args, **_kwargs: None),  # type: ignore[arg-type]
        secret_store=secrets,  # type: ignore[arg-type]
    )
    await registry.add_remote(
        "gmail",
        url="https://gmailmcp.googleapis.com/mcp/v1",
        auth_type="oauth",
        oauth_client_id="google-client-id",
        oauth_client_secret="google-client-secret",
    )

    url = await registry.begin_oauth("gmail")

    assert url.startswith("https://accounts.google.com/")
    assert ("gmail", "oauth_client") in secrets.text
    assert ("gmail", "oauth_client_static") in secrets.text
    await registry.shutdown()


@pytest.mark.asyncio
async def test_registry_completes_remote_oauth_and_registers_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secrets = _Secrets()
    tools = SimpleNamespace(
        registered=[],
        unregistered=[],
        register=lambda tool, **_kwargs: tools.registered.append(tool.name),
        unregister=lambda name: tools.unregistered.append(name),
    )

    class _Remote:
        def __init__(self, name: str, config: McpServerConfig, **_: Any) -> None:
            self.name = name
            self.config = config
            self.callback = asyncio.Event()

        async def connect(self):
            await self.callback.wait()
            return [
                SimpleNamespace(
                    name="notion-search",
                    description="Search Notion",
                    input_schema={"type": "object", "properties": {}},
                )
            ]

        async def wait_for_authorization_url(self, timeout: float = 30.0) -> str:
            return "https://notion.example/authorize"

        def submit_oauth_callback(self, code: str, state: str | None) -> None:
            assert code == "code"
            assert state == "state"
            self.callback.set()

        async def disconnect(self) -> None:
            self.callback.set()

        async def call(self, *_args: Any, **_kwargs: Any) -> str:
            return "ok"

    monkeypatch.setattr("agent.mcp.registry.RemoteMcpClient", _Remote)
    registry = McpServerRegistry(
        tmp_path / "mcp_servers.json",
        tools,  # type: ignore[arg-type]
        secret_store=secrets,  # type: ignore[arg-type]
    )
    await registry.sync_plugin_servers(
        [
            ActivePluginInfo(
                plugin_id="notion@official",
                plugin_dir=tmp_path / "notion",
                manifest={},
                module_path="notion",
                mcp_servers={
                    "notion": {
                        "transport": "streamable_http",
                        "url": "https://mcp.notion.com/mcp",
                        "auth": {"type": "oauth"},
                    }
                },
            )
        ]
    )

    authorization_url = await registry.begin_oauth("notion")
    assert authorization_url == "https://notion.example/authorize"
    assert registry.snapshot()[0]["status"] == "authorizing"
    registry.complete_oauth_callback("notion", code="code", state="state")
    for _ in range(20):
        if registry.snapshot()[0]["connected"]:
            break
        await asyncio.sleep(0.01)

    row = registry.snapshot()[0]
    assert row["connected"] is True
    assert row["tool_names"] == ["mcp_notion__notion-search"]
    assert tools.registered == ["mcp_notion__notion-search"]

    await registry.remove_plugin_servers(["notion"])
    assert registry.snapshot() == []
    assert tools.unregistered == ["mcp_notion__notion-search"]


@pytest.mark.asyncio
async def test_stale_remote_oauth_does_not_cancel_gateway_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _RemoteNeedsAuthorization:
        authorization_required = True

        def __init__(self, name: str, config: McpServerConfig, **_: Any) -> None:
            self.name = name
            self.config = config

        async def connect(self):
            raise asyncio.CancelledError

        async def disconnect(self) -> None:
            return None

    monkeypatch.setattr(
        "agent.mcp.registry.RemoteMcpClient",
        _RemoteNeedsAuthorization,
    )
    secrets = _Secrets()
    secrets.text[("notion", "oauth_tokens")] = "stale-token"
    registry = McpServerRegistry(
        tmp_path / "mcp_servers.json",
        SimpleNamespace(register=lambda *_args, **_kwargs: None),  # type: ignore[arg-type]
        secret_store=secrets,  # type: ignore[arg-type]
    )

    await registry.sync_plugin_servers(
        [
            ActivePluginInfo(
                plugin_id="notion@official",
                plugin_dir=tmp_path / "notion",
                manifest={},
                module_path="notion",
                mcp_servers={
                    "notion": {
                        "transport": "streamable_http",
                        "url": "https://mcp.notion.com/mcp",
                        "auth": {"type": "oauth"},
                    }
                },
            )
        ]
    )

    assert registry.snapshot()[0]["status"] == "authorization_required"
    assert ("notion", "oauth_tokens") not in secrets.text


def test_notion_is_a_standard_remote_mcp_catalog_entry() -> None:
    notion = CATALOG["notion"]

    assert notion.transport == "streamable_http"
    assert notion.requires_oauth is True
