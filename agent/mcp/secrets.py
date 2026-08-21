from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import keyring
from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


class McpSecretStore:
    """OS-backed MCP credential storage.

    On Windows, ``keyring`` uses Credential Manager/DPAPI, so access and refresh
    tokens are encrypted for the signed-in user instead of being written to the
    workspace JSON file.
    """

    _SERVICE = "xiaoman.mcp"
    _KINDS = (
        "env",
        "headers",
        "bearer",
        "oauth_tokens",
        "oauth_token_expiry",
        "oauth_client",
        "oauth_client_static",
    )

    def __init__(self, config_path: Path) -> None:
        resolved = str(config_path.resolve(strict=False)).casefold().encode("utf-8")
        self._namespace = hashlib.sha256(resolved).hexdigest()[:16]

    async def get_bundle(self, server_name: str, kind: str) -> dict[str, str]:
        value = await self._get(server_name, kind)
        if not value:
            return {}
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if not isinstance(loaded, dict):
            return {}
        return {str(key): str(item) for key, item in loaded.items()}

    async def set_bundle(
        self,
        server_name: str,
        kind: str,
        values: dict[str, str],
    ) -> None:
        if values:
            await self._set(
                server_name,
                kind,
                json.dumps(values, ensure_ascii=False, separators=(",", ":")),
            )
        else:
            await self.delete(server_name, kind)

    async def get_text(self, server_name: str, kind: str) -> str | None:
        return await self._get(server_name, kind)

    async def set_text(self, server_name: str, kind: str, value: str) -> None:
        if value:
            await self._set(server_name, kind, value)
        else:
            await self.delete(server_name, kind)

    async def delete(self, server_name: str, kind: str) -> None:
        account = self._account(server_name, kind)

        def remove() -> None:
            try:
                keyring.delete_password(self._SERVICE, account)
            except keyring.errors.PasswordDeleteError:
                pass

        await asyncio.to_thread(remove)

    async def delete_server(self, server_name: str) -> None:
        await asyncio.gather(*(self.delete(server_name, kind) for kind in self._KINDS))

    async def has_oauth_tokens(self, server_name: str) -> bool:
        return bool(await self.get_text(server_name, "oauth_tokens"))

    async def set_oauth_client(
        self,
        server_name: str,
        *,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scope: str = "",
    ) -> None:
        """Persist a pre-registered OAuth client without exposing it in config."""
        client = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret or None,
            redirect_uris=[redirect_uri],
            token_endpoint_auth_method=(
                "client_secret_post" if client_secret else "none"
            ),
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=scope or None,
            client_name="Xiaoman Agent",
        )
        await self.set_text(server_name, "oauth_client", client.model_dump_json())
        await self.set_text(server_name, "oauth_client_static", "1")

    async def has_static_oauth_client(self, server_name: str) -> bool:
        return bool(await self.get_text(server_name, "oauth_client_static"))

    async def _get(self, server_name: str, kind: str) -> str | None:
        account = self._account(server_name, kind)
        return await asyncio.to_thread(
            keyring.get_password,
            self._SERVICE,
            account,
        )

    async def _set(self, server_name: str, kind: str, value: str) -> None:
        account = self._account(server_name, kind)
        await asyncio.to_thread(
            keyring.set_password,
            self._SERVICE,
            account,
            value,
        )

    def _account(self, server_name: str, kind: str) -> str:
        return f"{self._namespace}:{server_name}:{kind}"


class KeyringOAuthTokenStorage(TokenStorage):
    def __init__(self, secrets: McpSecretStore, server_name: str) -> None:
        self._secrets = secrets
        self._server_name = server_name

    async def get_tokens(self) -> OAuthToken | None:
        raw = await self._secrets.get_text(self._server_name, "oauth_tokens")
        return OAuthToken.model_validate_json(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await self._secrets.set_text(
            self._server_name,
            "oauth_tokens",
            tokens.model_dump_json(),
        )
        if tokens.expires_in is None:
            await self._secrets.delete(self._server_name, "oauth_token_expiry")
            return
        await self._secrets.set_text(
            self._server_name,
            "oauth_token_expiry",
            str(time.time() + max(float(tokens.expires_in), 0.0)),
        )

    async def get_token_expiry_time(self) -> float | None:
        raw = await self._secrets.get_text(
            self._server_name,
            "oauth_token_expiry",
        )
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            await self._secrets.delete(self._server_name, "oauth_token_expiry")
            return None

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = await self._secrets.get_text(self._server_name, "oauth_client")
        return OAuthClientInformationFull.model_validate_json(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await self._secrets.set_text(
            self._server_name,
            "oauth_client",
            client_info.model_dump_json(),
        )


def redact_exception(exc: Exception) -> str:
    """Return a dashboard-safe error without accidentally serializing secrets."""
    text = str(exc).replace("\r", " ").replace("\n", " ").strip()
    return text[:500] or exc.__class__.__name__
