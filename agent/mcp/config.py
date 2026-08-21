from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import urlparse


McpTransport = Literal["stdio", "streamable_http", "sse"]
McpAuthType = Literal["none", "oauth", "bearer", "headers"]


@dataclass(frozen=True)
class McpServerConfig:
    """Validated transport configuration shared by manual and plugin MCP servers."""

    transport: McpTransport
    command: tuple[str, ...] = ()
    url: str = ""
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    env_keys: tuple[str, ...] = ()
    auth_type: McpAuthType = "none"
    scopes: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    header_names: tuple[str, ...] = ()

    @property
    def is_remote(self) -> bool:
        return self.transport != "stdio"

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "McpServerConfig":
        command = _strings(raw.get("command"))
        transport_raw = str(raw.get("transport") or "").strip().lower()
        if not transport_raw:
            transport_raw = "stdio" if command else "streamable_http"
        if transport_raw not in {"stdio", "streamable_http", "sse"}:
            raise ValueError(f"不支持的 MCP transport: {transport_raw}")
        transport = cast(McpTransport, transport_raw)

        auth_raw = raw.get("auth")
        auth: dict[str, Any]
        if isinstance(auth_raw, dict):
            auth = cast(dict[str, Any], auth_raw)
            auth_type_raw = str(auth.get("type") or "none").strip().lower()
        else:
            auth = {}
            auth_type_raw = str(auth_raw or "none").strip().lower()
        if auth_type_raw not in {"none", "oauth", "bearer", "headers"}:
            raise ValueError(f"不支持的 MCP auth type: {auth_type_raw}")
        auth_type = cast(McpAuthType, auth_type_raw)

        url = str(raw.get("url") or "").strip()
        if transport == "stdio":
            if not command:
                raise ValueError("stdio MCP server 必须提供 command")
            url = ""
            auth_type = "none"
        else:
            _validate_remote_url(url)
            if command:
                raise ValueError("远程 MCP server 不能同时提供 command")

        env = _string_dict(raw.get("env"))
        headers = _string_dict(raw.get("headers"))
        env_keys = tuple(sorted(set(_strings(raw.get("env_keys"))) | set(env)))
        header_names = tuple(
            sorted(set(_strings(raw.get("header_names"))) | set(headers))
        )
        return cls(
            transport=transport,
            command=tuple(command),
            url=url,
            cwd=str(raw.get("cwd") or "").strip() or None,
            env=env,
            env_keys=env_keys,
            auth_type=auth_type,
            scopes=str(auth.get("scopes") or raw.get("scopes") or "").strip(),
            headers=headers,
            header_names=header_names,
        )

    def persisted(self) -> dict[str, Any]:
        """Return a disk-safe representation containing references, never secrets."""
        if self.transport == "stdio":
            result: dict[str, Any] = {
                "transport": "stdio",
                "command": list(self.command),
            }
            if self.cwd:
                result["cwd"] = self.cwd
            if self.env_keys:
                result["env_keys"] = list(self.env_keys)
            return result

        result = {
            "transport": self.transport,
            "url": self.url,
            "auth": {"type": self.auth_type},
        }
        if self.scopes:
            cast(dict[str, str], result["auth"])["scopes"] = self.scopes
        if self.header_names:
            result["header_names"] = list(self.header_names)
        return result


def _validate_remote_url(value: str) -> None:
    if not value:
        raise ValueError("远程 MCP server 必须提供 url")
    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        return
    raise ValueError("远程 MCP URL 必须使用 HTTPS；仅本机地址允许 HTTP")


def _strings(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _string_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): str(item)
        for key, item in cast(dict[object, object], value).items()
        if str(key).strip()
    }
