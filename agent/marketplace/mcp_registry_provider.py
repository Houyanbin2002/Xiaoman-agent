from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import MarketplaceField, MarketplaceInstallMode, MarketplaceItem

_REGISTRY_BASE_URL = "https://registry.modelcontextprotocol.io/v0.1/servers"
_REGISTRY_TIMEOUT_SECONDS = 20
_REGISTRY_WARMUP_PAGES = 3


class McpRegistryProvider:
    """Read the official MCP Registry through a small, offline-first cache."""

    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        fetch_json: Callable[[str], dict[str, object]] | None = None,
        cache_ttl_seconds: int = 3600,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._cache_path = cache_path or (
            Path.home() / ".xiaoman" / "marketplace" / "mcp-registry.json"
        )
        self._fetch_json = fetch_json or _fetch_json
        self._cache_ttl_seconds = max(0, cache_ttl_seconds)
        self._now = now
        self._items: list[MarketplaceItem] | None = None

    def search(self, query: str = "", limit: int = 20) -> list[MarketplaceItem]:
        needle = query.strip().casefold()
        rows = self._load_items(needle) if needle else self._load_cached_items()
        if needle:
            rows = [row for row in rows if _matches(row, needle)]
        return rows[: max(0, min(limit, 100))]

    def get(self, item_id: str) -> MarketplaceItem | None:
        return next((row for row in self._load_items(item_id) if row.id == item_id), None)

    def refresh(self) -> list[MarketplaceItem]:
        payload = self._download_registry("")
        self._write_cache(payload)
        self._items = _parse_payload(payload)
        return list(self._items)

    def _load_items(self, query: str) -> list[MarketplaceItem]:
        if self._items is not None:
            matches = [row for row in self._items if _matches(row, query)]
            if matches:
                return list(self._items)
        cached = self._read_cache()
        if cached is not None and self._is_fresh(cached):
            self._items = _parse_payload(_payload(cached))
            if any(_matches(row, query) for row in self._items):
                return list(self._items)
        try:
            downloaded = self._download_registry(query)
            payload = _merge_payloads(_payload(cached or {}), downloaded)
            self._write_cache(payload)
            self._items = _parse_payload(payload)
            return list(self._items)
        except Exception:
            if cached is None:
                raise
            self._items = _parse_payload(_payload(cached))
            return list(self._items)

    def _load_cached_items(self) -> list[MarketplaceItem]:
        if self._items is not None:
            return list(self._items)
        cached = self._read_cache()
        self._items = _parse_payload(_payload(cached or {}))
        return list(self._items)

    def _download_registry(self, search_query: str) -> dict[str, object]:
        servers: list[object] = []
        cursor = ""
        for _page in range(_REGISTRY_WARMUP_PAGES):
            params = {"limit": "100"}
            if search_query:
                params["search"] = search_query
            if cursor:
                params["cursor"] = cursor
            try:
                page = self._fetch_json(
                    f"{_REGISTRY_BASE_URL}?{urlencode(params)}"
                )
            except (OSError, ValueError):
                if servers:
                    break
                raise
            page_servers = page.get("servers", [])
            if isinstance(page_servers, list):
                servers.extend(page_servers)
            if search_query:
                break
            cursor = _next_cursor(page)
            if not cursor:
                break
        return {"servers": servers}

    def _read_cache(self) -> dict[str, object] | None:
        try:
            loaded = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def _write_cache(self, payload: dict[str, object]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        document = {"fetched_at": self._now(), "payload": payload}
        temporary = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._cache_path)

    def _is_fresh(self, cached: dict[str, object]) -> bool:
        fetched_at = cached.get("fetched_at")
        return isinstance(fetched_at, (int, float)) and (
            self._now() - float(fetched_at) < self._cache_ttl_seconds
        )


def _fetch_json(url: str) -> dict[str, object]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(
        request,
        timeout=_REGISTRY_TIMEOUT_SECONDS,
    ) as response:  # noqa: S310 - fixed registry host
        loaded = json.loads(response.read().decode("utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("MCP Registry 返回格式无效")
    return loaded


def _payload(cached: dict[str, object]) -> dict[str, object]:
    payload = cached.get("payload")
    return payload if isinstance(payload, dict) else {"servers": []}


def _merge_payloads(
    existing: dict[str, object], incoming: dict[str, object]
) -> dict[str, object]:
    merged: dict[str, object] = {}
    for payload in (existing, incoming):
        servers = payload.get("servers")
        if not isinstance(servers, list):
            continue
        for envelope in servers:
            if not isinstance(envelope, dict):
                continue
            server = envelope.get("server", envelope)
            if not isinstance(server, dict):
                continue
            item_id = str(server.get("name", "")).strip()
            if item_id:
                merged[item_id] = envelope
    return {"servers": list(merged.values())}


def _next_cursor(payload: dict[str, object]) -> str:
    direct = payload.get("nextCursor") or payload.get("next_cursor")
    if isinstance(direct, str):
        return direct
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("nextCursor") or metadata.get("next_cursor")
        return nested if isinstance(nested, str) else ""
    return ""


def _parse_payload(payload: dict[str, object]) -> list[MarketplaceItem]:
    rows: list[MarketplaceItem] = []
    raw_servers = payload.get("servers")
    if not isinstance(raw_servers, list):
        return rows
    for envelope in raw_servers:
        if not isinstance(envelope, dict):
            continue
        server = envelope.get("server", envelope)
        if not isinstance(server, dict) or not _is_visible(envelope, server):
            continue
        item = _parse_server(server, envelope)
        if item is not None:
            rows.append(item)
    rows.sort(key=lambda item: (item.deprecated, item.name.casefold(), item.id))
    return rows


def _is_visible(envelope: dict[str, Any], server: dict[str, Any]) -> bool:
    status = str(server.get("status", "")).casefold()
    metadata = envelope.get("_meta")
    if isinstance(metadata, dict):
        official = metadata.get("io.modelcontextprotocol.registry/official")
        if isinstance(official, dict):
            status = str(official.get("status", status)).casefold()
    return status != "deleted"


def _parse_server(
    server: dict[str, Any], envelope: dict[str, Any]
) -> MarketplaceItem | None:
    item_id = str(server.get("name", "")).strip()
    if not item_id:
        return None
    name = str(server.get("title") or item_id.rsplit("/", 1)[-1]).strip()
    description = str(server.get("description", "")).strip()
    version = str(server.get("version", "")).strip()
    provider = item_id.split("/", 1)[0]
    metadata = envelope.get("_meta")
    status = str(server.get("status", "")).casefold()
    verified = False
    if isinstance(metadata, dict):
        official = metadata.get("io.modelcontextprotocol.registry/official")
        if isinstance(official, dict):
            status = str(official.get("status", status)).casefold()
            verified = bool(official.get("isLatest", True))
    source_url = _source_url(server)
    icon_url = _icon_url(server)
    install_mode, install_spec, fields, reason = _installation(server)
    return MarketplaceItem(
        id=item_id,
        kind="mcp",
        name=name,
        description=description,
        provider=provider,
        source_url=source_url,
        version=version,
        icon_url=icon_url,
        verified=verified,
        deprecated=status == "deprecated",
        install_mode=install_mode,
        configuration_fields=fields,
        unsupported_reason=reason,
        install_spec=install_spec,
    )


def _installation(
    server: dict[str, Any],
) -> tuple[
    MarketplaceInstallMode,
    dict[str, Any],
    tuple[MarketplaceField, ...],
    str,
]:
    remotes = server.get("remotes")
    if isinstance(remotes, list):
        for remote in remotes:
            if not isinstance(remote, dict):
                continue
            transport = _transport_type(remote.get("type"))
            url = str(remote.get("url", "")).strip()
            if transport != "streamable_http" or not url:
                continue
            auth_type = _auth_type(remote)
            fields = _configuration_fields(remote.get("headers"))
            mode: MarketplaceInstallMode = (
                "configure"
                if fields
                else "oauth"
                if auth_type == "oauth"
                else "direct"
            )
            spec: dict[str, Any] = {
                "transport": transport,
                "url": url,
                "auth_type": auth_type,
            }
            if fields:
                spec["header_fields"] = [field.name for field in fields]
            return mode, spec, fields, ""
    packages = server.get("packages")
    if isinstance(packages, list):
        for package in packages:
            if not isinstance(package, dict):
                continue
            transport = package.get("transport")
            transport_type = _transport_type(
                transport.get("type") if isinstance(transport, dict) else transport
            )
            if transport_type != "stdio":
                continue
            registry = str(
                package.get("registryType") or package.get("registry_type") or ""
            ).casefold()
            identifier = str(package.get("identifier", "")).strip()
            version = str(package.get("version") or server.get("version") or "").strip()
            fields = _configuration_fields(package.get("environmentVariables"))
            mode: MarketplaceInstallMode = "configure" if fields else "direct"
            if registry == "npm" and identifier:
                spec: dict[str, Any] = {
                    "transport": "stdio",
                    "registry": "npm",
                    "package": identifier,
                    "version": version,
                }
                if fields:
                    spec["environment_fields"] = [field.name for field in fields]
                return mode, spec, fields, ""
            if registry in {"pypi", "python"} and identifier:
                executable = str(
                    package.get("executable") or package.get("entryPoint") or ""
                ).strip()
                if executable:
                    spec = {
                        "transport": "stdio",
                        "registry": "pypi",
                        "package": identifier,
                        "version": version,
                        "executable": executable,
                    }
                    if fields:
                        spec["environment_fields"] = [field.name for field in fields]
                    return mode, spec, fields, ""
    return "unsupported", {}, (), "Registry 未提供小满可直接运行的 HTTP、npm 或 PyPI 入口"


def _configuration_fields(value: object) -> tuple[MarketplaceField, ...]:
    if not isinstance(value, list):
        return ()
    fields: list[MarketplaceField] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        fields.append(
            MarketplaceField(
                name=name,
                label=str(raw.get("description") or name),
                required=bool(raw.get("isRequired", raw.get("required", False))),
                secret=bool(raw.get("isSecret", raw.get("secret", False))),
                placeholder=str(raw.get("valueHint") or raw.get("placeholder") or ""),
            )
        )
    return tuple(fields)


def _auth_type(remote: dict[str, Any]) -> str:
    direct = str(remote.get("auth_type", "")).casefold()
    auth = remote.get("auth") or remote.get("authentication")
    if isinstance(auth, dict):
        direct = str(auth.get("type", direct)).casefold()
    elif isinstance(auth, str):
        direct = auth.casefold()
    return "oauth" if "oauth" in direct else "none"


def _transport_type(value: object) -> str:
    return str(value or "").strip().casefold().replace("-", "_")


def _source_url(server: dict[str, Any]) -> str:
    repository = server.get("repository")
    if isinstance(repository, dict):
        url = repository.get("url")
        if isinstance(url, str):
            return url
    for key in ("websiteUrl", "homepage", "sourceUrl"):
        value = server.get(key)
        if isinstance(value, str):
            return value
    return ""


def _icon_url(server: dict[str, Any]) -> str:
    icons = server.get("icons")
    if isinstance(icons, list) and icons and isinstance(icons[0], dict):
        src = icons[0].get("src")
        return src if isinstance(src, str) else ""
    return ""


def _matches(item: MarketplaceItem, needle: str) -> bool:
    haystack = " ".join(
        (
            item.id,
            item.name,
            item.description,
            item.provider,
            str(item.install_spec.get("package", "")),
        )
    ).casefold()
    return needle in haystack
