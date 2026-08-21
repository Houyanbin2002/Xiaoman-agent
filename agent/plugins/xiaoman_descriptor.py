from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from agent.mcp.config import McpServerConfig

logger = logging.getLogger(__name__)

XIAOMAN_PLUGIN_MANIFEST = ".xiaoman-plugin/plugin.json"
# Compatibility is intentionally isolated in this module.  New plugins and all
# user-facing documentation use the Xiaoman manifest exclusively.
_LEGACY_PLUGIN_MANIFEST = ".aka-plugin/plugin.json"
_LEGACY_RUNTIME_KEY = "akashic"


@dataclass(frozen=True)
class PluginDescriptor:
    name: str
    version: str
    description: str
    root: Path
    raw_manifest: dict[str, object]
    lifecycle_entry: Path | None = None
    lifecycle_class: str = ""
    skill_roots: tuple[Path, ...] = ()
    drift_skill_roots: tuple[Path, ...] = ()
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    manifest_format: Literal["xiaoman", "legacy"] = "xiaoman"
    legacy_data_env: bool = False

    @property
    def is_legacy(self) -> bool:
        return self.manifest_format == "legacy"

    @property
    def uses_legacy_data_env(self) -> bool:
        return self.is_legacy or self.legacy_data_env


def has_plugin_descriptor(plugin_root: Path) -> bool:
    """Return whether a directory declares a supported plugin manifest."""
    return any((plugin_root / item).is_file() for item in _manifest_paths())


def load_plugin_descriptor(plugin_root: Path) -> PluginDescriptor | None:
    manifest_path, manifest_format = _find_manifest(plugin_root)
    if manifest_path is None:
        return None
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("插件清单读取失败 (%s): %s", manifest_path, e)
        return None
    if not isinstance(loaded, dict):
        logger.warning("插件清单格式错误，期望 dict (%s)", manifest_path)
        return None

    raw = cast(dict[str, object], loaded)
    name = str(raw.get("name") or plugin_root.name).strip()
    if not name:
        logger.warning("xiaoman plugin manifest 缺少 name (%s)", manifest_path)
        return None

    version = str(raw.get("version") or "").strip()
    description = str(raw.get("description") or "").strip()
    paths = _as_dict(raw.get("paths"))
    xiaoman = _as_dict(
        raw.get("xiaoman")
        if manifest_format == "xiaoman"
        else raw.get(_LEGACY_RUNTIME_KEY)
    )
    lifecycle = _as_dict(xiaoman.get("lifecycle"))
    compatibility = _as_dict(xiaoman.get("compatibility"))

    lifecycle_entry = _resolve_optional_path(
        plugin_root,
        str(lifecycle.get("entry") or "").strip(),
    )
    lifecycle_class = str(lifecycle.get("class") or "").strip()
    skill_roots = _resolve_root_dirs(plugin_root, paths.get("skills"))
    drift_skill_roots = _resolve_root_dirs(plugin_root, paths.get("drift_skills"))
    mcp_servers = _load_mcp_servers(
        plugin_root,
        paths.get("mcp_servers"),
    )

    return PluginDescriptor(
        name=name,
        version=version,
        description=description,
        root=plugin_root,
        raw_manifest=raw,
        lifecycle_entry=lifecycle_entry,
        lifecycle_class=lifecycle_class,
        skill_roots=skill_roots,
        drift_skill_roots=drift_skill_roots,
        mcp_servers=mcp_servers,
        manifest_format=manifest_format,
        legacy_data_env=bool(compatibility.get("legacy_data_env", False)),
    )


def _manifest_paths() -> tuple[str, str]:
    return (XIAOMAN_PLUGIN_MANIFEST, _LEGACY_PLUGIN_MANIFEST)


def _find_manifest(
    plugin_root: Path,
) -> tuple[Path | None, Literal["xiaoman", "legacy"]]:
    primary = plugin_root / XIAOMAN_PLUGIN_MANIFEST
    if primary.is_file():
        return primary, "xiaoman"
    legacy = plugin_root / _LEGACY_PLUGIN_MANIFEST
    if legacy.is_file():
        return legacy, "legacy"
    return None, "xiaoman"


def _resolve_root_dirs(plugin_root: Path, raw_value: object) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for path_text in _as_str_list(raw_value):
        path = (plugin_root / path_text).resolve(strict=False)
        if path.is_dir():
            resolved.append(path)
    return tuple(resolved)


def _load_mcp_servers(
    plugin_root: Path,
    raw_value: object,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path_text in _as_str_list(raw_value):
        config_path = (plugin_root / path_text).resolve(strict=False)
        if not config_path.exists():
            continue
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("xiaoman plugin mcp 配置读取失败 (%s): %s", config_path, e)
            continue
        if not isinstance(loaded, dict):
            logger.warning("xiaoman plugin mcp 配置格式错误，期望 dict (%s)", config_path)
            continue
        servers = _as_dict(cast(dict[str, object], loaded).get("servers"))
        for server_name, server_value in servers.items():
            server = _normalize_server(plugin_root, server_name, server_value)
            if server is None:
                continue
            merged[server_name] = server
    return merged


def _normalize_server(
    plugin_root: Path,
    server_name: str,
    raw_value: object,
) -> dict[str, Any] | None:
    server = _as_dict(raw_value)
    command = _as_str_list(server.get("command"))
    normalized_command = (
        [_normalize_command_item(plugin_root, item) for item in command]
        if command
        else []
    )
    env = {
        str(key): str(value)
        for key, value in _as_dict(server.get("env")).items()
    }
    cwd_raw = str(server.get("cwd") or "").strip()
    cwd = (
        str((plugin_root / cwd_raw).resolve(strict=False))
        if cwd_raw and not Path(cwd_raw).is_absolute()
        else cwd_raw or str(plugin_root.resolve(strict=False))
    )
    auth = _as_dict(server.get("auth"))
    normalized: dict[str, Any] = {
        "transport": str(server.get("transport") or "").strip(),
        "command": normalized_command,
        "url": str(server.get("url") or "").strip(),
        "env": env,
        "cwd": cwd,
        "auth": {
            "type": str(auth.get("type") or "none").strip(),
            "scopes": str(auth.get("scopes") or "").strip(),
        },
        "headers": {
            str(key): str(value)
            for key, value in _as_dict(server.get("headers")).items()
        },
    }
    try:
        config = McpServerConfig.from_raw(normalized)
    except ValueError as exc:
        logger.warning("xiaoman plugin mcp server 配置无效 (%s): %s", server_name, exc)
        return None
    if config.transport == "stdio":
        return {
            "transport": "stdio",
            "command": list(config.command),
            "env": config.env,
            "cwd": config.cwd,
        }
    return {
        "transport": config.transport,
        "url": config.url,
        "auth": {"type": config.auth_type, "scopes": config.scopes},
        "headers": config.headers,
    }


def _resolve_optional_path(plugin_root: Path, value: str) -> Path | None:
    if not value:
        return None
    path = (plugin_root / value).resolve(strict=False)
    if not path.exists():
        logger.warning("xiaoman plugin lifecycle 入口不存在 (%s)", path)
        return None
    return path


def _normalize_command_item(plugin_root: Path, value: str) -> str:
    if not value:
        return value
    if Path(value).is_absolute():
        return value
    if "/" not in value and "\\" not in value and not value.startswith("."):
        return value
    return str((plugin_root / value).resolve(strict=False))


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return result
