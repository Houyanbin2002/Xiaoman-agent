"""Marketplace discovery and installation for independently published plugins.

The marketplace layer owns catalog checkout and catalog parsing.  It deliberately
does not know about runtime loading; installation is delegated to ``install`` so
direct Git installs and marketplace installs share one cache and registry path.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugins.install import (
    PluginInstallResult,
    clone_git_source,
    install_git_plugin,
    install_plugin_directory,
    normalize_git_source,
    xiaoman_plugins_root,
)

logger = logging.getLogger(__name__)

MARKETPLACE_MANIFEST = ".xiaoman-plugin/marketplace.json"
_MARKETPLACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class MarketplacePlugin:
    name: str
    source: str | dict[str, object]
    description: str = ""
    version: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketplaceDescriptor:
    name: str
    root: Path
    plugins: tuple[MarketplacePlugin, ...]
    owner_name: str = ""


@dataclass(frozen=True)
class MarketplaceRecord:
    name: str
    source: str
    ref_name: str = ""


def marketplace_registry_path(plugins_home: Path | None = None) -> Path:
    return (plugins_home or xiaoman_plugins_root()) / "marketplaces.json"


def marketplace_checkout_root(plugins_home: Path | None = None) -> Path:
    return (plugins_home or xiaoman_plugins_root()) / "marketplaces"


def load_marketplace_descriptor(root: Path) -> MarketplaceDescriptor:
    manifest_path = root / MARKETPLACE_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"市场缺少 {MARKETPLACE_MANIFEST}")
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"市场清单不是有效 JSON: {manifest_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("市场清单必须是 JSON 对象")
    raw = cast(dict[str, object], loaded)
    name = _validate_id(raw.get("name"), label="市场 name")
    owner = _as_dict(raw.get("owner"))
    raw_plugins = raw.get("plugins")
    if not isinstance(raw_plugins, list):
        raise ValueError("市场清单必须包含 plugins 数组")

    plugins: list[MarketplacePlugin] = []
    seen: set[str] = set()
    for item in raw_plugins:
        if not isinstance(item, dict):
            raise ValueError("市场插件项必须是 JSON 对象")
        plugin = _parse_marketplace_plugin(cast(dict[str, object], item))
        if plugin.name in seen:
            raise ValueError(f"市场内存在重复插件: {plugin.name}")
        seen.add(plugin.name)
        plugins.append(plugin)
    return MarketplaceDescriptor(
        name=name,
        root=root.resolve(strict=False),
        plugins=tuple(plugins),
        owner_name=str(owner.get("name") or "").strip(),
    )


def add_marketplace(
    *,
    source: str,
    ref_name: str = "",
    expected_name: str = "",
    plugins_home: Path | None = None,
) -> MarketplaceDescriptor:
    """Fetch a catalog and persist its checked-out, inspectable copy locally."""
    home = plugins_home or xiaoman_plugins_root()
    normalized_source = normalize_git_source(source)
    checkout_root = marketplace_checkout_root(home)
    checkout_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checkout_root, prefix="catalog-") as temp_dir:
        clone_root = Path(temp_dir) / "source"
        clone_git_source(
            source=normalized_source,
            destination=clone_root,
            ref_name=ref_name,
            sparse_paths=[],
        )
        descriptor = load_marketplace_descriptor(clone_root)
        if expected_name and descriptor.name != expected_name:
            raise ValueError("市场刷新后的名称发生变化，已拒绝替换本地市场")
        _replace_marketplace_checkout(
            source_root=clone_root,
            target_root=checkout_root / descriptor.name,
        )

    records = _load_marketplace_records(home)
    records[descriptor.name] = MarketplaceRecord(
        name=descriptor.name,
        source=normalized_source,
        ref_name=ref_name.strip(),
    )
    _write_marketplace_records(home, records)
    return load_marketplace_descriptor(checkout_root / descriptor.name)


def refresh_marketplace(
    name: str,
    *,
    plugins_home: Path | None = None,
) -> MarketplaceDescriptor:
    home = plugins_home or xiaoman_plugins_root()
    normalized_name = _validate_id(name, label="市场名称")
    record = _load_marketplace_records(home).get(normalized_name)
    if record is None:
        raise ValueError(f"市场不存在: {normalized_name}")
    refreshed = add_marketplace(
        source=record.source,
        ref_name=record.ref_name,
        expected_name=normalized_name,
        plugins_home=home,
    )
    return refreshed


def list_marketplaces(
    *,
    plugins_home: Path | None = None,
) -> list[dict[str, object]]:
    home = plugins_home or xiaoman_plugins_root()
    records = _load_marketplace_records(home)
    result: list[dict[str, object]] = []
    for name, record in sorted(records.items()):
        checkout = marketplace_checkout_root(home) / name
        try:
            descriptor = load_marketplace_descriptor(checkout)
        except ValueError:
            descriptor = None
        result.append(
            {
                "name": name,
                "source": _safe_source_label(record.source),
                "ref": record.ref_name,
                "available": descriptor is not None,
                "plugin_count": len(descriptor.plugins) if descriptor else 0,
                "owner": descriptor.owner_name if descriptor else "",
            }
        )
    return result


def list_marketplace_plugins(
    name: str,
    *,
    plugins_home: Path | None = None,
) -> list[dict[str, object]]:
    home = plugins_home or xiaoman_plugins_root()
    normalized_name = _validate_id(name, label="市场名称")
    if normalized_name not in _load_marketplace_records(home):
        raise ValueError(f"市场不存在: {normalized_name}")
    descriptor = load_marketplace_descriptor(
        marketplace_checkout_root(home) / normalized_name
    )
    return [
        {
            "name": plugin.name,
            "description": plugin.description,
            "version": plugin.version,
            "tags": list(plugin.tags),
        }
        for plugin in descriptor.plugins
    ]


def install_marketplace_plugin(
    *,
    marketplace: str,
    plugin_name: str,
    plugins_home: Path | None = None,
) -> PluginInstallResult:
    """Install one catalog entry without treating the catalog itself as code."""
    home = plugins_home or xiaoman_plugins_root()
    normalized_marketplace = _validate_id(marketplace, label="市场名称")
    normalized_plugin = _validate_id(plugin_name, label="插件名称")
    if normalized_marketplace not in _load_marketplace_records(home):
        raise ValueError(f"市场不存在: {normalized_marketplace}")
    descriptor = load_marketplace_descriptor(
        marketplace_checkout_root(home) / normalized_marketplace
    )
    entry = next(
        (item for item in descriptor.plugins if item.name == normalized_plugin),
        None,
    )
    if entry is None:
        raise ValueError(
            f"市场 {normalized_marketplace} 未提供插件: {normalized_plugin}"
        )

    if isinstance(entry.source, str):
        plugin_root = _resolve_relative_plugin_source(descriptor.root, entry.source)
        return install_plugin_directory(
            plugin_root=plugin_root,
            marketplace=normalized_marketplace,
            install_source=f"{normalized_marketplace}:{entry.source}",
            plugins_home=home,
        )

    source, ref_name, source_subdir = _resolve_external_plugin_source(entry.source)
    sparse_paths = [source_subdir] if source_subdir else []
    return install_git_plugin(
        source=source,
        marketplace=normalized_marketplace,
        ref_name=ref_name,
        sparse_paths=sparse_paths,
        source_subdir=source_subdir,
        plugins_home=home,
    )


def _parse_marketplace_plugin(raw: dict[str, object]) -> MarketplacePlugin:
    name = _validate_id(raw.get("name"), label="插件 name")
    source = raw.get("source")
    if isinstance(source, str):
        normalized_source: str | dict[str, object] = source.strip()
        if not normalized_source.startswith("./"):
            raise ValueError(f"插件 {name} 的相对 source 必须以 ./ 开头")
    elif isinstance(source, dict):
        normalized_source = {
            str(key): value
            for key, value in cast(dict[object, object], source).items()
            if isinstance(key, str)
        }
        _ = _resolve_external_plugin_source(normalized_source)
    else:
        raise ValueError(f"插件 {name} 缺少 source")
    tags_raw = raw.get("tags")
    tags = (
        tuple(
            item.strip() for item in tags_raw if isinstance(item, str) and item.strip()
        )
        if isinstance(tags_raw, list)
        else ()
    )
    return MarketplacePlugin(
        name=name,
        source=normalized_source,
        description=str(raw.get("description") or "").strip(),
        version=str(raw.get("version") or "").strip(),
        tags=tags,
    )


def _resolve_relative_plugin_source(marketplace_root: Path, source: str) -> Path:
    path_text = source.strip().replace("\\", "/")
    relative = Path(path_text)
    if (
        not path_text.startswith("./")
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError("市场插件 source 必须是市场仓库内的相对路径")
    root = marketplace_root.resolve(strict=False)
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("市场插件 source 不能越出市场仓库") from exc
    return candidate


def _resolve_external_plugin_source(
    source: dict[str, object],
) -> tuple[str, str, str]:
    kind = str(source.get("type") or source.get("source") or "").strip().lower()
    ref_name = str(source.get("sha") or source.get("ref") or "").strip()
    source_subdir = _safe_subdir(str(source.get("path") or "").strip())
    if kind == "github":
        repository = str(source.get("repo") or "").strip()
        return normalize_git_source(repository), ref_name, source_subdir
    if kind in {"git", "url", "git-subdir"}:
        url = str(source.get("url") or "").strip()
        if not url:
            raise ValueError("外部插件 source 缺少 url")
        if kind == "git-subdir" and not source_subdir:
            raise ValueError("git-subdir source 缺少 path")
        return normalize_git_source(url), ref_name, source_subdir
    raise ValueError("外部插件 source.type 仅支持 github、git 或 git-subdir")


def _replace_marketplace_checkout(*, source_root: Path, target_root: Path) -> None:
    stage = target_root.with_name(f".{target_root.name}.staging")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(source_root, stage, ignore=shutil.ignore_patterns(".git"))
    if target_root.exists():
        shutil.rmtree(target_root)
    stage.replace(target_root)


def _load_marketplace_records(home: Path) -> dict[str, MarketplaceRecord]:
    path = marketplace_registry_path(home)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("市场注册表读取失败 (%s): %s", path, exc)
        return {}
    if not isinstance(loaded, dict) or not isinstance(loaded.get("marketplaces"), dict):
        return {}
    records: dict[str, MarketplaceRecord] = {}
    for raw_name, raw_value in cast(
        dict[object, object], loaded["marketplaces"]
    ).items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, dict):
            continue
        item = cast(dict[object, object], raw_value)
        try:
            name = _validate_id(raw_name, label="市场名称")
        except ValueError:
            continue
        source = str(item.get("source") or "").strip()
        if not source:
            continue
        records[name] = MarketplaceRecord(
            name=name,
            source=source,
            ref_name=str(item.get("ref") or "").strip(),
        )
    return records


def _write_marketplace_records(
    home: Path,
    records: dict[str, MarketplaceRecord],
) -> None:
    path = marketplace_registry_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "marketplaces": {
            name: {"source": record.source, "ref": record.ref_name}
            for name, record in sorted(records.items())
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_id(value: object, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not _MARKETPLACE_ID.fullmatch(normalized):
        raise ValueError(
            f"{label}只能包含字母、数字、点、下划线或连字符，长度不超过 64"
        )
    return normalized


def _safe_subdir(value: str) -> str:
    if not value:
        return ""
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("外部插件 path 必须是仓库内的相对路径")
    return normalized


def _safe_source_label(value: str) -> str:
    """Return a displayable source without leaking URL credentials."""
    scheme, marker, rest = value.partition("://")
    if marker and "@" in rest:
        return f"{scheme}{marker}{rest.rsplit('@', 1)[-1]}"
    return value


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}
