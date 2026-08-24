from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import ModuleType
from typing import cast

from fastapi import FastAPI

logger = logging.getLogger(__name__)

def is_plugin_disabled(plugin_dir: Path) -> bool:
    return (plugin_dir / "plugin.disabled").exists()


def dashboard_plugin_dirs(project_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    plugins_root = project_root / "plugins"
    if plugins_root.is_dir():
        for plugin_dir in sorted(plugins_root.iterdir()):
            if not plugin_dir.is_dir() or is_plugin_disabled(plugin_dir):
                continue
            result[plugin_dir.name] = plugin_dir

    registry_path = Path.home() / ".xiaoman-plugin" / "registry.json"
    if not registry_path.exists():
        return result
    try:
        loaded = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("插件注册表读取失败 (%s): %s", registry_path, exc)
        return result
    if not isinstance(loaded, dict):
        return result
    raw_plugins = loaded.get("plugins")
    if not isinstance(raw_plugins, dict):
        return result
    for raw_plugin_id, raw_entry in sorted(raw_plugins.items()):
        if not isinstance(raw_plugin_id, str) or not isinstance(raw_entry, dict):
            continue
        entry = cast(dict[str, object], raw_entry)
        if str(entry.get("source_type") or "") != "installed":
            continue
        if entry.get("active") is False:
            continue
        plugin_root_text = str(entry.get("plugin_root") or "").strip()
        if not plugin_root_text:
            continue
        plugin_root = Path(plugin_root_text).resolve(strict=False)
        if not plugin_root.is_dir() or is_plugin_disabled(plugin_root):
            continue
        result[raw_plugin_id] = plugin_root
    return result


def install_plugin_dashboards(
    app: FastAPI,
    *,
    project_root: Path,
    workspace: Path,
) -> list[object]:
    closeables: list[object] = []
    for _plugin_id, plugin_dir in sorted(dashboard_plugin_dirs(project_root).items()):
        if not plugin_dashboard_enabled(app, plugin_dir):
            continue
        if (plugin_dir / "dashboard.py").exists():
            closeables.extend(load_plugin_dashboard(app, plugin_dir, workspace))
    return closeables


def plugin_dashboard_enabled(app: FastAPI, plugin_dir: Path) -> bool:
    dash_path = plugin_dir / "dashboard.py"
    if not dash_path.exists():
        return False
    try:
        module = _load_plugin_dashboard_module(plugin_dir)
    except Exception as exc:
        logger.warning("插件 dashboard 检查失败 (%s): %s", plugin_dir.name, exc)
        return False
    enabled = getattr(module, "plugin_enabled", None)
    if not callable(enabled):
        return True
    return bool(enabled(app))


def load_plugin_dashboard(
    app: FastAPI,
    plugin_dir: Path,
    workspace: Path,
) -> list[object]:
    try:
        module = _load_plugin_dashboard_module(plugin_dir)
        if hasattr(module, "register"):
            result = module.register(app, plugin_dir, workspace)
            logger.info("插件 dashboard 已挂载: %s", plugin_dir.name)
            return dashboard_closeables(result)
    except Exception as exc:
        logger.warning("插件 dashboard 挂载失败 (%s): %s", plugin_dir.name, exc)
    return []


def close_dashboard_value(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        _ = close()


def dashboard_closeables(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        items = cast(list[object], value)
        return [item for item in items if _is_dashboard_closeable(item)]
    if _is_dashboard_closeable(value):
        return [value]
    return []


def _is_dashboard_closeable(value: object) -> bool:
    return callable(getattr(value, "close", None))


def _load_plugin_dashboard_module(plugin_dir: Path) -> ModuleType:
    dash_path = plugin_dir / "dashboard.py"
    module_name = _dashboard_module_name(plugin_dir)
    spec = importlib.util.spec_from_file_location(
        module_name,
        dash_path,
        submodule_search_locations=[str(plugin_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {dash_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _dashboard_module_name(plugin_dir: Path) -> str:
    raw = str(plugin_dir.resolve(strict=False))
    normalized = "".join(char if char.isalnum() else "_" for char in raw)
    return f"xiaoman_dashboard_plugin_{normalized}"
