from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

from agent.plugins.memory import MemoryPlugin

MemoryPluginFactory = Callable[[], MemoryPlugin]
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_default_memory_plugin() -> MemoryPlugin:
    from plugins.default_memory.memory_plugin import MemoryPlugin as DefaultMemoryPlugin

    return DefaultMemoryPlugin()


_MEMORY_PLUGIN_WIRING: dict[str, MemoryPluginFactory] = {
    "default": _build_default_memory_plugin,
}


def resolve_memory_plugin(name: str) -> MemoryPlugin:
    normalized = (name or "akasha").strip() or "akasha"
    if normalized in _MEMORY_PLUGIN_WIRING:
        return _MEMORY_PLUGIN_WIRING[normalized]()
    plugin = _load_memory_plugin_from_dir(normalized)
    if plugin is None:
        choices = ", ".join(sorted(_MEMORY_PLUGIN_WIRING))
        raise ValueError(f"未知 memory engine: {normalized}；可选值: {choices}")
    return plugin


def _load_memory_plugin_from_dir(name: str) -> MemoryPlugin | None:
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"memory engine 名称非法: {name}")
    plugin_path = _PROJECT_ROOT / "plugins" / name / "memory_plugin.py"
    if not plugin_path.exists():
        return None
    module_name = f"akasic_memory_plugin_{name}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if hasattr(module, "create_memory_plugin"):
        plugin = module.create_memory_plugin()
    elif hasattr(module, "MemoryPlugin"):
        plugin = module.MemoryPlugin()
    else:
        raise ValueError(f"{plugin_path} 缺少 create_memory_plugin 或 MemoryPlugin")
    if not isinstance(plugin, MemoryPlugin):
        raise TypeError(f"{plugin_path} 未返回 MemoryPlugin")
    return plugin
