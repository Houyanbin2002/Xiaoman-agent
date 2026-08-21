from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import threading
from types import ModuleType
from typing import cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

_pending_plugins: list[tuple[Path, Path]] = []
_pending_plugins_lock = threading.Lock()


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


def build_plugin_panels_js(project_root: Path, plugin_dir: Path) -> None:
    esbuild_cmd: list[str] | None = None
    for ts_path in _iter_plugin_panel_sources(plugin_dir):
        js_path = ts_path.with_suffix(".js")
        if js_path.exists() and js_path.stat().st_mtime >= ts_path.stat().st_mtime:
            continue
        if esbuild_cmd is None:
            esbuild_cmd = _esbuild_command(project_root)
        if esbuild_cmd is None:
            with _pending_plugins_lock:
                _pending_plugins.append((project_root, plugin_dir))
            return
        _run_esbuild(esbuild_cmd, ts_path, js_path, f"{plugin_dir.name}/{ts_path.stem}")


async def compile_pending_plugins() -> None:
    with _pending_plugins_lock:
        if not _pending_plugins:
            return
        pending = _pending_plugins.copy()
        _pending_plugins.clear()
    first_root = pending[0][0]

    logger.info("正在安装前端构建工具 (npx esbuild)...")
    esbuild_cmd = _esbuild_command(first_root)
    if esbuild_cmd is None:
        logger.warning("esbuild unavailable: neither local install nor npx was found")
        return
    proc = await asyncio.create_subprocess_exec(
        *esbuild_cmd,
        "--version",
        cwd=str(first_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(
            "npx esbuild 不可用 (%d)，插件面板未编译:\n%s",
            proc.returncode,
            stderr.decode("utf-8", errors="replace")[:500],
        )
        return
    version = stdout.decode("utf-8", errors="replace").strip()
    logger.info("npx esbuild 就绪 (%s)，开始编译插件面板...", version)
    for _root, plugin_dir in pending:
        for ts_path in _iter_plugin_panel_sources(plugin_dir):
            js_path = ts_path.with_suffix(".js")
            if not (
                js_path.exists() and js_path.stat().st_mtime >= ts_path.stat().st_mtime
            ):
                _run_esbuild(
                    esbuild_cmd,
                    ts_path,
                    js_path,
                    f"{plugin_dir.name}/{ts_path.stem}",
                )


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
        build_plugin_panels_js(project_root, plugin_dir)
        if (plugin_dir / "dashboard.py").exists():
            closeables.extend(load_plugin_dashboard(app, plugin_dir, workspace))
    return closeables


def register_plugin_routes(app: FastAPI, *, project_root: Path) -> None:
    @app.get("/api/dashboard/plugins")
    def list_dashboard_plugins() -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for plugin_id, plugin_dir in sorted(
            dashboard_plugin_dirs(project_root).items()
        ):
            if not plugin_dashboard_enabled(app, plugin_dir):
                continue
            build_plugin_panels_js(project_root, plugin_dir)
            panels: list[dict[str, object]] = []
            for js_path in sorted(plugin_dir.glob("dashboard_panel*.js")):
                css_path = js_path.with_suffix(".css")
                panels.append(
                    {
                        "name": js_path.stem,
                        "js_version": str(js_path.stat().st_mtime_ns),
                        "has_css": css_path.exists(),
                    }
                )
            if panels:
                result.append({"id": plugin_id, "panels": panels})
        return result

    @app.get("/plugins/{plugin_id}/{panel_name}.js")
    def get_plugin_panel_js(plugin_id: str, panel_name: str) -> FileResponse:
        if not panel_name.startswith("dashboard_panel"):
            raise HTTPException(status_code=404, detail="plugin panel not found")
        plugin_dir = _resolve_plugin_dir(
            dashboard_plugin_dirs(project_root),
            plugin_id,
        )
        if is_plugin_disabled(plugin_dir) or not plugin_dashboard_enabled(
            app, plugin_dir
        ):
            raise HTTPException(status_code=404, detail="plugin panel not found")
        build_plugin_panels_js(project_root, plugin_dir)
        js_path = plugin_dir / f"{panel_name}.js"
        if not js_path.exists():
            raise HTTPException(status_code=404, detail="plugin panel not found")
        return FileResponse(js_path, media_type="application/javascript")

    @app.get("/plugins/{plugin_id}/{panel_name}.css")
    def get_plugin_panel_css(plugin_id: str, panel_name: str) -> FileResponse:
        if not panel_name.startswith("dashboard_panel"):
            raise HTTPException(status_code=404, detail="plugin panel css not found")
        plugin_dir = _resolve_plugin_dir(
            dashboard_plugin_dirs(project_root),
            plugin_id,
        )
        if is_plugin_disabled(plugin_dir) or not plugin_dashboard_enabled(
            app, plugin_dir
        ):
            raise HTTPException(status_code=404, detail="plugin panel css not found")
        css_path = plugin_dir / f"{panel_name}.css"
        if not css_path.exists():
            raise HTTPException(status_code=404, detail="plugin panel css not found")
        return FileResponse(css_path, media_type="text/css")


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


def _esbuild_command(project_root: Path) -> list[str] | None:
    bin_name = "esbuild.cmd" if os.name == "nt" else "esbuild"
    local_bin = project_root / "node_modules" / ".bin" / bin_name
    if local_bin.exists():
        return [str(local_bin)]
    if os.name == "nt":
        cmd_bin = shutil.which("cmd.exe") or shutil.which("cmd")
        npx_bin = shutil.which("npx.cmd") or shutil.which("npx")
        if cmd_bin and npx_bin:
            return [cmd_bin, "/d", "/s", "/c", "npx", "--yes", "esbuild"]
        return None
    npx_bin = shutil.which("npx")
    if npx_bin:
        return [npx_bin, "--yes", "esbuild"]
    return None


def _iter_plugin_panel_sources(plugin_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in plugin_dir.glob("dashboard_panel*")
        if path.suffix in {".ts", ".tsx"}
    )


def _run_esbuild(cmd: list[str], ts_path: Path, js_path: Path, name: str) -> None:
    try:
        result = subprocess.run(
            [
                *cmd,
                str(ts_path),
                f"--outfile={js_path}",
                "--bundle",
                "--platform=browser",
                "--target=es2020",
                "--format=esm",
                "--jsx=automatic",
                "--external:react",
                "--external:react-dom",
                "--external:react-dom/client",
                "--external:react/jsx-runtime",
                "--external:@xiaoman/dashboard-ui",
                "--external:@akashic/dashboard-ui",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("插件面板已编译: %s", name)
        else:
            logger.warning("插件面板编译失败 (%s):\n%s", name, result.stderr)
    except Exception as exc:
        logger.warning("插件面板编译异常 (%s): %s", name, exc)


def _resolve_plugin_dir(plugin_dirs: dict[str, Path], plugin_id: str) -> Path:
    if not plugin_id or "/" in plugin_id or "\\" in plugin_id:
        raise HTTPException(status_code=400, detail="invalid plugin id")
    windows_path = PureWindowsPath(plugin_id)
    if Path(plugin_id).is_absolute() or windows_path.drive or windows_path.root:
        raise HTTPException(status_code=400, detail="invalid plugin id")
    plugin_dir = plugin_dirs.get(plugin_id)
    if plugin_dir is None:
        raise HTTPException(status_code=404, detail="plugin not found")
    return plugin_dir


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
