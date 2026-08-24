from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugins.xiaoman_descriptor import (
    XIAOMAN_PLUGIN_MANIFEST,
    PluginDescriptor,
    load_plugin_descriptor,
)
from agent.plugins.global_registry import (
    load_plugin_registry,
    remove_plugin_registry_entry,
    upsert_plugin_registry_entry,
)


@dataclass(frozen=True)
class PluginInstallResult:
    plugin_name: str
    plugin_version: str
    marketplace: str
    installed_path: Path
    data_path: Path


@dataclass(frozen=True)
class PluginUninstallResult:
    plugin_id: str
    removed_path: Path
    data_path: Path | None
    data_removed: bool


_INSTALL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_INSTALL_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}$")
_GITHUB_REPOSITORY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)


def xiaoman_plugins_root() -> Path:
    return Path.home() / ".xiaoman-plugin"


def install_git_plugin(
    *,
    source: str,
    marketplace: str,
    ref_name: str = "",
    sparse_paths: list[str] | None = None,
    source_subdir: str = "",
    plugins_home: Path | None = None,
) -> PluginInstallResult:
    home = plugins_home or xiaoman_plugins_root()
    normalized_marketplace = _validate_install_id(marketplace, label="市场名称")
    normalized_source = normalize_git_source(source)
    marketplace_root = home / "marketplaces" / normalized_marketplace
    marketplace_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=marketplace_root, prefix="clone-") as clone_dir:
        clone_root = Path(clone_dir)
        clone_git_source(
            source=normalized_source,
            destination=clone_root,
            ref_name=ref_name,
            sparse_paths=sparse_paths or [],
        )
        plugin_root = _resolve_plugin_root(clone_root, source_subdir)
        return install_plugin_directory(
            plugin_root=plugin_root,
            marketplace=normalized_marketplace,
            install_source=normalized_source,
            legacy_name_hint=_source_name_hint(normalized_source),
            legacy_version_hint=_git_revision(clone_root),
            plugins_home=home,
        )


def install_plugin_directory(
    *,
    plugin_root: Path,
    marketplace: str,
    install_source: str,
    legacy_name_hint: str = "",
    legacy_version_hint: str = "",
    plugins_home: Path | None = None,
) -> PluginInstallResult:
    """Activate a verified plugin directory into the versioned local cache.

    Marketplace installation uses this for a relative plugin directory; direct
    Git installation reaches the same path after cloning.  Keeping activation
    here prevents the two install surfaces from developing different cache or
    registry semantics.
    """
    home = plugins_home or xiaoman_plugins_root()
    normalized_marketplace = _validate_install_id(marketplace, label="市场名称")
    resolved_root = plugin_root.resolve(strict=False)
    if not resolved_root.is_dir():
        raise ValueError(f"插件目录不存在: {plugin_root}")
    descriptor = load_plugin_descriptor(resolved_root)
    generated_manifest_kind = ""
    if descriptor is None and (resolved_root / "plugin.py").is_file():
        generated_manifest_kind = "legacy"
        descriptor = _infer_legacy_plugin_descriptor(
            resolved_root,
            name_hint=legacy_name_hint,
            version_hint=legacy_version_hint,
        )
    elif descriptor is None and (resolved_root / "SKILL.md").is_file():
        generated_manifest_kind = "skill_package"
        descriptor = _infer_skill_package_descriptor(
            resolved_root,
            name_hint=legacy_name_hint,
            version_hint=legacy_version_hint,
        )
    if descriptor is None:
        raise ValueError(f"插件缺少 {XIAOMAN_PLUGIN_MANIFEST} 或兼容旧版清单")
    _validate_install_id(descriptor.name, label="插件名称")
    _validate_version(descriptor.version)

    cache_root = home / "cache" / normalized_marketplace
    data_root = home / "data"
    cache_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    install_result = _activate_plugin_version(
        descriptor=descriptor,
        marketplace=normalized_marketplace,
        clone_root=resolved_root,
        cache_root=cache_root,
        data_root=data_root,
        generated_manifest_kind=generated_manifest_kind,
    )
    installed_descriptor = load_plugin_descriptor(install_result.installed_path) or descriptor
    plugin_id = f"{descriptor.name}@{normalized_marketplace}"
    _ = upsert_plugin_registry_entry(
        plugin_id,
        {
            "plugin_id": plugin_id,
            "name": installed_descriptor.name,
            "marketplace": normalized_marketplace,
            "source_type": "installed",
            "version": installed_descriptor.version,
            "description": installed_descriptor.description,
            "enabled": True,
            "local_disabled": False,
            "active": False,
            "plugin_root": str(install_result.installed_path),
            "data_dir": str(install_result.data_path),
            "lifecycle_entry": str(installed_descriptor.lifecycle_entry or ""),
            "manifest_format": (
                f"generated_{generated_manifest_kind}"
                if generated_manifest_kind
                else descriptor.manifest_format
            ),
            "capabilities": {
                "lifecycle": bool(installed_descriptor.lifecycle_entry),
                "skills": bool(
                    installed_descriptor.skill_roots
                    or installed_descriptor.drift_skill_roots
                ),
                "mcp": bool(installed_descriptor.mcp_servers),
            },
            "skills": _collect_skill_names(installed_descriptor.skill_roots),
            "drift_skills": _collect_skill_names(installed_descriptor.drift_skill_roots),
            "mcp_servers": sorted(installed_descriptor.mcp_servers.keys()),
            "install_source": install_source,
        },
        plugins_home=home,
    )
    return install_result


def uninstall_plugin(
    *,
    plugin_id: str,
    purge_data: bool = False,
    plugins_home: Path | None = None,
) -> PluginUninstallResult:
    """Remove one installed plugin while preserving its state by default."""
    home = plugins_home or xiaoman_plugins_root()
    registry = load_plugin_registry(home)
    entry = registry.get(plugin_id)
    if entry is None:
        raise ValueError(f"已安装插件不存在: {plugin_id}")
    if str(entry.get("source_type") or "") != "installed":
        raise ValueError("内置插件不能在此处卸载")
    plugin_root_text = str(entry.get("plugin_root") or "").strip()
    if not plugin_root_text:
        raise ValueError("插件注册表缺少 plugin_root")
    plugin_root = Path(plugin_root_text).resolve(strict=False)
    cache_root = (home / "cache").resolve(strict=False)
    _assert_within(plugin_root, cache_root, label="插件缓存路径")
    data_path = _registered_data_path(entry, home)
    if plugin_root.exists():
        _remove_tree(plugin_root)
    _remove_empty_cache_parents(plugin_root, cache_root)

    data_removed = False
    if purge_data and data_path is not None and data_path.exists():
        shutil.rmtree(data_path)
        data_removed = True
    _ = remove_plugin_registry_entry(plugin_id, plugins_home=home)
    return PluginUninstallResult(
        plugin_id=plugin_id,
        removed_path=plugin_root,
        data_path=data_path,
        data_removed=data_removed,
    )


def normalize_git_source(source: str) -> str:
    """Accept GitHub's concise ``owner/repository`` notation safely."""
    normalized = source.strip()
    if not normalized:
        raise ValueError("插件来源不能为空")
    scheme, marker, remainder = normalized.partition("://")
    authority = remainder.split("/", 1)[0] if marker else ""
    if marker and "@" in authority:
        raise ValueError("插件来源 URL 不能包含凭据，请使用 Git 凭据管理器")
    if _GITHUB_REPOSITORY.fullmatch(normalized):
        return f"https://github.com/{normalized}.git"
    return normalized


def clone_git_source(
    *,
    source: str,
    destination: Path,
    ref_name: str,
    sparse_paths: list[str],
) -> None:
    if sparse_paths:
        _run_git(
            [
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                source,
                str(destination),
            ]
        )
        _run_git(
            ["sparse-checkout", "set", *sparse_paths],
            cwd=destination,
        )
        _run_git(
            ["checkout", ref_name or "HEAD"],
            cwd=destination,
        )
        return
    _run_git(["clone", source, str(destination)])
    if ref_name:
        _run_git(["checkout", ref_name], cwd=destination)


def _resolve_plugin_root(clone_root: Path, source_subdir: str) -> Path:
    if not source_subdir:
        return clone_root
    relative = _safe_relative_path(source_subdir, label="插件子目录")
    root = clone_root.resolve(strict=False)
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("插件子目录不能越出仓库") from exc
    return candidate


def _activate_plugin_version(
    *,
    descriptor: PluginDescriptor,
    marketplace: str,
    clone_root: Path,
    cache_root: Path,
    data_root: Path,
    generated_manifest_kind: str,
) -> PluginInstallResult:
    data_path = data_root / f"{descriptor.name}-{marketplace}"
    data_path.mkdir(parents=True, exist_ok=True)
    plugin_base = cache_root / descriptor.name
    target_root = plugin_base / descriptor.version
    plugin_base.mkdir(parents=True, exist_ok=True)
    if target_root.exists():
        shutil.rmtree(target_root)
    _ = shutil.copytree(clone_root, target_root)
    if generated_manifest_kind == "skill_package":
        _materialize_skill_package(target_root, descriptor)
    if generated_manifest_kind:
        _write_generated_manifest(target_root, descriptor)
    installed_descriptor = load_plugin_descriptor(target_root) or descriptor
    _prepare_plugin_python_runtimes(target_root, installed_descriptor)
    _prepare_plugin_mcp_runtimes(target_root, installed_descriptor, data_path)
    _remove_old_versions(plugin_base, descriptor.version)
    return PluginInstallResult(
        plugin_name=descriptor.name,
        plugin_version=descriptor.version,
        marketplace=marketplace,
        installed_path=target_root,
        data_path=data_path,
    )


def _remove_old_versions(
    plugin_base: Path,
    active_version: str,
) -> None:
    for child in plugin_base.iterdir():
        if not child.is_dir() or child.name == active_version:
            continue
        shutil.rmtree(child)


def _prepare_plugin_python_runtimes(
    plugin_root: Path,
    descriptor: PluginDescriptor,
) -> None:
    """Prepare isolated dependencies declared by a native Xiaoman plugin."""
    for runtime_relpath in _manifest_path_list(descriptor, "python_runtimes"):
        relative = _safe_relative_path(runtime_relpath, label="Python 运行时目录")
        runtime_root = (plugin_root / relative).resolve(strict=False)
        _assert_within(
            runtime_root,
            plugin_root.resolve(strict=False),
            label="Python 运行时目录",
        )
        if not runtime_root.is_dir():
            raise ValueError(f"Python 运行时目录不存在: {runtime_relpath}")
        requirements = runtime_root / "requirements.txt"
        if not requirements.is_file():
            raise ValueError(
                f"Python 运行时缺少 requirements.txt: {runtime_relpath}"
            )
        _ensure_python_runtime(
            runtime_root,
            requirements,
            f"{descriptor.name}:{runtime_relpath}",
        )


def _prepare_plugin_mcp_runtimes(
    plugin_root: Path,
    descriptor: PluginDescriptor,
    data_path: Path,
) -> None:
    for config_relpath in _manifest_mcp_config_paths(descriptor):
        config_path = plugin_root / config_relpath
        if not config_path.exists():
            continue
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        loaded_dict = cast(dict[str, object], loaded)
        servers = loaded_dict.get("servers")
        if not isinstance(servers, dict):
            continue
        servers_dict = cast(dict[object, object], servers)
        changed = False
        for server_name, server_value in servers_dict.items():
            if not isinstance(server_value, dict):
                continue
            server_dict = cast(dict[str, object], server_value)
            _inject_plugin_env(server_dict, data_path, descriptor)
            if _prepare_single_mcp_server(
                plugin_root=plugin_root,
                server_name=str(server_name),
                server=server_dict,
            ):
                changed = True
                continue
            changed = True
        if changed:
            _ = config_path.write_text(
                json.dumps(loaded, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


def _manifest_mcp_config_paths(descriptor: PluginDescriptor) -> list[str]:
    return _manifest_path_list(descriptor, "mcp_servers")


def _manifest_path_list(
    descriptor: PluginDescriptor,
    key: str,
) -> list[str]:
    raw_paths = descriptor.raw_manifest.get("paths")
    if not isinstance(raw_paths, dict):
        return []
    raw_paths_dict = cast(dict[str, object], raw_paths)
    configs = raw_paths_dict.get(key)
    if isinstance(configs, str):
        stripped = configs.strip()
        return [stripped] if stripped else []
    if not isinstance(configs, list):
        return []
    result: list[str] = []
    for item in cast(list[object], configs):
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return result


def _prepare_single_mcp_server(
    *,
    plugin_root: Path,
    server_name: str,
    server: dict[str, object],
) -> bool:
    command = server.get("command")
    if not isinstance(command, list) or not command:
        return False
    command_items = [str(item) for item in cast(list[object], command)]
    if not _is_python_command(command_items[0]):
        return False
    runtime_root = _resolve_mcp_runtime_root(plugin_root, server, command_items)
    if runtime_root is None:
        return False
    requirements = runtime_root / "requirements.txt"
    if not requirements.exists():
        return False
    venv_python = _ensure_python_runtime(runtime_root, requirements, server_name)
    if command_items[0] == str(venv_python):
        return False
    command_items[0] = str(venv_python)
    server["command"] = command_items
    return True


def _inject_plugin_env(
    server: dict[str, object],
    data_path: Path,
    descriptor: PluginDescriptor,
) -> None:
    env = server.get("env")
    if not isinstance(env, dict):
        env = {}
        server["env"] = env
    env_dict = cast(dict[str, object], env)
    _ = env_dict.setdefault("XIAOMAN_PLUGIN_DATA_DIR", str(data_path))
    if descriptor.uses_legacy_data_env:
        # Old MCP plugins may read this exact variable.  It is only injected for
        # a legacy manifest and never documented as part of the Xiaoman API.
        _ = env_dict.setdefault("AKA_PLUGIN_DATA_DIR", str(data_path))


def _resolve_mcp_runtime_root(
    plugin_root: Path,
    server: dict[str, object],
    command_items: list[str],
) -> Path | None:
    candidates: list[Path] = []
    if len(command_items) >= 2:
        script_path = Path(command_items[1])
        if not script_path.is_absolute():
            candidates.append((plugin_root / script_path).resolve(strict=False).parent)
    cwd_raw = str(server.get("cwd") or "").strip()
    if cwd_raw:
        cwd_path = Path(cwd_raw)
        resolved_cwd = (
            cwd_path
            if cwd_path.is_absolute()
            else (plugin_root / cwd_path).resolve(strict=False)
        )
        candidates.append(resolved_cwd)
    candidates.append(plugin_root)
    for candidate in candidates:
        if (candidate / "requirements.txt").exists():
            return candidate
    return None


def _ensure_python_runtime(
    runtime_root: Path,
    requirements: Path,
    server_name: str,
) -> Path:
    venv_dir = runtime_root / ".venv"
    venv_python = _venv_python_path(venv_dir)
    if not venv_python.exists():
        _run_command(
            [sys.executable, "-m", "venv", str(venv_dir)],
            cwd=runtime_root,
            label=f"{server_name} venv",
        )
    _run_command(
        [str(venv_python), "-m", "pip", "install", "-r", str(requirements)],
        cwd=runtime_root,
        label=f"{server_name} pip install",
    )
    return venv_python


def _venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def _is_python_command(value: str) -> bool:
    name = Path(value).name.lower()
    return name in {"python", "python3", "python.exe"}


def _validate_install_id(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not _INSTALL_ID.fullmatch(normalized):
        raise ValueError(
            f"{label}只能包含字母、数字、点、下划线或连字符，长度不超过 64"
        )
    return normalized


def _validate_version(value: str) -> None:
    if not _INSTALL_VERSION.fullmatch(value):
        raise ValueError("插件 version 必须是安全的非空版本标识，长度不超过 64")


def _safe_relative_path(value: str, *, label: str) -> Path:
    normalized = value.strip().replace("\\", "/")
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label}必须是仓库内的相对路径")
    return path


def _registered_data_path(entry: dict[str, object], home: Path) -> Path | None:
    raw_path = str(entry.get("data_dir") or "").strip()
    if not raw_path:
        return None
    data_path = Path(raw_path).resolve(strict=False)
    _assert_within(data_path, (home / "data").resolve(strict=False), label="插件数据路径")
    return data_path


def _assert_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}不在 Xiaoman 插件目录内") from exc


def _remove_empty_cache_parents(plugin_root: Path, cache_root: Path) -> None:
    current = plugin_root.parent
    while current != cache_root:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _remove_tree(path: Path) -> None:
    """Remove a Git checkout even when Windows marks packed objects read-only."""

    def make_writable(function, value, _error) -> None:
        os.chmod(value, stat.S_IWRITE)
        function(value)

    shutil.rmtree(path, onexc=make_writable)


def _infer_legacy_plugin_descriptor(
    plugin_root: Path,
    *,
    name_hint: str,
    version_hint: str,
) -> PluginDescriptor:
    metadata = _read_legacy_plugin_metadata(plugin_root / "plugin.py")
    name = str(metadata.get("name") or name_hint or "legacy-plugin").strip()
    version = str(metadata.get("version") or version_hint or "legacy-0").strip()
    description = str(metadata.get("description") or "Legacy Xiaoman plugin").strip()
    _validate_install_id(name, label="插件名称")
    _validate_version(version)
    paths: dict[str, list[str]] = {}
    supports: list[str] = ["lifecycle"]
    if (plugin_root / "skills").is_dir():
        paths["skills"] = ["skills"]
        supports.append("skills")
    if (plugin_root / "drift" / "skills").is_dir():
        paths["drift_skills"] = ["drift/skills"]
        supports.append("drift")
    if (plugin_root / "mcp" / "servers.json").is_file():
        paths["mcp_servers"] = ["mcp/servers.json"]
        supports.append("mcp")
    raw_manifest: dict[str, object] = {
        "name": name,
        "version": version,
        "description": description,
        "paths": paths,
        "xiaoman": {
            "runtime": {"supports": supports},
            "lifecycle": {"entry": "plugin.py"},
            "compatibility": {"legacy_data_env": True},
        },
    }
    return PluginDescriptor(
        name=name,
        version=version,
        description=description,
        root=plugin_root,
        raw_manifest=raw_manifest,
        lifecycle_entry=plugin_root / "plugin.py",
        skill_roots=tuple(
            (plugin_root / path).resolve(strict=False)
            for path in paths.get("skills", [])
        ),
        drift_skill_roots=tuple(
            (plugin_root / path).resolve(strict=False)
            for path in paths.get("drift_skills", [])
        ),
        manifest_format="legacy",
        legacy_data_env=True,
    )


def _write_generated_manifest(
    plugin_root: Path,
    descriptor: PluginDescriptor,
) -> None:
    manifest_path = plugin_root / XIAOMAN_PLUGIN_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(descriptor.raw_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _materialize_skill_package(
    plugin_root: Path,
    descriptor: PluginDescriptor,
) -> None:
    source_skill = plugin_root / "SKILL.md"
    target_skill = plugin_root / "skills" / descriptor.name / "SKILL.md"
    target_skill.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_skill, target_skill)


def _read_legacy_plugin_metadata(plugin_path: Path) -> dict[str, str]:
    try:
        tree = ast.parse(plugin_path.read_text(encoding="utf-8"), filename=str(plugin_path))
    except (OSError, SyntaxError):
        return {}
    fields = {"name": "", "version": "", "description": ""}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                continue
            target = item.targets[0]
            if not isinstance(target, ast.Name) or target.id not in {"name", "version", "desc"}:
                continue
            if not isinstance(item.value, ast.Constant) or not isinstance(item.value.value, str):
                continue
            key = "description" if target.id == "desc" else target.id
            fields[key] = item.value.value.strip()
        if fields["name"]:
            return fields
    return fields


def _infer_skill_package_descriptor(
    plugin_root: Path,
    *,
    name_hint: str,
    version_hint: str,
) -> PluginDescriptor:
    metadata = _read_skill_metadata(plugin_root / "SKILL.md")
    name = str(metadata.get("name") or name_hint or "external-skill").strip()
    version = str(version_hint or "skill-0").strip()
    description = str(metadata.get("description") or "External Xiaoman skill").strip()
    _validate_install_id(name, label="技能名称")
    _validate_version(version)
    raw_manifest: dict[str, object] = {
        "name": name,
        "version": version,
        "description": description,
        "paths": {"skills": ["skills"]},
        "xiaoman": {"runtime": {"supports": ["skills"]}},
    }
    return PluginDescriptor(
        name=name,
        version=version,
        description=description,
        root=plugin_root,
        raw_manifest=raw_manifest,
        skill_roots=(plugin_root / "skills",),
    )


def _read_skill_metadata(skill_path: Path) -> dict[str, str]:
    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    header = re.match(r"^---\s*\r?\n(.*?)\r?\n---", content, flags=re.DOTALL)
    if header is None:
        return {}
    metadata: dict[str, str] = {}
    for field_name in ("name", "description"):
        match = re.search(
            rf"(?m)^{field_name}:\s*[\"']?([^\r\n\"']+)",
            header.group(1),
        )
        if match is not None:
            metadata[field_name] = match.group(1).strip()
    return metadata


def _source_name_hint(source: str) -> str:
    name = Path(source.rstrip("/").replace("\\", "/")).name
    if name.endswith(".git"):
        name = name[:-4]
    return name


def _git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        return ""
    revision = result.stdout.strip()
    return f"git-{revision}" if revision else ""


def _collect_skill_names(skill_roots: tuple[Path, ...]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for root in skill_roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not (child / "SKILL.md").exists():
                continue
            if child.name in seen:
                continue
            seen.add(child.name)
            names.append(child.name)
    return names


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode == 0:
        return
    raise RuntimeError(
        "git 命令失败: "
        + " ".join(args)
        + f"\nstdout:\n{result.stdout.strip()}\nstderr:\n{result.stderr.strip()}"
    )


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    label: str,
) -> None:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"{label} 失败: {' '.join(args)}"
        + f"\nstdout:\n{result.stdout.strip()}\nstderr:\n{result.stderr.strip()}"
    )
