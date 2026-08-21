from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from agent.skill_packages import install_skill_directory

from .models import MarketplaceKind
from .service import MarketplaceService

InstallStatus = Literal[
    "installed", "already_installed", "authorization_required", "unsupported"
]


class McpRegistry(Protocol):
    async def add_remote(self, name: str, **kwargs: Any) -> str: ...

    async def add(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class MarketplaceInstallResult:
    status: InstallStatus
    item_id: str
    kind: MarketplaceKind
    resource_name: str = ""
    message: str = ""

    def public(self) -> dict[str, str]:
        return {
            "status": self.status,
            "item_id": self.item_id,
            "kind": self.kind,
            "resource_name": self.resource_name,
            "message": self.message,
        }


class MarketplaceInstaller:
    def __init__(
        self,
        service: MarketplaceService,
        mcp_registry: McpRegistry,
        *,
        skills_root: Path | None = None,
        pypi_runtime_factory: Callable[[dict[str, Any], str], list[str]] | None = None,
    ) -> None:
        self._service = service
        self._mcp_registry = mcp_registry
        self._skills_root = skills_root
        self._pypi_runtime_factory = pypi_runtime_factory or _ensure_pypi_runtime

    async def install(
        self,
        kind: MarketplaceKind,
        item_id: str,
        configuration: dict[str, Any] | None = None,
    ) -> MarketplaceInstallResult:
        item = self._service.get(kind, item_id)
        if item is None:
            raise ValueError("市场中不存在该扩展能力")
        if item.installed:
            return MarketplaceInstallResult(
                status="already_installed",
                item_id=item.id,
                kind=kind,
                resource_name=item.name,
                message="该扩展能力已经安装",
            )
        if item.install_mode == "unsupported":
            return MarketplaceInstallResult(
                status="unsupported",
                item_id=item.id,
                kind=kind,
                resource_name=item.name,
                message=item.unsupported_reason,
            )
        if kind == "skill":
            return await self._install_skill(item.id)
        return await self._install_mcp(item.id, configuration or {})

    async def _install_skill(self, item_id: str) -> MarketplaceInstallResult:
        provider = self._service.skill_provider
        download = getattr(provider, "download", None)
        if not callable(download):
            raise RuntimeError("当前 Skill 市场来源不支持下载安装")
        item = self._service.get("skill", item_id)
        assert item is not None
        with tempfile.TemporaryDirectory(prefix="xiaoman-market-skill-") as temporary:
            skill_root = await asyncio.to_thread(download, item_id, Path(temporary))
            result = await asyncio.to_thread(
                install_skill_directory,
                skill_root=skill_root,
                source=item.source_url,
                revision=item.version,
                skills_root=self._skills_root,
            )
        return MarketplaceInstallResult(
            status="installed",
            item_id=item.id,
            kind="skill",
            resource_name=result.name,
            message=f"已安装 Skill {result.name}",
        )

    async def _install_mcp(
        self, item_id: str, configuration: dict[str, Any]
    ) -> MarketplaceInstallResult:
        item = self._service.get("mcp", item_id)
        assert item is not None
        missing = [
            field.label
            for field in item.configuration_fields
            if field.required and not str(configuration.get(field.name, "")).strip()
        ]
        if missing:
            raise ValueError(f"请先填写：{'、'.join(missing)}")
        spec = item.install_spec
        name = _server_name(str(configuration.get("name") or item.id))
        catalog_id = str(spec.get("catalog_id", ""))
        if catalog_id:
            from agent.mcp.catalog import install_catalog_server

            message = await install_catalog_server(catalog_id, self._mcp_registry)
            return MarketplaceInstallResult(
                "installed", item.id, "mcp", name, message
            )
        transport = str(spec.get("transport", ""))
        if transport == "streamable_http":
            auth_type = str(spec.get("auth_type", "none"))
            headers = _string_mapping(configuration.get("headers"))
            headers.update(
                {
                    field_name: str(configuration.get(field_name, ""))
                    for field_name in spec.get("header_fields", [])
                    if str(configuration.get(field_name, ""))
                }
            )
            message = await self._mcp_registry.add_remote(
                name,
                url=str(spec["url"]),
                transport="streamable_http",
                auth_type=auth_type,
                scopes=str(configuration.get("scopes", "")),
                bearer_token=str(configuration.get("bearer_token", "")),
                headers=headers,
                oauth_client_id=str(configuration.get("oauth_client_id", "")),
                oauth_client_secret=str(configuration.get("oauth_client_secret", "")),
            )
            status: InstallStatus = (
                "authorization_required" if auth_type == "oauth" else "installed"
            )
            return MarketplaceInstallResult(status, item.id, "mcp", name, message)
        registry = str(spec.get("registry", ""))
        if transport == "stdio" and registry == "npm":
            package = str(spec["package"])
            version = str(spec.get("version", ""))
            pinned = f"{package}@{version}" if version else package
            arguments = [
                str(configuration.get(field_name, ""))
                for field_name in spec.get("argument_fields", [])
                if str(configuration.get(field_name, ""))
            ]
            environment = {
                field_name: str(configuration.get(field_name, ""))
                for field_name in spec.get("environment_fields", [])
                if str(configuration.get(field_name, ""))
            }
            message = await self._mcp_registry.add(
                name,
                ["npx", "-y", pinned, *arguments],
                environment,
            )
            return MarketplaceInstallResult(
                "installed", item.id, "mcp", name, message
            )
        if transport == "stdio" and registry == "pypi":
            command = await asyncio.to_thread(self._pypi_runtime_factory, spec, name)
            message = await self._mcp_registry.add(name, command)
            return MarketplaceInstallResult(
                "installed", item.id, "mcp", name, message
            )
        return MarketplaceInstallResult(
            "unsupported", item.id, "mcp", name, item.unsupported_reason
        )


def _server_name(item_id: str) -> str:
    value = item_id.rsplit("/", 1)[-1].strip().casefold()
    value = re.sub(r"[^a-z0-9._-]+", "-", value).strip("-.")[:64]
    if not value:
        raise ValueError("无法从市场条目生成有效的 MCP 名称")
    return value


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _ensure_pypi_runtime(spec: dict[str, Any], name: str) -> list[str]:
    package = str(spec.get("package", "")).strip()
    version = str(spec.get("version", "")).strip()
    executable = str(spec.get("executable", "")).strip()
    if not package or not executable:
        raise ValueError("PyPI MCP 缺少包名或执行入口")
    root = Path.home() / ".xiaoman" / "mcp" / name
    venv = root / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = root / "marketplace-runtime.json"
    expected = {"package": package, "version": version, "executable": executable}
    if not python.is_file() or _read_json(marker) != expected:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        _run([sys.executable, "-m", "venv", str(venv)], root)
        requirement = f"{package}=={version}" if version else package
        _run([str(python), "-m", "pip", "install", requirement], root)
        marker.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    executable_path = venv / ("Scripts" if os.name == "nt" else "bin") / executable
    if os.name == "nt" and executable_path.suffix.lower() != ".exe":
        executable_path = executable_path.with_suffix(".exe")
    if executable_path.is_file():
        return [str(executable_path)]
    return [str(python), "-m", executable]


def _read_json(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or "安装 PyPI MCP 运行环境失败")
