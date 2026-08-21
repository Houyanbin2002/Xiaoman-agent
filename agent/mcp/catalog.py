from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.mcp.registry import McpServerRegistry


@dataclass(frozen=True)
class McpCatalogEntry:
    id: str
    name: str
    description: str
    provider: str
    transport: str
    requires_oauth: bool = False
    configuration: dict[str, Any] | None = None

    def public(self, installed_names: set[str]) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "provider": self.provider,
            "transport": self.transport,
            "requires_oauth": self.requires_oauth,
            "installed": self.id in installed_names,
            "configuration": self.configuration,
        }


CATALOG = {
    "markitdown": McpCatalogEntry(
        id="markitdown",
        name="文档解析",
        description="使用微软官方 MarkItDown MCP 读取 PDF、Word、Excel、PPT 和常见文本文件。",
        provider="Microsoft",
        transport="stdio",
    ),
    "notion": McpCatalogEntry(
        id="notion",
        name="Notion",
        description="连接 Notion 官方远程 MCP，搜索、读取、创建和更新工作区内容。",
        provider="Notion",
        transport="streamable_http",
        requires_oauth=True,
    ),
    "gmail": McpCatalogEntry(
        id="gmail",
        name="Gmail",
        description=(
            "连接 Google 官方 Gmail MCP，搜索邮件、读取会话并创建草稿；"
            "目前属于 Google Workspace Developer Preview。"
        ),
        provider="Google",
        transport="streamable_http",
        requires_oauth=True,
        configuration={
            "name": "gmail",
            "transport": "streamable_http",
            "url": "https://gmailmcp.googleapis.com/mcp/v1",
            "auth_type": "oauth",
            "scopes": (
                "https://www.googleapis.com/auth/gmail.readonly "
                "https://www.googleapis.com/auth/gmail.compose"
            ),
            "requires_oauth_client": True,
            "docs_url": (
                "https://developers.google.com/workspace/gmail/api/guides/"
                "configure-mcp-server"
            ),
        },
    ),
    "github": McpCatalogEntry(
        id="github",
        name="GitHub",
        description=(
            "连接 GitHub 官方远程 MCP，读取代码仓库、Issue 和 Pull Request，"
            "也可在授权范围内创建或更新内容。"
        ),
        provider="GitHub",
        transport="streamable_http",
        configuration={
            "name": "github",
            "transport": "streamable_http",
            "url": "https://api.githubcopilot.com/mcp/",
            "auth_type": "bearer",
            "docs_url": "https://github.com/github/github-mcp-server",
        },
    ),
    "obsidian": McpCatalogEntry(
        id="obsidian",
        name="Obsidian Vault",
        description=(
            "使用开源 MCPVault 直接读写本地 Obsidian Vault，无需安装 Obsidian 插件。"
        ),
        provider="MCPVault",
        transport="stdio",
        configuration={
            "name": "obsidian",
            "transport": "stdio",
            "command": [
                "npx",
                "-y",
                "@bitbonsai/mcpvault@latest",
                "C:\\path\\to\\your\\vault",
            ],
            "requires_vault_path": True,
            "docs_url": "https://github.com/bitbonsai/mcpvault",
        },
    ),
}

_MARKITDOWN_PACKAGE = "markitdown-mcp==0.0.1a4"


def list_catalog(installed_names: set[str]) -> list[dict[str, object]]:
    return [entry.public(installed_names) for entry in CATALOG.values()]


async def install_catalog_server(
    entry_id: str,
    registry: "McpServerRegistry",
) -> str:
    if entry_id not in CATALOG:
        raise ValueError(f"MCP 目录中不存在: {entry_id}")
    if CATALOG[entry_id].configuration is not None:
        raise ValueError("该 MCP 需要先填写连接参数")
    if entry_id == "notion":
        return await registry.add_remote(
            "notion",
            url="https://mcp.notion.com/mcp",
            transport="streamable_http",
            auth_type="oauth",
        )
    runtime_python = await asyncio.to_thread(_ensure_markitdown_runtime)
    return await registry.add(
        "markitdown",
        [str(runtime_python), "-m", "markitdown_mcp"],
    )


def uninstall_catalog_runtime(entry_id: str) -> bool:
    if entry_id != "markitdown":
        return False
    root = _mcp_runtime_root(entry_id)
    marker = root / "runtime.json"
    if not marker.is_file():
        return False
    shutil.rmtree(root)
    return True


def _ensure_markitdown_runtime() -> Path:
    root = _mcp_runtime_root("markitdown")
    venv = root / ".venv"
    python = _venv_python(venv)
    marker = root / "runtime.json"
    expected = {"package": _MARKITDOWN_PACKAGE}
    if python.is_file() and _read_json(marker) == expected:
        return python
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "venv", str(venv)], cwd=root)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            _MARKITDOWN_PACKAGE,
        ],
        cwd=root,
    )
    marker.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return python


def _mcp_runtime_root(name: str) -> Path:
    return Path.home() / ".xiaoman" / "mcp" / name


def _venv_python(venv: Path) -> Path:
    return (
        venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"
    )


def _read_json(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _run(command: list[str], *, cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"安装 MCP 运行环境失败: {detail or result.returncode}")
