from pathlib import Path

import pytest

from agent.mcp import catalog


class _Registry:
    def __init__(self) -> None:
        self.local: tuple[str, list[str]] | None = None
        self.remote: tuple[str, str, str, str] | None = None

    async def add(self, name: str, command: list[str]) -> str:
        self.local = (name, command)
        return "connected"

    async def add_remote(
        self,
        name: str,
        *,
        url: str,
        transport: str,
        auth_type: str,
    ) -> str:
        self.remote = (name, url, transport, auth_type)
        return "authorization required"


def test_catalog_exposes_standard_mcp_services() -> None:
    rows = catalog.list_catalog({"notion"})
    assert {row["id"] for row in rows} == {
        "markitdown",
        "notion",
        "gmail",
        "github",
        "obsidian",
    }
    notion = next(row for row in rows if row["id"] == "notion")
    assert notion["installed"] is True
    assert notion["requires_oauth"] is True
    gmail = next(row for row in rows if row["id"] == "gmail")
    assert gmail["configuration"]["url"] == "https://gmailmcp.googleapis.com/mcp/v1"
    github = next(row for row in rows if row["id"] == "github")
    assert github["configuration"] == {
        "name": "github",
        "transport": "streamable_http",
        "url": "https://api.githubcopilot.com/mcp/",
        "auth_type": "bearer",
        "docs_url": "https://github.com/github/github-mcp-server",
    }
    obsidian = next(row for row in rows if row["id"] == "obsidian")
    assert obsidian["configuration"]["command"][:3] == [
        "npx",
        "-y",
        "@bitbonsai/mcpvault@latest",
    ]


@pytest.mark.asyncio
async def test_markitdown_catalog_uses_official_isolated_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python = tmp_path / "python"
    monkeypatch.setattr(catalog, "_ensure_markitdown_runtime", lambda: python)
    registry = _Registry()

    await catalog.install_catalog_server("markitdown", registry)  # type: ignore[arg-type]

    assert registry.local == (
        "markitdown",
        [str(python), "-m", "markitdown_mcp"],
    )
