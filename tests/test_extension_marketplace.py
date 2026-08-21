from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import agent.marketplace.mcp_registry_provider as mcp_registry_provider
from agent.marketplace.curated_mcp_provider import CuratedMcpProvider
from agent.marketplace.installer import MarketplaceInstaller
from agent.marketplace.mcp_registry_provider import McpRegistryProvider
from agent.marketplace.models import MarketplaceItem
from agent.marketplace.service import CombinedMarketplaceProvider, MarketplaceService
from agent.marketplace.skills_provider import SkillsCliProvider


def _registry_payload(*servers: dict[str, object]) -> dict[str, object]:
    return {"servers": [{"server": server, "_meta": {}} for server in servers]}


def _command_result(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class _StaticProvider:
    def __init__(self, rows: list[MarketplaceItem]) -> None:
        self.rows = rows

    def search(self, query: str, limit: int) -> list[MarketplaceItem]:
        needle = query.casefold()
        return [
            row
            for row in self.rows
            if not needle or needle in f"{row.id} {row.name}".casefold()
        ][:limit]

    def get(self, item_id: str) -> MarketplaceItem | None:
        return next((row for row in self.rows if row.id == item_id), None)

    def refresh(self) -> list[MarketplaceItem]:
        return self.rows


class _OfflineProvider(_StaticProvider):
    def search(self, query: str, limit: int) -> list[MarketplaceItem]:
        raise OSError("offline")


class _UnexpectedProvider(_StaticProvider):
    def search(self, query: str, limit: int) -> list[MarketplaceItem]:
        raise AssertionError("exact curated match should not query remote registry")


class _FakeMcpRegistry:
    def __init__(self) -> None:
        self.remotes: list[tuple[str, dict[str, Any]]] = []
        self.local: list[tuple[str, list[str]]] = []
        self.local_env: list[dict[str, str]] = []

    async def add_remote(self, name: str, **kwargs: Any) -> str:
        self.remotes.append((name, kwargs))
        return "added remote"

    async def add(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        self.local.append((name, command))
        self.local_env.append(env or {})
        return "added local"


def test_mcp_registry_provider_maps_remote_item(tmp_path: Path) -> None:
    payload = _registry_payload(
        {
            "name": "io.example/notion",
            "title": "Notion",
            "description": "Workspace",
            "version": "1.0.0",
            "remotes": [
                {
                    "type": "streamable-http",
                    "url": "https://example.com/mcp",
                }
            ],
        }
    )
    provider = McpRegistryProvider(
        cache_path=tmp_path / "mcp.json",
        fetch_json=lambda _url: payload,
    )

    rows = provider.search("notion", limit=20)

    assert len(rows) == 1
    assert rows[0].kind == "mcp"
    assert rows[0].id == "io.example/notion"
    assert rows[0].install_mode == "direct"
    assert rows[0].install_spec == {
        "transport": "streamable_http",
        "url": "https://example.com/mcp",
        "auth_type": "none",
    }


def test_mcp_registry_fetch_allows_slow_registry_responses(monkeypatch) -> None:
    captured: dict[str, float] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"servers": []}'

    def fake_urlopen(_request, *, timeout: float):
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(mcp_registry_provider, "urlopen", fake_urlopen)

    assert mcp_registry_provider._fetch_json("https://registry.example") == {
        "servers": []
    }
    assert captured["timeout"] >= 20


def test_mcp_registry_refresh_keeps_successful_pages_on_later_timeout(
    tmp_path: Path,
) -> None:
    calls = 0

    def fetch(_url: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "servers": [
                    {
                        "server": {
                            "name": "io.example/first-page",
                            "description": "Cached before the timeout",
                            "version": "1.0.0",
                        }
                    }
                ],
                "metadata": {"nextCursor": "page-2"},
            }
        raise TimeoutError("registry page timed out")

    provider = McpRegistryProvider(
        cache_path=tmp_path / "mcp.json",
        fetch_json=fetch,
    )

    rows = provider.refresh()

    assert [row.id for row in rows] == ["io.example/first-page"]


def test_mcp_registry_refresh_caps_catalog_warmup_pages(tmp_path: Path) -> None:
    calls = 0

    def fetch(_url: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "servers": [
                {
                    "server": {
                        "name": f"io.example/server-{calls}",
                        "description": "Catalog entry",
                        "version": "1.0.0",
                    }
                }
            ],
            "metadata": {"nextCursor": f"page-{calls + 1}"},
        }

    provider = McpRegistryProvider(
        cache_path=tmp_path / "mcp.json",
        fetch_json=fetch,
    )

    rows = provider.refresh()

    assert calls == 3
    assert len(rows) == 3


def test_mcp_registry_provider_maps_npm_stdio_item(tmp_path: Path) -> None:
    payload = _registry_payload(
        {
            "name": "io.example/docs",
            "title": "Docs",
            "description": "Document tools",
            "version": "1.0.0",
            "packages": [
                {
                    "registryType": "npm",
                    "identifier": "@example/docs",
                    "version": "1.0.0",
                    "transport": {"type": "stdio"},
                }
            ],
        }
    )
    provider = McpRegistryProvider(
        cache_path=tmp_path / "mcp.json",
        fetch_json=lambda _url: payload,
    )

    row = provider.search("docs", limit=20)[0]

    assert row.install_mode == "direct"
    assert row.install_spec == {
        "transport": "stdio",
        "registry": "npm",
        "package": "@example/docs",
        "version": "1.0.0",
    }


def test_mcp_registry_provider_requires_declared_package_environment(
    tmp_path: Path,
) -> None:
    payload = _registry_payload(
        {
            "name": "io.example/search",
            "title": "Search",
            "description": "Search tools",
            "version": "1.0.0",
            "packages": [
                {
                    "registryType": "npm",
                    "identifier": "@example/search",
                    "version": "1.0.0",
                    "transport": {"type": "stdio"},
                    "environmentVariables": [
                        {
                            "name": "SEARCH_API_KEY",
                            "description": "Search API Key",
                            "isRequired": True,
                            "isSecret": True,
                        }
                    ],
                }
            ],
        }
    )
    provider = McpRegistryProvider(
        cache_path=tmp_path / "mcp.json",
        fetch_json=lambda _url: payload,
    )

    row = provider.search("search", 20)[0]

    assert row.install_mode == "configure"
    assert row.configuration_fields[0].name == "SEARCH_API_KEY"
    assert row.configuration_fields[0].secret is True
    assert row.install_spec["environment_fields"] == ["SEARCH_API_KEY"]


def test_mcp_registry_provider_uses_fresh_cache_without_network(
    tmp_path: Path,
) -> None:
    payload = _registry_payload(
        {
            "name": "io.example/docs",
            "title": "Docs",
            "description": "Docs",
            "version": "1.0.0",
            "remotes": [
                {"type": "streamable-http", "url": "https://example.com/mcp"}
            ],
        }
    )
    path = tmp_path / "mcp.json"
    assert McpRegistryProvider(
        cache_path=path,
        fetch_json=lambda _url: payload,
    ).search("docs", 20)

    def fail_fetch(_url: str) -> dict[str, object]:
        raise AssertionError("network should not run")

    assert McpRegistryProvider(
        cache_path=path,
        fetch_json=fail_fetch,
    ).search("docs", 20)


def test_mcp_registry_provider_uses_stale_cache_when_refresh_fails(
    tmp_path: Path,
) -> None:
    payload = _registry_payload(
        {
            "name": "io.example/docs",
            "title": "Docs",
            "description": "Docs",
            "version": "1.0.0",
            "remotes": [
                {"type": "streamable-http", "url": "https://example.com/mcp"}
            ],
        }
    )
    path = tmp_path / "mcp.json"
    first = McpRegistryProvider(
        cache_path=path,
        fetch_json=lambda _url: payload,
        cache_ttl_seconds=0,
    )
    assert first.search("docs", 20)

    def offline(_url: str) -> dict[str, object]:
        raise OSError("offline")

    second = McpRegistryProvider(
        cache_path=path,
        fetch_json=offline,
        cache_ttl_seconds=0,
    )
    assert second.search("docs", 20)[0].id == "io.example/docs"


def test_mcp_registry_provider_does_not_block_empty_catalog_without_cache(
    tmp_path: Path,
) -> None:
    def fail_fetch(_url: str) -> dict[str, object]:
        raise AssertionError("empty catalog should use local cache only")

    provider = McpRegistryProvider(
        cache_path=tmp_path / "missing.json",
        fetch_json=fail_fetch,
    )

    assert provider.search("", 20) == []


def test_skills_cli_provider_parses_results_without_cli_help_example(
    tmp_path: Path,
) -> None:
    output = (
        "Install with npx skills add <owner/repo@skill>\n"
        "vercel-labs/skills@react-best-practices 562K installs\n"
        "https://skills.sh/vercel-labs/skills/react-best-practices\n"
        "google-labs-code/stitch-skills@react:components 50K installs\n"
        "https://skills.sh/google-labs-code/stitch-skills/react:components\n"
    )
    provider = SkillsCliProvider(
        run_command=lambda *args, **kwargs: _command_result(output),
        cache_path=tmp_path / "skills.json",
    )

    rows = provider.search("react", 20)

    assert [row.id for row in rows] == [
        "vercel-labs/skills/react-best-practices",
        "google-labs-code/stitch-skills/react:components",
    ]
    assert rows[0].kind == "skill"
    assert rows[0].install_mode == "direct"


def test_skills_cli_provider_reuses_persistent_query_cache(tmp_path: Path) -> None:
    path = tmp_path / "skills.json"
    first = SkillsCliProvider(
        run_command=lambda *args, **kwargs: _command_result(
            "https://skills.sh/vercel-labs/skills/react-best-practices\n"
        ),
        cache_path=path,
    )
    assert first.search("react", 20)

    def fail_command(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("fresh query cache should avoid CLI")

    second = SkillsCliProvider(run_command=fail_command, cache_path=path)
    assert second.search("react", 20)[0].id == "vercel-labs/skills/react-best-practices"


def test_skills_cli_provider_downloads_selected_skill(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_download(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert env["CI"] == "1"
        assert env["NO_COLOR"] == "1"
        assert timeout == 600
        assert capture_output is True
        assert text is True
        assert check is False
        target = Path(cwd) / ".agents" / "skills" / "demo"
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text(
            "---\nname: demo\ndescription: Demo\n---\n", encoding="utf-8"
        )
        return _command_result()

    provider = SkillsCliProvider(
        run_command=fake_download,
        npx_command="C:/node/npx.cmd",
    )

    path = provider.download("owner/repo/demo", tmp_path)

    assert path.name == "demo"
    assert (path / "SKILL.md").is_file()
    assert commands[0][0] == "C:/node/npx.cmd"
    assert commands[0][-7:] == [
        "owner/repo",
        "--skill",
        "demo",
        "--agent",
        "universal",
        "--copy",
        "-y",
    ]


def test_service_merges_installed_state() -> None:
    row = MarketplaceItem(
        id="owner/repo/demo",
        kind="skill",
        name="demo",
        description="Demo",
        provider="owner",
        source_url="https://skills.sh/owner/repo/demo",
        install_mode="direct",
    )
    service = MarketplaceService(
        skill_provider=_StaticProvider([row]),
        mcp_provider=_StaticProvider([]),
        installed_skills=lambda: {"demo"},
        installed_mcp=lambda: set(),
    )

    result = service.search("skill", "demo", 20)[0]

    assert result.installed is True
    assert result.public()["installed"] is True


@pytest.mark.asyncio
async def test_installer_adds_remote_mcp_without_signal_source() -> None:
    registry = _FakeMcpRegistry()
    item = MarketplaceItem(
        id="io.example/remote",
        kind="mcp",
        name="Remote",
        description="Remote",
        provider="example",
        source_url="https://example.com/mcp",
        install_mode="direct",
        unsupported_reason="",
        install_spec={
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "auth_type": "none",
        },
    )
    service = MarketplaceService(
        _StaticProvider([]),
        _StaticProvider([item]),
    )
    installer = MarketplaceInstaller(service=service, mcp_registry=registry)

    result = await installer.install("mcp", item.id, {})

    assert result.status == "installed"
    assert registry.remotes == [
        (
            "remote",
            {
                "url": "https://example.com/mcp",
                "transport": "streamable_http",
                "auth_type": "none",
                "scopes": "",
                "bearer_token": "",
                "headers": {},
                "oauth_client_id": "",
                "oauth_client_secret": "",
            },
        )
    ]
    assert not hasattr(registry, "sources")


@pytest.mark.asyncio
async def test_installer_uses_pinned_npm_package() -> None:
    registry = _FakeMcpRegistry()
    item = MarketplaceItem(
        id="io.example/docs",
        kind="mcp",
        name="Docs",
        description="Docs",
        provider="example",
        install_mode="direct",
        unsupported_reason="",
        install_spec={
            "transport": "stdio",
            "registry": "npm",
            "package": "@example/docs",
            "version": "1.2.3",
        },
    )
    installer = MarketplaceInstaller(
        MarketplaceService(_StaticProvider([]), _StaticProvider([item])),
        registry,
    )

    result = await installer.install("mcp", item.id, {})

    assert result.status == "installed"
    assert registry.local == [("docs", ["npx", "-y", "@example/docs@1.2.3"])]


def test_curated_mcp_provider_exposes_real_oauth_and_configuration() -> None:
    provider = CuratedMcpProvider()

    notion = provider.get("notion")
    gmail = provider.get("gmail")
    github = provider.get("github")
    obsidian = provider.get("obsidian")

    assert notion is not None and notion.install_mode == "oauth"
    assert notion.install_spec["url"] == "https://mcp.notion.com/mcp"
    assert gmail is not None and gmail.install_mode == "configure"
    assert {field.name for field in gmail.configuration_fields} == {
        "oauth_client_id",
        "oauth_client_secret",
    }
    assert github is not None and github.install_mode == "configure"
    assert github.configuration_fields[0].name == "bearer_token"
    assert github.configuration_fields[0].secret is True
    assert github.install_spec["url"] == "https://api.githubcopilot.com/mcp/"
    assert obsidian is not None and obsidian.install_mode == "configure"
    assert obsidian.configuration_fields[0].name == "vault_path"


@pytest.mark.asyncio
async def test_installer_passes_required_configuration_to_npm_mcp() -> None:
    registry = _FakeMcpRegistry()
    item = CuratedMcpProvider().get("obsidian")
    assert item is not None
    installer = MarketplaceInstaller(
        MarketplaceService(_StaticProvider([]), _StaticProvider([item])),
        registry,
    )

    with pytest.raises(ValueError, match="Vault 路径"):
        await installer.install("mcp", item.id, {})

    result = await installer.install(
        "mcp", item.id, {"vault_path": "D:/notes"}
    )

    assert result.status == "installed"
    assert registry.local == [
        ("obsidian", ["npx", "-y", "@bitbonsai/mcpvault@latest", "D:/notes"])
    ]


def test_combined_provider_keeps_curated_results_when_registry_is_offline() -> None:
    curated = CuratedMcpProvider()
    combined = CombinedMarketplaceProvider(curated, _OfflineProvider([]))

    rows = combined.search("", 20)

    assert {row.id for row in rows} >= {"markitdown", "notion", "gmail", "github", "obsidian"}


def test_combined_provider_returns_exact_curated_match_without_remote_wait() -> None:
    combined = CombinedMarketplaceProvider(
        CuratedMcpProvider(), _UnexpectedProvider([])
    )

    rows = combined.search("notion", 20)

    assert [row.id for row in rows] == ["notion"]
