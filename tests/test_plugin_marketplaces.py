from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from agent.plugins.marketplaces import (
    add_marketplace,
    install_marketplace_plugin,
    list_marketplace_plugins,
    list_marketplaces,
    refresh_marketplace,
)


def test_marketplace_installs_a_relative_plugin_and_refreshes_catalog(
    tmp_path: Path,
) -> None:
    market = tmp_path / "market"
    _write_plugin(market / "plugins" / "feed", name="feed", version="1.0.0")
    _write_marketplace(
        market,
        name="team-tools",
        plugins=[
            {
                "name": "feed",
                "source": "./plugins/feed",
                "description": "订阅源管理",
                "version": "1.0.0",
                "tags": ["mcp"],
            }
        ],
    )
    _commit_repository(market)

    home = tmp_path / "plugins-home"
    descriptor = add_marketplace(source=str(market), plugins_home=home)

    assert descriptor.name == "team-tools"
    assert not (home / "marketplaces" / "team-tools" / ".git").exists()
    assert list_marketplaces(plugins_home=home) == [
        {
            "name": "team-tools",
            "source": str(market),
            "ref": "",
            "available": True,
            "plugin_count": 1,
            "owner": "Test Team",
        }
    ]
    assert list_marketplace_plugins("team-tools", plugins_home=home) == [
        {
            "name": "feed",
            "description": "订阅源管理",
            "version": "1.0.0",
            "tags": ["mcp"],
        }
    ]

    installed = install_marketplace_plugin(
        marketplace="team-tools",
        plugin_name="feed",
        plugins_home=home,
    )

    assert installed.installed_path == home / "cache" / "team-tools" / "feed" / "1.0.0"
    assert (installed.installed_path / ".xiaoman-plugin" / "plugin.json").exists()
    registry = json.loads((home / "registry.json").read_text(encoding="utf-8"))
    assert (
        registry["plugins"]["feed@team-tools"]["install_source"]
        == "team-tools:./plugins/feed"
    )

    _write_plugin(market / "plugins" / "calendar", name="calendar", version="1.0.0")
    _write_marketplace(
        market,
        name="team-tools",
        plugins=[
            {"name": "feed", "source": "./plugins/feed"},
            {"name": "calendar", "source": "./plugins/calendar"},
        ],
    )
    _commit_repository(market, message="add calendar")

    refreshed = refresh_marketplace("team-tools", plugins_home=home)

    assert [plugin.name for plugin in refreshed.plugins] == ["feed", "calendar"]


def test_marketplace_can_install_a_git_subdirectory_plugin(tmp_path: Path) -> None:
    plugin_repository = tmp_path / "external-plugin"
    _write_plugin(
        plugin_repository / "extensions" / "weather",
        name="weather",
        version="2.0.0",
    )
    _commit_repository(plugin_repository)

    market = tmp_path / "market"
    _write_marketplace(
        market,
        name="external-tools",
        plugins=[
            {
                "name": "weather",
                "source": {
                    "type": "git-subdir",
                    "url": str(plugin_repository),
                    "path": "extensions/weather",
                },
            }
        ],
    )
    _commit_repository(market)

    home = tmp_path / "plugins-home"
    add_marketplace(source=str(market), plugins_home=home)
    installed = install_marketplace_plugin(
        marketplace="external-tools",
        plugin_name="weather",
        plugins_home=home,
    )

    assert (
        installed.installed_path
        == home / "cache" / "external-tools" / "weather" / "2.0.0"
    )


def _write_marketplace(
    root: Path,
    *,
    name: str,
    plugins: list[dict[str, object]],
) -> None:
    manifest = root / ".xiaoman-plugin" / "marketplace.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "name": name,
                "owner": {"name": "Test Team"},
                "plugins": plugins,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_plugin(root: Path, *, name: str, version: str) -> None:
    manifest = root / ".xiaoman-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "name": name,
                "version": version,
                "description": f"{name} plugin",
                "xiaoman": {"runtime": {"supports": []}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _commit_repository(root: Path, *, message: str = "initial") -> None:
    if not (root / ".git").exists():
        _run_git(["init"], cwd=root)
        _run_git(["config", "user.name", "test"], cwd=root)
        _run_git(["config", "user.email", "test@example.com"], cwd=root)
    _run_git(["add", "."], cwd=root)
    _run_git(["commit", "-m", message], cwd=root)


def _run_git(args: list[str], *, cwd: Path) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
