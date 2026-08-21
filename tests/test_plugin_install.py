from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import agent.plugins.install as install_module
from agent.plugins.install import install_git_plugin, uninstall_plugin
from agent.plugins.xiaoman_descriptor import load_plugin_descriptor


def test_install_git_plugin_installs_into_cache_and_preserves_data(tmp_path: Path) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / ".xiaoman-plugin").mkdir(parents=True)
    (repo / "skills" / "feed-manage").mkdir(parents=True)
    (repo / "skills" / "feed-manage" / "SKILL.md").write_text(
        "---\nname: feed-manage\ndescription: feed\n---\nbody\n",
        encoding="utf-8",
    )
    (repo / ".xiaoman-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "feed",
                "version": "0.1.0",
                "description": "feed plugin",
                "paths": {"skills": ["skills"]},
                "xiaoman": {"runtime": {"supports": ["skills"]}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    home = tmp_path / "plugins-home"
    data_dir = home / "data" / "feed-lab"
    data_dir.mkdir(parents=True)
    (data_dir / "state.json").write_text('{"keep":true}\n', encoding="utf-8")

    result = install_git_plugin(
        source=str(repo),
        marketplace="lab",
        plugins_home=home,
    )

    assert result.plugin_name == "feed"
    assert result.installed_path == home / "cache" / "lab" / "feed" / "0.1.0"
    assert (result.installed_path / ".xiaoman-plugin" / "plugin.json").exists()
    assert (result.installed_path / "skills" / "feed-manage" / "SKILL.md").exists()
    assert (result.data_path / "state.json").read_text(encoding="utf-8").strip() == '{"keep":true}'
    registry = json.loads((home / "registry.json").read_text(encoding="utf-8"))
    entry = registry["plugins"]["feed@lab"]
    assert entry["plugin_id"] == "feed@lab"
    assert entry["install_source"] == str(repo)
    assert entry["skills"] == ["feed-manage"]
    assert entry["active"] is False


def test_install_git_plugin_prepares_mcp_venv_and_rewrites_python_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "feed-mcp"
    (repo / ".xiaoman-plugin").mkdir(parents=True)
    (repo / "mcp").mkdir(parents=True)
    (repo / "mcp" / "run_mcp.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "mcp" / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (repo / "mcp" / "servers.json").write_text(
        json.dumps(
            {
                "servers": {
                    "feed": {
                        "command": ["python", "mcp/run_mcp.py"],
                        "cwd": ".",
                        "env": {},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (repo / ".xiaoman-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "feed",
                "version": "0.1.0",
                "description": "feed plugin",
                "paths": {"mcp_servers": ["mcp/servers.json"]},
                "xiaoman": {"runtime": {"supports": ["mcp"]}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    calls: list[tuple[str, Path]] = []

    def _fake_run(args: list[str], *, cwd: Path, label: str) -> None:
        calls.append((label, cwd))
        if label.endswith("venv"):
            python_path = install_module._venv_python_path(cwd / ".venv")
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(install_module, "_run_command", _fake_run)

    result = install_git_plugin(
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    servers = json.loads(
        (result.installed_path / "mcp" / "servers.json").read_text(encoding="utf-8")
    )["servers"]
    expected_python = install_module._venv_python_path(
        result.installed_path / "mcp" / ".venv"
    )

    assert servers["feed"]["command"][0] == str(expected_python)
    assert servers["feed"]["env"]["XIAOMAN_PLUGIN_DATA_DIR"] == str(result.data_path)
    assert calls == [
        ("feed venv", result.installed_path / "mcp"),
        ("feed pip install", result.installed_path / "mcp"),
    ]
    registry = json.loads(
        ((tmp_path / "plugins-home") / "registry.json").read_text(encoding="utf-8")
    )
    assert registry["plugins"]["feed@lab"]["mcp_servers"] == ["feed"]


def test_install_git_plugin_prepares_declared_native_python_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "document-plugin"
    (repo / ".xiaoman-plugin").mkdir(parents=True)
    (repo / "runtime").mkdir()
    (repo / "runtime" / "requirements.txt").write_text(
        "markitdown==0.1.6\n",
        encoding="utf-8",
    )
    (repo / "plugin.py").write_text(
        "from agent.plugins import Plugin\nclass DocumentPlugin(Plugin):\n    pass\n",
        encoding="utf-8",
    )
    (repo / ".xiaoman-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "documents",
                "version": "1.0.0",
                "paths": {"python_runtimes": ["runtime"]},
                "xiaoman": {
                    "runtime": {"supports": ["tools"]},
                    "lifecycle": {
                        "entry": "plugin.py",
                        "class": "DocumentPlugin",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    calls: list[tuple[Path, Path, str]] = []

    def _fake_runtime(runtime_root: Path, requirements: Path, label: str) -> Path:
        calls.append((runtime_root, requirements, label))
        return install_module._venv_python_path(runtime_root / ".venv")

    monkeypatch.setattr(install_module, "_ensure_python_runtime", _fake_runtime)

    result = install_git_plugin(
        source=str(repo),
        marketplace="lab",
        plugins_home=tmp_path / "plugins-home",
    )

    runtime_root = result.installed_path / "runtime"
    assert calls == [
        (
            runtime_root,
            runtime_root / "requirements.txt",
            "documents:runtime",
        )
    ]


def test_install_git_plugin_accepts_a_legacy_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "legacy-plugin"
    (repo / ".aka-plugin").mkdir(parents=True)
    (repo / ".aka-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "legacy-feed",
                "version": "0.1.0",
                "description": "legacy plugin",
                "akashic": {"runtime": {"supports": []}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    result = install_git_plugin(
        source=str(repo),
        marketplace="compat",
        plugins_home=tmp_path / "plugins-home",
    )

    descriptor = load_plugin_descriptor(result.installed_path)
    assert descriptor is not None
    assert descriptor.is_legacy is True
    registry = json.loads(
        ((tmp_path / "plugins-home") / "registry.json").read_text(encoding="utf-8")
    )
    assert registry["plugins"]["legacy-feed@compat"]["manifest_format"] == "legacy"


def test_install_git_plugin_generates_a_manifest_for_legacy_plugin_py(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "legacy-skills"
    (repo / "skills" / "legacy-help").mkdir(parents=True)
    (repo / "skills" / "legacy-help" / "SKILL.md").write_text(
        "---\nname: legacy-help\ndescription: legacy help\n---\nbody\n",
        encoding="utf-8",
    )
    (repo / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class LegacySkillsPlugin(Plugin):\n"
        "    name = 'legacy-skills'\n"
        "    version = '1.2.3'\n"
        "    desc = 'legacy skills bundle'\n",
        encoding="utf-8",
    )
    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    result = install_git_plugin(
        source=str(repo),
        marketplace="compat",
        plugins_home=tmp_path / "plugins-home",
    )

    descriptor = load_plugin_descriptor(result.installed_path)
    assert descriptor is not None
    assert descriptor.name == "legacy-skills"
    assert descriptor.version == "1.2.3"
    assert descriptor.skill_roots == (result.installed_path / "skills",)
    assert descriptor.uses_legacy_data_env is True
    registry = json.loads(
        ((tmp_path / "plugins-home") / "registry.json").read_text(encoding="utf-8")
    )
    entry = registry["plugins"]["legacy-skills@compat"]
    assert entry["manifest_format"] == "generated_legacy"
    assert entry["skills"] == ["legacy-help"]


def test_install_git_skill_package_and_uninstall_preserves_data(tmp_path: Path) -> None:
    repo = tmp_path / "skill-source"
    skill_root = repo / "skills" / "find-skills"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: find-skills\ndescription: discover skills\n---\nbody\n",
        encoding="utf-8",
    )
    _run_git(["init"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    home = tmp_path / "plugins-home"
    installed = install_git_plugin(
        source=str(repo),
        marketplace="skills-sh",
        source_subdir="skills/find-skills",
        sparse_paths=["skills/find-skills"],
        plugins_home=home,
    )
    (installed.data_path / "state.json").write_text("{}", encoding="utf-8")

    assert (installed.installed_path / "skills" / "find-skills" / "SKILL.md").exists()
    registry = json.loads((home / "registry.json").read_text(encoding="utf-8"))
    assert registry["plugins"]["find-skills@skills-sh"]["manifest_format"] == "generated_skill_package"

    removed = uninstall_plugin(
        plugin_id="find-skills@skills-sh",
        plugins_home=home,
    )

    assert not installed.installed_path.exists()
    assert removed.data_removed is False
    assert (installed.data_path / "state.json").exists()
    registry = json.loads((home / "registry.json").read_text(encoding="utf-8"))
    assert registry["plugins"] == {}


def _run_git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if result.returncode == 0:
        return
    raise AssertionError(result.stderr or result.stdout)
