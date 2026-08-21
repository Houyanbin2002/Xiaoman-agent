from pathlib import Path

import pytest

from agent.skill_packages import install_skill_directory, uninstall_skill
from agent.skills import SkillsLoader


def _write_skill(root: Path, name: str = "demo-skill") -> Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo skill\n---\n\nUse the demo.\n",
        encoding="utf-8",
    )
    return root


def test_installed_skill_is_discovered_and_can_be_uninstalled(tmp_path: Path) -> None:
    source = _write_skill(tmp_path / "source")
    installed_root = tmp_path / "installed"
    result = install_skill_directory(
        skill_root=source,
        source="owner/repository",
        revision="abc123",
        skills_root=installed_root,
    )

    loader = SkillsLoader(
        tmp_path / "workspace",
        builtin_skills_dir=tmp_path / "builtin",
        installed_skills_dir=installed_root,
    )
    record = loader.load_skill_record("demo-skill")
    assert record is not None
    assert record.source == "installed"
    assert record.source_id == "owner/repository"
    assert result.installed_path == installed_root / "demo-skill"

    uninstall_skill("demo-skill", skills_root=installed_root)
    assert loader.load_skill_record("demo-skill") is None


def test_skill_installer_does_not_overwrite_unmanaged_directory(tmp_path: Path) -> None:
    source = _write_skill(tmp_path / "source")
    installed_root = tmp_path / "installed"
    _write_skill(installed_root / "demo-skill")

    with pytest.raises(ValueError, match="不是由安装器管理"):
        install_skill_directory(
            skill_root=source,
            source="owner/repository",
            skills_root=installed_root,
        )
