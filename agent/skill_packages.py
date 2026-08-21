from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_GITHUB_REPOSITORY = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_INSTALL_MARKER = ".xiaoman-skill.json"


@dataclass(frozen=True)
class SkillInstallResult:
    name: str
    source: str
    revision: str
    installed_path: Path


def installed_skills_root(root: Path | None = None) -> Path:
    return root or (Path.home() / ".xiaoman" / "skills")


def install_skill_from_git(
    *,
    source: str,
    ref_name: str = "",
    source_subdir: str = "",
    skills_root: Path | None = None,
) -> SkillInstallResult:
    normalized_source = _normalize_git_source(source)
    with tempfile.TemporaryDirectory(prefix="xiaoman-skill-") as temporary:
        clone_root = Path(temporary)
        sparse_paths = [source_subdir] if source_subdir else []
        _clone_git_source(
            source=normalized_source,
            destination=clone_root,
            ref_name=ref_name,
            sparse_paths=sparse_paths,
        )
        skill_root = _resolve_skill_root(clone_root, source_subdir)
        return install_skill_directory(
            skill_root=skill_root,
            source=normalized_source,
            revision=_git_revision(clone_root),
            skills_root=skills_root,
        )


def install_skill_directory(
    *,
    skill_root: Path,
    source: str,
    revision: str = "",
    skills_root: Path | None = None,
) -> SkillInstallResult:
    root = skill_root.resolve(strict=False)
    skill_file = root / "SKILL.md"
    if not skill_file.is_file():
        raise ValueError("Skill 目录必须直接包含 SKILL.md")
    name = _read_skill_name(skill_file, fallback=root.name)
    target_root = installed_skills_root(skills_root).resolve(strict=False)
    target_root.mkdir(parents=True, exist_ok=True)
    target = (target_root / name).resolve(strict=False)
    _assert_within(target, target_root)
    if target.exists():
        marker = _read_marker(target)
        if marker is None:
            raise ValueError(f"Skill {name!r} 已存在且不是由安装器管理，未覆盖")
        shutil.rmtree(target)
    shutil.copytree(root, target, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    marker = {
        "schema_version": 1,
        "name": name,
        "source": source,
        "revision": revision,
    }
    (target / _INSTALL_MARKER).write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return SkillInstallResult(
        name=name,
        source=source,
        revision=revision,
        installed_path=target,
    )


def uninstall_skill(name: str, *, skills_root: Path | None = None) -> Path:
    normalized = _validate_skill_name(name)
    root = installed_skills_root(skills_root).resolve(strict=False)
    target = (root / normalized).resolve(strict=False)
    _assert_within(target, root)
    if not target.is_dir() or _read_marker(target) is None:
        raise ValueError(f"独立安装的 Skill 不存在: {normalized}")
    shutil.rmtree(target)
    return target


def installed_skill_metadata(skill_root: Path) -> dict[str, Any] | None:
    return _read_marker(skill_root)


def _read_skill_name(skill_file: Path, *, fallback: str) -> str:
    content = skill_file.read_text(encoding="utf-8")
    name = fallback
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            loaded = cast(object, yaml.safe_load(parts[1]) or {})
            if isinstance(loaded, dict):
                name = str(cast(dict[object, object], loaded).get("name") or fallback)
    return _validate_skill_name(name)


def _read_marker(skill_root: Path) -> dict[str, Any] | None:
    marker = skill_root / _INSTALL_MARKER
    if not marker.is_file():
        return None
    try:
        loaded = cast(object, json.loads(marker.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    return {str(key): value for key, value in cast(dict[object, Any], loaded).items()}


def _validate_skill_name(value: str) -> str:
    normalized = value.strip()
    if not _SKILL_NAME.fullmatch(normalized):
        raise ValueError(
            "Skill 名称只能包含字母、数字、点、下划线或连字符，长度不超过 64"
        )
    return normalized


def _normalize_git_source(source: str) -> str:
    normalized = source.strip()
    if not normalized:
        raise ValueError("Skill 来源不能为空")
    _, marker, remainder = normalized.partition("://")
    authority = remainder.split("/", 1)[0] if marker else ""
    if marker and "@" in authority:
        raise ValueError("Skill 来源 URL 不能包含凭据，请使用 Git 凭据管理器")
    if _GITHUB_REPOSITORY.fullmatch(normalized):
        return f"https://github.com/{normalized}.git"
    return normalized


def _clone_git_source(
    *,
    source: str,
    destination: Path,
    ref_name: str,
    sparse_paths: list[str],
) -> None:
    if sparse_paths:
        _run_git(
            ["clone", "--filter=blob:none", "--no-checkout", source, str(destination)]
        )
        _run_git(["sparse-checkout", "set", *sparse_paths], cwd=destination)
        _run_git(["checkout", ref_name or "HEAD"], cwd=destination)
        return
    _run_git(["clone", source, str(destination)])
    if ref_name:
        _run_git(["checkout", ref_name], cwd=destination)


def _resolve_skill_root(clone_root: Path, source_subdir: str) -> Path:
    root = clone_root.resolve(strict=False)
    if not source_subdir:
        return root
    relative = Path(source_subdir.strip().replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Skill 子目录必须是仓库内的相对路径")
    candidate = (root / relative).resolve(strict=False)
    _assert_within(candidate, root)
    return candidate


def _git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Git 操作失败: {detail or result.returncode}")


def _assert_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("目标路径越出允许目录") from exc
