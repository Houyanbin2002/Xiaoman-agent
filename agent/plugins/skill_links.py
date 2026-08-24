from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast
from uuid import uuid4

from agent.plugins.manager import ActivePluginInfo
from agent.plugins.skill_paths import (
    plugin_skill_materialization_path,
    read_managed_skill_copy_marker,
    write_managed_skill_copy_marker,
)

logger = logging.getLogger(__name__)

PluginSkillPolicyKind = Literal["plugin_loaded", "memory_engine"]


@dataclass(frozen=True)
class PluginSkillPolicy:
    kind: PluginSkillPolicyKind
    engine: str = ""


@dataclass(frozen=True)
class PluginSkillSyncResult:
    expected: int = 0
    created: int = 0
    repaired: int = 0
    removed: int = 0
    skipped: int = 0


class PluginSkillLinker:
    def __init__(
        self,
        *,
        workspace: Path,
        plugin_roots: Sequence[Path],
        memory_engine: object | None,
    ) -> None:
        self._workspace_skills = workspace / "skills"
        self._workspace_drift_skills = workspace / "drift" / "skills"
        self._plugin_roots = [root.resolve(strict=False) for root in plugin_roots]
        self._memory_engine = memory_engine

    # 将已生效插件的普通 skill 和 drift skill 物化到 workspace。
    def sync(
        self,
        active_plugins: Sequence[ActivePluginInfo],
    ) -> PluginSkillSyncResult:
        normal = self._sync_links(
            workspace_skills=self._workspace_skills,
            expected=self._build_expected_links(
                active_plugins,
                plugin_subpath=("skills",),
                manifest_key="skills",
            ),
            managed_subpath=("skills",),
        )
        drift = self._sync_links(
            workspace_skills=self._workspace_drift_skills,
            expected=self._build_expected_links(
                active_plugins,
                plugin_subpath=("drift", "skills"),
                manifest_key="drift_skills",
            ),
            managed_subpath=("drift", "skills"),
        )
        return PluginSkillSyncResult(
            expected=normal.expected + drift.expected,
            created=normal.created + drift.created,
            repaired=normal.repaired + drift.repaired,
            removed=normal.removed + drift.removed,
            skipped=normal.skipped + drift.skipped,
        )

    def _sync_links(
        self,
        *,
        workspace_skills: Path,
        expected: Mapping[str, Path],
        managed_subpath: Sequence[str],
    ) -> PluginSkillSyncResult:
        created = 0
        repaired = 0
        skipped = 0

        if expected:
            workspace_skills.mkdir(parents=True, exist_ok=True)

        materialized = {
            plugin_skill_materialization_path(
                workspace_skills, logical_name
            ).name: target
            for logical_name, target in expected.items()
        }
        for logical_name, target in expected.items():
            link = plugin_skill_materialization_path(workspace_skills, logical_name)
            action = self._ensure_link(
                link,
                target,
                logical_name=logical_name,
                managed_subpath=managed_subpath,
            )
            if action == "created":
                created += 1
            elif action == "repaired":
                repaired += 1
            elif action == "skipped":
                skipped += 1

        removed = self._cleanup_stale_links(
            workspace_skills,
            materialized,
            managed_subpath,
        )
        return PluginSkillSyncResult(
            expected=len(expected),
            created=created,
            repaired=repaired,
            removed=removed,
            skipped=skipped,
        )

    def _build_expected_links(
        self,
        active_plugins: Sequence[ActivePluginInfo],
        *,
        plugin_subpath: Sequence[str],
        manifest_key: str,
    ) -> dict[str, Path]:
        expected: dict[str, Path] = {}
        for plugin in active_plugins:
            if not _is_safe_name(plugin.plugin_id):
                logger.warning("插件 skill 跳过非法 plugin_id: %s", plugin.plugin_id)
                continue
            policy = parse_skill_policy(plugin.plugin_id, plugin.manifest, manifest_key)
            if not self._policy_enabled(plugin.plugin_id, policy):
                continue
            for skill_dir in _iter_plugin_skill_dirs(plugin, plugin_subpath):
                if not _is_safe_name(skill_dir.name):
                    logger.warning(
                        "插件 skill 跳过非法 skill 名称: %s/%s",
                        plugin.plugin_id,
                        skill_dir.name,
                    )
                    continue
                if plugin.declares_xiaoman_plugin and manifest_key == "skills":
                    link_name = skill_dir.name
                elif plugin.declares_xiaoman_plugin and manifest_key == "drift_skills":
                    link_name = f"{plugin.plugin_id.split('@', 1)[0]}:{skill_dir.name}"
                else:
                    link_name = f"{plugin.plugin_id}:{skill_dir.name}"
                target = skill_dir.resolve(strict=False)
                existing = expected.get(link_name)
                if existing is not None and existing != target:
                    logger.warning("插件 skill 名称重复，保留第一项: %s", link_name)
                    continue
                expected[link_name] = target
        return expected

    def _policy_enabled(
        self,
        plugin_id: str,
        policy: PluginSkillPolicy,
    ) -> bool:
        if policy.kind == "plugin_loaded":
            return True
        engine_name = _memory_engine_name(self._memory_engine, plugin_id)
        return bool(policy.engine and engine_name == policy.engine)

    def _ensure_link(
        self,
        link: Path,
        target: Path,
        *,
        logical_name: str,
        managed_subpath: Sequence[str],
    ) -> str:
        if link.is_symlink():
            current = _readlink_target(link)
            if current is not None and _same_path(current, target):
                return "unchanged"
            if current is None or not any(
                _is_under_plugin_skills(current, root, managed_subpath)
                for root in self._plugin_roots
            ):
                logger.warning("插件 skill 与用户软链接冲突，已保留用户路径: %s", link)
                return "skipped"
            try:
                link.unlink()
            except OSError as e:
                logger.warning("插件 skill 软链接删除失败 (%s): %s", link, e)
                return "skipped"
            return (
                "repaired"
                if self._create_link(link, target, logical_name=logical_name)
                else "skipped"
            )

        if link.exists():
            metadata = read_managed_skill_copy_marker(link)
            if metadata is None:
                logger.warning("插件 skill 与用户目录冲突，已保留用户路径: %s", link)
                return "skipped"
            if self._managed_copy_matches(
                link,
                target,
                logical_name=logical_name,
            ):
                return "unchanged"
            if not _remove_existing_path(link):
                return "skipped"
            return (
                "repaired"
                if self._create_link(link, target, logical_name=logical_name)
                else "skipped"
            )

        return (
            "created"
            if self._create_link(link, target, logical_name=logical_name)
            else "skipped"
        )

    def _create_link(
        self,
        link: Path,
        target: Path,
        *,
        logical_name: str,
    ) -> bool:
        if _prefer_managed_copy():
            try:
                self._create_managed_copy(
                    link,
                    target,
                    logical_name=logical_name,
                )
            except (OSError, shutil.Error) as copy_error:
                logger.warning(
                    "插件 skill 受管副本创建失败 (%s -> %s): %s",
                    link,
                    target,
                    copy_error,
                )
                return False
            return True

        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as symlink_error:
            try:
                self._create_managed_copy(
                    link,
                    target,
                    logical_name=logical_name,
                )
            except (OSError, shutil.Error) as copy_error:
                logger.warning(
                    "插件 skill 物化失败 (%s -> %s): symlink=%s; copy=%s",
                    link,
                    target,
                    symlink_error,
                    copy_error,
                )
                return False
            logger.info(
                "插件 skill 无法创建软链接，已降级为受管副本 (%s -> %s): %s",
                link,
                target,
                symlink_error,
            )
        return True

    def _create_managed_copy(
        self,
        link: Path,
        target: Path,
        *,
        logical_name: str,
    ) -> None:
        fingerprint = _directory_fingerprint(target)
        # Do not repeat the potentially long encoded logical name here. On
        # Windows that made nested files hit MAX_PATH during the copy fallback.
        temporary = link.with_name(f".skill-{uuid4().hex[:12]}.tmp")
        try:
            _ = shutil.copytree(
                target,
                temporary,
                ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
            )
            write_managed_skill_copy_marker(
                temporary,
                logical_name=logical_name,
                target=target,
                fingerprint=fingerprint,
            )
            _ = temporary.replace(link)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def _managed_copy_matches(
        self,
        link: Path,
        target: Path,
        *,
        logical_name: str,
    ) -> bool:
        metadata = read_managed_skill_copy_marker(link)
        if metadata is None or metadata["logical_name"] != logical_name:
            return False
        recorded_target = Path(metadata["target"])
        if not _same_path(recorded_target, target):
            return False
        try:
            return metadata["fingerprint"] == _directory_fingerprint(target)
        except OSError:
            return False

    def _cleanup_stale_links(
        self,
        workspace_skills: Path,
        expected: Mapping[str, Path],
        managed_subpath: Sequence[str],
    ) -> int:
        if not workspace_skills.exists():
            return 0
        removed = 0
        for item in list(workspace_skills.iterdir()):
            if item.name in expected:
                continue
            if not self._is_managed_materialization(item, managed_subpath):
                continue
            if not _remove_existing_path(item):
                continue
            removed += 1
        return removed

    def _is_managed_materialization(
        self,
        path: Path,
        managed_subpath: Sequence[str],
    ) -> bool:
        if not path.is_symlink():
            # A managed-copy marker is written only after the copy succeeds and
            # is the ownership boundary used by the rest of the skill loader.
            # Do not require its recorded target to remain under a currently
            # configured plugin root: roots can be removed, and Windows may
            # normalize short and long path aliases differently between runs.
            return read_managed_skill_copy_marker(path) is not None

        target = _readlink_target(path)
        if target is None:
            return False
        return any(
            _is_under_plugin_skills(target, root, managed_subpath)
            for root in self._plugin_roots
        )


def parse_skill_policy(
    plugin_id: str,
    manifest: Mapping[str, object],
    section: str = "skills",
) -> PluginSkillPolicy:
    raw_skills = manifest.get(section)
    if not isinstance(raw_skills, dict):
        return PluginSkillPolicy(kind="plugin_loaded")
    skills = cast(Mapping[str, object], raw_skills)
    raw_policy = skills.get("enabled_when")
    if not isinstance(raw_policy, dict):
        return PluginSkillPolicy(kind="plugin_loaded")
    policy = cast(Mapping[str, object], raw_policy)

    kind = str(policy.get("kind") or "plugin_loaded").strip()
    if kind == "plugin_loaded":
        return PluginSkillPolicy(kind="plugin_loaded")
    if kind == "memory_engine":
        engine = str(policy.get("engine") or "").strip()
        if engine:
            return PluginSkillPolicy(kind="memory_engine", engine=engine)

    logger.warning(
        "插件 %s 的 %s.enabled_when 无效，使用 plugin_loaded",
        plugin_id,
        section,
    )
    return PluginSkillPolicy(kind="plugin_loaded")


def _iter_plugin_skill_dirs(
    plugin: ActivePluginInfo,
    plugin_subpath: Sequence[str],
) -> list[Path]:
    result: list[Path] = []
    roots = _resolve_skill_roots(plugin, plugin_subpath)
    for skills_dir in roots:
        if not skills_dir.is_dir():
            continue
        for child in sorted(skills_dir.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            if not (child / "SKILL.md").exists():
                continue
            result.append(child)
    return result


def _resolve_skill_roots(
    plugin: ActivePluginInfo,
    plugin_subpath: Sequence[str],
) -> tuple[Path, ...]:
    if tuple(plugin_subpath) == ("skills",):
        if plugin.declares_xiaoman_plugin:
            return plugin.skill_roots
        if plugin.skill_roots:
            return plugin.skill_roots
    if tuple(plugin_subpath) == ("drift", "skills"):
        if plugin.declares_xiaoman_plugin:
            return plugin.drift_skill_roots
        if plugin.drift_skill_roots:
            return plugin.drift_skill_roots
    return (plugin.plugin_dir.joinpath(*plugin_subpath),)


def _memory_engine_name(
    memory_engine: object | None,
    plugin_id: str,
) -> str:
    if memory_engine is None:
        return ""
    describe = getattr(memory_engine, "describe", None)
    if not callable(describe):
        return ""
    try:
        descriptor = describe()
    except Exception as e:
        logger.warning("插件 %s 读取 memory engine 描述失败: %s", plugin_id, e)
        return ""
    return str(getattr(descriptor, "name", "") or "")


def _is_safe_name(name: str) -> bool:
    value = name.strip()
    return bool(value) and "/" not in value and "\\" not in value and ".." not in value


def _readlink_target(link: Path) -> Path | None:
    try:
        raw = link.readlink()
    except OSError as e:
        logger.warning("读取软链接失败 (%s): %s", link, e)
        return None
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (link.parent / raw).resolve(strict=False)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _prefer_managed_copy() -> bool:
    """Use ownership-marked copies where Windows link semantics are unstable."""
    return os.name == "nt"


def _remove_existing_path(path: Path) -> bool:
    try:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as e:
        logger.warning("插件 skill 覆盖旧路径失败 (%s): %s", path, e)
        return False
    return True


def _directory_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _is_under_plugin_skills(
    target: Path,
    plugin_root: Path,
    managed_subpath: Sequence[str],
) -> bool:
    normalized_target = target.resolve(strict=False)
    normalized_root = plugin_root.resolve(strict=False)
    try:
        relative = normalized_target.relative_to(normalized_root)
    except ValueError:
        return False
    parts = relative.parts
    expected = tuple(managed_subpath)
    return (
        len(parts) >= len(expected) + 1 and parts[-(len(expected) + 1) : -1] == expected
    )
