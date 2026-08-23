from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import yaml
from agent.plugins.manager import ActivePluginInfo
from agent.plugins.skill_links import (
    PluginSkillLinker,
    remove_plugin_skill_materializations,
)
from agent.plugins.skill_paths import (
    is_plugin_skill_materialized,
    plugin_skill_materialization_path,
    write_managed_skill_copy_marker,
)
from agent.skills import SkillsLoader
from proactive_v2.drift_state import DriftStateStore


def _write_plugin_skill(
    plugin_root: Path,
    plugin_id: str,
    skill_name: str,
    *,
    body: str = "plugin skill body",
) -> Path:
    skill_dir = plugin_root / plugin_id / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: 插件技能\n---\n{body}\n",
        encoding="utf-8",
    )
    return plugin_root / plugin_id


def test_remove_plugin_skill_materializations_only_removes_owned_copies(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins" / "demo"
    owned_target = plugin_root / "skills" / "owned"
    other_target = tmp_path / "plugins" / "other" / "skills" / "kept"
    owned_target.mkdir(parents=True)
    other_target.mkdir(parents=True)
    skills_dir = workspace / "skills"
    owned_copy = skills_dir / "owned"
    kept_copy = skills_dir / "kept"
    owned_copy.mkdir(parents=True)
    kept_copy.mkdir(parents=True)
    write_managed_skill_copy_marker(
        owned_copy,
        logical_name="owned",
        target=owned_target,
        fingerprint="owned-fingerprint",
    )
    write_managed_skill_copy_marker(
        kept_copy,
        logical_name="kept",
        target=other_target,
        fingerprint="kept-fingerprint",
    )

    removed = remove_plugin_skill_materializations(
        workspace=workspace,
        plugin_root=plugin_root,
    )

    assert removed == 1
    assert not owned_copy.exists()
    assert kept_copy.exists()


def _write_plugin_drift_skill(
    plugin_root: Path,
    plugin_id: str,
    skill_name: str,
    *,
    body: str = "plugin drift skill body",
) -> Path:
    skill_dir = plugin_root / plugin_id / "drift" / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {plugin_id}:{skill_name}\n"
        "description: 插件 Drift 技能\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return plugin_root / plugin_id


def _plugin_info(
    plugin_id: str,
    plugin_dir: Path,
    manifest: dict[str, object] | None = None,
) -> ActivePluginInfo:
    return ActivePluginInfo(
        plugin_id=plugin_id,
        plugin_dir=plugin_dir,
        manifest=manifest or {},
        module_path=f"test_{plugin_id}",
    )


def _memory_engine(name: str) -> object:
    return SimpleNamespace(describe=lambda: SimpleNamespace(name=name))


def _materialized_skill_path(skills_dir: Path, logical_name: str) -> Path:
    return plugin_skill_materialization_path(skills_dir, logical_name)


def _assert_materialized(skills_dir: Path, logical_name: str) -> Path:
    materialized = _materialized_skill_path(skills_dir, logical_name)
    assert is_plugin_skill_materialized(
        materialized,
        logical_name=logical_name,
    )
    return materialized


def test_plugin_skill_linker_materializes_workspace_skill(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("foo", plugin_dir)])

    materialized = _materialized_skill_path(workspace / "skills", "foo:bar")
    assert result.expected == 1
    assert result.created == 1
    assert is_plugin_skill_materialized(materialized, logical_name="foo:bar")
    loader = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin")
    assert loader.load_skill_body("foo:bar") == "plugin skill body"


def test_plugin_skill_linker_removes_stale_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("foo", plugin_dir)])

    result = linker.sync([])

    assert result.removed == 1
    assert not _materialized_skill_path(workspace / "skills", "foo:bar").exists()


def test_plugin_skill_linker_removes_stale_managed_copy_after_root_change(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    stale_copy = _materialized_skill_path(workspace / "skills", "foo:bar")
    stale_copy.mkdir(parents=True)
    (stale_copy / "SKILL.md").write_text("stale managed copy", encoding="utf-8")
    write_managed_skill_copy_marker(
        stale_copy,
        logical_name="foo:bar",
        target=tmp_path / "removed-plugin-root" / "foo" / "skills" / "bar",
        fingerprint="stale-fingerprint",
    )

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[tmp_path / "current-plugin-root"],
        memory_engine=None,
    ).sync([])

    assert result.removed == 1
    assert not stale_copy.exists()


def test_plugin_skill_linker_removes_broken_plugin_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "gone", "bar")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("gone", plugin_dir)])
    materialized = _assert_materialized(workspace / "skills", "gone:bar")
    shutil.rmtree(plugin_dir)

    result = linker.sync([])

    assert result.removed == 1
    assert not materialized.exists()


def test_plugin_skill_linker_preserves_user_skill_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    user_skill = _materialized_skill_path(workspace / "skills", "foo:bar")
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("user body", encoding="utf-8")

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("foo", plugin_dir)])

    assert result.skipped == 1
    assert not is_plugin_skill_materialized(user_skill, logical_name="foo:bar")
    assert (user_skill / "SKILL.md").read_text(encoding="utf-8") == "user body"


def test_plugin_skill_linker_filters_by_memory_engine(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "akasha", "memory")
    manifest: dict[str, object] = {
        "skills": {
            "enabled_when": {
                "kind": "memory_engine",
                "engine": "akasha",
            }
        }
    }
    plugin = _plugin_info("akasha", plugin_dir, manifest)

    disabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])
    enabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("akasha"),
    ).sync([plugin])
    removed = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])

    assert disabled.expected == 0
    assert enabled.expected == 1
    assert removed.removed == 1
    assert not _materialized_skill_path(
        workspace / "skills",
        "akasha:memory",
    ).exists()


def test_xiaoman_plugin_skill_is_exposed_with_bare_name(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    plugin_dir = cache_root / "lab" / "feed" / "0.1.0"
    skill_dir = plugin_dir / "skills" / "feed-manage"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: feed-manage\ndescription: feed skill\n---\nbody\n",
        encoding="utf-8",
    )
    plugin = ActivePluginInfo(
        plugin_id="feed@lab",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="feed",
        declares_xiaoman_plugin=True,
        skill_roots=(plugin_dir / "skills",),
    )

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[cache_root],
        memory_engine=None,
    ).sync([plugin])

    assert result.expected == 1
    _assert_materialized(workspace / "skills", "feed-manage")
    assert not _materialized_skill_path(
        workspace / "skills",
        "feed@lab:feed-manage",
    ).exists()


def test_xiaoman_plugin_skill_sync_removes_old_prefixed_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    plugin_dir = cache_root / "lab" / "feed" / "0.1.0"
    skill_dir = plugin_dir / "skills" / "feed-manage"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: feed-manage\ndescription: feed skill\n---\nbody\n",
        encoding="utf-8",
    )
    legacy_plugin = ActivePluginInfo(
        plugin_id="feed@lab",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="feed_legacy",
        skill_roots=(plugin_dir / "skills",),
    )
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[cache_root],
        memory_engine=None,
    )
    linker.sync([legacy_plugin])
    old_materialization = _assert_materialized(
        workspace / "skills",
        "feed@lab:feed-manage",
    )
    plugin = ActivePluginInfo(
        plugin_id="feed@lab",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="feed",
        declares_xiaoman_plugin=True,
        skill_roots=(plugin_dir / "skills",),
    )

    result = linker.sync([plugin])

    assert result.created == 1
    assert result.removed == 1
    _assert_materialized(workspace / "skills", "feed-manage")
    assert not old_materialization.exists()


def test_xiaoman_plugin_drift_skill_uses_bare_plugin_name(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    plugin_dir = cache_root / "github" / "emotion" / "0.1.0"
    skill_dir = plugin_dir / "drift" / "skills" / "feedback-preference-context"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: emotion:feedback-preference-context\n"
        "description: drift skill\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    plugin = ActivePluginInfo(
        plugin_id="emotion@github",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="emotion",
        declares_xiaoman_plugin=True,
        drift_skill_roots=(plugin_dir / "drift" / "skills",),
    )

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[cache_root],
        memory_engine=None,
    ).sync([plugin])

    assert result.expected == 1
    _assert_materialized(
        workspace / "drift" / "skills",
        "emotion:feedback-preference-context",
    )
    assert not _materialized_skill_path(
        workspace / "drift" / "skills",
        "emotion@github:feedback-preference-context",
    ).exists()


def test_plugin_drift_skill_linker_materializes_workspace_skill(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_drift_skill(plugin_root, "foo", "daily")

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("foo", plugin_dir)])

    materialized = _materialized_skill_path(
        workspace / "drift" / "skills",
        "foo:daily",
    )
    store = DriftStateStore(workspace / "drift")
    skills = store.scan_skills()
    skill_dir = store.skill_dir_for("foo:daily")

    assert result.expected == 1
    assert result.created == 1
    assert is_plugin_skill_materialized(materialized, logical_name="foo:daily")
    assert {skill.name for skill in skills} == {"foo:daily"}
    assert skill_dir is not None
    assert skill_dir == materialized
    assert "plugin drift skill body" in (skill_dir / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_plugin_drift_skill_linker_removes_stale_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_drift_skill(plugin_root, "foo", "daily")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("foo", plugin_dir)])

    result = linker.sync([])

    assert result.removed == 1
    assert not _materialized_skill_path(
        workspace / "drift" / "skills",
        "foo:daily",
    ).exists()


def test_plugin_drift_skill_linker_preserves_user_skill_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_drift_skill(plugin_root, "foo", "daily")
    user_skill = _materialized_skill_path(
        workspace / "drift" / "skills",
        "foo:daily",
    )
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("user body", encoding="utf-8")

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("foo", plugin_dir)])

    assert result.skipped == 1
    assert not is_plugin_skill_materialized(user_skill, logical_name="foo:daily")
    assert (user_skill / "SKILL.md").read_text(encoding="utf-8") == "user body"


def test_plugin_drift_skill_linker_filters_by_memory_engine(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_drift_skill(plugin_root, "akasha", "daily")
    manifest: dict[str, object] = {
        "drift_skills": {
            "enabled_when": {
                "kind": "memory_engine",
                "engine": "akasha",
            }
        }
    }
    plugin = _plugin_info("akasha", plugin_dir, manifest)

    disabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])
    enabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("akasha"),
    ).sync([plugin])

    assert disabled.expected == 0
    assert enabled.expected == 1
    _assert_materialized(workspace / "drift" / "skills", "akasha:daily")


def test_default_memory_audit_drift_skill_is_gated_by_memory_engine(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = Path(__file__).parents[1] / "plugins"
    plugin_dir = plugin_root / "default_memory"
    loaded = yaml.safe_load((plugin_dir / "manifest.yaml").read_text(encoding="utf-8"))
    manifest = cast(dict[str, object], loaded)
    plugin = _plugin_info("default_memory", plugin_dir, manifest)
    materialized = _materialized_skill_path(
        workspace / "drift" / "skills",
        "default_memory:audit-dirty-memories",
    )

    disabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("akasha"),
    ).sync([plugin])
    enabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])
    skills = DriftStateStore(workspace / "drift").scan_skills()

    assert disabled.expected == 0
    assert enabled.expected == 1
    assert is_plugin_skill_materialized(
        materialized,
        logical_name="default_memory:audit-dirty-memories",
    )
    assert {skill.name for skill in skills} == {"default_memory:audit-dirty-memories"}

    removed = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("akasha"),
    ).sync([plugin])

    assert removed.removed == 1
    assert not materialized.exists()


def test_emotion_feedback_drift_skill_is_exposed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache_root = tmp_path / "cache"
    plugin_dir = cache_root / "github" / "emotion" / "0.1.0"
    skill_dir = plugin_dir / "drift" / "skills" / "feedback-preference-context"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: emotion:feedback-preference-context\n"
        "description: 情绪反馈 drift skill\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    (scripts_dir / "sample_feedback_context.py").write_text(
        "print('ok')\n",
        encoding="utf-8",
    )
    plugin = ActivePluginInfo(
        plugin_id="emotion@github",
        plugin_dir=plugin_dir,
        manifest={},
        module_path="emotion",
        declares_xiaoman_plugin=True,
        drift_skill_roots=(plugin_dir / "drift" / "skills",),
    )

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[cache_root],
        memory_engine=None,
    ).sync([plugin])
    skills = DriftStateStore(workspace / "drift").scan_skills()
    linked_skill_dir = DriftStateStore(workspace / "drift").skill_dir_for(
        "emotion:feedback-preference-context"
    )

    assert result.expected >= 1
    _assert_materialized(
        workspace / "drift" / "skills",
        "emotion:feedback-preference-context",
    )
    assert "emotion:feedback-preference-context" in {skill.name for skill in skills}
    assert linked_skill_dir is not None
    assert (linked_skill_dir / "scripts" / "sample_feedback_context.py").exists()


def test_default_memory_audit_script_uses_drift_journal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    drift_dir = workspace / "drift"
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True)
    DriftStateStore(drift_dir)
    memory_db = memory_dir / "memory2.db"
    conn = sqlite3.connect(memory_db)
    try:
        _ = conn.execute(
            """
            CREATE TABLE memory_items (
                id TEXT PRIMARY KEY,
                memory_type TEXT,
                summary TEXT,
                source_ref TEXT,
                happened_at TEXT,
                status TEXT
            )
            """
        )
        _ = conn.executemany(
            """
            INSERT INTO memory_items (
                id, memory_type, summary, source_ref, happened_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "m1",
                    "fact",
                    "用户喜欢极简实现",
                    "telegram:1",
                    "2026-07-01",
                    "active",
                ),
                (
                    "m2",
                    "fact",
                    "用户正在调 Drift",
                    "telegram:2",
                    "2026-07-02",
                    "active",
                ),
                (
                    "m3",
                    "fact",
                    "旧 post response",
                    "telegram:3@post_response",
                    "2026-07-03",
                    "active",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    script = (
        Path(__file__).parents[1]
        / "plugins"
        / "default_memory"
        / "drift"
        / "skills"
        / "audit-dirty-memories"
        / "scripts"
        / "sample_memory_for_audit.py"
    )
    sampled = _run_audit_script(script, "sample", "--drift-dir", str(drift_dir))
    item = cast(dict[str, object], sampled["item"])
    memory_id = str(item["id"])
    store = DriftStateStore(drift_dir)
    store.save_finish(
        skill_used="default_memory:audit-dirty-memories",
        status="completed",
        briefing=f"审计 memory_id={memory_id}，结果 clean",
        message_result="silent",
        scratchpad_update="下次继续随机抽样。",
        global_note_update=None,
        now_utc=datetime.now(timezone.utc),
        cursor_update={
            "next_action": "sample",
            "active_memory_id": None,
        },
        journal_append=[
            {
                "entry_type": "memory_audited",
                "key": memory_id,
                "payload": {"result": "clean"},
            }
        ],
    )
    sampled_again = _run_audit_script(script, "sample", "--drift-dir", str(drift_dir))
    journal = _load_audit_journal(drift_dir / "drift.db")
    cursor = _load_audit_cursor(drift_dir / "drift.db")

    assert sampled["found"] is True
    assert memory_id in {"m1", "m2"}
    required = cast(dict[str, object], sampled["journal_append_required"])
    assert required["entry_type"] == "memory_audited"
    if sampled_again["found"] is True:
        next_item = cast(dict[str, object], sampled_again["item"])
        assert str(next_item["id"]) != memory_id
    assert journal[memory_id]["result"] == "clean"
    assert cursor["next_action"] == "sample"
    assert not (
        drift_dir
        / "skill_state"
        / "default_memory:audit-dirty-memories"
        / "history.json"
    ).exists()


def _run_audit_script(script: Path, *args: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(result.stdout)
    return cast(dict[str, object], loaded)


def _load_audit_journal(drift_db: Path) -> dict[str, dict[str, object]]:
    conn = sqlite3.connect(drift_db)
    try:
        rows = conn.execute(
            """
            SELECT key, payload_json
            FROM skill_journal
            WHERE skill_name = 'default_memory:audit-dirty-memories'
              AND entry_type = 'memory_audited'
            """
        ).fetchall()
    finally:
        conn.close()
    return {
        str(row[0]): cast(dict[str, object], json.loads(str(row[1] or "{}")))
        for row in rows
    }


def _load_audit_cursor(drift_db: Path) -> dict[str, object]:
    conn = sqlite3.connect(drift_db)
    try:
        row = conn.execute(
            """
            SELECT cursor_json
            FROM skill_continuum
            WHERE skill_name = 'default_memory:audit-dirty-memories'
            """
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    loaded = json.loads(str(row[0] or "{}"))
    return cast(dict[str, object], loaded)
