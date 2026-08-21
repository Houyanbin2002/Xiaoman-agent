from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from agent.capabilities import CapabilityCatalog, CapabilityRouter
from agent.plugins.skill_paths import write_managed_skill_copy_marker
from agent.skills import SkillsLoader
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool


class _Tool(Tool):
    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calendar_id": {"type": "string"},
                "start_time": {"type": "string"},
            },
        }

    async def execute(self, **_: Any) -> str:
        return "ok"


def _write_skill(root: Path, name: str, description: str) -> Path:
    skill = root / "skills" / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return skill


def _skills(root: Path) -> SkillsLoader:
    return SkillsLoader(
        root,
        builtin_skills_dir=root / "none",
        installed_skills_dir=root / "installed",
    )


def test_catalog_searches_tools_skills_and_parameter_names(tmp_path: Path) -> None:
    _write_skill(tmp_path, "meeting-workflow", "整理会议纪要并提取后续行动")
    registry = ToolRegistry()
    registry.register(_Tool("calendar_create", "创建日历日程"))
    catalog = CapabilityCatalog(
        registry,
        _skills(tmp_path),
    )

    assert catalog.search("calendar_id", top_k=1)[0].record.name == "calendar_create"
    meeting = catalog.search("整理会议纪要", top_k=3)
    assert any(
        match.record.kind == "skill" and match.record.name == "meeting-workflow"
        for match in meeting
    )
    inventory = _skills(tmp_path).build_skills_inventory_summary()
    assert "meeting-workflow" not in inventory
    assert "Skill 1 个" in inventory


def test_catalog_reflects_runtime_install_and_uninstall(tmp_path: Path) -> None:
    registry = ToolRegistry()
    skills = _skills(tmp_path)
    catalog = CapabilityCatalog(registry, skills)

    registry.register(
        _Tool("remote_notes", "读取远程笔记"),
        source_type="mcp",
        source_name="notes",
    )
    skill_dir = _write_skill(tmp_path, "notes-review", "复盘远程笔记")
    assert {match.record.name for match in catalog.search("远程笔记", top_k=5)} == {
        "remote_notes",
        "notes-review",
    }

    registry.unregister("remote_notes")
    shutil.rmtree(skill_dir)
    assert catalog.search("远程笔记", top_k=5) == []


def test_materialized_plugin_skill_keeps_plugin_provenance(tmp_path: Path) -> None:
    target = (
        tmp_path
        / "cache"
        / "market"
        / "notes-plugin"
        / "1.0.0"
        / "skills"
        / "review"
    )
    target.mkdir(parents=True)
    materialized = _write_skill(tmp_path, "review", "复盘笔记")
    write_managed_skill_copy_marker(
        materialized,
        logical_name="review",
        target=target,
        fingerprint="test",
    )

    record = _skills(tmp_path).load_skill_record("review")

    assert record is not None
    assert record.source == "plugin"
    assert record.source_id == "notes-plugin"


def test_router_preloads_tool_and_activates_high_confidence_skill(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "capability-management", "添加并连接远程 MCP 工具")
    registry = ToolRegistry()
    registry.register(_Tool("mcp_add", "添加并连接远程 MCP 工具"))
    router = CapabilityRouter(
        CapabilityCatalog(
            registry,
            _skills(tmp_path),
        )
    )

    route = router.route("请用 capability-management 执行 mcp_add")

    assert route.active_skills == ("capability-management",)
    assert route.preloaded_tools == ("mcp_add",)
    assert "尚未执行任何操作" in route.prompt()


def test_tool_search_returns_tool_unlocks_and_skill_candidates(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "meeting-workflow", "整理会议并提取行动")
    registry = ToolRegistry()
    registry.register(_Tool("meeting_export", "导出会议和行动"))
    search = ToolSearchTool(
        registry,
        _skills(tmp_path),
    )

    result = json.loads(asyncio.run(search.execute(query="会议行动", top_k=5)))

    assert result["unlocked"] == ["meeting_export"]
    assert [skill["name"] for skill in result["skills"]] == ["meeting-workflow"]
    assert "load_skill" in result["next_action"]
