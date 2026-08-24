from __future__ import annotations

from agent.skills import SkillsLoader
from agent.background.subagent_executor import SubagentExecutor
from agent.tool_bundles import build_readonly_research_tools
from agent.tools.base import Tool
from agent.tools.meta import register_common_meta_tools
from agent.tools.registry import ToolRegistry
from agent.tools.skill_loader import LoadSkillTool
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    build_registration_result,
)
from core.net.http import SharedHttpResources


class CommonMetaToolsetProvider(ToolsetProvider):
    def __init__(self, readonly_tools: dict[str, Tool]) -> None:
        self._readonly_tools = readonly_tools

    def register(self, registry: ToolRegistry, deps: ToolsetDeps):
        before = registry.get_registered_names()
        skills = SkillsLoader(deps.workspace)
        push_tool = register_common_meta_tools(
            registry,
            self._readonly_tools,
            deps.session_store,
            push_tool=deps.push_tool,
            skills=skills,
        )
        registry.register(
            LoadSkillTool(skills),
            always_on=True,
            risk="read-only",
            search_hint="技能 skill SKILL.md 使用能力 先 load_skill 不要 read_file 猜路径",
        )

        # 主模型不支持多模态时，注册视觉工具供模型调用
        if deps.vl_provider is not None and deps.vl_model:
            from agent.tools.vision import ReadImageVisionTool

            registry.register(
                ReadImageVisionTool(
                    vl_provider=deps.vl_provider,
                    vl_model=deps.vl_model,
                ),
                always_on=True,
                risk="read-only",
                search_hint="看图 识图 图片内容 视觉识别 VL",
            )

        return build_registration_result(
            registry=registry,
            source_name="meta_common",
            before=before,
            extras={"push_tool": push_tool},
        )


class TaskExecutorToolsetProvider(ToolsetProvider):
    def register(self, registry: ToolRegistry, deps: ToolsetDeps):
        before = registry.get_registered_names()
        config = deps.config
        http_resources = deps.http_resources
        if config is None or http_resources is None:
            raise ValueError("task executor toolset 缺少必要依赖")
        subagent_executor = SubagentExecutor(
            provider=deps.agent_provider or deps.provider,
            workspace=deps.workspace,
            model=config.agent_model or config.model,
            max_tokens=config.max_tokens,
            fetch_requester=http_resources.external_default,
            multimodal=config.multimodal,
            execution_guard=config.execution_guard,
        )
        return build_registration_result(
            registry=registry,
            source_name="task_executor",
            before=before,
            extras={"task_executor": subagent_executor},
        )
def build_readonly_tools(
    http_resources: SharedHttpResources,
    *,
    multimodal: bool = True,
    vl_available: bool = False,
) -> dict[str, Tool]:
    return {
        tool.name: tool
        for tool in build_readonly_research_tools(
            fetch_requester=http_resources.external_default,
            include_list_dir=True,
            multimodal=multimodal,
            vl_available=vl_available,
        )
    }
