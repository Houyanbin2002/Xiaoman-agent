from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from agent.looping.core import AgentLoop
    from agent.mcp.registry import McpServerRegistry
    from agent.plugins.manager import PluginManager
    from core.memory.runtime import MemoryRuntime

logger = logging.getLogger(__name__)


async def activate_plugins_for_core(
    *,
    plugin_manager: "PluginManager | None",
    loop: "AgentLoop",
    tools: ToolRegistry,
    mcp_registry: "McpServerRegistry",
    workspace: Path | None,
    memory_runtime: "MemoryRuntime",
) -> None:
    """Load plugins and attach their runtime contributions to the core agent."""
    if plugin_manager is None:
        return

    await plugin_manager.load_all()
    _sync_plugin_skills(
        plugin_manager=plugin_manager,
        workspace=workspace,
        memory_runtime=memory_runtime,
    )
    await _sync_plugin_mcp_servers(
        plugin_manager=plugin_manager,
        mcp_registry=mcp_registry,
    )
    _sync_global_registry(plugin_manager)
    logger.info("插件加载完成: %d 个", plugin_manager.loaded_count)
    _attach_plugin_modules(loop=loop, plugin_manager=plugin_manager)
    _attach_tool_hooks(loop=loop, tools=tools, plugin_manager=plugin_manager)


def _sync_plugin_skills(
    *,
    plugin_manager: "PluginManager",
    workspace: Path | None,
    memory_runtime: "MemoryRuntime",
) -> None:
    if workspace is None:
        return
    from agent.plugins.skill_links import PluginSkillLinker

    link_result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=plugin_manager.plugin_dirs,
        memory_engine=getattr(memory_runtime, "engine", None),
    ).sync(plugin_manager.active_plugins())
    logger.info(
        "插件 skill 同步完成: expected=%d created=%d repaired=%d removed=%d skipped=%d",
        link_result.expected,
        link_result.created,
        link_result.repaired,
        link_result.removed,
        link_result.skipped,
    )


async def _sync_plugin_mcp_servers(
    *,
    plugin_manager: "PluginManager",
    mcp_registry: "McpServerRegistry",
) -> None:
    sync_plugin_servers = getattr(mcp_registry, "sync_plugin_servers", None)
    if not callable(sync_plugin_servers):
        return
    sync_result = sync_plugin_servers(plugin_manager.active_plugins())
    if inspect.isawaitable(sync_result):
        await sync_result


def _sync_global_registry(plugin_manager: "PluginManager") -> None:
    sync_global_registry = getattr(plugin_manager, "sync_global_registry", None)
    if not callable(sync_global_registry):
        return
    registry_path = sync_global_registry()
    logger.info("插件全局注册表已同步: %s", registry_path)


def _attach_plugin_modules(
    *,
    loop: "AgentLoop",
    plugin_manager: "PluginManager",
) -> None:
    loop.add_before_turn_plugin_modules(plugin_manager.before_turn_modules)
    loop.add_before_reasoning_plugin_modules(plugin_manager.before_reasoning_modules)
    loop.add_prompt_render_plugin_modules(plugin_manager.prompt_render_modules)
    loop.add_before_step_plugin_modules(plugin_manager.before_step_modules)
    loop.add_after_step_plugin_modules(plugin_manager.after_step_modules)
    loop.add_after_reasoning_plugin_modules(plugin_manager.after_reasoning_modules)
    loop.add_after_turn_plugin_modules(plugin_manager.after_turn_modules)


def _attach_tool_hooks(
    *,
    loop: "AgentLoop",
    tools: ToolRegistry,
    plugin_manager: "PluginManager",
) -> None:
    if not plugin_manager.tool_hooks:
        return
    loop.add_tool_hooks(plugin_manager.tool_hooks)
    task_tool = tools.get_tool("task_create")
    task_runtime = getattr(task_tool, "runtime", None)
    task_executor = getattr(task_runtime, "subagent_executor", None)
    if task_executor is not None and hasattr(task_executor, "add_tool_hooks"):
        cast_add_hooks = getattr(task_executor, "add_tool_hooks")
        cast_add_hooks(plugin_manager.tool_hooks)
