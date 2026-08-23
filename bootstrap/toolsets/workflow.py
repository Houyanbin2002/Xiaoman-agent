from __future__ import annotations

from agent.tools.registry import ToolRegistry
from agent.tools.workflow import TaskCreateTool, TaskManageTool
from agent.workflows.runtime import WorkflowRuntime
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.runtime.execution_guard import ExecutionGuardConfig
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    build_registration_result,
)
from infra.persistence.workflow_store import WorkflowStore


class WorkflowToolsetProvider(ToolsetProvider):
    run_after = ("task_executor",)

    def register(self, registry: ToolRegistry, deps: ToolsetDeps):
        before = registry.get_registered_names()
        if deps.push_tool is None or deps.agent_loop_provider is None:
            raise RuntimeError(
                "WorkflowToolsetProvider requires push_tool and agent_loop_provider"
            )
        config = getattr(deps, "config", None)
        guard = (
            config.execution_guard
            if config is not None
            else ExecutionGuardConfig()
        ).normalized()
        runtime = WorkflowRuntime(
            store=WorkflowStore(deps.workspace / "langgraph-workflow-index.db"),
            graph_runtime=LangGraphRuntime(deps.workspace / "langgraph-workflows.db"),
            agent_loop_provider=deps.agent_loop_provider,
            push_tool=deps.push_tool,
            tool_registry=registry,
            subagent_executor=deps.task_executor,
            trace_recorder=getattr(deps, "trace_recorder", None),
            max_concurrency=guard.workflow_max_concurrency,
            step_timeout_seconds=guard.workflow_step_timeout_seconds,
            max_subagent_steps=guard.workflow_max_subagent_steps,
        )
        registry.register(
            TaskCreateTool(runtime),
            always_on=True,
            risk="write",
            search_hint="任务中心 后台任务 长任务 多步骤 任务拆解 状态机 持久化",
        )
        registry.register(
            TaskManageTool(runtime),
            always_on=True,
            risk="external-side-effect",
            search_hint="任务进度 后台任务 继续 同意 拒绝 审批 回答 重试 取消",
        )
        return build_registration_result(
            registry=registry,
            source_name="workflow",
            before=before,
            extras={"workflow_runtime": runtime},
        )
