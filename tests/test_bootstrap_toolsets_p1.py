from __future__ import annotations
from typing import Any, cast

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.tools.registry import ToolRegistry
from bootstrap.toolsets.protocol import (
    ToolsetRegistrationResult,
    build_registration_result,
)
from bootstrap.toolsets.schedule import SchedulerToolsetProvider
from bootstrap.toolsets.workflow import WorkflowToolsetProvider
from bootstrap.tools import _ordered_toolset_providers, build_registered_tools
from bus.event_bus import EventBus


def test_scheduler_toolset_provider_registers_expected_tools(tmp_path: Path):
    registry = ToolRegistry()
    scheduler = SimpleNamespace()

    result = SchedulerToolsetProvider().register(
        registry,
        cast(
            Any,
            SimpleNamespace(
                config=None,
                workspace=tmp_path,
                scheduler=scheduler,
            ),
        ),
    )

    assert result.source_name == "schedule"
    assert set(result.tool_names) == {
        "schedule",
        "list_schedules",
        "cancel_schedule",
    }
    assert result.always_on_names == []


def test_build_registered_tools_uses_toolset_providers(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    class _MemoryProvider:
        def register(self, registry, deps):
            calls.append("memory")
            runtime = SimpleNamespace(engine=object())
            return ToolsetRegistrationResult(
                source_name="memory",
                tool_names=[],
                extras={"memory_runtime": runtime},
            )

    class _MetaProvider:
        def __init__(self, readonly_tools):
            self._readonly_tools = readonly_tools

        def register(self, registry, deps):
            calls.append("meta")
            return ToolsetRegistrationResult(source_name="meta_common")

    task_executor = object()

    class _TaskExecutorProvider:
        def register(self, registry, deps):
            calls.append("task_executor")
            return ToolsetRegistrationResult(
                source_name="task_executor",
                extras={"task_executor": task_executor},
            )

    class _ScheduleProvider:
        def register(self, registry, deps):
            calls.append("schedule")
            return ToolsetRegistrationResult(source_name="schedule")

    class _McpProvider:
        def register(self, registry, deps):
            calls.append("mcp")
            return ToolsetRegistrationResult(
                source_name="mcp",
                extras={"mcp_registry": object()},
            )

    class _WorkflowProvider:
        def register(self, registry, deps):
            assert deps.task_executor is task_executor
            calls.append("workflow")
            return ToolsetRegistrationResult(source_name="workflow")

    monkeypatch.setattr(
        "bootstrap.tools.resolve_memory_toolset_provider",
        lambda name: _MemoryProvider(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.resolve_toolset_provider",
        lambda name, readonly_tools=None: {
            "meta_common": _MetaProvider(readonly_tools),
            "task_executor": _TaskExecutorProvider(),
            "schedule": _ScheduleProvider(),
            "mcp": _McpProvider(),
            "workflow": _WorkflowProvider(),
        }[name],
    )
    monkeypatch.setattr("bootstrap.tools.build_readonly_tools", lambda *_, **__: {})
    monkeypatch.setattr(
        "bootstrap.tools.build_scheduler",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.build_peer_agent_resources",
        lambda *_args, **_kwargs: (None, None),
    )

    runtime = build_registered_tools(
            config=cast(Any, SimpleNamespace(proactive=SimpleNamespace())),
            workspace=tmp_path,
            http_resources=cast(Any, SimpleNamespace()),
            bus=cast(Any, SimpleNamespace(chat_lane=None)),
            provider=object(),
            light_provider=object(),
            session_store=object(),
            tools=ToolRegistry(),
            event_publisher=EventBus(),
            agent_loop_provider=lambda: None,
    )
    push_tool = runtime.push_tool
    scheduler = runtime.scheduler
    mcp_registry = runtime.mcp_registry
    memory_runtime = runtime.memory_runtime
    peer_pm = runtime.peer_process_manager
    peer_poller = runtime.peer_poller

    assert calls == [
        "memory",
        "meta",
        "task_executor",
        "schedule",
        "mcp",
        "workflow",
    ]
    assert push_tool is not None
    assert scheduler is not None
    assert mcp_registry is not None
    assert memory_runtime.engine is not None
    assert peer_pm is None
    assert peer_poller is None
    assert runtime.workflow_runtime is None


def test_build_registration_result_uses_public_registry_names():
    registry = SimpleNamespace(
        get_registered_names=lambda: {"a", "b", "always"},
        get_always_on_names=lambda: {"always"},
    )

    result = build_registration_result(
        registry=cast(Any, registry),
        source_name="demo",
        before={"a"},
    )

    assert result.tool_names == ["always", "b"]
    assert result.always_on_names == ["always"]


def test_toolset_dependencies_are_ordered_without_service_lookup():
    class _Provider:
        def __init__(self, *run_after: str) -> None:
            self.run_after = run_after

        def register(self, registry, deps):
            raise AssertionError("ordering does not register providers")

    ordered = _ordered_toolset_providers(
        [
            ("workflow", cast(Any, _Provider("task_executor"))),
            ("schedule", cast(Any, _Provider())),
            ("task_executor", cast(Any, _Provider())),
        ]
    )

    assert [name for name, _ in ordered] == [
        "schedule",
        "task_executor",
        "workflow",
    ]


def test_toolset_order_rejects_duplicate_entries():
    provider = cast(Any, SimpleNamespace(run_after=()))
    with pytest.raises(ValueError, match="重复"):
        _ordered_toolset_providers([("mcp", provider), ("mcp", provider)])


def test_workflow_toolset_exposes_only_unified_task_tools(tmp_path: Path):
    registry = ToolRegistry()
    task_executor = object()

    result = WorkflowToolsetProvider().register(
        registry,
        cast(
            Any,
            SimpleNamespace(
                workspace=tmp_path,
                push_tool=object(),
                agent_loop_provider=lambda: None,
                task_executor=task_executor,
            ),
        ),
    )

    assert set(result.tool_names) == {"task_create", "task_manage"}
    assert set(result.always_on_names) == {"task_create", "task_manage"}
    assert "spawn" not in registry.get_registered_names()
    runtime = result.extras["workflow_runtime"]
    assert runtime.subagent_executor is task_executor
    runtime.store.close()
