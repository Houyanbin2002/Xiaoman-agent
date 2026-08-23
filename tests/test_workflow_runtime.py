from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent.tools.workflow import (
    TaskCreateTool,
    TaskManageTool,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.workflows.runtime import WorkflowRuntime
from core.workflow.models import (
    StepExecutor,
    StepKind,
    StepSpec,
    StepStatus,
    WorkflowStatus,
)
from infra.persistence.workflow_store import WorkflowStore


def _spec(
    step_id: str,
    *,
    kind: StepKind = StepKind.AGENT,
    depends_on: tuple[str, ...] = (),
    max_attempts: int = 2,
    executor: StepExecutor = StepExecutor.AGENT,
    profile: str = "research",
    allowed_tools: tuple[str, ...] = (),
) -> StepSpec:
    return StepSpec(
        id=step_id,
        title=f"Step {step_id}",
        description=f"Execute {step_id}",
        kind=kind,
        depends_on=depends_on,
        max_attempts=max_attempts,
        executor=executor,
        profile=profile,
        allowed_tools=allowed_tools,
    )


async def _eventually(predicate, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.01)


def test_store_advances_dependencies_and_waiting_steps(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    workflow = store.create_workflow(
        name="daily review",
        goal="finish review",
        steps=[
            _spec("collect"),
            _spec("feeling", kind=StepKind.WAIT_USER, depends_on=("collect",)),
            _spec("adjust", depends_on=("feeling",)),
        ],
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
    )

    claimed = store.claim_runnable_steps(limit=3)
    assert [(step.id, step.status) for _, step in claimed] == [
        ("collect", StepStatus.RUNNING)
    ]

    store.complete_step(workflow.id, "collect", output={"load": 42})
    waiting = store.prepare_human_steps()
    assert [step.id for _, step in waiting] == ["feeling"]
    assert store.require_workflow(workflow.id).status == WorkflowStatus.WAITING

    store.respond_to_step(workflow.id, "feeling", response="有点疲劳")
    assert store.require_workflow(workflow.id).status == WorkflowStatus.RUNNING
    claimed = store.claim_runnable_steps(limit=3)
    assert [step.id for _, step in claimed] == ["adjust"]

    completed = store.complete_step(workflow.id, "adjust", output="降负荷 10%")
    assert completed.status == WorkflowStatus.SUCCEEDED
    assert completed.steps[1].output == {"response": "有点疲劳"}
    event_types = [event.event_type for event in store.list_events(workflow.id)]
    assert "step_waiting" in event_types
    assert "step_succeeded" in event_types
    store.close()


def test_store_replan_preserves_completed_steps_and_replaces_pending_tail(
    tmp_path: Path,
):
    store = WorkflowStore(tmp_path / "workflows.db")
    workflow = store.create_workflow(
        name="adaptive research",
        goal="adjust the remaining plan from evidence",
        steps=[
            _spec("collect"),
            _spec("obsolete", depends_on=("collect",)),
        ],
        session_key="cli:1",
        channel="cli",
        chat_id="1",
    )
    claimed = store.claim_runnable_steps(limit=1)
    assert claimed[0][1].id == "collect"
    completed = store.complete_step(workflow.id, "collect", output="new evidence")

    replanned = store.replan_workflow(
        workflow.id,
        remaining_steps=[_spec("analyze", depends_on=("collect",))],
        expected_revision=completed.revision,
        reason="collected evidence invalidated the old step",
    )

    assert [step.id for step in replanned.steps] == ["collect", "analyze"]
    assert replanned.steps[0].status == StepStatus.SUCCEEDED
    assert replanned.steps[0].output == "new evidence"
    assert replanned.steps[1].status == StepStatus.PENDING
    assert replanned.status == WorkflowStatus.RUNNING
    assert store.claim_runnable_steps(limit=1)[0][1].id == "analyze"
    event = next(
        item
        for item in store.list_events(workflow.id)
        if item.event_type == "workflow_replanned"
    )
    assert event.payload["preserved_step_ids"] == ["collect"]
    assert event.payload["replaced_step_ids"] == ["obsolete"]
    store.close()


def test_store_replan_rejects_stale_revision(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    workflow = store.create_workflow(
        name="revision guard",
        goal="prevent lost updates",
        steps=[_spec("old")],
        session_key="cli:1",
        channel="cli",
        chat_id="1",
    )

    with pytest.raises(ValueError, match="revision 已变化"):
        store.replan_workflow(
            workflow.id,
            remaining_steps=[_spec("new")],
            expected_revision=workflow.revision - 1,
            reason="stale planner output",
        )
    assert [step.id for step in store.require_workflow(workflow.id).steps] == ["old"]
    store.close()


def test_runtime_rejects_too_many_subagent_steps(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        subagent_executor=_FakeSubagentExecutor(),
        max_subagent_steps=2,
    )

    with pytest.raises(ValueError, match="最多允许 2 个 SubAgent"):
        runtime.create_workflow(
            name="too many children",
            goal="fan out",
            steps=[
                _spec(f"child-{index}", executor=StepExecutor.SUBAGENT)
                for index in range(3)
            ],
            session_key="dashboard:test",
            channel="dashboard",
            chat_id="test",
        )

    asyncio.run(runtime.aclose())


def test_store_discards_late_agent_results_after_cancellation(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    workflow = store.create_workflow(
        name="cancel race",
        goal="keep cancellation terminal",
        steps=[_spec("work")],
        session_key="cli:1",
        channel="cli",
        chat_id="1",
    )
    assert store.claim_runnable_steps(limit=1)[0][1].status == StepStatus.RUNNING

    cancelled = store.cancel_workflow(workflow.id, reason="user stopped it")
    cancelled_revision = cancelled.revision

    after_complete = store.complete_step(workflow.id, "work", output="late result")
    after_failure = store.fail_step(
        workflow.id,
        "work",
        error="late failure",
        retry_delay_seconds=0,
    )

    assert after_complete.status == WorkflowStatus.CANCELLED
    assert after_failure.status == WorkflowStatus.CANCELLED
    assert after_failure.revision == cancelled_revision
    assert after_failure.steps[0].status == StepStatus.CANCELLED
    assert after_failure.steps[0].output is None
    assert after_failure.steps[0].error == "user stopped it"
    assert {event.event_type for event in store.list_events(workflow.id)}.isdisjoint(
        {"step_succeeded", "step_failed", "step_retry_scheduled"}
    )
    store.close()


def test_store_only_deletes_terminal_workflow_history(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    workflow = store.create_workflow(
        name="history",
        goal="keep active execution safe",
        steps=[_spec("work")],
        session_key="cli:1",
        channel="cli",
        chat_id="1",
    )

    with pytest.raises(ValueError, match="先取消"):
        store.delete_workflow(workflow.id)
    store.cancel_workflow(workflow.id, reason="test cleanup")
    assert store.delete_workflow(workflow.id) is True
    assert store.get_workflow(workflow.id) is None
    assert store.delete_workflow(workflow.id) is False
    store.close()


def test_store_rejects_dependency_cycles(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    with pytest.raises(ValueError, match="存在环"):
        store.create_workflow(
            name="cycle",
            goal="invalid",
            steps=[
                _spec("a", depends_on=("b",)),
                _spec("b", depends_on=("a",)),
            ],
            session_key="cli:1",
            channel="cli",
            chat_id="1",
        )
    store.close()


def test_store_recovers_interrupted_steps_and_blocks_after_retries(tmp_path: Path):
    db_path = tmp_path / "workflows.db"
    store = WorkflowStore(db_path)
    workflow = store.create_workflow(
        name="recover",
        goal="recover after restart",
        steps=[_spec("work", max_attempts=2)],
        session_key="cli:1",
        channel="cli",
        chat_id="1",
    )
    claimed = store.claim_runnable_steps(limit=1)
    assert claimed[0][1].attempt_count == 1
    store.close()

    restored = WorkflowStore(db_path)
    assert restored.recover_interrupted() == 1
    pending = restored.require_workflow(workflow.id).steps[0]
    assert pending.status == StepStatus.PENDING
    claimed = restored.claim_runnable_steps(limit=1)
    assert claimed[0][1].attempt_count == 2
    blocked = restored.fail_step(
        workflow.id,
        "work",
        error="still broken",
        retry_delay_seconds=0,
    )
    assert blocked.status == WorkflowStatus.BLOCKED
    assert blocked.steps[0].status == StepStatus.FAILED
    restored.close()


def test_store_migrates_and_persists_allowed_tools(tmp_path: Path):
    db_path = tmp_path / "workflows.db"
    store = WorkflowStore(db_path)
    legacy = store.create_workflow(
        name="legacy",
        goal="open an old database",
        steps=[_spec("read")],
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        auto_start=False,
    )
    store.close()

    with sqlite3.connect(db_path) as db:
        db.execute("ALTER TABLE workflow_steps DROP COLUMN allowed_tools_json")

    migrated = WorkflowStore(db_path)
    assert migrated.require_workflow(legacy.id).steps[0].allowed_tools == ()
    current = migrated.create_workflow(
        name="current",
        goal="persist an explicit permission",
        steps=[_spec("write", allowed_tools=("write_data",))],
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        auto_start=False,
    )
    migrated.close()

    restored = WorkflowStore(db_path)
    assert restored.require_workflow(current.id).steps[0].allowed_tools == (
        "write_data",
    )
    restored.close()


class _FakeLoop:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.disabled_tools_seen: list[list[str]] = []

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        busy_session_key: str | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        omit_user_turn: bool = False,
        skip_post_memory: bool = False,
        skip_memory_retrieval: bool = False,
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
    ) -> str:
        self.prompts.append(content)
        self.disabled_tools_seen.append(list(disabled_tools or []))
        if "当前步骤: Step collect" in content:
            return "collected data"
        return "final recommendation"


class _FakePush:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def execute(self, **kwargs: Any) -> str:
        self.messages.append(str(kwargs.get("message") or ""))
        return "文本已发送"


class _CancellableLoop:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        busy_session_key: str | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        omit_user_turn: bool = False,
        skip_post_memory: bool = False,
        skip_memory_retrieval: bool = False,
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
    ) -> str:
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return content


class _FakeSubagentExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "独立调研已经完成"


class _ReadDataTool(Tool):
    name = "read_data"
    description = "Read test data"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs)


class _WriteDataTool(Tool):
    name = "write_data"
    description = "Write test data"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs)


class _OtherWriteTool(Tool):
    name = "other_write"
    description = "Write other test data"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs)


class _ToolSearchStub(Tool):
    name = "tool_search"
    description = "Search registered tools"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs)


def _permission_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_ReadDataTool(), risk="read-only")
    registry.register(_WriteDataTool(), risk="write")
    registry.register(_OtherWriteTool(), risk="external-side-effect")
    registry.register(_ToolSearchStub(), risk="read-only", always_on=True)
    return registry


@pytest.mark.asyncio
async def test_agent_step_defaults_to_read_only_and_keeps_tool_search_confined(
    tmp_path: Path,
):
    store = WorkflowStore(tmp_path / "workflows.db")
    loop = _FakeLoop()
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: loop,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        tool_registry=_permission_registry(),
        poll_interval_seconds=0.01,
    )
    workflow = runtime.create_workflow(
        name="safe offline step",
        goal="read without side effects",
        steps=[_spec("inspect")],
        session_key="cli:test",
        channel="cli",
        chat_id="test",
    )

    worker = asyncio.create_task(runtime.run())
    await _eventually(
        lambda: store.require_workflow(workflow.id).status == WorkflowStatus.SUCCEEDED
    )

    disabled = set(loop.disabled_tools_seen[0])
    assert "read_data" not in disabled
    assert "tool_search" not in disabled
    assert {"write_data", "other_write"} <= disabled
    assert {"task_create", "task_manage", "message_push"} <= disabled
    await runtime.aclose()
    await worker


@pytest.mark.asyncio
async def test_task_create_rejects_side_effect_tool_without_direct_approval(
    tmp_path: Path,
):
    store = WorkflowStore(tmp_path / "workflows.db")
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        tool_registry=_permission_registry(),
    )
    result = await TaskCreateTool(runtime).execute(
        name="unsafe task",
        goal="write without approval",
        steps=[
            {
                "id": "write",
                "title": "Write",
                "description": "Write data",
                "kind": "agent",
                "allowed_tools": ["write_data"],
            }
        ],
    )

    assert result.startswith("错误：")
    assert "必须直接依赖 approval" in result
    assert store.list_workflows() == []
    await runtime.aclose()


@pytest.mark.asyncio
async def test_approved_step_opens_only_its_declared_side_effect_tool(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    loop = _FakeLoop()
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: loop,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        tool_registry=_permission_registry(),
        poll_interval_seconds=0.01,
    )
    workflow = runtime.create_workflow(
        name="approved write",
        goal="write one approved record",
        steps=[
            _spec("approve", kind=StepKind.APPROVAL),
            _spec(
                "write",
                depends_on=("approve",),
                allowed_tools=("write_data",),
            ),
        ],
        session_key="cli:test",
        channel="cli",
        chat_id="test",
    )
    store.prepare_human_steps()
    store.approve_step(workflow.id, "approve", approved=True)

    worker = asyncio.create_task(runtime.run())
    await _eventually(
        lambda: store.require_workflow(workflow.id).status == WorkflowStatus.SUCCEEDED
    )

    disabled = set(loop.disabled_tools_seen[0])
    assert "write_data" not in disabled
    assert "other_write" in disabled
    assert "read_data" not in disabled
    assert {"task_create", "task_manage", "message_push"} <= disabled
    await runtime.aclose()
    await worker


@pytest.mark.asyncio
async def test_approval_prompt_names_downstream_non_read_only_permissions(
    tmp_path: Path,
):
    store = WorkflowStore(tmp_path / "workflows.db")
    push = _FakePush()
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=push,  # type: ignore[arg-type]
        tool_registry=_permission_registry(),
    )
    runtime.create_workflow(
        name="informed approval",
        goal="show the exact permission",
        steps=[
            _spec("approve", kind=StepKind.APPROVAL),
            _spec(
                "write",
                depends_on=("approve",),
                allowed_tools=("write_data",),
            ),
        ],
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
    )
    store.prepare_human_steps()

    await runtime._deliver_waiting_prompts()

    assert len(push.messages) == 1
    assert "Step write：write_data" in push.messages[0]
    assert "未列出的写入或外部副作用工具仍保持禁用" in push.messages[0]
    assert "other_write" not in push.messages[0]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_task_create_rejects_unknown_and_reserved_allowed_tools(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        tool_registry=_permission_registry(),
    )
    create = TaskCreateTool(runtime)
    base_step = {
        "id": "work",
        "title": "Work",
        "description": "Do work",
        "kind": "agent",
    }

    unknown = await create.execute(
        name="unknown",
        goal="reject unknown tool",
        steps=[{**base_step, "allowed_tools": ["missing_tool"]}],
    )
    reserved = await create.execute(
        name="reserved",
        goal="reject reserved tool",
        steps=[{**base_step, "allowed_tools": ["task_manage"]}],
    )

    assert "不存在的工具" in unknown
    assert "保留工具" in reserved
    assert store.list_workflows() == []
    await runtime.aclose()


@pytest.mark.asyncio
async def test_runtime_rechecks_approval_output_before_opening_write_tool(
    tmp_path: Path,
):
    db_path = tmp_path / "workflows.db"
    store = WorkflowStore(db_path)
    loop = _FakeLoop()
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: loop,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        tool_registry=_permission_registry(),
        poll_interval_seconds=0.01,
    )
    workflow = runtime.create_workflow(
        name="tampered approval",
        goal="fail closed",
        steps=[
            _spec("approve", kind=StepKind.APPROVAL),
            _spec(
                "write",
                depends_on=("approve",),
                max_attempts=1,
                allowed_tools=("write_data",),
            ),
        ],
        session_key="cli:test",
        channel="cli",
        chat_id="test",
    )
    store.prepare_human_steps()
    store.approve_step(workflow.id, "approve", approved=True)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE workflow_steps SET output_json = ? WHERE workflow_id = ? AND id = ?",
            ('{"approved":false}', workflow.id, "approve"),
        )

    worker = asyncio.create_task(runtime.run())
    await _eventually(
        lambda: store.require_workflow(workflow.id).status == WorkflowStatus.BLOCKED
    )

    failed = store.require_workflow(workflow.id).steps[-1]
    assert "缺少已批准" in failed.error
    assert loop.prompts == []
    await runtime.aclose()
    await worker


@pytest.mark.asyncio
async def test_runtime_executes_agent_steps_and_notifies(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    loop = _FakeLoop()
    push = _FakePush()
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: loop,
        push_tool=push,  # type: ignore[arg-type]
        poll_interval_seconds=0.01,
        max_concurrency=2,
    )
    workflow = store.create_workflow(
        name="training review",
        goal="review today's training",
        steps=[_spec("collect"), _spec("adjust", depends_on=("collect",))],
        session_key="telegram:1",
        channel="telegram",
        chat_id="1",
    )

    task = asyncio.create_task(runtime.run())
    await _eventually(
        lambda: store.require_workflow(workflow.id).status == WorkflowStatus.SUCCEEDED
    )
    await _eventually(lambda: any("已经完成" in message for message in push.messages))

    assert len(loop.prompts) == 2
    assert "collected data" in loop.prompts[1]
    await runtime.aclose()
    await task


@pytest.mark.asyncio
async def test_task_manage_cancel_stops_active_execution_and_preserves_state(
    tmp_path: Path,
):
    store = WorkflowStore(tmp_path / "workflows.db")
    loop = _CancellableLoop()
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: loop,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        poll_interval_seconds=0.01,
    )
    workflow = store.create_workflow(
        name="long task",
        goal="wait until cancelled",
        steps=[_spec("work")],
        session_key="dashboard:test",
        channel="dashboard",
        chat_id="test",
    )
    manage = TaskManageTool(runtime)
    worker = asyncio.create_task(runtime.run())
    await asyncio.wait_for(loop.started.wait(), timeout=1.0)

    result = await manage.execute(
        action="cancel",
        task_id=workflow.id[:8],
        note="user stopped it",
    )

    assert loop.cancelled.is_set()
    assert '"status": "cancelled"' in result
    cancelled = store.require_workflow(workflow.id)
    assert cancelled.status == WorkflowStatus.CANCELLED
    assert cancelled.steps[0].status == StepStatus.CANCELLED
    assert cancelled.steps[0].error == "user stopped it"

    await runtime.aclose()
    await worker


@pytest.mark.asyncio
async def test_runtime_uses_subagent_as_a_tracked_step_executor(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    push = _FakePush()
    executor = _FakeSubagentExecutor()
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=push,  # type: ignore[arg-type]
        subagent_executor=executor,
        poll_interval_seconds=0.01,
    )
    workflow = runtime.create_workflow(
        name="research task",
        goal="produce a durable report",
        steps=[
            _spec("approve", kind=StepKind.APPROVAL),
            _spec(
                "research",
                executor=StepExecutor.SUBAGENT,
                profile="general",
                depends_on=("approve",),
            ),
        ],
        session_key="dashboard:test",
        channel="dashboard",
        chat_id="test",
    )
    store.prepare_human_steps()
    store.approve_step(workflow.id, "approve", approved=True)

    task = asyncio.create_task(runtime.run())
    await _eventually(
        lambda: store.require_workflow(workflow.id).status == WorkflowStatus.SUCCEEDED
    )

    completed = store.require_workflow(workflow.id).steps[1]
    assert completed.executor == StepExecutor.SUBAGENT
    assert completed.profile == "general"
    assert completed.output == "独立调研已经完成"
    assert len(executor.calls) == 1
    assert executor.calls[0]["profile"] == "general"
    assert executor.calls[0]["execution_id"].endswith("-research")
    await runtime.aclose()
    await task


@pytest.mark.asyncio
async def test_task_create_rejects_privileged_subagent_without_approval(
    tmp_path: Path,
):
    store = WorkflowStore(tmp_path / "workflows.db")
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        subagent_executor=_FakeSubagentExecutor(),
    )

    result = await TaskCreateTool(runtime).execute(
        name="unapproved scripting",
        goal="run code",
        steps=[
            {
                "id": "script",
                "title": "Script",
                "description": "Run a script",
                "kind": "agent",
                "executor": "subagent",
                "profile": "scripting",
            }
        ],
    )

    assert result.startswith("错误：")
    assert "subagent profile=scripting" in result
    assert "approval" in result
    assert store.list_workflows() == []
    await runtime.aclose()


@pytest.mark.asyncio
async def test_task_tools_create_and_resume_waiting_task(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        poll_interval_seconds=0.01,
    )
    create = TaskCreateTool(runtime)
    manage = TaskManageTool(runtime)

    result = await create.execute(
        name="check in",
        goal="collect subjective state",
        channel="telegram",
        chat_id="7",
        steps=[
            {
                "id": "feeling",
                "title": "Feeling",
                "description": "How do you feel?",
                "kind": "wait_user",
            }
        ],
    )
    workflow_id = store.list_workflows(session_key="telegram:7")[0].id
    assert workflow_id in result
    store.prepare_human_steps()

    resumed = await manage.execute(
        action="respond",
        task_id=workflow_id[:8],
        step_id="feeling",
        response="good",
        channel="telegram",
        chat_id="7",
    )
    assert '"status": "succeeded"' in resumed
    assert store.require_workflow(workflow_id).steps[0].output == {"response": "good"}
    await runtime.aclose()


@pytest.mark.asyncio
async def test_task_manage_replan_replaces_only_unfinished_steps(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=_FakePush(),  # type: ignore[arg-type]
    )
    workflow = runtime.create_workflow(
        name="coarse plan",
        goal="adapt at a step boundary",
        steps=[_spec("old")],
        session_key="dashboard:test",
        channel="dashboard",
        chat_id="test",
    )

    result = await TaskManageTool(runtime).execute(
        action="replan",
        task_id=workflow.id[:8],
        expected_revision=workflow.revision,
        reason="new evidence requires a different approach",
        steps=[
            {
                "id": "new",
                "title": "New approach",
                "description": "Use the new evidence",
                "kind": "agent",
            }
        ],
    )

    assert '"status": "running"' in result
    assert [step.id for step in store.require_workflow(workflow.id).steps] == ["new"]
    assert "workflow_replanned" in {
        event.event_type for event in store.list_events(workflow.id)
    }
    await runtime.aclose()


@pytest.mark.asyncio
async def test_replan_requires_fresh_approval_for_new_side_effect(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=_FakePush(),  # type: ignore[arg-type]
        tool_registry=_permission_registry(),
    )
    workflow = runtime.create_workflow(
        name="approved old plan",
        goal="do one approved write",
        steps=[
            _spec("old_approval", kind=StepKind.APPROVAL),
            _spec(
                "old_write",
                depends_on=("old_approval",),
                allowed_tools=("write_data",),
            ),
        ],
        session_key="dashboard:test",
        channel="dashboard",
        chat_id="test",
    )
    store.prepare_human_steps()
    approved = store.approve_step(
        workflow.id,
        "old_approval",
        approved=True,
    )

    with pytest.raises(ValueError, match="本次计划中新建的 approval"):
        runtime.replan_workflow(
            workflow.id,
            remaining_steps=[
                _spec(
                    "new_write",
                    depends_on=("old_approval",),
                    allowed_tools=("write_data",),
                )
            ],
            expected_revision=approved.revision,
            reason="change write target",
        )

    await runtime.aclose()


@pytest.mark.asyncio
async def test_task_tools_are_the_unified_public_surface(tmp_path: Path):
    store = WorkflowStore(tmp_path / "workflows.db")
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: None,
        push_tool=_FakePush(),  # type: ignore[arg-type]
    )
    create = TaskCreateTool(runtime)
    manage = TaskManageTool(runtime)

    assert create.name == "task_create"
    assert manage.name == "task_manage"
    result = await create.execute(
        name="background research",
        goal="produce report",
        channel="dashboard",
        chat_id="test",
        auto_start=False,
        steps=[
            {
                "id": "research",
                "title": "Research",
                "description": "Research independently",
                "kind": "agent",
                "executor": "subagent",
                "profile": "research",
            }
        ],
    )

    assert '"created": true' in result
    stored = store.list_workflows(session_key="dashboard:test")[0].steps[0]
    assert stored.executor == StepExecutor.SUBAGENT
    assert stored.profile == "research"
    await runtime.aclose()
