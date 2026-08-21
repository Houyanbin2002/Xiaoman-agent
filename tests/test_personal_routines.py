from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.tools.personal import (
    PersonalContextTool,
    PersonalRecordTool,
    PersonalRoutineTool,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.workflows.personal import PersonalRoutineService, RoutineKind
from agent.workflows.runtime import WorkflowRuntime
from core.personal.models import PersonalEntityType, RecordStatus
from core.personal.service import PersonalDataService
from core.workflow.models import StepKind
from infra.persistence.personal_store import PersonalStore
from infra.persistence.workflow_store import WorkflowStore


class _Push:
    async def execute(self, **_: Any) -> str:
        return "ok"


class _PersonalRecordStub(Tool):
    name = "personal_record"
    description = "Store a personal record"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs)


def _routines(tmp_path: Path) -> tuple[WorkflowRuntime, PersonalRoutineService]:
    tools = ToolRegistry()
    tools.register(_PersonalRecordStub(), risk="write")
    runtime = WorkflowRuntime(
        store=WorkflowStore(tmp_path / "workflows.db"),
        agent_loop_provider=lambda: None,
        push_tool=_Push(),  # type: ignore[arg-type]
        tool_registry=tools,
    )
    return runtime, PersonalRoutineService(runtime)


def test_morning_routine_has_review_gate_and_is_deduplicated(tmp_path: Path):
    runtime, routines = _routines(tmp_path)
    first, created = routines.create(
        RoutineKind.MORNING_BRIEF,
        session_key="dashboard:owner",
        channel="dashboard",
        chat_id="owner",
        local_date="2026-07-10",
    )
    same, created_again = routines.create(
        RoutineKind.MORNING_BRIEF,
        session_key="dashboard:owner",
        channel="dashboard",
        chat_id="owner",
        local_date="2026-07-10",
    )

    assert created is True
    assert created_again is False
    assert same.id == first.id
    assert [step.kind for step in first.steps] == [
        StepKind.AGENT,
        StepKind.AGENT,
        StepKind.WAIT_USER,
        StepKind.APPROVAL,
        StepKind.AGENT,
    ]
    assert first.steps[-1].depends_on == (
        "draft_plan",
        "user_adjustment",
        "approve_plan",
    )
    assert first.steps[-1].allowed_tools == ("personal_record",)
    runtime.store.close()


def test_commitment_routine_requires_candidate_and_approval(tmp_path: Path):
    runtime, routines = _routines(tmp_path)
    with pytest.raises(ValueError, match="requires candidate"):
        routines.create(
            RoutineKind.CAPTURE_COMMITMENT,
            session_key="dashboard:owner",
            channel="dashboard",
            chat_id="owner",
        )
    workflow, _ = routines.create(
        RoutineKind.CAPTURE_COMMITMENT,
        session_key="dashboard:owner",
        channel="dashboard",
        chat_id="owner",
        candidate="下周五前完成项目报告",
    )
    assert workflow.steps[1].kind == StepKind.APPROVAL
    assert workflow.name == "记录一项待办"
    assert workflow.steps[1].title == "确认待办"
    assert workflow.context["timezone"] == "Asia/Shanghai"
    assert workflow.steps[2].depends_on == ("normalize", "approve")
    assert workflow.steps[2].allowed_tools == ("personal_record",)
    runtime.store.close()


@pytest.mark.asyncio
async def test_personal_tools_read_write_and_require_confirmation(tmp_path: Path):
    service = PersonalDataService(PersonalStore(tmp_path / "personal.db"))
    write_tool = PersonalRecordTool(service)
    read_tool = PersonalContextTool(service)
    created_payload = json.loads(
        await write_tool.execute(
            action="create",
            entity_type="commitment",
            title="Finish report",
            summary="Finish the weekly report",
            data={"state": "open"},
            channel="dashboard",
            chat_id="owner",
        )
    )
    record_id = created_payload["record"]["id"]
    context = json.loads(await read_tool.execute(entity_type="commitment"))
    assert context["count"] == 1
    assert context["records"][0]["entity_type"] == PersonalEntityType.COMMITMENT

    restricted_payload = json.loads(
        await write_tool.execute(
            action="create",
            entity_type="profile",
            title="Private account detail",
            summary="Account metadata",
            data={"provider": "example"},
            data_category="account",
            channel="dashboard",
            chat_id="owner",
        )
    )
    assert restricted_payload["record"]["access_policy"] == "owner_only"
    restricted_context = json.loads(await read_tool.execute(entity_type="profile"))
    assert restricted_context["count"] == 0
    assert restricted_context["restricted_count"] == 1

    proposed_follow_up = json.loads(
        await write_tool.execute(
            action="create",
            entity_type="proactive_intent",
            title="Inferred follow-up",
            summary="Assistant inferred this might be useful",
            data={"enabled": True, "status": "active"},
        )
    )
    assert proposed_follow_up["record"]["data"]["enabled"] is False
    assert proposed_follow_up["record"]["data"]["status"] == "proposed"

    denied = await write_tool.execute(action="forget", record_id=record_id)
    assert denied.startswith("错误：")
    forgotten = json.loads(
        await write_tool.execute(
            action="forget", record_id=record_id, user_confirmed=True
        )
    )
    assert forgotten["record"]["status"] == RecordStatus.FORGOTTEN
    service.close()


@pytest.mark.asyncio
async def test_personal_routine_tool_returns_task_id(tmp_path: Path):
    runtime, routines = _routines(tmp_path)
    tool = PersonalRoutineTool(routines)
    payload = json.loads(
        await tool.execute(
            routine="evening_review",
            local_date="2026-07-10",
            channel="dashboard",
            chat_id="owner",
        )
    )
    assert payload["created"] is True
    assert payload["task_id"]
    runtime.store.close()
