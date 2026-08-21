from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.permissions import PermissionClassifier, PermissionGuardHook, PermissionService
from agent.tool_hooks import HookContext, ToolExecutionRequest


def test_permission_classifier_distinguishes_reads_writes_and_deletes(
    tmp_path: Path,
) -> None:
    classifier = PermissionClassifier(tmp_path)

    read = classifier.classify(
        "shell",
        {"command": "rg TODO .", "description": "查找待办"},
        "external-side-effect",
    )
    project_write = classifier.classify(
        "write_file",
        {"path": str(tmp_path / "notes.txt"), "content": "ok"},
        "write",
    )
    external_write = classifier.classify(
        "edit_file",
        {"path": str(tmp_path.parent / "outside.txt")},
        "write",
    )
    deletion = classifier.classify(
        "shell",
        {"command": 'del "E:\\Desktop\\ThrottleStop.ini"'},
        "external-side-effect",
    )

    assert read.risk == "low"
    assert project_write.risk == "medium"
    assert external_write.risk == "high"
    assert deletion.category == "delete"
    assert deletion.risk == "high"


@pytest.mark.asyncio
async def test_permission_guard_waits_for_one_bound_approval(tmp_path: Path) -> None:
    service = PermissionService()
    queue = service.open("dashboard:chat-1")
    hook = PermissionGuardHook(
        classifier=PermissionClassifier(tmp_path),
        service=service,
        risk_resolver=lambda _name: "external-side-effect",
    )
    request = ToolExecutionRequest(
        call_id="call-1",
        tool_name="shell",
        arguments={"command": "Remove-Item secret.txt", "description": "删除文件"},
        source="passive",
        session_key="dashboard:chat-1",
        channel="dashboard",
        chat_id="chat-1",
        permission_mode="auto_approve",
    )

    task = asyncio.create_task(
        hook.run(
            HookContext(
                event="pre_tool_use",
                request=request,
                current_arguments=dict(request.arguments),
            )
        )
    )
    event = await asyncio.wait_for(queue.get(), timeout=1)

    assert event["type"] == "approval_request"
    assert event["category"] == "delete"
    assert service.resolve(
        session_key="dashboard:chat-1",
        approval_id=event["approval_id"],
        approved=True,
    )
    outcome = await task
    assert outcome.decision == "pass"
    assert "approved" in outcome.reason
    assert service.snapshots("dashboard:chat-1") == []


@pytest.mark.asyncio
async def test_full_access_records_decision_without_waiting(tmp_path: Path) -> None:
    service = PermissionService()
    hook = PermissionGuardHook(
        classifier=PermissionClassifier(tmp_path),
        service=service,
        risk_resolver=lambda _name: "external-side-effect",
    )
    request = ToolExecutionRequest(
        call_id="call-2",
        tool_name="shell",
        arguments={"command": "rm -rf build", "description": "删除构建目录"},
        source="passive",
        session_key="dashboard:chat-2",
        channel="dashboard",
        chat_id="chat-2",
        permission_mode="full_access",
    )

    outcome = await hook.run(
        HookContext(
            event="pre_tool_use",
            request=request,
            current_arguments=dict(request.arguments),
        )
    )

    assert outcome.decision == "pass"
    assert outcome.reason == "permission:full_access:allowed:delete:high"
    assert service.snapshots("dashboard:chat-2") == []
