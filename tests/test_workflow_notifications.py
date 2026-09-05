from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.tools.message_push import MessagePushTool
from agent.workflows.runtime import WorkflowRuntime
from bootstrap.dashboard_management.routes.workflows import register_workflow_routes
from core.workflow.models import StepKind, StepSpec, WorkflowStatus
from infra.persistence.workflow_store import WorkflowStore
import infra.persistence.workflow_notifications as notification_module


def create(store, *, waiting=False, name="report"):
    workflow = store.create_workflow(
        name=name,
        goal="生成报告",
        session_key="test",
        channel="test",
        chat_id="recipient",
        steps=[
            StepSpec(
                id="work",
                title="报告",
                description="生成报告",
                kind=StepKind.WAIT_USER if waiting else StepKind.AGENT,
            )
        ],
    )
    if waiting:
        store.prepare_human_steps()
    else:
        store.claim_workflow_steps(workflow.id)
        store.complete_step(workflow.id, "work", output="已生成报告")
    return store.require_workflow(workflow.id)


def enqueue(store, workflow):
    return store.notifications.ensure(
        workflow_id=workflow.id,
        step_id="",
        source_version=str(workflow.revision),
        target_status=workflow.status.value,
        channel=workflow.channel,
        chat_id=workflow.chat_id,
        message="已保存的报告通知",
    )


def worker(store, push):
    return WorkflowRuntime(
        store=store, agent_loop_provider=lambda: None, push_tool=push
    )


@pytest.mark.asyncio
async def test_accepted_persists_and_does_not_resend_after_reopen(tmp_path):
    path = tmp_path / "state.db"
    store = WorkflowStore(path)
    workflow = create(store)
    sent = []

    async def sender(*args):
        sent.append(args)

    push = MessagePushTool()
    push.register_channel("test", text=sender)
    await worker(store, push)._deliver_terminal_notifications()
    first = store.notifications.list_for_workflow(workflow.id)[0]
    assert first["status"] == "accepted"
    assert first["is_current"]
    assert store.require_workflow(workflow.id).notified_status == "succeeded"
    assert store.require_workflow(workflow.id).revision == workflow.revision
    store.close()
    reopened = WorkflowStore(path)
    await worker(reopened, push)._deliver_terminal_notifications()
    assert len(sent) == 1
    reopened.close()


def test_two_database_connections_claim_once_and_reject_stale_result(tmp_path):
    path = tmp_path / "state.db"
    first, second = WorkflowStore(path), WorkflowStore(path)
    workflow = create(first)
    notification_id = enqueue(first, workflow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(
            pool.map(
                lambda store: store.notifications.claim(notification_id),
                [first, second],
            )
        )
    assert sum(item is not None for item in claims) == 1
    attempt = next(item for item in claims if item)
    first.notifications.finish(
        notification_id, attempt["version"] - 1, status="accepted"
    )
    assert first.require_workflow(workflow.id).notified_status is None
    second.notifications.finish(notification_id, attempt["version"], status="accepted")
    assert first.require_workflow(workflow.id).notified_status == "succeeded"
    first.close()
    second.close()


@pytest.mark.asyncio
async def test_crash_after_remote_send_recovers_unknown_not_replay(
    tmp_path, monkeypatch
):
    now = [1000.0]
    monkeypatch.setattr(
        notification_module, "time", SimpleNamespace(time=lambda: now[0])
    )
    path = tmp_path / "state.db"
    store = WorkflowStore(path)
    workflow = create(store)
    notification_id = enqueue(store, workflow)
    attempt = store.notifications.claim(notification_id)
    assert attempt
    store.close()  # Crash window: remote may have accepted, finish never ran.
    store = WorkflowStore(path)
    store.notifications.recover_expired()
    assert store.notifications.list_for_workflow(workflow.id)[0]["status"] == "sending"
    now[0] += 301
    await worker(store, MessagePushTool())._deliver_terminal_notifications()
    row = store.notifications.list_for_workflow(workflow.id)[0]
    assert row["status"] == "unknown"
    assert row["attempts"] == 1
    assert store.notifications.due_ids() == []
    store.notifications.finish(notification_id, attempt["version"], status="accepted")
    assert store.require_workflow(workflow.id).notified_status is None
    store.close()


@pytest.mark.asyncio
async def test_preflight_failure_backoff_is_bounded(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(
        notification_module, "time", SimpleNamespace(time=lambda: now[0])
    )
    store = WorkflowStore(tmp_path / "state.db")
    workflow = create(store)
    runtime = worker(store, MessagePushTool())  # Known unsent: no registered channel.
    await runtime._deliver_terminal_notifications()
    for expected, advance in [(1, 0), (1, 29), (2, 1), (2, 119), (3, 1), (3, 10000)]:
        now[0] += advance
        await runtime._deliver_terminal_notifications()
        row = store.notifications.list_for_workflow(workflow.id)[0]
        assert row["attempts"] == expected
        assert row["status"] == "failed"
    assert row["next_attempt_at"] is None
    assert store.require_workflow(workflow.id).status == WorkflowStatus.SUCCEEDED
    assert store.require_workflow(workflow.id).steps[0].attempt_count == 1
    store.close()


@pytest.mark.asyncio
async def test_manual_notification_retry_does_not_execute_task(tmp_path):
    store = WorkflowStore(tmp_path / "state.db")
    workflow = create(store)
    push = MessagePushTool()
    runtime = worker(store, push)
    await runtime._deliver_terminal_notifications()
    row = store.notifications.list_for_workflow(workflow.id)[0]
    before = store.require_workflow(workflow.id).steps[0].to_dict()
    sent = []

    async def sender(*args):
        sent.append(args)

    push.register_channel("test", text=sender)
    store.notifications.resolve(
        workflow.id, row["id"], expected_version=row["version"], action="retry"
    )
    with pytest.raises(ValueError, match="状态已变化"):
        store.notifications.resolve(
            workflow.id, row["id"], expected_version=row["version"], action="retry"
        )
    await runtime._deliver_terminal_notifications()
    assert sent == [("recipient", row["message"])]
    assert store.require_workflow(workflow.id).steps[0].to_dict() == before
    assert len(store.notifications.list_for_workflow(workflow.id)) == 1
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_sender_error_or_cancellation_is_unknown(tmp_path, cancel):
    store = WorkflowStore(tmp_path / "state.db")
    workflow = create(store)
    push = MessagePushTool()
    entered = asyncio.Event()

    async def sender(*args):
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise ConnectionError("response lost after send")

    push.register_channel("test", text=sender)
    runtime = worker(store, push)
    task = asyncio.create_task(runtime._deliver_terminal_notifications())
    await entered.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    await runtime._deliver_terminal_notifications()
    row = store.notifications.list_for_workflow(workflow.id)[0]
    assert row["status"] == "unknown"
    assert row["attempts"] == 1
    with pytest.raises(ValueError, match="重复通知风险"):
        store.notifications.resolve(
            workflow.id, row["id"], expected_version=row["version"], action="retry"
        )
    store.notifications.resolve(
        workflow.id, row["id"], expected_version=row["version"], action="received"
    )
    assert (
        store.notifications.list_for_workflow(workflow.id)[0]["status"] == "confirmed"
    )
    assert store.require_workflow(workflow.id).notified_status == "succeeded"
    store.close()


@pytest.mark.asyncio
async def test_waiting_notification_checks_cancel_after_chat_lane_queue(tmp_path):
    store = WorkflowStore(tmp_path / "state.db")
    workflow = create(store, waiting=True)
    sent = []

    class Lane:
        async def run_non_passive(self, channel, chat_id, callback):
            store.cancel_workflow(workflow.id)
            return await callback()

    async def sender(*args):
        sent.append(args)

    push = MessagePushTool(chat_lane=Lane())
    push.register_channel("test", text=sender)
    runtime = worker(store, push)
    await runtime._deliver_waiting_prompts()
    await runtime._deliver_terminal_notifications()
    assert sent == []
    row = store.notifications.list_for_workflow(workflow.id)[0]
    assert not row["is_current"]
    with pytest.raises(ValueError):
        store.notifications.resolve(
            workflow.id, row["id"], expected_version=row["version"], action="retry"
        )
    store.close()


@pytest.mark.asyncio
async def test_waiting_receipt_is_atomic_and_preserves_source_version(tmp_path):
    store = WorkflowStore(tmp_path / "state.db")
    workflow = create(store, waiting=True)
    push = MessagePushTool()

    async def sender(*args):
        pass

    push.register_channel("test", text=sender)
    runtime = worker(store, push)
    await runtime._deliver_waiting_prompts()
    await runtime._deliver_waiting_prompts()
    row = store.notifications.list_for_workflow(workflow.id)[0]
    assert row["status"] == "accepted" and row["is_current"]
    assert store.require_workflow(workflow.id).steps[0].notified_at
    assert (
        store.require_workflow(workflow.id).steps[0].updated_at
        == workflow.steps[0].updated_at
    )
    store.close()


def test_notification_api_unknown_requires_consent_and_cross_task_is_rejected(tmp_path):
    store = WorkflowStore(tmp_path / "state.db")
    workflow = create(store)
    other = create(store, name="other")
    notification_id = enqueue(store, workflow)
    attempt = store.notifications.claim(notification_id)
    store.notifications.finish(notification_id, attempt["version"], status="unknown")
    runtime = worker(store, MessagePushTool())
    app = FastAPI()
    register_workflow_routes(app, SimpleNamespace(workflow_runtime=runtime))
    with TestClient(app) as client:
        row = client.get(f"/api/dashboard/control/tasks/{workflow.id}").json()[
            "notifications"
        ][0]
        body = {"expected_version": row["version"], "action": "retry"}
        url = f"/api/dashboard/control/tasks/{workflow.id}/notifications/{notification_id}"
        assert client.post(url, json=body).status_code == 409
        assert (
            client.post(url.replace(workflow.id, other.id), json=body).status_code
            == 409
        )
        body["confirm_unknown"] = True
        assert client.post(url, json=body).status_code == 200
        assert client.post(url, json=body).status_code == 409
        rows = client.get("/api/dashboard/control/tasks").json()
        assert (
            next(item for item in rows if item["id"] == workflow.id)["notifications"][
                0
            ]["status"]
            == "pending"
        )
    assert runtime._wake.is_set()
    store.close()


@pytest.mark.asyncio
async def test_exhausted_notifications_do_not_starve_new_workflows(tmp_path):
    store = WorkflowStore(tmp_path / "state.db")
    for index in range(25):
        workflow = create(store, name=str(index))
        notification_id = enqueue(store, workflow)
        attempt = store.notifications.claim(notification_id)
        store.notifications.finish(
            notification_id, attempt["version"], status="unknown"
        )
    latest = create(store, name="new")
    sent = []

    async def sender(*args):
        sent.append(args)

    push = MessagePushTool()
    push.register_channel("test", text=sender)
    await worker(store, push)._deliver_terminal_notifications()
    assert len(sent) == 1
    assert store.require_workflow(latest.id).notified_status == "succeeded"
    store.close()


@pytest.mark.asyncio
async def test_slow_notification_does_not_block_new_workflow_or_shutdown(tmp_path):
    store = WorkflowStore(tmp_path / "state.db")
    done = create(store)
    entered, executed = asyncio.Event(), asyncio.Event()

    async def sender(*args):
        entered.set()
        await asyncio.Event().wait()

    class Loop:
        async def process_direct(self, *args, **kwargs):
            executed.set()
            return "新任务已完成"

    push = MessagePushTool()
    push.register_channel("test", text=sender)
    runtime = WorkflowRuntime(
        store=store, agent_loop_provider=lambda: Loop(), push_tool=push
    )
    runner = asyncio.create_task(runtime.run())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        store.create_workflow(
            name="new",
            goal="不应等待慢通知",
            session_key="test",
            channel="test",
            chat_id="recipient",
            steps=[
                StepSpec(id="new", title="new", description="new", kind=StepKind.AGENT)
            ],
        )
        runtime.wake()
        await asyncio.wait_for(executed.wait(), 2)
    finally:
        await asyncio.wait_for(runtime.aclose(), 2)
        await runner
    reopened = WorkflowStore(tmp_path / "state.db")
    assert reopened.notifications.list_for_workflow(done.id)[0]["status"] == "unknown"
    reopened.close()
