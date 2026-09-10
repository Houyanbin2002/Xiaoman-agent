from __future__ import annotations

import json
from typing import Any, cast

import pytest

from agent.runtime.delegation import (
    CONTEXT_KEY,
    DelegationBudgetExceeded,
    DelegationLedger,
    DelegationScope,
    current_scope,
    new_scope,
)
from agent.runtime.execution_guard import ExecutionGuardConfig
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.runtime.model_step import run_model_step
from agent.core.passive_turn import AgentExecutionKernel
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.ports import LLMConfig, LLMServices
from agent.subagent import SubAgent
from agent.tools.registry import ToolRegistry
from agent.tools.workflow import TaskCreateTool
from agent.workflows.runtime import WorkflowRuntime
from core.llm import LLMResponse, ToolCall
from core.llm.request_budget import before_transport_retry
from core.workflow.models import StepExecutor, StepKind, StepSpec, StepStatus
from infra.persistence.workflow_store import WorkflowStore
from infra.providers.llm_provider import LLMProvider as OpenAIProvider


def step(name="work", *, child=False, dependencies=()):
    return StepSpec(
        id=name,
        title=name,
        description="bounded test",
        kind=StepKind.AGENT,
        executor=StepExecutor.SUBAGENT if child else StepExecutor.AGENT,
        depends_on=dependencies,
    )


class Push:
    async def execute(self, **kwargs):
        return "ok"


def runtime(tmp_path, *, allowed=False):
    return WorkflowRuntime(
        store=WorkflowStore(tmp_path / "index.db"),
        graph_runtime=LangGraphRuntime(tmp_path / "workflow.db"),
        agent_loop_provider=lambda: None,
        push_tool=Push(),
        delegation_guard=ExecutionGuardConfig(autonomous_delegation=allowed),
    )


def create(rt, steps, **kwargs):
    return rt.create_workflow(
        name="test",
        goal="test",
        steps=steps,
        session_key="test:1",
        channel="test",
        chat_id="1",
        **kwargs,
    )


def test_ledger_atomic_admission_and_restart(tmp_path):
    ledger = DelegationLedger(tmp_path / "budget.db")
    ledger.create("parent", limit=1000, reserve=100, max_children=2)
    ledger.reserve("parent", 200, child=False)
    ledger.claim_child("parent", "a")
    ledger.claim_child("parent", "a")  # retry does not consume another child slot
    ledger.claim_child("parent", "b")
    with pytest.raises(DelegationBudgetExceeded):
        ledger.claim_child("parent", "c")
    ledger.reserve("parent", 400, child=True)
    other = DelegationLedger(tmp_path / "budget.db")
    with pytest.raises(DelegationBudgetExceeded):
        other.reserve("parent", 400, child=True)
    other.settle("parent", 400, 100)
    assert ledger.used("parent") == 300
    other.create("parent", limit=99999, reserve=0, max_children=99)
    with pytest.raises(DelegationBudgetExceeded):
        other.reserve("parent", 601, child=True)
    ledger.reserve("parent", 1200, child=False)  # never hard-stop the main reply


@pytest.mark.asyncio
async def test_default_off_allows_sequential_workflow_not_parallel_or_forgery(tmp_path):
    rt = runtime(tmp_path)
    try:
        create(rt, [step("read"), step("summarize", dependencies=("read",))])
        with pytest.raises(ValueError, match="多 Agent"):
            create(rt, [step(child=True)])
        with pytest.raises(ValueError, match="并行"):
            create(rt, [step("a"), step("b")])
        result = await TaskCreateTool(rt).execute(
            name="forged",
            goal="test",
            context={CONTEXT_KEY: {"allowed": True, "budget_id": "fake"}},
            steps=[
                {
                    "id": "a",
                    "title": "test",
                    "description": "test",
                    "executor": "subagent",
                }
            ],
        )
        assert "错误" in result
        wf = rt.store.list_workflows()[0]
        with pytest.raises(ValueError, match="多 Agent"):
            rt.replan_workflow(
                wf.id,
                remaining_steps=[step(child=True)],
                expected_revision=wf.revision,
                reason="bypass",
            )
    finally:
        await rt.aclose()


@pytest.mark.asyncio
async def test_turn_policy_persisted_without_recursive_authority(tmp_path):
    rt = runtime(tmp_path)
    ledger = DelegationLedger(tmp_path / "delegation-budget.db")
    scope = new_scope(ledger, ExecutionGuardConfig(), allowed=True)
    token = current_scope.set(scope)
    try:
        result = await TaskCreateTool(rt).execute(
            name="authorized",
            goal="test",
            steps=[
                {
                    "id": "a",
                    "title": "test",
                    "description": "test",
                    "executor": "subagent",
                }
            ],
        )
        wf = rt.store.require_workflow(json.loads(result)["task_id"])
        assert wf.context[CONTEXT_KEY] == scope.metadata()
        child_token = current_scope.set(DelegationScope(ledger, scope.key, True, True))
        try:
            result = await TaskCreateTool(rt).execute(
                name="recursive", goal="test", steps=[]
            )
            assert "递归" in result
        finally:
            current_scope.reset(child_token)
    finally:
        current_scope.reset(token)
        await rt.aclose()


@pytest.mark.asyncio
async def test_usage_and_retry_reservation_are_shared(tmp_path):
    ledger = DelegationLedger(tmp_path / "budget.db")
    scope = new_scope(ledger, ExecutionGuardConfig())
    token = current_scope.set(scope)

    class Provider:
        async def chat(self, **kwargs: Any):
            retry = before_transport_retry.get()
            assert retry is not None
            retry()  # transport retried once with unknown billing
            return LLMResponse(
                content="ok", input_tokens=20, output_tokens=10, total_tokens=30
            )

    messages = [{"role": "user", "content": "test"}]
    try:
        await run_model_step(
            cast(Any, Provider()),
            messages=messages,
            tools=[],
            model="mock",
            max_tokens=100,
            source="passive",
            iteration=1,
        )
        estimate = 100 + len(
            json.dumps([messages, []], ensure_ascii=False).encode("utf-8")
        )
        assert ledger.used(scope.key) == estimate + 30
        assert before_transport_retry.get() is None
    finally:
        current_scope.reset(token)


@pytest.mark.asyncio
async def test_exhausted_workflow_is_not_retried_or_marked_success(tmp_path):
    rt = runtime(tmp_path, allowed=True)
    try:
        wf = create(rt, [step(child=True)])
        key = wf.context[CONTEXT_KEY]["budget_id"]
        rt._graph_runtime.delegation_ledger.reserve(key, 500_000, child=False)
        claimed = rt.store.claim_workflow_steps(wf.id, limit=1)[0][1]
        await rt._execute_step_bound(wf, claimed)
        result = rt.store.require_workflow(wf.id).steps[0]
        assert result.status == StepStatus.FAILED
        assert result.attempt_count == 1
        assert result.next_run_at is None
        assert "预算" in result.error
    finally:
        await rt.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "low", "max"])
async def test_real_kernel_workflow_child_chain_uses_one_budget(tmp_path, effort):
    rt = runtime(tmp_path)
    parent_runtime = LangGraphRuntime(tmp_path / "parent.db")
    child_runtime = LangGraphRuntime(tmp_path / "child.db")
    child_calls = []

    class ChildProvider:
        async def chat(self, **kwargs):
            child_calls.append(kwargs)
            return LLMResponse(content="child done", total_tokens=100)

    child = SubAgent(
        provider=cast(Any, ChildProvider()),
        model="qwen3.7-plus",
        tools=[],
        graph_runtime=child_runtime,
    )

    class Executor:
        async def execute(
            self, *, task, label, profile, execution_id, reasoning_effort
        ):
            return await child.run(
                task, execution_id=execution_id, reasoning_effort=reasoning_effort
            )

    rt.subagent_executor = Executor()

    class ParentProvider:
        calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    total_tokens=100,
                    tool_calls=[
                        ToolCall(
                            id="create",
                            name="task_create",
                            arguments={
                                "name": "test",
                                "goal": "test",
                                "steps": [
                                    {
                                        "id": "child",
                                        "title": "test",
                                        "description": "test",
                                        "kind": "agent",
                                        "executor": "subagent",
                                    }
                                ],
                            },
                        )
                    ],
                )
            return LLMResponse(content="created", total_tokens=100)

    provider = ParentProvider()
    registry = ToolRegistry()
    registry.register(TaskCreateTool(rt))
    kernel = AgentExecutionKernel(
        llm=LLMServices(
            provider=cast(Any, provider), light_provider=cast(Any, provider)
        ),
        llm_config=LLMConfig(model="deepseek-v4-pro-0813", max_iterations=3),
        tools=registry,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=0,
        graph_runtime=parent_runtime,
    )
    try:
        result = await kernel.run(
            [{"role": "user", "content": "test"}],
            autonomous_delegation=True,
            reasoning_effort=effort,
        )
        wf = rt.store.list_workflows()[0]
        key = result.metadata["delegation"]["budget_id"]
        assert wf.context[CONTEXT_KEY]["budget_id"] == key
        assert parent_runtime.delegation_ledger.used(key) == 200
        claimed = rt.store.claim_workflow_steps(wf.id, limit=1)[0][1]
        await rt._execute_step_bound(wf, claimed)
        final = rt.store.require_workflow(wf.id).steps[0]
        assert final.status == StepStatus.SUCCEEDED, final.error
        assert parent_runtime.delegation_ledger.used(key) == 300
        assert child_calls[0]["disable_thinking"] is (effort == "none")
        if effort == "low":
            assert child_calls[0]["extra_body"]["thinking_budget"] == 4096
        assert current_scope.get() is None
    finally:
        await rt.aclose()
        await parent_runtime.aclose()
        await child_runtime.aclose()


@pytest.mark.asyncio
async def test_transport_retry_obeys_budget_and_sdk_does_not_retry(monkeypatch):
    from types import SimpleNamespace

    requests = []
    constructors = []

    async def failing_create(**kwargs):
        requests.append(kwargs)
        raise TimeoutError("synthetic timeout")

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=failing_create))
    )

    def make_client(**kwargs):
        constructors.append(kwargs)
        return fake

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr("infra.providers.llm_provider.AsyncOpenAI", make_client)
    monkeypatch.setattr("infra.providers.llm_provider.asyncio.sleep", no_sleep)
    provider = OpenAIProvider(
        api_key="test-only", base_url="https://example.invalid", max_retries=2
    )
    provider.reconfigure(api_key="test-only", base_url="https://example.invalid")
    assert all(item["max_retries"] == 0 for item in constructors)

    def deny_retry():
        raise DelegationBudgetExceeded("budget exhausted")

    token = before_transport_retry.set(deny_retry)
    try:
        with pytest.raises(DelegationBudgetExceeded):
            await provider.chat(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
                model="test",
                max_tokens=10,
            )
        assert len(requests) == 1
    finally:
        before_transport_retry.reset(token)
