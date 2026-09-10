from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from agent.config import load_config
from agent.subagent import SubAgent
from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.ports import LLMConfig, LLMServices
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.runtime.reasoning_policy import (
    ReasoningPolicy,
    ReasoningPolicyConfig,
    normalize_reasoning_effort,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from core.llm import LLMResponse, ToolCall


class _Provider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(content="done")


def _reasoner(
    provider: _Provider,
    *,
    model: str = "deepseek-v4-pro-0813",
    config: ReasoningPolicyConfig | None = None,
) -> DefaultReasoner:
    return DefaultReasoner(
        llm=LLMServices(
            provider=cast(Any, provider),
            light_provider=cast(Any, provider),
        ),
        llm_config=LLMConfig(
            model=model,
            max_iterations=2,
            reasoning=config or ReasoningPolicyConfig(default_effort="medium"),
        ),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=0,
    )


def test_policy_uses_config_default_without_task_classification() -> None:
    policy = ReasoningPolicy(ReasoningPolicyConfig(default_effort="low"))

    decision = policy.decide("", source="passive", model="deepseek-v4-pro-0813")

    assert decision.requested_effort == "low"
    assert decision.effective_effort == "high"
    assert decision.reason == "global_default"


def test_explicit_turn_selection_wins_over_global_default() -> None:
    decision = ReasoningPolicy(ReasoningPolicyConfig(default_effort="medium")).decide(
        "max",
        source="passive",
        session_key="dashboard:1",
        model="deepseek-v4-pro-0813",
    )

    assert decision.enabled is True
    assert decision.requested_effort == "max"
    assert decision.effective_effort == "max"
    assert decision.reason == "turn_or_session_override"


def test_none_explicitly_disables_thinking() -> None:
    decision = ReasoningPolicy().decide(
        "none",
        source="passive",
        model="deepseek-v4-pro-0813",
    )

    assert decision.enabled is False
    assert decision.request_extra_body("deepseek-v4-pro-0813") == {}


@pytest.mark.parametrize(
    ("requested", "effective"),
    [
        ("minimal", "high"),
        ("low", "high"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
    ],
)
def test_deepseek_v4_clamps_to_supported_efforts(
    requested: str,
    effective: str,
) -> None:
    decision = ReasoningPolicy().decide(
        requested,
        source="passive",
        model="deepseek-v4-pro-0813",
    )

    assert decision.effective_effort == effective
    assert decision.request_extra_body("deepseek-v4-pro-0813") == {
        "enable_thinking": True,
        "reasoning_effort": effective,
    }


def test_qwen_hybrid_maps_effort_to_thinking_budget() -> None:
    decision = ReasoningPolicy().decide(
        "high",
        source="passive",
        model="qwen3.7-plus",
    )

    assert decision.request_extra_body("qwen3.7-plus") == {
        "enable_thinking": True,
        "thinking_budget": 16_384,
    }


def test_executor_override_wins_over_inherited_parent_effort() -> None:
    policy = ReasoningPolicy(
        ReasoningPolicyConfig(
            default_effort="medium",
            subagent_effort="low",
            workflow_effort="high",
        )
    )

    subagent = policy.decide(
        "max",
        source="subagent",
        session_key="subagent:1",
        model="qwen3.7-plus",
    )
    workflow = policy.decide(
        "low",
        source="passive",
        session_key="workflow:1",
        model="qwen3.7-plus",
    )

    assert subagent.requested_effort == "low"
    assert subagent.reason == "subagent_override"
    assert workflow.requested_effort == "high"
    assert workflow.reason == "workflow_override"


@pytest.mark.parametrize(
    "effort",
    ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
)
def test_reasoning_policy_accepts_supported_efforts(effort: str) -> None:
    assert normalize_reasoning_effort(effort) == effort


def test_reasoning_policy_rejects_unknown_effort() -> None:
    with pytest.raises(ValueError, match="思考等级必须是"):
        normalize_reasoning_effort("adaptive")


def test_agent_kernel_forwards_selected_reasoning_to_every_model_step() -> None:
    provider = _Provider()
    reasoner = _reasoner(provider)

    result = asyncio.run(
        reasoner.run(
            [{"role": "user", "content": "完成任务"}],
            request_text="完成任务",
            reasoning_effort="max",
            tool_event_session_key="dashboard:1",
        )
    )

    assert provider.calls[0]["disable_thinking"] is False
    assert provider.calls[0]["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "max",
    }
    assert result.metadata["reasoning_policy"] == {
        "enabled": True,
        "requested_effort": "max",
        "effective_effort": "max",
        "reason": "turn_or_session_override",
    }


@pytest.mark.parametrize("iterations", [1, 2])
def test_agent_kernel_keeps_same_reasoning_after_tool_round(iterations: int) -> None:
    class _TwoStepProvider(_Provider):
        async def chat(self, **kwargs: Any) -> LLMResponse:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="probe", arguments={})],
                )
            return LLMResponse(content="done")

    class _ProbeTool(Tool):
        name = "probe"
        description = "Probe"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **_kwargs: Any) -> str:
            return "ok"

    provider = _TwoStepProvider()
    reasoner = _reasoner(provider)
    reasoner._tools.register(cast(Any, _ProbeTool()))
    reasoner._llm_config.max_iterations = iterations

    asyncio.run(
        reasoner.run(
            [{"role": "user", "content": "先调用工具再回答"}],
            request_text="先调用工具再回答",
            reasoning_effort="max",
            tool_event_session_key="dashboard:tool-round",
        )
    )

    assert len(provider.calls) == 2
    assert all(
        call["extra_body"]["reasoning_effort"] == "max" for call in provider.calls
    )


def test_agent_kernel_explicitly_disables_selected_none() -> None:
    provider = _Provider()
    reasoner = _reasoner(provider)

    result = asyncio.run(
        reasoner.run(
            [{"role": "user", "content": "你好"}],
            request_text="你好",
            reasoning_effort="none",
            tool_event_session_key="dashboard:1",
        )
    )

    assert provider.calls[0]["disable_thinking"] is True
    assert provider.calls[0]["extra_body"] == {}
    assert result.metadata["reasoning_policy"]["requested_effort"] == "none"


def test_load_config_reads_explicit_reasoning_defaults(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
provider = "openai"
model = "test-model"

[agent.reasoning]
default_effort = "high"
subagent_effort = "low"
workflow_effort = "max"
""".strip(),
        encoding="utf-8",
    )

    reasoning = load_config(config_path).reasoning

    assert reasoning.default_effort == "high"
    assert reasoning.subagent_effort == "low"
    assert reasoning.workflow_effort == "max"


@pytest.mark.asyncio
async def test_checkpoint_resume_keeps_original_reasoning_selection(tmp_path) -> None:
    class _InterruptibleProvider(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.block = True

        async def chat(self, **kwargs: Any) -> LLMResponse:
            self.calls.append(kwargs)
            if self.block:
                self.started.set()
                await asyncio.Future()
            return LLMResponse(content="resumed")

    provider = _InterruptibleProvider()
    runtime = LangGraphRuntime(tmp_path / "reasoning-checkpoints.db")
    first = SubAgent(
        provider=cast(Any, provider),
        model="deepseek-v4-pro-0813",
        tools=[],
        graph_runtime=runtime,
    )
    interrupted = asyncio.create_task(
        first.run(
            "durable reasoning task",
            execution_id="reasoning-resume-1",
            reasoning_effort="max",
        )
    )
    await asyncio.wait_for(provider.started.wait(), timeout=1.0)
    interrupted.cancel()
    with pytest.raises(asyncio.CancelledError):
        await interrupted

    provider.block = False
    graph = await first._kernel._graph_executor._compiled_graph()
    graph_config = {"configurable": {"thread_id": "agent:subagent:reasoning-resume-1"}}
    snapshot = await graph.aget_state(graph_config)
    saved_budget_id = snapshot.values["delegation"]["budget_id"]
    used_before_restart = runtime.delegation_ledger.used(saved_budget_id)
    assert used_before_restart > 0  # cancelled requests retain a reservation
    await runtime.aclose()
    runtime = LangGraphRuntime(tmp_path / "reasoning-checkpoints.db")
    restored = SubAgent(
        provider=cast(Any, provider),
        model="deepseek-v4-pro-0813",
        tools=[],
        graph_runtime=runtime,
    )
    result = await restored.run(
        "durable reasoning task",
        execution_id="reasoning-resume-1",
        reasoning_effort="none",
    )

    assert result == "resumed"
    restored_graph = await restored._kernel._graph_executor._compiled_graph()
    restored_snapshot = await restored_graph.aget_state(graph_config)
    assert restored_snapshot.values["delegation"]["budget_id"] == saved_budget_id
    assert runtime.delegation_ledger.used(saved_budget_id) > used_before_restart
    assert len(provider.calls) == 2
    assert all(call["disable_thinking"] is False for call in provider.calls)
    assert all(
        call["extra_body"]["reasoning_effort"] == "max" for call in provider.calls
    )
    await runtime.aclose()


@pytest.mark.parametrize("effort", ["none", "low", "max"])
@pytest.mark.asyncio
async def test_tool_creation_inherits_requested_not_provider_mapped_effort(effort):
    captured = {}

    class Creation(Tool):
        name = "task_create"
        description = "test trusted context"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            captured.update(kwargs)
            return "ok"

    class Provider(_Provider):
        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="create-1",
                            name="task_create",
                            arguments={"current_reasoning_effort": "high"},
                        )
                    ],
                )
            return LLMResponse(content="done")

    provider = Provider()
    reasoner = _reasoner(provider)
    reasoner._tools.register(Creation())
    try:
        result = await reasoner.run(
            [{"role": "user", "content": "test"}], reasoning_effort=effort
        )
        assert captured["current_reasoning_effort"] == effort
        assert result.metadata["delegation"]["allowed"] is False
        assert len(provider.calls) == 2  # no extra complexity-classifier call
    finally:
        await reasoner._graph_executor._runtime.aclose()


@pytest.mark.parametrize(
    "legacy", ["enable_thinking = false", 'thinking = { type = "disabled" }']
)
def test_legacy_explicit_disable_survives_new_policy(tmp_path, legacy):
    path = tmp_path / "config.toml"
    path.write_text(
        'provider = "openai"\nmodel = "test"\n[llm.main]\n' + legacy, encoding="utf-8"
    )
    assert load_config(path).reasoning.default_effort == "none"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[agent.reasoning]\ndefault_effort = "high"',
        encoding="utf-8",
    )
    assert load_config(path).reasoning.default_effort == "high"


@pytest.mark.asyncio
async def test_subagent_applies_configured_override_to_request():
    provider = _Provider()
    agent = SubAgent(
        provider=cast(Any, provider),
        model="qwen3.7-plus",
        tools=[],
        reasoning_config=ReasoningPolicyConfig(subagent_effort="low"),
    )
    assert await agent.run("任务", reasoning_effort="max") == "done"
    assert provider.calls[0]["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 4096,
    }
