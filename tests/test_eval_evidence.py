from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval.analysis import failure_hotspots
from eval.dataset import load_cases
from eval.fixtures import _EvalRecoveryTool
from eval.judge import RubricJudgment
from eval.models import AgentRun, EvalCase
from eval.runner import EvalHarness, ProcessDirectExecutor, render_markdown


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,content,status",
    [
        ("completed", "结果", "completed"),
        ("completed", "", "incomplete"),
        ("tool_loop", "已经检查一部分", "incomplete"),
        ("budget_exhausted", "未完成", "incomplete"),
        ("unknown", "有文字并非完成", "incomplete"),
        ("error", "失败", "error"),
    ],
)
async def test_live_executor_preserves_kernel_status_and_full_tool_evidence(
    reason, content, status
):
    class Loop:
        async def process_direct_outbound(self, *_args, **_kwargs):
            return SimpleNamespace(
                content=content,
                metadata={
                    "context_retry": {"exit_reason": reason},
                    "tool_chain": [
                        {
                            "calls": [
                                {
                                    "name": "read",
                                    "status": "error",
                                    "result": "source unavailable",
                                },
                                {
                                    "name": "fallback",
                                    "status": "success",
                                    "result": "full evidence",
                                },
                            ]
                        }
                    ],
                },
            )

    trace_store = SimpleNamespace(
        get_trace=lambda _id: SimpleNamespace(status="completed", metadata={}),
        list_events=lambda _id: [],
    )
    run = await ProcessDirectExecutor(Loop(), trace_store=trace_store)(
        EvalCase("test", "test", "test")
    )
    assert run.status == status
    assert run.tools[0].error == "source unavailable"
    assert run.tools[1].output == "full evidence"
    assert run.tools[1].status == "completed"
    assert run.metadata["tool_evidence_source"] == "outbound.tool_chain"


@pytest.mark.asyncio
async def test_legacy_text_adapter_does_not_invent_completion():
    class Loop:
        async def process_direct(self, *_args, **_kwargs):
            return "有回复"

    run = await ProcessDirectExecutor(Loop())(EvalCase("test", "test", "test"))
    assert run.status == "incomplete"
    assert run.metadata["exit_reason"] == "unknown"


@pytest.mark.asyncio
async def test_judge_reasons_and_low_quality_survive_high_composite_reward():
    case = EvalCase.from_dict(
        {
            "case_id": "quality",
            "input": "读取配置",
            "tags": ["execution"],
            "rubric": [
                {"id": "status", "check": "status", "hard": True, "weight": 100},
                {
                    "id": "quality",
                    "description": "交付内容",
                    "evaluator": "llm",
                    "threshold": 0.6,
                },
            ],
            "expected": {"status": "completed"},
        }
    )
    summary = await EvalHarness(
        judge=lambda _case, _run: RubricJudgment(
            {"quality": 0.2}, {"quality": "仅说完成，没有配置内容"}
        )
    ).run([case], lambda _case: AgentRun(response="已完成"))
    result = summary.results[0]
    assert result.passed and result.reward > 0.9
    assert summary.metrics["quality_assessed_cases"] == 1
    assert summary.metrics["quality_passed_cases"] == 0
    assert summary.metrics["rubric_acceptance_cases"] == 0
    assert failure_hotspots([case], summary)["failed_or_low_reward"] == 1
    assert "仅说完成，没有配置内容" in render_markdown(summary)


@pytest.mark.asyncio
async def test_judge_unavailable_is_not_a_quality_pass():
    case = EvalCase.from_dict(
        {
            "case_id": "quality",
            "rubric": [
                {"id": "quality", "check": "response_contains", "evaluator": "llm"}
            ],
            "expected": {"response_contains": ["ok"]},
        }
    )
    summary = await EvalHarness().run([case], lambda _case: AgentRun(response="ok"))
    assert summary.metrics["quality_assessed_cases"] == 0
    assert summary.metrics["quality_passed_cases"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {"status": "success", "data": {"port": 2236}},
        {"status": "empty", "data": []},
        {"status": "unavailable", "message": "快照不存在"},
    ],
)
async def test_recovery_fixture_returns_bound_evidence(result):
    tool = _EvalRecoveryTool("fallback", failing_tool="read", result=result)
    payload = json.loads(await tool.execute())
    assert payload.items() >= result.items()
    assert payload["ok"] is (result["status"] != "unavailable")


@pytest.mark.asyncio
async def test_recovery_fixture_missing_data_is_not_empty_or_success():
    payload = json.loads(
        await _EvalRecoveryTool("fallback", failing_tool="read").execute()
    )
    assert payload["status"] == "unavailable" and payload["ok"] is False
    assert "data" not in payload


@pytest.mark.parametrize(
    "result",
    [
        {"status": "success"},
        {"status": "success", "data": []},
        {"status": "empty"},
        {"status": "empty", "data": [1]},
        {"status": "unavailable", "data": []},
        {"status": "completed"},
    ],
)
def test_recovery_fixture_rejects_inconsistent_evidence(result):
    with pytest.raises(ValueError):
        _EvalRecoveryTool("fallback", failing_tool="read", result=result)


def test_recovery_v2_has_explicit_evidence_and_distinguishes_three_outcomes():
    cases = load_cases("eval/datasets/recovery_v2_judge.jsonl")
    assert len(cases) == 8
    assert {case.metadata["fixture"]["result"]["status"] for case in cases} == {
        "success",
        "empty",
        "unavailable",
    }
    assert all(not case.replay for case in cases)
