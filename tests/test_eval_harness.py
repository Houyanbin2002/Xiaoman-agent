from __future__ import annotations

import pytest

from eval.dataset import load_cases, mine_hard_cases, write_cases
from eval.models import AgentRun, EvalCase, ToolCall
from eval.runner import EvalHarness, ReplayExecutor
from eval.store import EvalResultStore
from eval.publishers import LangfuseScorePublisher
from eval.judge import OpenAICompatibleRubricJudge
from eval.compare import compare
from eval.analysis import failure_hotspots


@pytest.mark.asyncio
async def test_smoke_dataset_is_fully_evaluable():
    cases = load_cases("eval/datasets/smoke.jsonl")
    summary = await EvalHarness(dataset_name="smoke", version="v1").run(cases, ReplayExecutor())

    assert summary.total == 6
    assert summary.passed == 6
    assert summary.pass_rate == 1.0
    assert summary.mean_reward > 0.95
    assert summary.metrics["recovery"] > 0.9
    assert summary.slices["memory"]["pass_rate"] == 1.0
    assert summary.slices["workflow"]["hard_fail_rate"] == 0.0


@pytest.mark.asyncio
async def test_hard_gate_vetoes_fluent_but_unsafe_run():
    case = EvalCase.from_dict(
        {
            "case_id": "safety.veto",
            "title": "unsafe",
            "input": "delete",
            "expected": {"response_contains": ["不能"], "forbidden_tools": ["delete"]},
        }
    )
    run = {"response": "不能这样做", "tools": [{"name": "delete"}]}
    summary = await EvalHarness().run([case], lambda _case: AgentRun.from_value(run))
    result = summary.results[0]

    assert result.passed is False
    assert result.reward == 0.0
    assert any(score.hard and not score.passed for score in result.scores)


def test_dataset_round_trip_and_hard_case_mining(tmp_path):
    case = EvalCase.from_dict({"case_id": "x", "title": "x", "input": "hello"})
    path = tmp_path / "cases.jsonl"
    write_cases(path, [case])
    assert load_cases(path)[0].case_id == "x"

    candidates = mine_hard_cases([{"case_id": "x", "reward": 0.2, "passed": False}])
    assert candidates[0]["review_status"] == "needs_human_review"


@pytest.mark.asyncio
async def test_result_store_keeps_low_reward_cases(tmp_path):
    case = EvalCase.from_dict(
        {
            "case_id": "store.case",
            "title": "store",
            "input": "hello",
            "expected": {"response_contains": ["ok"]},
            "replay": {"response": "no"},
        }
    )
    summary = await EvalHarness().run([case], ReplayExecutor())
    store = EvalResultStore(tmp_path / "eval.sqlite")
    try:
        assert store.save(summary) == 1
        assert store.low_reward_cases(threshold=0.9)[0]["case_id"] == "store.case"
    finally:
        store.close()


def test_langfuse_publisher_uses_trace_scores():
    class Client:
        def __init__(self):
            self.calls = []

        def create_score(self, **kwargs):
            self.calls.append(kwargs)

    client = Client()
    case = EvalCase.from_dict(
        {
            "case_id": "lf",
            "title": "lf",
            "input": "x",
            "expected": {"response_contains": ["ok"]},
        }
    )
    run = AgentRun(response="ok", trace_id="trace-1")
    summary = EvalHarness().run_sync([case], lambda _case: run)
    LangfuseScorePublisher(client).publish(summary)
    assert client.calls[0]["trace_id"] == "trace-1"
    assert client.calls[0]["name"] == "response_contains"


def test_langfuse_publisher_maps_local_trace_id_to_remote_trace_id():
    class Client:
        def __init__(self):
            self.calls = []

        def create_trace_id(self, *, seed):
            return f"remote:{seed}"

        def create_score(self, **kwargs):
            self.calls.append(kwargs)

    client = Client()
    case = EvalCase.from_dict({"case_id": "lf-map", "title": "lf", "input": "x", "expected": {"response_contains": ["ok"]}})
    summary = EvalHarness().run_sync([case], lambda _case: AgentRun(response="ok", trace_id="local-1"))
    LangfuseScorePublisher(client).publish(summary)
    assert client.calls
    assert {call["trace_id"] for call in client.calls} == {"remote:local-1"}


def test_openai_compatible_judge_parses_structured_scores():
    class _Message:
        content = '{"scores":{"quality":0.86},"reasons":{"quality":"完整"}}'

    class _Response:
        choices = [type("Choice", (), {"message": _Message()})()]

    class _Completions:
        def create(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            assert kwargs["model"] == "judge-model"
            return _Response()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    case = EvalCase.from_dict(
        {
            "case_id": "judge",
            "title": "judge",
            "input": "summarize",
            "rubric": [
                {"id": "quality", "description": "完整清晰", "evaluator": "llm"}
            ],
        }
    )
    judge = OpenAICompatibleRubricJudge(
        api_key="test",
        base_url="https://example.invalid/v1",
        model="judge-model",
        client=_Client(),
    )
    assert judge(case, AgentRun(response="summary")) == {"quality": 0.86}


def test_openai_compatible_judge_retries_malformed_json():
    class _Message:
        def __init__(self, content):
            self.content = content

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            content = (
                "not json"
                if self.calls == 1
                else '{"scores":{"quality":0.75}}'
            )
            message = _Message(content)
            return type(
                "Response",
                (),
                {"choices": [type("Choice", (), {"message": message})()]},
            )()

    completions = _Completions()
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    case = EvalCase.from_dict(
        {
            "case_id": "judge-retry",
            "title": "judge retry",
            "input": "summarize",
            "rubric": [
                {"id": "quality", "description": "完整清晰", "evaluator": "llm"}
            ],
        }
    )
    judge = OpenAICompatibleRubricJudge(
        api_key="test",
        base_url="https://example.invalid/v1",
        model="judge-model",
        client=client,
    )

    assert judge(case, AgentRun(response="summary")) == {"quality": 0.75}
    assert completions.calls == 2


@pytest.mark.asyncio
async def test_judge_failure_preserves_agent_run_and_uses_deterministic_fallback():
    case = EvalCase.from_dict(
        {
            "case_id": "judge-fallback",
            "title": "judge fallback",
            "input": "summarize",
            "expected": {"response_contains": ["ok"]},
            "rubric": [
                {
                    "id": "quality",
                    "description": "包含预期内容",
                    "evaluator": "llm",
                    "check": "response_contains",
                    "hard": True,
                }
            ],
        }
    )

    def failing_judge(_case, _run):
        raise RuntimeError("malformed response")

    summary = await EvalHarness(judge=failing_judge).run(
        [case],
        lambda _case: AgentRun(response="ok", status="success"),
    )
    result = summary.results[0]

    assert result.passed is True
    assert result.run.response == "ok"
    assert result.run.status == "success"
    assert result.scores[0].source == "deterministic_rubric_fallback"
    assert result.error.startswith("judge_degraded:")


@pytest.mark.asyncio
async def test_rubric_is_single_contract_with_deterministic_and_llm_backends():
    case = EvalCase.from_dict(
        {
            "case_id": "rubric.backends",
            "title": "rubric",
            "input": "summarize",
            "expected": {"forbidden_tools": ["delete"]},
            "rubric": [
                {
                    "id": "quality",
                    "description": "摘要完整清晰",
                    "evaluator": "llm",
                    "weight": 2,
                }
            ],
        }
    )
    summary = await EvalHarness(judge=lambda _case, _run: {"quality": 0.9}).run(
        [case], lambda _case: AgentRun(response="summary")
    )
    result = summary.results[0]

    assert result.passed is True
    assert {score.name for score in result.scores} == {"forbidden_tools", "quality"}
    assert next(score for score in result.scores if score.name == "quality").source == "rubric_judge"


@pytest.mark.asyncio
async def test_regression_gate_detects_case_failure():
    case = EvalCase.from_dict(
        {
            "case_id": "gate",
            "title": "gate",
            "input": "x",
            "expected": {"response_contains": ["ok"], "forbidden_tools": ["delete"]},
        }
    )
    baseline = await EvalHarness().run([case], lambda _case: AgentRun(response="ok"))
    candidate = await EvalHarness().run([case], lambda _case: AgentRun(response="no", tools=(ToolCall("delete"),)))
    result = compare(baseline, candidate)
    assert result.passed is False
    assert "case_failed:gate" in result.regressions


def test_tool_capability_aliases_match_concrete_installation_names():
    case = EvalCase.from_dict(
        {
            "case_id": "aliases",
            "title": "aliases",
            "input": "schedule",
            "expected": {
                "required_tools": ["schedule_reminder", "conversation_search"],
                "trajectory": {
                    "required_order": ["schedule_reminder", "conversation_search"],
                    "hard": True,
                },
            },
        }
    )
    run = AgentRun(
        response="ok",
        tools=(ToolCall("schedule"), ToolCall("search_messages")),
    )
    summary = EvalHarness().run_sync([case], lambda _case: run)
    result = summary.results[0]

    assert result.passed is True
    assert all(score.passed for score in result.scores if score.name in {"required_tools", "trajectory"})


@pytest.mark.asyncio
async def test_failure_hotspots_group_declared_optimization_modes():
    case = EvalCase.from_dict(
        {
            "case_id": "hotspot",
            "title": "hotspot",
            "input": "x",
            "metadata": {"slice": "memory", "failure_modes": ["preference_missed"]},
            "tags": ["memory", "preference"],
            "expected": {"response_contains": ["Python"]},
        }
    )
    summary = await EvalHarness().run([case], lambda _case: AgentRun(response="wrong"))
    report = failure_hotspots([case], summary)
    assert report["failed_or_low_reward"] == 1
    assert report["hot_failure_modes"] == [("preference_missed", 1)]
