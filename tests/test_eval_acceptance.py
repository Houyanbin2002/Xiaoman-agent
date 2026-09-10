from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from eval.calibration import calibrate
from eval.compare import compare
from eval.dataset import load_cases
from eval.evidence import check_evidence
from eval.experiment import aggregate_repeats, select_split
from eval.fixtures import LiveEvalFixtureManager, PreparedFixture
from eval.judge import RubricJudgment
from eval.models import AgentRun, EvalCase, ToolCall
from eval.outcomes import acceptance_exit_code
from eval.runner import EvalHarness, ProcessDirectExecutor


def task_case(**overrides):
    return EvalCase.from_dict(
        {
            "case_id": "task",
            "input": "give JSON answer",
            "expected": {
                "status": "completed",
                "evidence": {
                    "facts": {"source": "response_json", "contains": {"passed": 22}}
                },
            },
            "rubric": [{"id": "quality", "evaluator": "llm", "threshold": 0.75}],
            "metadata": {"acceptance": {"objective_checks": ["facts", "quality"]}},
            **overrides,
        }
    )


def harness(value=1):
    return EvalHarness(
        judge=lambda *_: RubricJudgment({"quality": value}, {"quality": "evidence"})
    )


def test_soft_quality_failure_blocks_default_gate_not_legacy_hard_gate():
    summary = harness(0.2).run_sync(
        [task_case()], lambda _: AgentRun(response='{"passed":22}')
    )
    assert summary.pass_rate == 1
    assert acceptance_exit_code(summary) == 1
    assert acceptance_exit_code(summary, "hard") == 0
    assert summary.results[0].assessment["objective_outcome"] == "not_completed"


def test_missing_judge_and_empty_rubric_cannot_pass_acceptance():
    for case in (task_case(), EvalCase("empty", "empty", "hi")):
        summary = EvalHarness().run_sync(
            [case], lambda _: AgentRun(response='{"passed":22}')
        )
        assert acceptance_exit_code(summary) == 1


def test_blocked_is_acceptable_but_not_objective_completion():
    case = task_case(
        expected={
            "evidence": {
                "unavailable": {
                    "source": "state",
                    "contains": {"service": "unavailable"},
                }
            }
        },
        metadata={
            "acceptance": {
                "blocked_checks": ["unavailable"],
                "expected_outcome": "blocked",
                "blocked_reason": "No service",
            }
        },
    )
    summary = harness().run_sync(
        [case],
        lambda _: AgentRun(response="Cannot verify", state={"service": "unavailable"}),
    )
    assert acceptance_exit_code(summary) == 0
    assert summary.metrics["objective_completed_cases"] == 0
    assert summary.metrics["objective_blocked_cases"] == 1


def test_reply_quality_is_independent_of_kernel_timeout():
    summary = harness().run_sync(
        [task_case()], lambda _: AgentRun(response="Timed out", status="incomplete")
    )
    assert summary.metrics["quality_passed_cases"] == 1
    assert summary.metrics["execution_completed_cases"] == 0
    assert acceptance_exit_code(summary) == 1


@pytest.mark.parametrize(
    "response",
    ['{"passed":24}', '{"passed":"22"}', '{"passed":true}', "包含22，但没有答案"],
)
def test_keywords_do_not_replace_typed_facts(response):
    assert not check_evidence(
        {"source": "response_json", "contains": {"passed": 22}},
        AgentRun(response=response),
    )


@pytest.mark.parametrize(
    "tool",
    [
        ToolCall("read", status="failed", output={"status": "success", "data": 22}),
        ToolCall("read", output={"status": "unavailable"}),
        ToolCall(
            "read",
            arguments={"file": "wrong"},
            output={"status": "success", "data": 22},
        ),
        ToolCall(
            "read",
            arguments={"file": "right"},
            output={"status": "success", "data": 22},
            error="failure",
        ),
    ],
)
def test_tool_evidence_checks_status_args_and_data(tool):
    assert not check_evidence(
        {
            "source": "tool",
            "name": "read",
            "status": "completed",
            "arguments": {"file": "right"},
            "output": {"status": "success", "data": 22},
        },
        AgentRun(tools=(tool,)),
    )


def test_llm_check_does_not_shadow_deterministic_assertion():
    case = task_case(
        expected={"response_contains": ["must-have"]},
        rubric=[{"id": "quality", "evaluator": "llm", "check": "response_contains"}],
    )
    summary = harness().run_sync([case], lambda _: AgentRun(response="wrong"))
    assert {s.name for s in summary.results[0].scores} == {
        "quality",
        "response_contains",
    }
    assert acceptance_exit_code(summary) == 1


def test_family_split_is_disjoint_and_order_stable():
    cases = [
        replace(task_case(), case_id=str(n), metadata={"family": f"family-{n//2}"})
        for n in range(30)
    ]
    dev = {c.case_id for c in select_split(cases, "dev")}
    holdout = {c.case_id for c in select_split(cases, "holdout")}
    assert not dev & holdout and dev | holdout == {c.case_id for c in cases}
    assert dev == {c.case_id for c in select_split(list(reversed(cases)), "dev")}
    for n in range(0, 30, 2):
        assert (str(n) in dev) == (str(n + 1) in dev)


def test_repeat_aggregate_retains_bad_trial_and_unique_ids():
    case = task_case()
    old = harness(0.2).run_sync(
        [case], lambda _: AgentRun(response='{"passed":22}', latency_ms=200)
    )
    new = harness().run_sync(
        [case], lambda _: AgentRun(response='{"passed":22}', latency_ms=100)
    )
    summary = aggregate_repeats([case], [old, new], {})
    assert summary.metrics["acceptance_trial_rate"] == 0.5
    assert summary.metrics["unstable_cases"] == 1
    assert summary.metrics["all_repeats_accepted_cases"] == 0
    assert [r.case_id for r in summary.results] == ["task::r1", "task::r2"]
    assert acceptance_exit_code(summary) == 1


def test_comparison_rejects_changed_fixture_and_quality_regression():
    base = harness().run_sync(
        [task_case()], lambda _: AgentRun(response='{"passed":22}')
    )
    candidate = harness(0.2).run_sync(
        [task_case()], lambda _: AgentRun(response='{"passed":22}')
    )
    assert "quality_failed:task" in compare(base, candidate).regressions
    assert not compare(
        replace(base, manifest={"dataset_sha256": "a"}),
        replace(candidate, manifest={"dataset_sha256": "b"}),
    ).passed


@pytest.mark.asyncio
async def test_scenario_shares_memory_across_sessions_and_cleans_up_on_error():
    sessions, cleaned, outputs = [], [], []

    class Loop:
        async def process_direct_outbound(self, text, **kwargs):
            sessions.append(kwargs["session_key"])
            outputs.append(text)
            return SimpleNamespace(
                content=text, metadata={"context_retry": {"exit_reason": "completed"}}
            )

    class Fixtures:
        async def prepare(self, case, **kwargs):
            return kwargs["session_key"]

        async def observe(self, prepared):
            return SimpleNamespace(state={"prepared": prepared})

        def disabled_tools(self, prepared):
            return []

        async def cleanup(self, prepared):
            cleaned.append(prepared)

    case = task_case(
        metadata={
            "turns": [
                {"input": "first"},
                {"input": "second"},
                {"input": "third", "session": "new"},
            ]
        }
    )
    run = await ProcessDirectExecutor(Loop(), fixture_manager=Fixtures())(case)
    assert sessions[0] == sessions[1] and sessions[0] != sessions[2]
    assert len(cleaned) == 2 and len(run.metadata["turn_runs"]) == 3
    assert run.metadata["turn_runs"][0]["response"] == "first"
    assert check_evidence(
        {
            "source": "turn",
            "index": 0,
            "assert": {"source": "response_regex", "pattern": "first"},
        },
        run,
    )


@pytest.mark.asyncio
async def test_artifact_observer_reads_file_not_model_claim(tmp_path):
    manager = LiveEvalFixtureManager(SimpleNamespace(workspace=tmp_path))
    prepared = PreparedFixture(kind="artifact", fixture={"paths": ["result.json"]})
    result = await manager.observe(prepared)
    assert result.state["artifacts"]["result.json"] == {"exists": False}
    (tmp_path / "result.json").write_text('{"passed":22}', encoding="utf-8")
    result = await manager.observe(prepared)
    assert result.state["artifacts"]["result.json"]["json"] == {"passed": 22}
    assert len(result.state["artifacts"]["result.json"]["sha256"]) == 64
    with pytest.raises(ValueError):
        manager._artifact_path("../outside.json")


def test_calibration_draft_is_not_reported_as_human_ground_truth():
    cases = load_cases("eval/datasets/judge_calibration_v1.jsonl")
    report = calibrate(cases, lambda *_: RubricJudgment({"quality": 1}, {}))
    assert report["human_agreement_rate"] is None
    assert report["human_reviewed_assessed"] == 0
    assert any(not row["agreement"] for row in report["cases"])


def test_acceptance_dataset_has_frozen_holdout_and_no_canned_agent_responses():
    cases = load_cases("eval/datasets/acceptance_v2.jsonl")
    assert len(cases) == 9
    assert len(select_split(cases, "dev")) == 6
    assert len(select_split(cases, "holdout")) == 3
    assert all(not c.replay for c in cases)
    assert all(c.metadata.get("acceptance") for c in cases)
