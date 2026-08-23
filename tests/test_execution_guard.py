from __future__ import annotations

from agent.runtime.execution_guard import (
    ExecutionGuard,
    ExecutionGuardConfig,
    bound_tool_result,
)
from agent.tools.base import ToolResult


def _call(
    call_id: str,
    *,
    name: str = "lookup",
    status: str = "success",
    result: str = "same result",
) -> dict:
    return {
        "call_id": call_id,
        "name": name,
        "arguments": {"query": "same"},
        "status": status,
        "result": result,
    }


def test_same_readonly_signature_warns_then_stops() -> None:
    guard = ExecutionGuard(
        ExecutionGuardConfig(same_signature_warn=2, same_signature_stop=3)
    )
    state = guard.initial_state()

    first = guard.before_tool_batch(
        state, [_call("c1")], risk_resolver=lambda _name: "read-only"
    )
    assert first.stop_reason == ""
    after_first = guard.after_tool_round(first.state, [_call("c1")])
    assert after_first.hint == ""

    second = guard.before_tool_batch(
        after_first.state, [_call("c2")], risk_resolver=lambda _name: "read-only"
    )
    assert second.stop_reason == ""
    after_second = guard.after_tool_round(second.state, [_call("c2")])
    assert "更换方法" in after_second.hint

    third = guard.before_tool_batch(
        after_second.state, [_call("c3")], risk_resolver=lambda _name: "read-only"
    )
    assert third.stop_reason == "tool_call_loop"


def test_duplicate_side_effect_is_blocked_before_second_execution() -> None:
    guard = ExecutionGuard()
    state = guard.after_tool_round(guard.initial_state(), [_call("c1")]).state

    decision = guard.before_tool_batch(
        state,
        [_call("c2")],
        risk_resolver=lambda _name: "external-side-effect",
    )

    assert decision.stop_reason == "duplicate_side_effect"


def test_three_failures_disable_non_polling_tool() -> None:
    guard = ExecutionGuard(ExecutionGuardConfig(no_progress_rounds=6))
    state = guard.initial_state()
    decision = None
    for index in range(3):
        decision = guard.after_tool_round(
            state,
            [
                _call(
                    f"c{index}",
                    status="error",
                    result=f"failure {index}",
                )
            ],
        )
        state = decision.state

    assert decision is not None
    assert decision.disabled_tools == ("lookup",)


def test_tool_result_is_deterministically_bounded_with_head_and_tail() -> None:
    source = "head-" + ("x" * 5_000) + "-tail"

    first = bound_tool_result(ToolResult(text=source), 1_000)
    second = bound_tool_result(ToolResult(text=source), 1_000)

    assert first.text == second.text
    assert len(first.text) <= 1_000
    assert "original_chars=" in first.text
    assert "head-" in first.text
    assert "-tail" in first.text


def test_resume_preserves_budgets_but_resets_wall_clock() -> None:
    guard = ExecutionGuard()
    state = guard.after_tool_round(guard.initial_state(), [_call("c1")]).state
    previous_started_at = state["started_at"]

    resumed = guard.resume_state(state)

    assert resumed["tool_calls_total"] == 1
    assert resumed["recent_rounds"] == state["recent_rounds"]
    assert resumed["started_at"] >= previous_started_at
