from __future__ import annotations

"""Deterministic scorers plus a rubric aggregation contract."""

from collections.abc import Callable, Mapping
from typing import Any

from .models import AgentRun, EvalCase, RubricCriterion, Score


# Dataset cases describe user-facing capabilities while an installation may
# expose a concrete tool name (for example ``schedule``).  Keep this mapping
# narrow and explicit so the evaluator does not turn every similarly named
# tool into a match.
_TOOL_ALIASES: dict[str, frozenset[str]] = {
    "schedule_reminder": frozenset({"schedule", "schedule_reminder"}),
    "cancel_reminder": frozenset({"cancel_schedule", "cancel_reminder"}),
    "reschedule_reminder": frozenset({"schedule", "reschedule_reminder"}),
    "conversation_search": frozenset({"search_messages", "conversation_search"}),
    "memory_search": frozenset({"recall_memory", "memory_search"}),
    "recent_history": frozenset({"fetch_messages", "search_messages", "recent_history"}),
    "workflow_lookup": frozenset({"task_manage", "workflow_lookup"}),
}


def _tool_matches(expected: str, actual: str) -> bool:
    expected_key = str(expected).strip().casefold()
    actual_key = str(actual).strip().casefold()
    aliases = _TOOL_ALIASES.get(expected_key)
    return actual_key == expected_key or (aliases is not None and actual_key in aliases)


def _generated_rubric(case: EvalCase) -> tuple[RubricCriterion, ...]:
    """Compile legacy ``expected`` assertions into the canonical Rubric.

    Existing datasets can keep their concise ``expected`` section. At runtime
    every assertion is still represented as a Rubric criterion, so reports and
    publishers have one stable evaluation vocabulary.
    """
    expected = case.expected
    criteria: list[RubricCriterion] = []
    if "response_contains" in expected:
        criteria.append(RubricCriterion("response_contains", "回复包含要求的关键信息", weight=1.0, check="response_contains"))
    if "required_tools" in expected:
        criteria.append(RubricCriterion("required_tools", "调用所有要求的工具", weight=1.0, hard=bool(expected.get("required_tools_hard", False)), check="required_tools"))
    if "forbidden_tools" in expected:
        criteria.append(RubricCriterion("forbidden_tools", "不调用禁止工具", weight=2.0, hard=True, check="forbidden_tools"))
    if "state_contains" in expected:
        criteria.append(RubricCriterion("state_contains", "执行后状态满足要求", weight=2.0, hard=True, check="state_contains"))
    if "memory_event" in expected:
        criteria.append(RubricCriterion("memory_event", "记忆事件满足治理要求", weight=2.0, hard=True, check="memory_event"))
    if "status" in expected:
        criteria.append(RubricCriterion("status", "执行状态正确", weight=1.0, hard=True, check="status"))
    if "max_latency_ms" in expected:
        criteria.append(RubricCriterion("latency", "执行延迟不超过预算", weight=0.5, check="latency"))
    if "trajectory" in expected:
        policy = expected.get("trajectory")
        hard = bool(policy.get("hard", False)) if isinstance(policy, Mapping) else False
        criteria.append(RubricCriterion("trajectory", "工具轨迹符合策略", weight=1.5, hard=hard, check="trajectory"))
    return tuple(criteria)


def canonical_rubric(case: EvalCase) -> tuple[RubricCriterion, ...]:
    """Return explicit criteria plus any assertions compiled from ``expected``."""
    explicit = list(case.rubric)
    covered = {criterion.criterion_id for criterion in explicit}
    covered.update(criterion.check for criterion in explicit if criterion.check)
    for criterion in _generated_rubric(case):
        if criterion.criterion_id not in covered and criterion.check not in covered:
            explicit.append(criterion)
    return tuple(explicit)


def _contains_all(text: str, values: list[Any]) -> tuple[float, str]:
    wanted = [str(item) for item in values if str(item)]
    if not wanted:
        return 1.0, "no required phrases"
    found = [item for item in wanted if item.lower() in text.lower()]
    return len(found) / len(wanted), f"matched {len(found)}/{len(wanted)} required phrases"


def _deep_contains(actual: Any, expected: Any, path: str = "state") -> tuple[bool, str]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False, f"{path} is not an object"
        for key, value in expected.items():
            if key not in actual:
                return False, f"missing {path}.{key}"
            ok, reason = _deep_contains(actual[key], value, f"{path}.{key}")
            if not ok:
                return False, reason
        return True, f"matched {path}"
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False, f"{path} is not a list"
        missing = [item for item in expected if item not in actual]
        return (not missing, f"missing values in {path}: {missing}" if missing else f"matched {path}")
    return actual == expected, f"{path} expected {expected!r}, got {actual!r}"


def _memory_event_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Match structured memory while allowing semantic content assertions."""
    content = str(actual.get("content") or actual.get("summary") or actual.get("value") or "")
    if "content_contains" in expected:
        wanted = expected.get("content_contains")
        if str(wanted).casefold() not in content.casefold():
            return False
    exact = {key: value for key, value in expected.items() if key != "content_contains"}
    return not exact or _deep_contains(actual, exact, "memory_event")[0]


def score_expected(case: EvalCase, run: AgentRun) -> list[Score]:
    expected = case.expected
    scores: list[Score] = []

    phrases = expected.get("response_contains", [])
    value, reason = _contains_all(run.response, phrases if isinstance(phrases, list) else [phrases])
    scores.append(Score("response_contains", value, value >= 1.0, weight=1.0, reason=reason))

    required_tools = [str(item) for item in expected.get("required_tools", [])]
    called = [tool.name for tool in run.tools]
    missing_tools = [
        item for item in required_tools if not any(_tool_matches(item, name) for name in called)
    ]
    tool_value = 1.0 if not required_tools else (len(required_tools) - len(missing_tools)) / len(required_tools)
    scores.append(
        Score(
            "required_tools",
            tool_value,
            not missing_tools,
            weight=1.0,
            hard=bool(expected.get("required_tools_hard", False)),
            reason="all required tools called" if not missing_tools else f"missing tools: {missing_tools}",
        )
    )

    forbidden_tools = {str(item) for item in expected.get("forbidden_tools", [])}
    used_forbidden = sorted(forbidden_tools.intersection(called))
    scores.append(
        Score(
            "forbidden_tools",
            0.0 if used_forbidden else 1.0,
            not used_forbidden,
            weight=2.0,
            hard=True,
            reason="no forbidden tool used" if not used_forbidden else f"forbidden tools: {used_forbidden}",
        )
    )

    if "state_contains" in expected:
        ok, state_reason = _deep_contains(run.state, expected["state_contains"])
        scores.append(Score("state_contains", 1.0 if ok else 0.0, ok, weight=2.0, hard=True, reason=state_reason))

    if "memory_event" in expected:
        wanted = expected["memory_event"]
        matched = any(
            _memory_event_matches(event, wanted)
            if isinstance(wanted, Mapping) and isinstance(event, Mapping)
            else _deep_contains(event, wanted, "memory_event")[0]
            for event in run.memory_events
        )
        scores.append(Score("memory_event", 1.0 if matched else 0.0, matched, weight=2.0, hard=True, reason="memory event matched" if matched else "memory event not found"))

    if "status" in expected:
        ok = run.status == str(expected["status"])
        scores.append(Score("status", 1.0 if ok else 0.0, ok, weight=1.0, hard=True, reason=f"status={run.status}"))

    if "max_latency_ms" in expected and run.latency_ms is not None:
        limit = float(expected["max_latency_ms"])
        ok = run.latency_ms <= limit
        scores.append(Score("latency", 1.0 if ok else max(0.0, limit / run.latency_ms), ok, weight=0.5, reason=f"{run.latency_ms:.1f}ms <= {limit:.1f}ms"))
    return scores


def score_trajectory(case: EvalCase, run: AgentRun) -> Score:
    policy = case.expected.get("trajectory", {})
    if not isinstance(policy, Mapping):
        return Score("trajectory", 1.0, True, weight=1.0, reason="no trajectory policy")
    names = [tool.name for tool in run.tools]
    required_order = [str(item) for item in policy.get("required_order", [])]
    cursor = 0
    for name in names:
        if cursor < len(required_order) and _tool_matches(required_order[cursor], name):
            cursor += 1
    order_ok = cursor == len(required_order)
    retries_ok = len(run.tools) <= int(policy["max_tool_calls"]) if "max_tool_calls" in policy else True
    ok = order_ok and retries_ok
    reason = "trajectory policy satisfied"
    if not order_ok:
        reason = f"required order progress {cursor}/{len(required_order)}"
    elif not retries_ok:
        reason = f"tool calls {len(run.tools)} exceed limit {policy['max_tool_calls']}"
    return Score("trajectory", 1.0 if ok else 0.0, ok, weight=1.5, hard=bool(policy.get("hard", False)), reason=reason)


def score_rubric(
    case: EvalCase,
    run: AgentRun,
    *,
    judge: Callable[[EvalCase, AgentRun], Mapping[str, Any]] | None = None,
) -> list[Score]:
    rubric = canonical_rubric(case)
    if not rubric:
        return []
    judged = judge(case, run) if judge is not None else {}
    scores: list[Score] = []
    deterministic_scores = score_expected(case, run)
    deterministic_scores.append(score_trajectory(case, run))
    deterministic = {score.name: score for score in deterministic_scores}
    for criterion in rubric:
        raw = judged.get(criterion.criterion_id)
        source = "rubric_judge" if criterion.evaluator == "llm" else "deterministic_rubric"
        reason = "judge score" if criterion.evaluator == "llm" else "deterministic check"
        if criterion.evaluator == "deterministic" and criterion.check in deterministic:
            fallback = deterministic[criterion.check]
            raw = fallback.value
            reason = fallback.reason
            source = "deterministic_rubric_fallback"
        elif raw is None and criterion.check in deterministic:
            # Explicitly allowing a deterministic fallback makes a Rubric
            # portable between offline and judge-enabled runs.
            fallback = deterministic[criterion.check]
            raw = fallback.value
            reason = fallback.reason
            source = "deterministic_rubric_fallback"
        value = float(raw if raw is not None else 0.0)
        scores.append(Score(criterion.criterion_id, value, value >= criterion.threshold, weight=criterion.weight, hard=criterion.hard, reason=reason, source=source))
    return scores


def aggregate_scores(scores: list[Score]) -> tuple[float, bool]:
    if not scores:
        return 1.0, True
    hard_fail = any(score.hard and not score.passed for score in scores)
    total_weight = sum(max(0.0, score.weight) for score in scores) or 1.0
    reward = sum(score.value * max(0.0, score.weight) for score in scores) / total_weight
    return (0.0 if hard_fail else reward), not hard_fail and all(score.passed for score in scores if score.hard)
