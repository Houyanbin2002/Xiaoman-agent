from __future__ import annotations

"""Evidence-based acceptance. Never infer business success from fluent text."""

from collections.abc import Sequence
from typing import Any

from .models import AgentRun, EvalCase, EvalSummary, Score


def assess(
    case: EvalCase, run: AgentRun, scores: Sequence[Score], error: str
) -> dict[str, Any]:
    by_id = {score.name: score for score in scores}
    llm_ids = {item.criterion_id for item in case.rubric if item.evaluator == "llm"}

    def verified(ids: list[str]) -> bool:
        return bool(ids) and all(
            name in by_id
            and by_id[name].passed
            and by_id[name].source != "unassessed"
            and (name not in llm_ids or by_id[name].source == "rubric_judge")
            for name in ids
        )

    contract = case.metadata.get("acceptance", {})
    objective_ids = list(contract.get("objective_checks", ()))
    blocked_ids = list(contract.get("blocked_checks", ()))
    delivery_ids = list(contract.get("delivery_checks", ()))
    outcome = "unassessed"
    if objective_ids or blocked_ids:
        outcome = "not_completed"
        if verified(blocked_ids):
            outcome = "blocked"
        elif verified(objective_ids) and run.status == "completed":
            outcome = "completed"
    elif "objective_outcome" in case.expected:
        outcome = run.objective_outcome
    delivery = "not_applicable"
    if delivery_ids:
        delivery = "delivered" if verified(delivery_ids) else "unconfirmed"
    elif "delivery_status" in case.expected:
        delivery = run.delivery_status

    # Quality remains meaningful for an honest blocked or interrupted response.
    quality_assessed = bool(llm_ids) and all(
        name in by_id and by_id[name].source == "rubric_judge" for name in llm_ids
    )
    expected_outcome = str(contract.get("expected_outcome", "completed"))
    outcome_ok = outcome == expected_outcome if objective_ids or blocked_ids else True
    accepted = (
        bool(scores)
        and not error
        and run.status == "completed"
        and all(score.passed and score.source != "unassessed" for score in scores)
        and (not llm_ids or quality_assessed)
        and outcome_ok
    )
    return {
        "objective_outcome": outcome,
        "objective_evidence": objective_ids,
        "blocked_reason": (
            str(contract.get("blocked_reason", "")) if outcome == "blocked" else ""
        ),
        "delivery_status": delivery,
        "delivery_evidence": delivery_ids,
        "quality_assessed": quality_assessed,
        "quality_passed": quality_assessed and verified(list(llm_ids)),
        "accepted": accepted,
        "coverage_level": str(case.metadata.get("coverage_level", "unspecified")),
    }


def acceptance_exit_code(summary: EvalSummary, gate: str = "acceptance") -> int:
    if not summary.total:
        return 2
    if gate == "hard":
        return 0 if summary.pass_rate == 1.0 else 1
    return (
        0
        if all(result.assessment.get("accepted", False) for result in summary.results)
        else 1
    )
