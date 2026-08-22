from __future__ import annotations

"""Regression gates for prompt, model, tool-policy and memory changes."""

from dataclasses import dataclass

from .models import EvalSummary


@dataclass(frozen=True)
class EvalComparison:
    passed: bool
    pass_rate_delta: float
    reward_delta: float
    regressions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "pass_rate_delta": self.pass_rate_delta,
            "reward_delta": self.reward_delta,
            "regressions": list(self.regressions),
        }


def compare(
    baseline: EvalSummary,
    candidate: EvalSummary,
    *,
    max_reward_drop: float = 0.02,
    max_pass_rate_drop: float = 0.0,
) -> EvalComparison:
    """Fail a promotion when a candidate regresses a golden case or aggregate gate."""
    baseline_by_id = {result.case_id: result for result in baseline.results}
    candidate_by_id = {result.case_id: result for result in candidate.results}
    regressions: list[str] = []
    for case_id, old in baseline_by_id.items():
        new = candidate_by_id.get(case_id)
        if new is None:
            regressions.append(f"missing_case:{case_id}")
            continue
        if old.passed and not new.passed:
            regressions.append(f"case_failed:{case_id}")
        old_hard = {score.name for score in old.scores if score.hard and not score.passed}
        new_hard = {score.name for score in new.scores if score.hard and not score.passed}
        for name in sorted(new_hard - old_hard):
            regressions.append(f"hard_gate:{case_id}:{name}")
    pass_rate_delta = candidate.pass_rate - baseline.pass_rate
    reward_delta = candidate.mean_reward - baseline.mean_reward
    passed = (
        not regressions
        and pass_rate_delta >= -abs(max_pass_rate_drop)
        and reward_delta >= -abs(max_reward_drop)
    )
    if pass_rate_delta < -abs(max_pass_rate_drop):
        regressions.append("aggregate_pass_rate")
    if reward_delta < -abs(max_reward_drop):
        regressions.append("aggregate_reward")
    return EvalComparison(passed, pass_rate_delta, reward_delta, tuple(regressions))
