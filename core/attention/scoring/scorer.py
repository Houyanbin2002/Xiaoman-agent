from __future__ import annotations

from dataclasses import dataclass, field

from core.attention.actions import ActionCandidate, ActionRisk


@dataclass(frozen=True)
class ScoringWeights:
    positive: dict[str, float] = field(
        default_factory=lambda: {
            "relevance": 0.18,
            "urgency": 0.14,
            "severity": 0.15,
            "goal_alignment": 0.12,
            "preference_fit": 0.08,
            "freshness": 0.08,
            "window_fit": 0.12,
            "historical_acceptance": 0.13,
        }
    )
    negative: dict[str, float] = field(
        default_factory=lambda: {
            "interruption_cost": 0.35,
            "repetition_penalty": 0.30,
            "uncertainty_penalty": 0.20,
            "risk_penalty": 0.25,
        }
    )


class UtilityScorer:
    _RISK_PENALTY = {
        ActionRisk.READ_ONLY: 0.0,
        ActionRisk.NOTIFY: 0.08,
        ActionRisk.REVERSIBLE_WRITE: 0.25,
        ActionRisk.EXTERNAL_WRITE: 0.45,
        ActionRisk.DESTRUCTIVE: 0.85,
    }

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score(
        self,
        candidate: ActionCandidate,
        *,
        policy_adjustment: float = 0.0,
    ) -> ActionCandidate:
        features = dict(candidate.features)
        features["risk_penalty"] = self._RISK_PENALTY[candidate.risk]
        positive = {
            key: self.weights.positive[key] * float(features.get(key, 0.0))
            for key in self.weights.positive
        }
        negative = {
            key: self.weights.negative[key] * float(features.get(key, 0.0))
            for key in self.weights.negative
        }
        total = sum(positive.values()) - sum(negative.values()) + policy_adjustment
        components = {
            **positive,
            **{key: -value for key, value in negative.items()},
            "policy_adjustment": policy_adjustment,
        }
        return candidate.with_score(round(total, 6), components)


__all__ = ["ScoringWeights", "UtilityScorer"]
