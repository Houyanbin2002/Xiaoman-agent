from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from core.attention._shared import clamp01, parse_datetime, utc_iso
from core.attention.policies import PolicyRule, PolicyStatus


class PolicyLearner:
    """Promote and decay learned policies using evidence, never scenario names."""

    def __init__(
        self,
        *,
        activation_threshold: float = 0.75,
        minimum_observations: int = 3,
        stale_after_days: int = 30,
        decay_per_stale_period: float = 0.08,
    ) -> None:
        self.activation_threshold = clamp01(activation_threshold)
        self.minimum_observations = max(1, int(minimum_observations))
        self.stale_after_days = max(1, int(stale_after_days))
        self.decay_per_stale_period = clamp01(decay_per_stale_period)

    def observe(
        self,
        policy: PolicyRule,
        *,
        confidence: float,
        observed_at: datetime,
        direct_user_instruction: bool = False,
    ) -> PolicyRule:
        if policy.user_locked or policy.status in {
            PolicyStatus.REJECTED,
            PolicyStatus.EXPIRED,
        }:
            return policy
        count = policy.observation_count + 1
        combined = clamp01(
            (policy.confidence * policy.observation_count + clamp01(confidence)) / count
        )
        status = policy.status
        if direct_user_instruction or (
            count >= self.minimum_observations and combined >= self.activation_threshold
        ):
            status = PolicyStatus.ACTIVE
        return replace(
            policy,
            confidence=combined,
            observation_count=count,
            last_observed_at=utc_iso(observed_at),
            status=status,
        )

    def decay(self, policy: PolicyRule, *, now: datetime) -> PolicyRule:
        if policy.user_locked or policy.source == "user":
            return policy
        observed = parse_datetime(policy.last_observed_at)
        last_decay = parse_datetime(policy.metadata.get("last_decay_at"))
        reference = max(
            (item for item in (observed, last_decay) if item is not None),
            default=None,
        )
        if reference is None:
            return policy
        elapsed_days = max(0, (now.astimezone(timezone.utc) - reference).days)
        periods = elapsed_days // self.stale_after_days
        if periods <= 0:
            return policy
        confidence = clamp01(policy.confidence - periods * self.decay_per_stale_period)
        status = policy.status
        if confidence < 0.25:
            status = PolicyStatus.EXPIRED
        elif confidence < self.activation_threshold and status == PolicyStatus.ACTIVE:
            status = PolicyStatus.PROPOSED
        metadata = dict(policy.metadata)
        metadata["last_decay_at"] = utc_iso(now)
        return replace(
            policy,
            confidence=confidence,
            status=status,
            metadata=metadata,
        )


__all__ = ["PolicyLearner"]
