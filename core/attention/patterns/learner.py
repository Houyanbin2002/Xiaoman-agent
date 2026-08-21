from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from core.attention._shared import clamp01, parse_datetime, utc_iso
from core.attention.patterns.models import BehaviorPattern, PatternSource, PatternStatus


class PatternLearner:
    """Apply evidence and decay without embedding scenario-specific rules."""

    def __init__(
        self,
        *,
        activation_threshold: float = 0.75,
        minimum_observations: int = 3,
        stale_after_days: int = 30,
        decay_per_stale_period: float = 0.08,
    ) -> None:
        self.activation_threshold = clamp01(activation_threshold)
        self.minimum_observations = max(1, minimum_observations)
        self.stale_after_days = max(1, stale_after_days)
        self.decay_per_stale_period = clamp01(decay_per_stale_period)

    def observe(
        self,
        pattern: BehaviorPattern,
        *,
        confidence: float,
        observed_at: datetime,
    ) -> BehaviorPattern:
        if pattern.user_locked or pattern.status in {
            PatternStatus.REJECTED,
            PatternStatus.EXPIRED,
        }:
            return pattern
        count = pattern.observation_count + 1
        incoming = clamp01(confidence)
        combined = clamp01(
            (pattern.confidence * pattern.observation_count + incoming) / count
        )
        status = pattern.status
        if count >= self.minimum_observations and combined >= self.activation_threshold:
            status = PatternStatus.ACTIVE
        return replace(
            pattern,
            confidence=combined,
            observation_count=count,
            last_observed_at=utc_iso(observed_at),
            status=status,
        )

    def decay(self, pattern: BehaviorPattern, *, now: datetime) -> BehaviorPattern:
        if pattern.user_locked or pattern.source == PatternSource.USER:
            return pattern
        observed = parse_datetime(pattern.last_observed_at)
        last_decay = parse_datetime(pattern.metadata.get("last_decay_at"))
        reference = max(
            (item for item in (observed, last_decay) if item is not None),
            default=None,
        )
        if reference is None:
            return pattern
        elapsed_days = max(0, (now.astimezone(timezone.utc) - reference).days)
        periods = elapsed_days // self.stale_after_days
        if periods <= 0:
            return pattern
        confidence = clamp01(pattern.confidence - periods * self.decay_per_stale_period)
        status = pattern.status
        if confidence < 0.25:
            status = PatternStatus.EXPIRED
        elif confidence < self.activation_threshold and status == PatternStatus.ACTIVE:
            status = PatternStatus.PROPOSED
        metadata = dict(pattern.metadata)
        metadata["last_decay_at"] = utc_iso(now)
        return replace(
            pattern,
            confidence=confidence,
            status=status,
            metadata=metadata,
        )


__all__ = ["PatternLearner"]
