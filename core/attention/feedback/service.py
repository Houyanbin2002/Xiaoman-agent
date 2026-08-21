from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from core.attention._shared import clamp01, parse_datetime
from core.attention.actions import ActionPlanStatus
from core.attention.feedback.models import AttentionFeedback, FeedbackKind
from core.attention.patterns import BehaviorPattern, PatternStatus
from core.attention.policies import PolicyRule, PolicyStatus
from core.attention.ports import AttentionRepository


class FeedbackService:
    """Record feedback and update only the dimension the feedback describes."""

    def __init__(self, repository: AttentionRepository) -> None:
        self.repository = repository

    def record(
        self,
        *,
        plan_id: str,
        kind: FeedbackKind,
        note: str = "",
        metadata: dict[str, object] | None = None,
        now: datetime | None = None,
    ) -> AttentionFeedback:
        feedback = AttentionFeedback.create(
            plan_id=plan_id,
            kind=kind,
            note=note,
            metadata=dict(metadata or {}),
            created_at=now,
        )
        self.repository.add_feedback(feedback)
        plan = self.repository.get_plan(plan_id)
        if plan is None:
            return feedback
        for policy_id in plan.policy_ids:
            policy = self.repository.get_policy(policy_id)
            if policy is None or policy.user_locked:
                continue
            updated_policy = self._update_policy(policy, kind)
            if updated_policy != policy:
                self.repository.upsert_policy(updated_policy)
        window = self.repository.get_window(plan.opportunity_id)
        if window is None or not window.source_pattern_id:
            return feedback
        pattern = self.repository.get_pattern(window.source_pattern_id)
        if pattern is None or pattern.user_locked:
            return feedback
        updated = self._update_pattern(pattern, kind)
        if updated != pattern:
            self.repository.upsert_pattern(updated)
        return feedback

    def decision_attributes(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Build generic history features for the next attention decision.

        The profile is capability based, so new MCP-backed abilities inherit
        the same learning loop without adding scenario-specific code.
        """
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        plans = self.repository.list_plans(limit=1000)
        feedback = self.repository.list_feedback()
        latest_feedback: dict[str, AttentionFeedback] = {}
        for item in feedback:
            existing = latest_feedback.get(item.plan_id)
            if existing is None or item.created_at > existing.created_at:
                latest_feedback[item.plan_id] = item

        recent_cutoff = current - timedelta(hours=24)
        history_cutoff = current - timedelta(days=90)
        by_scope: dict[str, list] = {}
        frequency_counts: dict[str, int] = {}
        for plan in plans:
            created = parse_datetime(plan.created_at)
            if created is None or created < history_cutoff:
                continue
            domain = self._plan_domain(plan.signal_ids)
            scope_keys = (
                plan.capability_id,
                f"{plan.capability_id}|domain:{domain}",
            )
            for key in scope_keys:
                by_scope.setdefault(key, []).append(plan)
            updated = parse_datetime(plan.updated_at)
            delivered = max(created, updated) if updated is not None else created
            if plan.status == ActionPlanStatus.SUCCEEDED and delivered >= recent_cutoff:
                for key in scope_keys:
                    frequency_counts[key] = frequency_counts.get(key, 0) + 1

        preferences: dict[str, dict[str, float]] = {}
        positive_kinds = {
            FeedbackKind.ACCEPTED,
            FeedbackKind.OPENED,
            FeedbackKind.COMPLETED,
        }
        negative_kinds = {
            FeedbackKind.IGNORED,
            FeedbackKind.DISLIKED,
            FeedbackKind.INACCURATE,
        }
        for scope_key, scoped_plans in by_scope.items():
            positive = 0.0
            negative = 0.0
            extra_frequency_penalty = 0.0
            latest_success: datetime | None = None
            is_domain_scope = "|domain:" in scope_key
            for plan in scoped_plans:
                created = parse_datetime(plan.created_at)
                updated = parse_datetime(plan.updated_at)
                delivered = (
                    max(created, updated)
                    if created is not None and updated is not None
                    else updated or created
                )
                if plan.status == ActionPlanStatus.SUCCEEDED and delivered is not None:
                    latest_success = max(latest_success or delivered, delivered)
                item = latest_feedback.get(plan.id)
                if item is None:
                    continue
                feedback_at = parse_datetime(item.created_at) or current
                age_days = max(
                    0.0,
                    (current - feedback_at).total_seconds() / 86400,
                )
                # Recent reactions should matter most.  A stale dislike from
                # months ago must not permanently freeze proactive behavior.
                feedback_weight = 0.5 ** (age_days / 30.0)
                if item.kind in positive_kinds:
                    positive += feedback_weight
                elif item.kind in negative_kinds:
                    negative += feedback_weight
                elif item.kind == FeedbackKind.DEFERRED:
                    positive += 0.25 * feedback_weight
                    negative += 0.25 * feedback_weight
                if is_domain_scope and item.kind == FeedbackKind.TOO_FREQUENT:
                    extra_frequency_penalty += 0.35 * feedback_weight
                if is_domain_scope and item.kind == FeedbackKind.WRONG_TIME:
                    extra_frequency_penalty += 0.2 * feedback_weight

            total = positive + negative
            acceptance = (positive + 2.0) / (total + 4.0)
            count_24h = frequency_counts.get(scope_key, 0)
            recentness = 0.0
            if latest_success is not None:
                hours = max(0.0, (current - latest_success).total_seconds() / 3600)
                recentness = max(0.0, 1.0 - hours / 6.0)
            if is_domain_scope:
                repetition = min(
                    1.0,
                    recentness * 0.7
                    + min(1.0, count_24h / 4.0) * 0.3
                    + extra_frequency_penalty,
                )
            else:
                # Global volume is a weak signal only.  Strong suppression is
                # learned per domain so one noisy source cannot silence every
                # other kind of help.
                repetition = min(
                    0.4,
                    recentness * 0.25 + min(1.0, count_24h / 8.0) * 0.15,
                )
            preferences[scope_key] = {
                "historical_acceptance": acceptance,
                "preference_fit": acceptance,
                "repetition_penalty": repetition,
            }
        return {
            "capability_preferences": preferences,
            "frequency_counts": frequency_counts,
        }

    def _plan_domain(self, signal_ids: tuple[str, ...]) -> str:
        totals: dict[str, float] = {}
        for signal_id in signal_ids:
            signal = self.repository.get_signal(signal_id)
            if signal is None:
                continue
            totals[signal.domain] = totals.get(signal.domain, 0.0) + signal.confidence
        return max(totals, key=lambda item: (totals[item], item)) if totals else "general"

    @staticmethod
    def _update_pattern(
        pattern: BehaviorPattern,
        kind: FeedbackKind,
    ) -> BehaviorPattern:
        if kind in {FeedbackKind.ACCEPTED, FeedbackKind.OPENED, FeedbackKind.COMPLETED}:
            return replace(pattern, confidence=clamp01(pattern.confidence + 0.03))
        if kind == FeedbackKind.WRONG_TIME:
            confidence = clamp01(pattern.confidence - 0.15)
            return replace(
                pattern,
                confidence=confidence,
                status=(
                    PatternStatus.PROPOSED if confidence < 0.75 else pattern.status
                ),
            )
        if kind == FeedbackKind.TOO_FREQUENT:
            metadata = dict(pattern.metadata)
            metadata["frequency_feedback"] = "reduce"
            return replace(pattern, metadata=metadata)
        return pattern

    @staticmethod
    def _update_policy(policy: PolicyRule, kind: FeedbackKind) -> PolicyRule:
        if kind in {FeedbackKind.ACCEPTED, FeedbackKind.OPENED, FeedbackKind.COMPLETED}:
            return replace(policy, confidence=clamp01(policy.confidence + 0.03))
        if kind in {FeedbackKind.DISLIKED, FeedbackKind.INACCURATE}:
            confidence = clamp01(policy.confidence - 0.2)
            return replace(
                policy,
                confidence=confidence,
                status=(PolicyStatus.PROPOSED if confidence < 0.75 else policy.status),
            )
        if kind == FeedbackKind.IGNORED:
            return replace(policy, confidence=clamp01(policy.confidence - 0.03))
        if kind == FeedbackKind.TOO_FREQUENT:
            metadata = dict(policy.metadata)
            metadata["frequency_feedback"] = "reduce"
            return replace(policy, metadata=metadata)
        return policy


__all__ = ["FeedbackService"]
