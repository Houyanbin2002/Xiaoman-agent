from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from core.attention.actions.models import ActionCandidate, ActionRisk
from core.attention.policies.models import (
    DecisionContext,
    PolicyDecision,
    PolicyEffect,
    PolicyRule,
)


class PolicyEngine:
    """Evaluate a deliberately small declarative policy language."""

    def evaluate(
        self,
        *,
        candidate: ActionCandidate,
        context: DecisionContext,
        policies: Iterable[PolicyRule],
    ) -> PolicyDecision:
        kernel = self._kernel_decision(candidate, context)
        if not kernel.allowed:
            return kernel
        adjustment = kernel.score_adjustment
        approval = kernel.require_approval
        deferred = False
        reasons = list(kernel.reasons)
        matched: list[str] = []
        active = sorted(
            (rule for rule in policies if rule.is_active_at(context.now)),
            key=lambda item: (-item.priority, item.id),
        )
        for rule in active:
            if not self._matches(rule, candidate, context):
                continue
            matched.append(rule.id)
            reasons.append(f"policy:{rule.id}:{rule.effect.value}")
            if rule.effect == PolicyEffect.ADJUST_SCORE:
                adjustment += rule.score_adjustment
                continue
            if rule.effect == PolicyEffect.REQUIRE_APPROVAL:
                approval = True
                continue
            if rule.effect == PolicyEffect.LIMIT_FREQUENCY:
                maximum = int(rule.metadata.get("max_count") or 1)
                key = str(rule.metadata.get("counter_key") or candidate.capability_id)
                counts = context.attributes.get("frequency_counts") or {}
                if int(counts.get(key, 0)) >= maximum:
                    return PolicyDecision(
                        allowed=False,
                        require_approval=approval,
                        score_adjustment=adjustment,
                        reasons=tuple(reasons + ["frequency_limit"]),
                        matched_policy_ids=tuple(matched),
                    )
                continue
            if rule.effect == PolicyEffect.DEFER:
                deferred = True
                return PolicyDecision(
                    allowed=False,
                    require_approval=approval,
                    deferred=True,
                    score_adjustment=adjustment,
                    reasons=tuple(reasons),
                    matched_policy_ids=tuple(matched),
                )
            if rule.effect == PolicyEffect.DENY:
                return PolicyDecision(
                    allowed=False,
                    require_approval=approval,
                    deferred=deferred,
                    score_adjustment=adjustment,
                    reasons=tuple(reasons),
                    matched_policy_ids=tuple(matched),
                )
            if rule.effect == PolicyEffect.ALLOW:
                return PolicyDecision(
                    allowed=True,
                    require_approval=approval,
                    deferred=deferred,
                    score_adjustment=adjustment,
                    reasons=tuple(reasons),
                    matched_policy_ids=tuple(matched),
                )
        return PolicyDecision(
            allowed=True,
            require_approval=approval,
            deferred=deferred,
            score_adjustment=adjustment,
            reasons=tuple(reasons),
            matched_policy_ids=tuple(matched),
        )

    @staticmethod
    def _kernel_decision(
        candidate: ActionCandidate,
        context: DecisionContext,
    ) -> PolicyDecision:
        severity = candidate.features.get("severity", 0.0)
        if context.do_not_disturb and not (
            severity >= 0.8 and context.allow_high_priority
        ):
            return PolicyDecision(allowed=False, reasons=("do_not_disturb",))
        if context.focus_active and severity < 0.8:
            return PolicyDecision(allowed=False, reasons=("focus_active",))
        approval = False
        if candidate.risk in {ActionRisk.EXTERNAL_WRITE, ActionRisk.DESTRUCTIVE}:
            approval = context.permission_mode != "full"
        elif candidate.risk == ActionRisk.REVERSIBLE_WRITE:
            approval = context.permission_mode == "ask"
        return PolicyDecision(
            allowed=True,
            require_approval=approval,
            reasons=(("kernel_approval",) if approval else ()),
        )

    def _matches(
        self,
        rule: PolicyRule,
        candidate: ActionCandidate,
        context: DecisionContext,
    ) -> bool:
        values = {
            "domain": candidate.domain,
            "action_type": candidate.action_type,
            "capability_id": candidate.capability_id,
            "risk": candidate.risk.value,
            "scene": context.scene,
            "channel": context.channel,
        }
        if any(
            not self._value_matches(values.get(key), expected)
            for key, expected in rule.scope.items()
        ):
            return False
        for key, expected in rule.conditions.items():
            if key == "severity_min" and candidate.features.get(
                "severity", 0.0
            ) < float(expected):
                return False
            if key == "severity_max" and candidate.features.get(
                "severity", 0.0
            ) > float(expected):
                return False
            if key == "confidence_min" and candidate.features.get(
                "confidence", 0.0
            ) < float(expected):
                return False
            if key == "focus_active" and context.focus_active is not bool(expected):
                return False
            if key == "do_not_disturb" and context.do_not_disturb is not bool(expected):
                return False
            if key == "scene" and not self._value_matches(context.scene, expected):
                return False
            if key.startswith("attribute."):
                attribute = key.removeprefix("attribute.")
                if not self._value_matches(context.attributes.get(attribute), expected):
                    return False
        return True

    @staticmethod
    def _value_matches(actual: Any, expected: Any) -> bool:
        if expected in (None, "", "*"):
            return True
        if isinstance(expected, (list, tuple, set)):
            return actual in expected or "*" in expected
        return actual == expected


__all__ = ["PolicyEngine"]
