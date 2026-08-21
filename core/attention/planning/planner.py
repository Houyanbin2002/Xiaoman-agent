from __future__ import annotations

import hashlib
from collections.abc import Iterable
from statistics import fmean
from core.attention.actions import ActionCandidate, ActionCapability
from core.attention.opportunities import OpportunityWindow
from core.attention.signals import AttentionSignal


class ActionPlanner:
    """Match open action manifests to signals and opportunity windows."""

    def generate(
        self,
        *,
        signals: Iterable[AttentionSignal],
        windows: Iterable[OpportunityWindow],
        capabilities: Iterable[ActionCapability],
        preference_features: dict[str, dict[str, float]] | None = None,
    ) -> list[ActionCandidate]:
        signal_by_id = {item.id: item for item in signals}
        preferences = preference_features or {}
        candidates: list[ActionCandidate] = []
        for window in windows:
            scoped = [
                signal_by_id[item] for item in window.signal_ids if item in signal_by_id
            ]
            for capability in capabilities:
                compatible = self._compatible_signals(capability, scoped, window)
                if not compatible and (
                    not capability.can_propose_without_signal
                    or not window.source_pattern_id
                ):
                    continue
                # A notification is an atomic decision about one item of
                # evidence.  Bundling unrelated signals causes hidden items to
                # be acknowledged even though the user never saw them.
                groups = [[item] for item in compatible] if compatible else [[]]
                for group in groups:
                    domain = self._domain(group)
                    if not capability.supports(
                        domain=domain,
                        scene=window.scene,
                        available_minutes=window.available_minutes,
                    ):
                        continue
                    estimated = min(
                        window.available_minutes,
                        capability.maximum_minutes,
                        max(capability.minimum_minutes, capability.default_minutes),
                    )
                    scoped_preferences = dict(preferences.get(capability.id, {}))
                    scoped_preferences.update(
                        preferences.get(f"{capability.id}|domain:{domain}", {})
                    )
                    features = self._features(
                        group,
                        window,
                        capability,
                        scoped_preferences,
                    )
                    signal_ids = tuple(item.id for item in group)
                    title = group[0].summary if group else capability.name
                    reason = self._reason(group, window, capability)
                    candidate_id = self._candidate_id(
                        capability.id,
                        window.id,
                        signal_ids,
                    )
                    candidates.append(
                        ActionCandidate(
                            id=candidate_id,
                            capability_id=capability.id,
                            action_type=capability.action_type,
                            domain=domain,
                            risk=capability.risk,
                            title=title,
                            reason=reason,
                            signal_ids=signal_ids,
                            opportunity_id=window.id,
                            estimated_minutes=estimated,
                            inputs={
                                "signal_ids": list(signal_ids),
                                "signal_summaries": [item.summary for item in group],
                                "evidence": [
                                    evidence
                                    for item in group
                                    for evidence in item.evidence
                                ],
                                "available_minutes": window.available_minutes,
                                "scene": window.scene,
                            },
                            features=features,
                        )
                    )
        return candidates

    @staticmethod
    def _compatible_signals(
        capability: ActionCapability,
        signals: list[AttentionSignal],
        window: OpportunityWindow,
    ) -> list[AttentionSignal]:
        result: list[AttentionSignal] = []
        for signal in signals:
            if signal.suggested_capabilities and not (
                capability.id in signal.suggested_capabilities
                or capability.action_type in signal.suggested_capabilities
            ):
                continue
            if not capability.supports(
                domain=signal.domain,
                scene=window.scene,
                available_minutes=window.available_minutes,
            ):
                continue
            result.append(signal)
        return sorted(
            result,
            key=lambda item: (
                -(item.urgency + item.severity + item.confidence),
                item.id,
            ),
        )[:5]

    @staticmethod
    def _domain(signals: list[AttentionSignal]) -> str:
        if not signals:
            return "general"
        totals: dict[str, float] = {}
        for signal in signals:
            totals[signal.domain] = totals.get(signal.domain, 0.0) + signal.confidence
        return max(totals, key=lambda item: (totals[item], item))

    @staticmethod
    def _features(
        signals: list[AttentionSignal],
        window: OpportunityWindow,
        capability: ActionCapability,
        preferences: dict[str, float],
    ) -> dict[str, float]:
        if signals:
            relevance = fmean(item.confidence * item.actionability for item in signals)
            urgency = max(item.urgency for item in signals)
            severity = max(item.severity for item in signals)
            confidence = fmean(item.confidence for item in signals)
            freshness = fmean(item.freshness for item in signals)
            goal_alignment = fmean(
                float(item.metadata.get("goal_alignment") or 0.5) for item in signals
            )
        else:
            relevance = float(preferences.get("relevance", 0.5))
            urgency = 0.0
            severity = 0.0
            confidence = window.confidence
            freshness = 1.0
            goal_alignment = float(preferences.get("goal_alignment", 0.5))
        raw_repetition = max(
            0.0,
            min(float(preferences.get("repetition_penalty", 0.0)), 1.0),
        )
        # Repeated low-value nudges should quickly become quiet, while a new
        # high-severity signal must still be able to break through. This is a
        # generic severity interaction, not a domain-specific exception.
        effective_repetition = raw_repetition * (1.0 - severity * severity)
        return {
            "relevance": max(0.0, min(relevance, 1.0)),
            "urgency": max(0.0, min(urgency, 1.0)),
            "severity": max(0.0, min(severity, 1.0)),
            "confidence": max(0.0, min(confidence, 1.0)),
            "goal_alignment": max(0.0, min(goal_alignment, 1.0)),
            "preference_fit": max(
                0.0,
                min(float(preferences.get("preference_fit", 0.5)), 1.0),
            ),
            "freshness": max(0.0, min(freshness, 1.0)),
            "window_fit": window.confidence,
            "historical_acceptance": max(
                0.0,
                min(float(preferences.get("historical_acceptance", 0.5)), 1.0),
            ),
            "interruption_cost": max(
                0.0,
                min(float(capability.interruption_cost), 1.0),
            ),
            "repetition_penalty": max(
                0.0,
                min(effective_repetition, 1.0),
            ),
            "uncertainty_penalty": max(0.0, min(1.0 - confidence, 1.0)),
        }

    @staticmethod
    def _reason(
        signals: list[AttentionSignal],
        window: OpportunityWindow,
        capability: ActionCapability,
    ) -> str:
        if signals:
            return (
                f"{window.scene}机会窗口有效；"
                f"{len(signals)} 个可信信号可由 {capability.name} 处理"
            )
        return f"{window.scene}机会窗口有效；{capability.name}支持无事件候选"

    @staticmethod
    def _candidate_id(
        capability_id: str,
        window_id: str,
        signal_ids: tuple[str, ...],
    ) -> str:
        raw = "|".join((capability_id, window_id, *signal_ids))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"cand_{digest}"


__all__ = ["ActionPlanner"]
