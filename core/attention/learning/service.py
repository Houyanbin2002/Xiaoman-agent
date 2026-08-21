from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from core.attention._shared import clamp01, parse_datetime
from core.attention.learning.identity import build_rule_identity
from core.attention.learning.models import AttentionObservation, ObservationKind
from core.attention.learning.policy_learner import PolicyLearner
from core.attention.patterns import (
    BehaviorPattern,
    PatternLearner,
    PatternSource,
    PatternStatus,
    RecurrenceSpec,
)
from core.attention.policies import (
    PolicyEffect,
    PolicyRule,
    PolicyStatus,
)
from core.attention.ports import AttentionRepository
from core.attention.signals import AttentionSignal

_LEARNABLE_EFFECTS = frozenset(
    {
        PolicyEffect.DENY,
        PolicyEffect.REQUIRE_APPROVAL,
        PolicyEffect.ADJUST_SCORE,
        PolicyEffect.DEFER,
        PolicyEffect.LIMIT_FREQUENCY,
    }
)
_EXPLICIT_ACTIVATION_CONFIDENCE = 0.75


class AttentionLearningService:
    """Turn traceable observations into versioned, expiring decision knowledge."""

    def __init__(
        self,
        repository: AttentionRepository,
        *,
        pattern_learner: PatternLearner | None = None,
        policy_learner: PolicyLearner | None = None,
    ) -> None:
        self.repository = repository
        self.pattern_learner = pattern_learner or PatternLearner()
        self.policy_learner = policy_learner or PolicyLearner()

    def ingest_many(
        self,
        raw_items: Sequence[Mapping[str, Any]],
        *,
        source_type: str,
        source_ref: str,
        observed_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        trust_user_evidence: bool = False,
    ) -> list[AttentionObservation]:
        now = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        accepted: list[AttentionObservation] = []
        for raw in raw_items[:20]:
            observation = self._parse_observation(
                raw,
                source_type=source_type,
                # Observation identity must not depend on array order. A
                # consolidation retry may reorder equivalent items; using the
                # stable source record lets the repository deduplicate them.
                source_ref=source_ref,
                observed_at=now,
                metadata=metadata,
                trust_user_evidence=trust_user_evidence,
            )
            if observation is None:
                continue
            if not self.repository.add_observation(observation):
                continue
            if observation.kind == ObservationKind.OPPORTUNITY:
                self._apply_opportunity(observation, now=now)
            else:
                self._apply_policy(observation, now=now)
            accepted.append(observation)
        return accepted

    def ingest_signal(self, signal: AttentionSignal) -> list[AttentionObservation]:
        raw_items: list[Mapping[str, Any]] = []
        pattern = signal.metadata.get("pattern_observation")
        if isinstance(pattern, Mapping):
            raw_items.append(
                {
                    "type": "opportunity",
                    "statement": signal.summary,
                    "confidence": signal.confidence,
                    **dict(pattern),
                }
            )
        policy = signal.metadata.get("policy_observation")
        if isinstance(policy, Mapping):
            raw_items.append(
                {
                    "type": "policy",
                    "statement": signal.summary,
                    "confidence": signal.confidence,
                    **dict(policy),
                }
            )
        if not raw_items:
            return []
        observed_at = parse_datetime(signal.occurred_at) or datetime.now(timezone.utc)
        return self.ingest_many(
            raw_items,
            source_type=signal.source.type,
            source_ref=signal.id,
            observed_at=observed_at,
            metadata={"signal_id": signal.id, "signal_kind": signal.kind},
        )

    def refresh_lifecycle(self, *, now: datetime) -> None:
        for pattern in self.repository.list_patterns():
            updated = self.pattern_learner.decay(pattern, now=now)
            if updated != pattern:
                self.repository.upsert_pattern(updated)
        for policy in self.repository.list_policies():
            updated = self.policy_learner.decay(policy, now=now)
            if updated != policy:
                self.repository.upsert_policy(updated)

    def _parse_observation(
        self,
        raw: Mapping[str, Any],
        *,
        source_type: str,
        source_ref: str,
        observed_at: datetime,
        metadata: Mapping[str, Any] | None,
        trust_user_evidence: bool,
    ) -> AttentionObservation | None:
        try:
            kind = ObservationKind(str(raw.get("type") or "").strip().lower())
        except ValueError:
            return None
        statement = str(raw.get("statement") or raw.get("evidence") or "").strip()
        if not statement:
            return None
        try:
            confidence = clamp01(raw.get("confidence"), 0.5)
            verified_user_evidence = (
                trust_user_evidence
                and raw.get("_user_evidence_verified") is True
                and str(raw.get("origin") or "explicit_user")
                in {"explicit_user", "user_correction"}
                and confidence >= _EXPLICIT_ACTIVATION_CONFIDENCE
            )
            payload = (
                self._opportunity_payload(raw, explicit=verified_user_evidence)
                if kind == ObservationKind.OPPORTUNITY
                else self._policy_payload(raw, direct=verified_user_evidence)
            )
        except (TypeError, ValueError):
            return None
        identity = build_rule_identity(kind, payload)
        evidence_source_ref = source_ref
        source_message_id = str(raw.get("source_message_id") or "").strip()
        if verified_user_evidence and source_message_id:
            evidence_source_ref = f"{source_ref}#message:{source_message_id}"
        return AttentionObservation.create(
            kind=kind,
            rule_key=identity.slot_key,
            variant_key=identity.variant_key,
            statement=statement,
            confidence=confidence,
            explicit=verified_user_evidence,
            source_type=source_type,
            source_ref=evidence_source_ref,
            observed_at=observed_at,
            payload=payload,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _opportunity_payload(
        raw: Mapping[str, Any],
        *,
        explicit: bool,
    ) -> dict[str, Any]:
        recurrence_raw = raw.get("recurrence")
        if not isinstance(recurrence_raw, Mapping):
            recurrence_raw = {
                "timezone": raw.get("timezone"),
                "days": raw.get("days"),
                "start": raw.get("start"),
                "end": raw.get("end"),
            }
        recurrence = RecurrenceSpec.from_dict(dict(recurrence_raw))
        return {
            "scene": str(raw.get("scene") or "neutral")[:120],
            "kind": str(raw.get("kind") or "availability_pattern")[:120],
            "recurrence": recurrence.to_dict(),
            "available_minutes": max(
                1,
                min(int(raw.get("available_minutes") or 15), 1440),
            ),
            "explicit_user_statement": explicit,
            "expires_at": str(raw.get("expires_at") or "")[:80],
        }

    @staticmethod
    def _policy_payload(
        raw: Mapping[str, Any],
        *,
        direct: bool,
    ) -> dict[str, Any]:
        effect = PolicyEffect(str(raw.get("effect") or "adjust_score"))
        if effect not in _LEARNABLE_EFFECTS:
            raise ValueError("effect is not learnable")
        scope = raw.get("scope")
        conditions = raw.get("conditions")
        if not isinstance(scope, Mapping) or not isinstance(conditions, Mapping):
            raise ValueError("policy scope and conditions must be objects")
        score_adjustment = max(
            -0.5,
            min(float(raw.get("score_adjustment") or 0.0), 0.5),
        )
        probe = PolicyRule.create(
            effect=effect,
            scope=dict(scope),
            conditions=dict(conditions),
            score_adjustment=score_adjustment,
            source="user" if direct else "learned",
        )
        return {
            "scope": dict(probe.scope),
            "conditions": dict(probe.conditions),
            "effect": effect.value,
            "priority": max(0, min(int(raw.get("priority") or 50), 200)),
            "score_adjustment": score_adjustment,
            "user_directive": direct,
            "expires_at": str(raw.get("expires_at") or "")[:80],
            "metadata": {
                key: value
                for key, value in dict(raw.get("metadata") or {}).items()
                if key in {"max_count", "counter_key", "window_hours"}
            },
        }

    def _apply_opportunity(
        self,
        observation: AttentionObservation,
        *,
        now: datetime,
    ) -> None:
        payload = observation.payload
        pattern_id = f"pat_learned_{observation.variant_key}"
        current = self.repository.get_pattern(pattern_id)
        explicit = bool(payload.get("explicit_user_statement"))
        siblings = self._patterns_for_slot(observation.rule_key, exclude_id=pattern_id)
        if current is None:
            status = PatternStatus.ACTIVE if explicit else PatternStatus.PROPOSED
            updated = BehaviorPattern.create(
                pattern_id=pattern_id,
                kind=str(payload["kind"]),
                scene=str(payload["scene"]),
                recurrence=RecurrenceSpec.from_dict(dict(payload["recurrence"])),
                available_minutes=int(payload["available_minutes"]),
                confidence=observation.confidence,
                source=(PatternSource.USER if explicit else PatternSource.LEARNED),
                status=status,
                observation_count=1,
                last_observed_at=now,
                expires_at=parse_datetime(payload.get("expires_at")),
                user_locked=explicit,
                metadata=self._evidence_metadata(observation),
            )
        elif explicit:
            updated = replace(
                current,
                kind=str(payload["kind"]),
                scene=str(payload["scene"]),
                recurrence=RecurrenceSpec.from_dict(dict(payload["recurrence"])),
                available_minutes=int(payload["available_minutes"]),
                confidence=max(current.confidence, observation.confidence),
                observation_count=current.observation_count + 1,
                source=PatternSource.USER,
                status=PatternStatus.ACTIVE,
                last_observed_at=now.isoformat(),
                expires_at=(payload.get("expires_at") or current.expires_at or None),
                user_locked=True,
                metadata=self._merge_evidence(current.metadata, observation),
            )
        elif current.user_locked or current.source == PatternSource.USER:
            return
        else:
            updated = self.pattern_learner.observe(
                current,
                confidence=observation.confidence,
                observed_at=now,
            )
            updated = replace(
                updated,
                expires_at=(payload.get("expires_at") or updated.expires_at or None),
                metadata=self._merge_evidence(updated.metadata, observation),
            )

        if explicit:
            superseded = self._suspend_pattern_variants(
                siblings,
                superseded_by=updated.id,
            )
            updated = self._mark_pattern_supersedes(updated, superseded)
        elif updated.status == PatternStatus.ACTIVE:
            locked_active = any(
                self._is_authoritative_pattern(item)
                and item.status == PatternStatus.ACTIVE
                for item in siblings
            )
            if locked_active:
                updated = replace(updated, status=PatternStatus.PROPOSED)
            else:
                superseded = self._suspend_pattern_variants(
                    [
                        item
                        for item in siblings
                        if not self._is_authoritative_pattern(item)
                    ],
                    superseded_by=updated.id,
                )
                updated = self._mark_pattern_supersedes(updated, superseded)
        self.repository.upsert_pattern(updated)

    def _apply_policy(
        self,
        observation: AttentionObservation,
        *,
        now: datetime,
    ) -> None:
        payload = observation.payload
        policy_id = f"pol_learned_{observation.variant_key}"
        current = self.repository.get_policy(policy_id)
        direct = bool(payload.get("user_directive"))
        siblings = self._policies_for_slot(observation.rule_key, exclude_id=policy_id)
        if current is None:
            updated = PolicyRule.create(
                policy_id=policy_id,
                effect=PolicyEffect(str(payload["effect"])),
                scope=dict(payload["scope"]),
                conditions=dict(payload["conditions"]),
                priority=int(payload["priority"]),
                score_adjustment=float(payload["score_adjustment"]),
                status=(PolicyStatus.ACTIVE if direct else PolicyStatus.PROPOSED),
                confidence=observation.confidence,
                observation_count=1,
                last_observed_at=now,
                effective_from=now,
                expires_at=parse_datetime(payload.get("expires_at")),
                source="user" if direct else "learned",
                user_locked=direct,
                metadata={
                    **dict(payload.get("metadata") or {}),
                    **self._evidence_metadata(observation),
                },
            )
        elif direct:
            updated = replace(
                current,
                effect=PolicyEffect(str(payload["effect"])),
                scope=dict(payload["scope"]),
                conditions=dict(payload["conditions"]),
                priority=int(payload["priority"]),
                score_adjustment=float(payload["score_adjustment"]),
                status=PolicyStatus.ACTIVE,
                confidence=max(current.confidence, observation.confidence),
                observation_count=current.observation_count + 1,
                last_observed_at=now.isoformat(),
                effective_from=current.effective_from or now.isoformat(),
                expires_at=(payload.get("expires_at") or current.expires_at or None),
                source="user",
                user_locked=True,
                metadata={
                    **dict(payload.get("metadata") or {}),
                    **self._merge_evidence(current.metadata, observation),
                },
            )
        elif current.user_locked or current.source == "user":
            return
        else:
            updated = self.policy_learner.observe(
                current,
                confidence=observation.confidence,
                observed_at=now,
                direct_user_instruction=False,
            )
            updated = replace(
                updated,
                expires_at=(payload.get("expires_at") or updated.expires_at or None),
                metadata=self._merge_evidence(updated.metadata, observation),
            )

        if direct:
            superseded = self._suspend_policy_variants(
                siblings,
                superseded_by=updated.id,
            )
            updated = self._mark_policy_supersedes(updated, superseded)
        elif updated.status == PolicyStatus.ACTIVE:
            locked_active = any(
                self._is_authoritative_policy(item)
                and item.status == PolicyStatus.ACTIVE
                for item in siblings
            )
            if locked_active:
                updated = replace(updated, status=PolicyStatus.PROPOSED)
            else:
                superseded = self._suspend_policy_variants(
                    [
                        item
                        for item in siblings
                        if not self._is_authoritative_policy(item)
                    ],
                    superseded_by=updated.id,
                )
                updated = self._mark_policy_supersedes(updated, superseded)
        self.repository.upsert_policy(updated)

    def _patterns_for_slot(
        self,
        slot_key: str,
        *,
        exclude_id: str,
    ) -> list[BehaviorPattern]:
        return [
            item
            for item in self.repository.list_patterns()
            if item.id != exclude_id and self._pattern_slot_key(item) == slot_key
        ]

    @staticmethod
    def _is_authoritative_pattern(pattern: BehaviorPattern) -> bool:
        return pattern.user_locked or pattern.source == PatternSource.USER

    @staticmethod
    def _is_authoritative_policy(policy: PolicyRule) -> bool:
        return policy.user_locked or policy.source == "user"

    def _policies_for_slot(
        self,
        slot_key: str,
        *,
        exclude_id: str,
    ) -> list[PolicyRule]:
        return [
            item
            for item in self.repository.list_policies()
            if item.id != exclude_id and self._policy_slot_key(item) == slot_key
        ]

    @staticmethod
    def _pattern_slot_key(pattern: BehaviorPattern) -> str:
        stored = str(pattern.metadata.get("slot_key") or "").strip()
        if stored:
            return stored
        return build_rule_identity(
            ObservationKind.OPPORTUNITY,
            {
                "kind": pattern.kind,
                "scene": pattern.scene,
                "recurrence": pattern.recurrence.to_dict(),
                "available_minutes": pattern.available_minutes,
            },
        ).slot_key

    @staticmethod
    def _policy_slot_key(policy: PolicyRule) -> str:
        stored = str(policy.metadata.get("slot_key") or "").strip()
        if stored:
            return stored
        return build_rule_identity(
            ObservationKind.POLICY,
            {
                "scope": policy.scope,
                "conditions": policy.conditions,
                "effect": policy.effect.value,
                "priority": policy.priority,
                "score_adjustment": policy.score_adjustment,
                "metadata": {
                    key: value
                    for key, value in policy.metadata.items()
                    if key in {"max_count", "counter_key", "window_hours"}
                },
            },
        ).slot_key

    def _suspend_pattern_variants(
        self,
        patterns: Sequence[BehaviorPattern],
        *,
        superseded_by: str,
    ) -> list[BehaviorPattern]:
        suspended: list[BehaviorPattern] = []
        for pattern in patterns:
            if pattern.status not in {PatternStatus.ACTIVE, PatternStatus.PROPOSED}:
                continue
            metadata = dict(pattern.metadata)
            metadata["superseded_by"] = superseded_by
            updated = replace(
                pattern,
                status=PatternStatus.SUSPENDED,
                metadata=metadata,
            )
            self.repository.upsert_pattern(updated)
            suspended.append(updated)
        return suspended

    def _suspend_policy_variants(
        self,
        policies: Sequence[PolicyRule],
        *,
        superseded_by: str,
    ) -> list[PolicyRule]:
        suspended: list[PolicyRule] = []
        for policy in policies:
            if policy.status not in {PolicyStatus.ACTIVE, PolicyStatus.PROPOSED}:
                continue
            metadata = dict(policy.metadata)
            metadata["superseded_by"] = superseded_by
            updated = replace(
                policy,
                status=PolicyStatus.SUSPENDED,
                metadata=metadata,
            )
            self.repository.upsert_policy(updated)
            suspended.append(updated)
        return suspended

    @staticmethod
    def _mark_pattern_supersedes(
        pattern: BehaviorPattern,
        superseded: Sequence[BehaviorPattern],
    ) -> BehaviorPattern:
        metadata = dict(pattern.metadata)
        metadata.pop("superseded_by", None)
        if superseded:
            metadata["supersedes_id"] = superseded[-1].id
        return replace(pattern, metadata=metadata)

    @staticmethod
    def _mark_policy_supersedes(
        policy: PolicyRule,
        superseded: Sequence[PolicyRule],
    ) -> PolicyRule:
        metadata = dict(policy.metadata)
        metadata.pop("superseded_by", None)
        if superseded:
            metadata["supersedes_id"] = superseded[-1].id
        version = max(
            [policy.version, *(item.version + 1 for item in superseded)],
        )
        return replace(policy, metadata=metadata, version=version)

    @staticmethod
    def _evidence_metadata(observation: AttentionObservation) -> dict[str, Any]:
        return {
            "slot_key": observation.rule_key,
            "variant_key": observation.variant_key,
            "evidence_refs": [observation.source_ref],
            "explicit_user_evidence": observation.explicit,
            "last_statement": observation.statement,
            "last_observation_id": observation.id,
        }

    @classmethod
    def _merge_evidence(
        cls,
        metadata: Mapping[str, Any],
        observation: AttentionObservation,
    ) -> dict[str, Any]:
        result = dict(metadata)
        refs = [str(item) for item in result.get("evidence_refs") or []]
        refs.append(observation.source_ref)
        result.update(cls._evidence_metadata(observation))
        result["evidence_refs"] = list(dict.fromkeys(refs))[-20:]
        return result


__all__ = ["AttentionLearningService"]
