from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from core.attention._shared import utc_iso


class ActionRisk(StrEnum):
    READ_ONLY = "read_only"
    NOTIFY = "notify"
    REVERSIBLE_WRITE = "reversible_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class ActionPlanStatus(StrEnum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    DEFERRED = "deferred"
    EXPIRED = "expired"
    FAILED = "failed"


_TRANSITIONS: dict[ActionPlanStatus, set[ActionPlanStatus]] = {
    ActionPlanStatus.PROPOSED: {
        ActionPlanStatus.PENDING_APPROVAL,
        ActionPlanStatus.APPROVED,
        ActionPlanStatus.EXECUTING,
        ActionPlanStatus.SKIPPED,
        ActionPlanStatus.DEFERRED,
        ActionPlanStatus.EXPIRED,
    },
    ActionPlanStatus.PENDING_APPROVAL: {
        ActionPlanStatus.APPROVED,
        ActionPlanStatus.SKIPPED,
        ActionPlanStatus.EXPIRED,
    },
    ActionPlanStatus.APPROVED: {
        ActionPlanStatus.EXECUTING,
        ActionPlanStatus.SKIPPED,
        ActionPlanStatus.EXPIRED,
    },
    ActionPlanStatus.EXECUTING: {
        ActionPlanStatus.SUCCEEDED,
        ActionPlanStatus.FAILED,
    },
    ActionPlanStatus.DEFERRED: {
        ActionPlanStatus.PROPOSED,
        ActionPlanStatus.EXPIRED,
        ActionPlanStatus.SKIPPED,
    },
    ActionPlanStatus.SUCCEEDED: set(),
    ActionPlanStatus.SKIPPED: set(),
    ActionPlanStatus.EXPIRED: set(),
    ActionPlanStatus.FAILED: set(),
}


@dataclass(frozen=True)
class ActionCapability:
    id: str
    name: str
    description: str
    provider: str
    action_type: str
    risk: ActionRisk
    auto_execute: bool
    supported_domains: tuple[str, ...] = ("*",)
    supported_scenes: tuple[str, ...] = ("*",)
    minimum_minutes: int = 1
    maximum_minutes: int = 24 * 60
    default_minutes: int = 5
    interruption_cost: float = 0.1
    can_propose_without_signal: bool = False
    required_inputs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, *, domain: str, scene: str, available_minutes: int) -> bool:
        domain_ok = "*" in self.supported_domains or domain in self.supported_domains
        scene_ok = "*" in self.supported_scenes or scene in self.supported_scenes
        return domain_ok and scene_ok and available_minutes >= self.minimum_minutes

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "provider": self.provider,
            "action_type": self.action_type,
            "risk": self.risk.value,
            "auto_execute": self.auto_execute,
            "supported_domains": list(self.supported_domains),
            "supported_scenes": list(self.supported_scenes),
            "minimum_minutes": self.minimum_minutes,
            "maximum_minutes": self.maximum_minutes,
            "default_minutes": self.default_minutes,
            "interruption_cost": self.interruption_cost,
            "can_propose_without_signal": self.can_propose_without_signal,
            "required_inputs": list(self.required_inputs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ActionCandidate:
    id: str
    capability_id: str
    action_type: str
    domain: str
    risk: ActionRisk
    title: str
    reason: str
    signal_ids: tuple[str, ...]
    opportunity_id: str
    estimated_minutes: int
    inputs: dict[str, Any]
    features: dict[str, float]
    score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)

    def with_score(
        self,
        score: float,
        components: dict[str, float],
    ) -> "ActionCandidate":
        return replace(self, score=score, score_components=dict(components))


@dataclass(frozen=True)
class ActionPlan:
    id: str
    idempotency_key: str
    signal_ids: tuple[str, ...]
    opportunity_id: str
    capability_id: str
    action_type: str
    decision_reason: str
    score: float
    score_components: dict[str, float]
    risk: ActionRisk
    approval: str
    status: ActionPlanStatus
    inputs: dict[str, Any]
    policy_ids: tuple[str, ...]
    created_at: str
    expires_at: str | None
    updated_at: str
    result: dict[str, Any] | None = None
    error: str = ""

    @classmethod
    def from_candidate(
        cls,
        candidate: ActionCandidate,
        *,
        idempotency_key: str,
        status: ActionPlanStatus,
        approval: str,
        policy_ids: tuple[str, ...],
        created_at: str,
        expires_at: str | None,
    ) -> "ActionPlan":
        return cls(
            id=f"act_{uuid.uuid4().hex}",
            idempotency_key=idempotency_key,
            signal_ids=candidate.signal_ids,
            opportunity_id=candidate.opportunity_id,
            capability_id=candidate.capability_id,
            action_type=candidate.action_type,
            decision_reason=candidate.reason,
            score=candidate.score,
            score_components=dict(candidate.score_components),
            risk=candidate.risk,
            approval=approval,
            status=status,
            inputs=dict(candidate.inputs),
            policy_ids=policy_ids,
            created_at=created_at,
            expires_at=expires_at,
            updated_at=created_at,
        )

    def transition(
        self,
        status: ActionPlanStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str = "",
        updated_at: str | None = None,
    ) -> "ActionPlan":
        if status == self.status:
            return self
        if status not in _TRANSITIONS[self.status]:
            raise ValueError(
                f"invalid action plan transition: {self.status.value} -> {status.value}"
            )
        return replace(
            self,
            status=status,
            result=dict(result) if result is not None else self.result,
            error=error,
            updated_at=updated_at or utc_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "idempotency_key": self.idempotency_key,
            "signal_ids": list(self.signal_ids),
            "opportunity_id": self.opportunity_id,
            "capability_id": self.capability_id,
            "action_type": self.action_type,
            "decision_reason": self.decision_reason,
            "score": self.score,
            "score_components": dict(self.score_components),
            "risk": self.risk.value,
            "approval": self.approval,
            "status": self.status.value,
            "inputs": dict(self.inputs),
            "policy_ids": list(self.policy_ids),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionPlan":
        return cls(
            id=str(value["id"]),
            idempotency_key=str(value["idempotency_key"]),
            signal_ids=tuple(str(item) for item in value.get("signal_ids") or []),
            opportunity_id=str(value.get("opportunity_id") or ""),
            capability_id=str(value.get("capability_id") or ""),
            action_type=str(value.get("action_type") or ""),
            decision_reason=str(value.get("decision_reason") or ""),
            score=float(value.get("score") or 0.0),
            score_components={
                str(key): float(item)
                for key, item in dict(value.get("score_components") or {}).items()
            },
            risk=ActionRisk(str(value.get("risk") or "read_only")),
            approval=str(value.get("approval") or "not_required"),
            status=ActionPlanStatus(str(value.get("status") or "proposed")),
            inputs=dict(value.get("inputs") or {}),
            policy_ids=tuple(str(item) for item in value.get("policy_ids") or []),
            created_at=str(value["created_at"]),
            expires_at=str(value["expires_at"]) if value.get("expires_at") else None,
            updated_at=str(value.get("updated_at") or value["created_at"]),
            result=(
                dict(value["result"]) if isinstance(value.get("result"), dict) else None
            ),
            error=str(value.get("error") or ""),
        )


__all__ = [
    "ActionCandidate",
    "ActionCapability",
    "ActionPlan",
    "ActionPlanStatus",
    "ActionRisk",
]
