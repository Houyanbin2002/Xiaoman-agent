from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from core.attention._shared import parse_datetime, utc_iso


class PolicyEffect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    ADJUST_SCORE = "adjust_score"
    DEFER = "defer"
    LIMIT_FREQUENCY = "limit_frequency"


class PolicyStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REJECTED = "rejected"
    EXPIRED = "expired"


_SCOPE_FIELDS = frozenset(
    {"domain", "action_type", "capability_id", "risk", "scene", "channel"}
)
_CONDITION_FIELDS = frozenset(
    {
        "severity_min",
        "severity_max",
        "confidence_min",
        "focus_active",
        "do_not_disturb",
        "scene",
    }
)


@dataclass(frozen=True)
class DecisionContext:
    now: datetime
    scene: str = "neutral"
    focus_active: bool = False
    do_not_disturb: bool = False
    allow_high_priority: bool = True
    channel: str = "dashboard"
    permission_mode: str = "ask"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyRule:
    id: str
    scope: dict[str, Any]
    conditions: dict[str, Any]
    effect: PolicyEffect
    priority: int
    score_adjustment: float
    version: int
    enabled: bool
    status: PolicyStatus
    confidence: float
    observation_count: int
    last_observed_at: str | None
    effective_from: str | None
    expires_at: str | None
    source: str
    user_locked: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown_scope = set(self.scope) - _SCOPE_FIELDS
        unknown_conditions = {
            key
            for key in self.conditions
            if key not in _CONDITION_FIELDS and not key.startswith("attribute.")
        }
        if unknown_scope:
            raise ValueError(
                f"unsupported policy scope fields: {sorted(unknown_scope)}"
            )
        if unknown_conditions:
            raise ValueError(
                "unsupported policy condition fields: " f"{sorted(unknown_conditions)}"
            )
        if self.effect == PolicyEffect.ALLOW and self.source != "user":
            raise ValueError("only user-authored policies may use allow")

    @classmethod
    def create(
        cls,
        *,
        effect: PolicyEffect,
        scope: dict[str, Any] | None = None,
        conditions: dict[str, Any] | None = None,
        priority: int = 50,
        score_adjustment: float = 0.0,
        version: int = 1,
        enabled: bool = True,
        status: PolicyStatus | None = None,
        confidence: float = 1.0,
        observation_count: int = 1,
        last_observed_at: datetime | None = None,
        effective_from: datetime | None = None,
        expires_at: datetime | None = None,
        source: str = "user",
        user_locked: bool = False,
        metadata: dict[str, Any] | None = None,
        policy_id: str | None = None,
    ) -> "PolicyRule":
        resolved_status = status or (
            PolicyStatus.ACTIVE if source == "user" else PolicyStatus.PROPOSED
        )
        return cls(
            id=policy_id or f"pol_{uuid.uuid4().hex}",
            scope=dict(scope or {}),
            conditions=dict(conditions or {}),
            effect=effect,
            priority=max(0, min(int(priority), 1000)),
            score_adjustment=float(score_adjustment),
            version=max(1, int(version)),
            enabled=bool(enabled),
            status=resolved_status,
            confidence=max(0.0, min(float(confidence), 1.0)),
            observation_count=max(1, int(observation_count)),
            last_observed_at=(
                utc_iso(last_observed_at) if last_observed_at is not None else None
            ),
            effective_from=(
                utc_iso(effective_from) if effective_from is not None else None
            ),
            expires_at=utc_iso(expires_at) if expires_at is not None else None,
            source=source.strip() or "user",
            user_locked=bool(user_locked),
            metadata=dict(metadata or {}),
        )

    def is_active_at(self, now: datetime) -> bool:
        if not self.enabled or self.status != PolicyStatus.ACTIVE:
            return False
        current = parse_datetime(now.isoformat())
        starts = parse_datetime(self.effective_from)
        ends = parse_datetime(self.expires_at)
        if current is None or (starts is not None and current < starts):
            return False
        return ends is None or current <= ends

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": dict(self.scope),
            "conditions": dict(self.conditions),
            "effect": self.effect.value,
            "priority": self.priority,
            "score_adjustment": self.score_adjustment,
            "version": self.version,
            "enabled": self.enabled,
            "status": self.status.value,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "last_observed_at": self.last_observed_at,
            "effective_from": self.effective_from,
            "expires_at": self.expires_at,
            "source": self.source,
            "user_locked": self.user_locked,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PolicyRule":
        return cls(
            id=str(value["id"]),
            scope=dict(value.get("scope") or {}),
            conditions=dict(value.get("conditions") or {}),
            effect=PolicyEffect(str(value.get("effect") or "allow")),
            priority=int(value.get("priority") or 0),
            score_adjustment=float(value.get("score_adjustment") or 0.0),
            version=max(1, int(value.get("version") or 1)),
            enabled=bool(value.get("enabled", True)),
            status=PolicyStatus(
                str(
                    value.get("status")
                    or (
                        "active"
                        if str(value.get("source") or "user") == "user"
                        else "proposed"
                    )
                )
            ),
            confidence=max(0.0, min(float(value.get("confidence", 1.0)), 1.0)),
            observation_count=max(1, int(value.get("observation_count") or 1)),
            last_observed_at=(
                str(value["last_observed_at"])
                if value.get("last_observed_at")
                else None
            ),
            effective_from=(
                str(value["effective_from"]) if value.get("effective_from") else None
            ),
            expires_at=str(value["expires_at"]) if value.get("expires_at") else None,
            source=str(value.get("source") or "user"),
            user_locked=bool(value.get("user_locked", False)),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool = True
    require_approval: bool = False
    deferred: bool = False
    score_adjustment: float = 0.0
    reasons: tuple[str, ...] = ()
    matched_policy_ids: tuple[str, ...] = ()


__all__ = [
    "DecisionContext",
    "PolicyDecision",
    "PolicyEffect",
    "PolicyRule",
    "PolicyStatus",
]
