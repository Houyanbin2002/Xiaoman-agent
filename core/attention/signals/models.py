from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from core.attention._shared import clamp01, parse_datetime, positive_int, utc_iso


class SignalValence(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass(frozen=True)
class SignalSource:
    type: str
    name: str
    reference: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "name": self.name,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SignalSource":
        return cls(
            type=str(value.get("type") or "unknown"),
            name=str(value.get("name") or "unknown"),
            reference=str(value.get("reference") or ""),
        )


@dataclass(frozen=True)
class AttentionSignal:
    id: str
    kind: str
    domain: str
    occurred_at: str
    expires_at: str | None
    valence: SignalValence
    severity: float
    urgency: float
    actionability: float
    confidence: float
    freshness: float
    estimated_attention_minutes: int
    risk_domain: str
    summary: str
    source: SignalSource
    evidence: tuple[dict[str, Any], ...] = ()
    suggested_capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("signal id must not be empty")
        if not self.kind.strip() or "." not in self.kind:
            raise ValueError("signal kind must be a namespaced value")
        if not self.domain.strip():
            raise ValueError("signal domain must not be empty")
        if parse_datetime(self.occurred_at) is None:
            raise ValueError("signal occurred_at must be timezone-aware ISO time")
        if self.expires_at and parse_datetime(self.expires_at) is None:
            raise ValueError("signal expires_at must be timezone-aware ISO time")

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        domain: str,
        summary: str,
        source: SignalSource,
        occurred_at: datetime | None = None,
        expires_at: datetime | None = None,
        valence: SignalValence = SignalValence.NEUTRAL,
        severity: float = 0.0,
        urgency: float = 0.0,
        actionability: float = 0.5,
        confidence: float = 0.7,
        freshness: float = 1.0,
        estimated_attention_minutes: int = 5,
        risk_domain: str = "general",
        evidence: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        suggested_capabilities: list[str] | tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        signal_id: str | None = None,
    ) -> "AttentionSignal":
        return cls(
            id=signal_id or f"sig_{uuid.uuid4().hex}",
            kind=kind.strip(),
            domain=domain.strip(),
            occurred_at=utc_iso(occurred_at),
            expires_at=utc_iso(expires_at) if expires_at is not None else None,
            valence=valence,
            severity=clamp01(severity),
            urgency=clamp01(urgency),
            actionability=clamp01(actionability),
            confidence=clamp01(confidence),
            freshness=clamp01(freshness, 1.0),
            estimated_attention_minutes=positive_int(
                estimated_attention_minutes,
                5,
            ),
            risk_domain=risk_domain.strip() or "general",
            summary=summary.strip(),
            source=source,
            evidence=tuple(evidence),
            suggested_capabilities=tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in suggested_capabilities
                    if str(item).strip()
                )
            ),
            metadata=dict(metadata or {}),
        )

    def is_active_at(self, now: datetime) -> bool:
        current = parse_datetime(now.isoformat())
        occurred = parse_datetime(self.occurred_at)
        expires = parse_datetime(self.expires_at)
        if current is None or occurred is None or occurred > current:
            return False
        return expires is None or expires >= current

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "domain": self.domain,
            "occurred_at": self.occurred_at,
            "expires_at": self.expires_at,
            "valence": self.valence.value,
            "severity": self.severity,
            "urgency": self.urgency,
            "actionability": self.actionability,
            "confidence": self.confidence,
            "freshness": self.freshness,
            "estimated_attention_minutes": self.estimated_attention_minutes,
            "risk_domain": self.risk_domain,
            "summary": self.summary,
            "source": self.source.to_dict(),
            "evidence": list(self.evidence),
            "suggested_capabilities": list(self.suggested_capabilities),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttentionSignal":
        return cls(
            id=str(value["id"]),
            kind=str(value["kind"]),
            domain=str(value.get("domain") or "general"),
            occurred_at=str(value["occurred_at"]),
            expires_at=(str(value["expires_at"]) if value.get("expires_at") else None),
            valence=SignalValence(str(value.get("valence") or "neutral")),
            severity=clamp01(value.get("severity")),
            urgency=clamp01(value.get("urgency")),
            actionability=clamp01(value.get("actionability"), 0.5),
            confidence=clamp01(value.get("confidence"), 0.7),
            freshness=clamp01(value.get("freshness"), 1.0),
            estimated_attention_minutes=positive_int(
                value.get("estimated_attention_minutes"),
                5,
            ),
            risk_domain=str(value.get("risk_domain") or "general"),
            summary=str(value.get("summary") or ""),
            source=SignalSource.from_dict(dict(value.get("source") or {})),
            evidence=tuple(dict(item) for item in value.get("evidence") or []),
            suggested_capabilities=tuple(
                str(item) for item in value.get("suggested_capabilities") or []
            ),
            metadata=dict(value.get("metadata") or {}),
        )


__all__ = ["AttentionSignal", "SignalSource", "SignalValence"]
