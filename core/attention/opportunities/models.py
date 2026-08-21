from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from core.attention._shared import clamp01, parse_datetime, positive_int


class OpportunityKind(StrEnum):
    RECURRING = "recurring"
    EVENT = "event"
    TEMPORARY = "temporary"
    CONDITIONAL = "conditional"


class OpportunityStatus(StrEnum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    CONSUMED = "consumed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class OpportunityWindow:
    id: str
    kind: OpportunityKind
    scene: str
    available_from: str
    available_until: str
    available_minutes: int
    confidence: float
    status: OpportunityStatus
    source_pattern_id: str | None = None
    signal_ids: tuple[str, ...] = ()
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active_at(self, now: datetime) -> bool:
        if self.status != OpportunityStatus.ACTIVE:
            return False
        current = parse_datetime(now.isoformat())
        starts = parse_datetime(self.available_from)
        ends = parse_datetime(self.available_until)
        return bool(current and starts and ends and starts <= current <= ends)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "scene": self.scene,
            "available_from": self.available_from,
            "available_until": self.available_until,
            "available_minutes": self.available_minutes,
            "confidence": self.confidence,
            "status": self.status.value,
            "source_pattern_id": self.source_pattern_id,
            "signal_ids": list(self.signal_ids),
            "validation": dict(self.validation),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OpportunityWindow":
        return cls(
            id=str(value["id"]),
            kind=OpportunityKind(str(value.get("kind") or "event")),
            scene=str(value.get("scene") or "neutral"),
            available_from=str(value["available_from"]),
            available_until=str(value["available_until"]),
            available_minutes=positive_int(value.get("available_minutes"), 15),
            confidence=clamp01(value.get("confidence")),
            status=OpportunityStatus(str(value.get("status") or "active")),
            source_pattern_id=(
                str(value["source_pattern_id"])
                if value.get("source_pattern_id")
                else None
            ),
            signal_ids=tuple(str(item) for item in value.get("signal_ids") or []),
            validation=dict(value.get("validation") or {}),
            metadata=dict(value.get("metadata") or {}),
        )


__all__ = ["OpportunityKind", "OpportunityStatus", "OpportunityWindow"]
