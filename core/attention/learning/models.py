from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from core.attention._shared import clamp01, utc_iso


class ObservationKind(StrEnum):
    OPPORTUNITY = "opportunity"
    POLICY = "policy"


@dataclass(frozen=True)
class AttentionObservation:
    """One traceable piece of evidence used to learn an attention rule."""

    id: str
    kind: ObservationKind
    rule_key: str
    statement: str
    confidence: float
    explicit: bool
    source_type: str
    source_ref: str
    observed_at: str
    payload: dict[str, Any]
    variant_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: ObservationKind,
        rule_key: str,
        variant_key: str = "",
        statement: str,
        confidence: float,
        explicit: bool,
        source_type: str,
        source_ref: str,
        payload: dict[str, Any],
        observed_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AttentionObservation":
        normalized_payload = dict(payload)
        material = json.dumps(
            {
                "kind": kind.value,
                "source_ref": source_ref,
                "rule_key": rule_key,
                "variant_key": variant_key or rule_key,
                "payload": normalized_payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
        return cls(
            id=f"obs_{digest}",
            kind=kind,
            rule_key=rule_key.strip(),
            statement=statement.strip()[:1000],
            confidence=clamp01(confidence),
            explicit=bool(explicit),
            source_type=source_type.strip() or "unknown",
            source_ref=source_ref.strip(),
            observed_at=utc_iso(observed_at),
            payload=normalized_payload,
            variant_key=(variant_key or rule_key).strip(),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "rule_key": self.rule_key,
            "variant_key": self.variant_key,
            "statement": self.statement,
            "confidence": self.confidence,
            "explicit": self.explicit,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "observed_at": self.observed_at,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttentionObservation":
        return cls(
            id=str(value["id"]),
            kind=ObservationKind(str(value["kind"])),
            rule_key=str(value.get("rule_key") or ""),
            statement=str(value.get("statement") or ""),
            confidence=clamp01(value.get("confidence")),
            explicit=bool(value.get("explicit", False)),
            source_type=str(value.get("source_type") or "unknown"),
            source_ref=str(value.get("source_ref") or ""),
            observed_at=str(value["observed_at"]),
            payload=dict(value.get("payload") or {}),
            variant_key=str(value.get("variant_key") or value.get("rule_key") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


__all__ = ["AttentionObservation", "ObservationKind"]
