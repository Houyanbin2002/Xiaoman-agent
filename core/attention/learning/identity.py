from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.attention.learning.models import ObservationKind


@dataclass(frozen=True)
class AttentionRuleIdentity:
    """Stable identity for one logical rule slot and one concrete version."""

    slot_key: str
    variant_key: str


def build_rule_identity(
    kind: ObservationKind,
    payload: Mapping[str, Any],
) -> AttentionRuleIdentity:
    if kind == ObservationKind.OPPORTUNITY:
        recurrence = dict(payload.get("recurrence") or {})
        slot = {
            "kind": str(payload.get("kind") or "availability_pattern"),
            "scene": str(payload.get("scene") or "neutral"),
            "timezone": str(recurrence.get("timezone") or "Asia/Shanghai"),
            "days": sorted(str(item) for item in recurrence.get("days") or ()),
        }
        variant = {
            **slot,
            "start": str(recurrence.get("start") or "00:00"),
            "end": str(recurrence.get("end") or "00:30"),
            "available_minutes": int(payload.get("available_minutes") or 15),
        }
    else:
        slot = {
            "scope": _canonical(payload.get("scope") or {}),
            "conditions": _canonical(payload.get("conditions") or {}),
        }
        variant = {
            **slot,
            "effect": str(payload.get("effect") or "adjust_score"),
            "priority": int(payload.get("priority") or 50),
            "score_adjustment": float(payload.get("score_adjustment") or 0.0),
            "metadata": _canonical(payload.get("metadata") or {}),
        }
    return AttentionRuleIdentity(
        slot_key=_digest(kind.value, "slot", slot),
        variant_key=_digest(kind.value, "variant", variant),
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = [_canonical(item) for item in value]
        return sorted(normalized, key=_sort_key)
    return value


def _sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(kind: str, level: str, value: Mapping[str, Any]) -> str:
    material = json.dumps(
        {"kind": kind, "level": level, "value": value},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


__all__ = ["AttentionRuleIdentity", "build_rule_identity"]
