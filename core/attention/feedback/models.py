from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from core.attention._shared import utc_iso


class FeedbackKind(StrEnum):
    ACCEPTED = "accepted"
    OPENED = "opened"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    IGNORED = "ignored"
    DISLIKED = "disliked"
    TOO_FREQUENT = "too_frequent"
    WRONG_TIME = "wrong_time"
    INACCURATE = "inaccurate"


@dataclass(frozen=True)
class AttentionFeedback:
    id: str
    plan_id: str
    kind: FeedbackKind
    created_at: str
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        plan_id: str,
        kind: FeedbackKind,
        note: str = "",
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> "AttentionFeedback":
        return cls(
            id=f"fb_{uuid.uuid4().hex}",
            plan_id=plan_id,
            kind=kind,
            created_at=utc_iso(created_at),
            note=note.strip(),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "plan_id": self.plan_id,
            "kind": self.kind.value,
            "created_at": self.created_at,
            "note": self.note,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttentionFeedback":
        return cls(
            id=str(value["id"]),
            plan_id=str(value["plan_id"]),
            kind=FeedbackKind(str(value["kind"])),
            created_at=str(value["created_at"]),
            note=str(value.get("note") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


__all__ = ["AttentionFeedback", "FeedbackKind"]
