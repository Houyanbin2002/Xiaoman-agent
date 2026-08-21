from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.attention._shared import clamp01, parse_datetime, positive_int, utc_iso

_DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class PatternStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PatternSource(StrEnum):
    USER = "user"
    LEARNED = "learned"
    IMPORTED = "imported"


@dataclass(frozen=True)
class RecurrenceSpec:
    timezone: str
    days: tuple[str, ...]
    start: str
    end: str

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"invalid recurrence timezone: {self.timezone}") from exc
        if not self.days or any(day not in _DAY_NAMES for day in self.days):
            raise ValueError("recurrence days must use mon..sun")
        self._parse_clock(self.start)
        self._parse_clock(self.end)

    @staticmethod
    def _parse_clock(value: str) -> time:
        try:
            return time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"invalid recurrence clock: {value}") from exc

    def interval_containing(self, now: datetime) -> tuple[datetime, datetime] | None:
        zone = ZoneInfo(self.timezone)
        local = now.astimezone(zone)
        start_clock = self._parse_clock(self.start)
        end_clock = self._parse_clock(self.end)
        candidates = (local.date(), local.date() - timedelta(days=1))
        for anchor in candidates:
            if _DAY_NAMES[anchor.weekday()] not in self.days:
                continue
            start_local = datetime.combine(anchor, start_clock, tzinfo=zone)
            end_date = anchor if end_clock > start_clock else anchor + timedelta(days=1)
            end_local = datetime.combine(end_date, end_clock, tzinfo=zone)
            if start_local <= local <= end_local:
                return (
                    start_local.astimezone(timezone.utc),
                    end_local.astimezone(timezone.utc),
                )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timezone": self.timezone,
            "days": list(self.days),
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecurrenceSpec":
        return cls(
            timezone=str(value.get("timezone") or "Asia/Shanghai"),
            days=tuple(str(item).lower() for item in value.get("days") or ()),
            start=str(value.get("start") or "00:00"),
            end=str(value.get("end") or "00:30"),
        )


@dataclass(frozen=True)
class BehaviorPattern:
    id: str
    kind: str
    scene: str
    recurrence: RecurrenceSpec
    available_minutes: int
    confidence: float
    observation_count: int
    source: PatternSource
    status: PatternStatus
    last_observed_at: str | None
    valid_from: str | None
    expires_at: str | None
    user_locked: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        scene: str,
        recurrence: RecurrenceSpec,
        available_minutes: int,
        confidence: float,
        source: PatternSource,
        observation_count: int = 1,
        status: PatternStatus | None = None,
        last_observed_at: datetime | None = None,
        valid_from: datetime | None = None,
        expires_at: datetime | None = None,
        user_locked: bool = False,
        metadata: dict[str, Any] | None = None,
        pattern_id: str | None = None,
    ) -> "BehaviorPattern":
        resolved_status = status or (
            PatternStatus.ACTIVE
            if source == PatternSource.USER or confidence >= 0.75
            else PatternStatus.PROPOSED
        )
        return cls(
            id=pattern_id or f"pat_{uuid.uuid4().hex}",
            kind=kind.strip() or "availability_pattern",
            scene=scene.strip() or "neutral",
            recurrence=recurrence,
            available_minutes=positive_int(available_minutes, 15),
            confidence=clamp01(confidence),
            observation_count=max(1, int(observation_count)),
            source=source,
            status=resolved_status,
            last_observed_at=(
                utc_iso(last_observed_at) if last_observed_at is not None else None
            ),
            valid_from=utc_iso(valid_from) if valid_from is not None else None,
            expires_at=utc_iso(expires_at) if expires_at is not None else None,
            user_locked=bool(user_locked),
            metadata=dict(metadata or {}),
        )

    def is_eligible_at(self, now: datetime) -> bool:
        if self.status != PatternStatus.ACTIVE:
            return False
        current = parse_datetime(now.isoformat())
        valid_from = parse_datetime(self.valid_from)
        expires = parse_datetime(self.expires_at)
        if current is None or (valid_from is not None and current < valid_from):
            return False
        return expires is None or expires >= current

    def interval_containing(self, now: datetime) -> tuple[datetime, datetime] | None:
        if not self.is_eligible_at(now):
            return None
        return self.recurrence.interval_containing(now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "scene": self.scene,
            "recurrence": self.recurrence.to_dict(),
            "available_minutes": self.available_minutes,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "source": self.source.value,
            "status": self.status.value,
            "last_observed_at": self.last_observed_at,
            "valid_from": self.valid_from,
            "expires_at": self.expires_at,
            "user_locked": self.user_locked,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorPattern":
        return cls(
            id=str(value["id"]),
            kind=str(value.get("kind") or "availability_pattern"),
            scene=str(value.get("scene") or "neutral"),
            recurrence=RecurrenceSpec.from_dict(dict(value.get("recurrence") or {})),
            available_minutes=positive_int(value.get("available_minutes"), 15),
            confidence=clamp01(value.get("confidence")),
            observation_count=max(1, int(value.get("observation_count") or 1)),
            source=PatternSource(str(value.get("source") or "learned")),
            status=PatternStatus(str(value.get("status") or "proposed")),
            last_observed_at=(
                str(value["last_observed_at"])
                if value.get("last_observed_at")
                else None
            ),
            valid_from=str(value["valid_from"]) if value.get("valid_from") else None,
            expires_at=str(value["expires_at"]) if value.get("expires_at") else None,
            user_locked=bool(value.get("user_locked", False)),
            metadata=dict(value.get("metadata") or {}),
        )


__all__ = [
    "BehaviorPattern",
    "PatternSource",
    "PatternStatus",
    "RecurrenceSpec",
]
