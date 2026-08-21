from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from core.personal.models import MemoryKind, PersonalRecord

_CORE_KINDS = {
    MemoryKind.FACT,
    MemoryKind.PREFERENCE,
    MemoryKind.REQUESTED,
    MemoryKind.RELATIONSHIP,
}


@dataclass(frozen=True)
class CoreMemorySelection:
    records: tuple[PersonalRecord, ...]
    estimated_chars: int
    dropped_count: int


class PersonalCoreMemorySelector:
    """Select the small always-visible subset from the governed personal store."""

    def __init__(self, max_chars: int = 3200) -> None:
        self.max_chars = max(600, int(max_chars))

    def select(
        self,
        records: list[PersonalRecord],
        *,
        now: datetime | None = None,
    ) -> CoreMemorySelection:
        current = _aware(now or datetime.now(timezone.utc))
        candidates = [
            record
            for record in records
            if _kind(record) in _CORE_KINDS and not _expired(record, current)
        ]
        candidates.sort(
            key=lambda record: (
                _priority(record),
                record.updated_at,
                record.id,
            ),
            reverse=True,
        )

        selected: list[PersonalRecord] = []
        seen: set[str] = set()
        used = 160
        for record in candidates:
            content = _content(record)
            normalized = " ".join(content.casefold().split())
            if not normalized or normalized in seen:
                continue
            estimated = len(content) + 12
            if used + estimated > self.max_chars:
                continue
            selected.append(record)
            seen.add(normalized)
            used += estimated
        return CoreMemorySelection(
            records=tuple(selected),
            estimated_chars=used,
            dropped_count=max(0, len(candidates) - len(selected)),
        )


def _priority(record: PersonalRecord) -> float:
    kind = _kind(record)
    scores: dict[MemoryKind, float] = {
        MemoryKind.REQUESTED: 80.0,
        MemoryKind.RELATIONSHIP: 58.0,
        MemoryKind.PREFERENCE: 52.0,
        MemoryKind.FACT: 45.0,
    }
    kind_score = scores.get(kind, 0.0) if kind is not None else 0.0
    source_score = 10.0 if record.source.source in {"user", "dashboard"} else 0.0
    return (
        kind_score
        + (100.0 if record.user_locked else 0.0)
        + max(0.0, min(1.0, float(record.confidence))) * 20.0
        + (8.0 if record.last_confirmed_at else 0.0)
        + source_score
    )


def _kind(record: PersonalRecord) -> MemoryKind | None:
    try:
        return MemoryKind(str(record.data.get("kind") or ""))
    except ValueError:
        return None


def _content(record: PersonalRecord) -> str:
    return str(record.data.get("content") or record.summary or record.title).strip()


def _expired(record: PersonalRecord, now: datetime) -> bool:
    if not record.expires_at:
        return False
    raw = record.expires_at
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        expires_at = datetime.fromisoformat(raw)
    except ValueError:
        return False
    return _aware(expires_at) <= now


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
