from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from core.personal.models import PersonalEntityType, PersonalRecord, RecordStatus
from core.personal.service import PersonalDataService

_TODAY_TYPES = {
    PersonalEntityType.COMMITMENT,
    PersonalEntityType.DAILY_PLAN,
    PersonalEntityType.CALENDAR_EVENT,
    PersonalEntityType.HEALTH_OBSERVATION,
    PersonalEntityType.CHECK_IN,
}
_DONE_STATES = {"done", "completed", "cancelled", "closed", "inactive"}


def _parse_date(value: Any, timezone: ZoneInfo) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if len(raw) == 10:
            return date.fromisoformat(raw)
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone)
        return parsed.astimezone(timezone).date()
    except ValueError:
        return None


@dataclass(frozen=True)
class PersonalTodayResult:
    local_date: str
    timezone: str
    records: tuple[PersonalRecord, ...]
    counts: dict[str, int]
    overdue_count: int
    sources: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.local_date,
            "timezone": self.timezone,
            "records": [record.to_dict() for record in self.records],
            "counts": self.counts,
            "overdue_count": self.overdue_count,
            "sources": self.sources,
        }


class PersonalTodayService:
    """Date-scoped read model over the canonical personal record store."""

    def __init__(self, personal_data: PersonalDataService) -> None:
        self.personal_data = personal_data

    def get(
        self,
        *,
        local_date: str,
        timezone_name: str,
        now: datetime | None = None,
    ) -> PersonalTodayResult:
        del now
        timezone = ZoneInfo(timezone_name)
        target = date.fromisoformat(local_date)
        selected: list[tuple[date | None, PersonalRecord]] = []
        overdue = 0
        for record in self.personal_data.list(
            statuses=[RecordStatus.ACTIVE],
            limit=1000,
        ):
            if record.entity_type not in _TODAY_TYPES:
                continue
            data = record.data
            state = str(data.get("state") or data.get("status") or "").lower()
            if state in _DONE_STATES:
                continue
            record_date = self._record_date(record, timezone)
            if record.entity_type == PersonalEntityType.COMMITMENT:
                if record_date is not None and record_date > target:
                    continue
                if record_date is not None and record_date < target:
                    overdue += 1
            elif record_date != target:
                continue
            selected.append((record_date, record))
        selected.sort(
            key=lambda item: (
                item[0] is None,
                item[0] or target,
                item[1].title,
            )
        )
        records = tuple(record for _, record in selected)
        counts: dict[str, int] = {}
        sources: dict[str, int] = {}
        for record in records:
            counts[record.entity_type.value] = counts.get(record.entity_type.value, 0) + 1
            sources[record.source.source] = sources.get(record.source.source, 0) + 1
        return PersonalTodayResult(
            local_date=target.isoformat(),
            timezone=timezone_name,
            records=records,
            counts=counts,
            overdue_count=overdue,
            sources=sources,
        )

    @staticmethod
    def _record_date(record: PersonalRecord, timezone: ZoneInfo) -> date | None:
        fields = (
            ("plan_date",)
            if record.entity_type == PersonalEntityType.DAILY_PLAN
            else ("due_at", "start_at", "observed_at", "date")
        )
        for field in fields:
            parsed = _parse_date(record.data.get(field), timezone)
            if parsed is not None:
                return parsed
        return None
