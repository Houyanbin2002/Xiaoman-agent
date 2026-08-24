from __future__ import annotations

from dataclasses import dataclass

from core.personal.models import PersonalRecord


@dataclass(frozen=True)
class PersonalRecordChanged:
    record: PersonalRecord
    change: str
    actor: str
