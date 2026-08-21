from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Protocol, runtime_checkable

from core.personal.models import (
    AccessPolicy,
    MemoryKind,
    PersonalRecord,
    RecordStatus,
)


@dataclass(frozen=True)
class PersonalRecallSignals:
    semantic: float = 0.0
    keyword: float = 0.0
    entity_context: float = 0.0
    use_success: float = 0.0


@dataclass(frozen=True)
class PersonalMemoryHit:
    record: PersonalRecord
    score: float
    signals: PersonalRecallSignals


@dataclass(frozen=True)
class PersonalMemoryQueryResult:
    text_block: str = ""
    hits: tuple[PersonalMemoryHit, ...] = ()


@runtime_checkable
class PersonalMemoryRetrievalApi(Protocol):
    def retrieve_personal_memory(
        self,
        query: str,
        *,
        limit: int = 6,
        context_tags: set[str] | None = None,
        now: datetime | None = None,
    ) -> PersonalMemoryQueryResult: ...


class GovernedPersonalMemoryRetriever:
    """Type-aware ranking over the canonical personal memory records.

    Candidate generation can later provide vector similarities, while governance,
    expiry and type-specific time semantics remain deterministic here.
    """

    def retrieve(
        self,
        records: list[PersonalRecord],
        *,
        query: str,
        semantic_scores: dict[str, float] | None = None,
        context_tags: set[str] | None = None,
        limit: int = 6,
        now: datetime | None = None,
    ) -> list[PersonalMemoryHit]:
        current = _aware(now or datetime.now(timezone.utc))
        query_terms = _terms(query)
        wanted_tags = {item.casefold() for item in (context_tags or set())}
        semantic = semantic_scores or {}
        hits: list[PersonalMemoryHit] = []
        for record in records:
            if not _recallable(record, current):
                continue
            text = _content(record)
            keyword_score = _keyword_score(query_terms, _terms(text))
            record_tags = {
                str(item).casefold()
                for item in record.data.get("tags", [])
                if str(item).strip()
            }
            entity_score = (
                len(wanted_tags & record_tags) / max(1, len(wanted_tags))
                if wanted_tags
                else 0.0
            )
            success_count = _int(record.data.get("recall_success_count"))
            failure_count = _int(record.data.get("recall_failure_count"))
            use_success = (success_count + 1.0) / (success_count + failure_count + 2.0)
            signals = PersonalRecallSignals(
                semantic=_clamp(semantic.get(record.id, 0.0)),
                keyword=keyword_score,
                entity_context=entity_score,
                use_success=use_success,
            )
            score = _rank(record, signals, current)
            if score <= 0.0 or (
                signals.semantic <= 0.0
                and signals.keyword <= 0.0
                and signals.entity_context <= 0.0
            ):
                continue
            hits.append(PersonalMemoryHit(record=record, score=score, signals=signals))
        hits.sort(key=lambda item: (item.score, item.record.updated_at), reverse=True)
        return hits[: max(1, int(limit))]


def _rank(
    record: PersonalRecord,
    signals: PersonalRecallSignals,
    now: datetime,
) -> float:
    kind = _kind(record)
    importance_by_kind: dict[MemoryKind, float] = {
        MemoryKind.REQUESTED: 1.0,
        MemoryKind.RELATIONSHIP: 0.8,
        MemoryKind.PREFERENCE: 0.8,
        MemoryKind.FACT: 0.7,
        MemoryKind.HISTORICAL_EVENT: 0.55,
        MemoryKind.EPISODE: 0.45,
        MemoryKind.TEMPORARY_STATE: 0.5,
    }
    importance = importance_by_kind.get(kind, 0.0) if kind is not None else 0.0
    if record.user_locked:
        importance = 1.0
    temporal = _temporal_score(record, kind, now)
    return _clamp(
        0.35 * signals.semantic
        + 0.2 * signals.keyword
        + 0.15 * signals.entity_context
        + 0.1 * importance
        + 0.1 * _clamp(record.confidence)
        + 0.05 * signals.use_success
        + 0.05 * temporal
    )


def _temporal_score(
    record: PersonalRecord,
    kind: MemoryKind | None,
    now: datetime,
) -> float:
    if kind in {
        MemoryKind.FACT,
        MemoryKind.PREFERENCE,
        MemoryKind.REQUESTED,
        MemoryKind.RELATIONSHIP,
    }:
        return 1.0
    updated_at = _parse_datetime(record.updated_at)
    if updated_at is None:
        return 0.5
    age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
    half_life = 14.0 if kind is MemoryKind.TEMPORARY_STATE else 180.0
    return math.exp(-math.log(2.0) * age_days / half_life)


def _recallable(record: PersonalRecord, now: datetime) -> bool:
    if record.status is not RecordStatus.ACTIVE:
        return False
    if record.access_policy in {AccessPolicy.CONFIRM_READ, AccessPolicy.OWNER_ONLY}:
        return False
    if _kind(record) is MemoryKind.PROCEDURE:
        return False
    expires_at = _parse_datetime(record.expires_at)
    return expires_at is None or expires_at > now


def _keyword_score(query_terms: set[str], record_terms: set[str]) -> float:
    if not query_terms or not record_terms:
        return 0.0
    overlap = query_terms & record_terms
    return len(overlap) / max(1, len(query_terms))


def _terms(text: str) -> set[str]:
    normalized = str(text or "").casefold()
    terms = set(re.findall(r"[a-z0-9_.-]{2,}", normalized))
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        if len(chunk) <= 4:
            terms.add(chunk)
        terms.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return terms


def _kind(record: PersonalRecord) -> MemoryKind | None:
    try:
        return MemoryKind(str(record.data.get("kind") or ""))
    except ValueError:
        return None


def _content(record: PersonalRecord) -> str:
    return str(record.data.get("content") or record.summary or record.title)


def _parse_datetime(raw: str | None) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return _aware(datetime.fromisoformat(text))
    except ValueError:
        return None


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _int(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
