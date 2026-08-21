from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from core.memory.personal_core import CoreMemorySelection, PersonalCoreMemorySelector
from core.memory.personal_retrieval import (
    GovernedPersonalMemoryRetriever,
    PersonalMemoryHit,
    PersonalMemoryQueryResult,
)
from core.personal.governance import MemoryGovernanceService
from core.personal.models import (
    AccessPolicy,
    DataCategory,
    MemoryData,
    MemoryKind,
    PersonalRecord,
    RecordSource,
)

_KIND_LABELS: tuple[tuple[MemoryKind, str], ...] = (
    (MemoryKind.FACT, "用户事实"),
    (MemoryKind.PREFERENCE, "用户偏好"),
    (MemoryKind.REQUESTED, "用户明确要求长期记住的关键内容"),
    (MemoryKind.RELATIONSHIP, "关系记忆"),
    (MemoryKind.HISTORICAL_EVENT, "重要经历"),
    (MemoryKind.TEMPORARY_STATE, "临时状态"),
)

_TAG_KIND = {
    "identity": MemoryKind.FACT,
    "preference": MemoryKind.PREFERENCE,
    "relationship": MemoryKind.RELATIONSHIP,
    "long_term_health": MemoryKind.FACT,
    "project_context": MemoryKind.FACT,
    "correction": MemoryKind.FACT,
}


@dataclass(frozen=True)
class LongTermMemorySyncResult:
    created: int = 0
    unchanged: int = 0
    conflicts: int = 0
    skipped: int = 0
    expired: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "created": self.created,
            "unchanged": self.unchanged,
            "conflicts": self.conflicts,
            "skipped": self.skipped,
            "expired": self.expired,
        }


class GovernedLongTermMemory:
    """Canonical long-term memory backed by governed personal records."""

    def __init__(
        self,
        *,
        governance: MemoryGovernanceService,
        core_max_chars: int = 3200,
    ) -> None:
        self.governance = governance
        self._core_selector = PersonalCoreMemorySelector(core_max_chars)
        self._personal_retriever = GovernedPersonalMemoryRetriever()

    def render_prompt(self) -> str:
        records = self._prompt_visible_records()
        return self._render_records(records) if records else ""

    def core_selection(self) -> CoreMemorySelection:
        self.governance.personal_data.expire_due()
        return self._core_selector.select(self._readable_records())

    def retrieve_personal_memory(
        self,
        query: str,
        *,
        limit: int = 6,
        context_tags: set[str] | None = None,
        now: datetime | None = None,
    ) -> PersonalMemoryQueryResult:
        return self._search_personal_memory(
            query,
            limit=limit,
            context_tags=context_tags,
            now=now,
            exclude_core=True,
        )

    def search_personal_memory(
        self,
        query: str,
        *,
        limit: int = 6,
        context_tags: set[str] | None = None,
        now: datetime | None = None,
        semantic_scores: Mapping[str, float] | None = None,
    ) -> PersonalMemoryQueryResult:
        """Search all governed personal memory, including the always-visible core."""

        return self._search_personal_memory(
            query,
            limit=limit,
            context_tags=context_tags,
            now=now,
            exclude_core=False,
            semantic_scores=semantic_scores,
        )

    def personal_memory_records(self) -> list[PersonalRecord]:
        """Return governance-filtered records for the semantic side index."""

        self.governance.personal_data.expire_due()
        return self._readable_records()

    def remember_explicit(
        self,
        summary: str,
        *,
        memory_kind: str = "",
        source_ref: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[str, PersonalRecord | None]:
        """Persist an explicit user memory in the canonical governed store."""

        content = summary.strip()
        if not content:
            return "empty", None
        kind = {
            "profile": MemoryKind.FACT,
            "fact": MemoryKind.FACT,
            "preference": MemoryKind.PREFERENCE,
            "event": MemoryKind.HISTORICAL_EVENT,
            "relationship": MemoryKind.RELATIONSHIP,
            "temporary": MemoryKind.TEMPORARY_STATE,
            "requested": MemoryKind.REQUESTED,
        }.get(memory_kind.strip().lower(), MemoryKind.REQUESTED)
        raw_metadata = dict(metadata or {})
        raw_attributes = raw_metadata.get("attributes")
        attributes = (
            {str(key): value for key, value in raw_attributes.items()}
            if isinstance(raw_attributes, Mapping)
            else {}
        )
        expires_at = str(raw_metadata.get("expires_at") or "").strip() or None
        result = self.governance.propose(
            memory=MemoryData(
                kind=kind,
                content=content,
                tags=["explicit", "user-confirmed"],
                category=DataCategory.GENERAL,
                subject=str(raw_metadata.get("subject") or "").strip(),
                predicate=str(raw_metadata.get("predicate") or "").strip(),
                value=str(raw_metadata.get("value") or "").strip(),
                scope=str(raw_metadata.get("scope") or "").strip(),
                attributes=attributes,
            ),
            summary=content,
            source=RecordSource(
                "explicit_memory",
                source_ref.strip() or "memorize_tool",
            ),
            confidence=1.0,
            data_category=DataCategory.GENERAL,
            valid_from=str(raw_metadata.get("valid_from") or "").strip() or None,
            expires_at=expires_at,
            actor="user",
            user_confirmed=True,
            explicit_replaces=bool(raw_metadata.get("replaces")),
        )
        return result.status, result.record

    def forget_explicit(
        self,
        ids: Sequence[str],
    ) -> tuple[list[str], list[str], list[dict[str, object]]]:
        """Forget canonical personal memories without affecting other domains."""

        affected: list[str] = []
        missing: list[str] = []
        items: list[dict[str, object]] = []
        for item_id in dict.fromkeys(
            str(item).strip() for item in ids if str(item).strip()
        ):
            record = self.governance.personal_data.get(item_id)
            if record is None or record.entity_type.value != "memory":
                missing.append(item_id)
                continue
            forgotten = self.governance.forget(
                item_id,
                actor="user",
                reason="explicit forget_memory request",
            )
            affected.append(item_id)
            items.append(forgotten.to_dict())
        return affected, missing, items

    def _search_personal_memory(
        self,
        query: str,
        *,
        limit: int,
        context_tags: set[str] | None,
        now: datetime | None,
        exclude_core: bool,
        semantic_scores: Mapping[str, float] | None = None,
    ) -> PersonalMemoryQueryResult:
        _ = self.governance.personal_data.expire_due()
        readable = self._readable_records()
        core_ids: set[str] = (
            {
                record.id
                for record in self._core_selector.select(readable, now=now).records
            }
            if exclude_core
            else set()
        )
        hits = [
            hit
            for hit in self._personal_retriever.retrieve(
                readable,
                query=query,
                semantic_scores=dict(semantic_scores or {}),
                context_tags=context_tags,
                limit=max(limit * 2, 8),
                now=now,
            )
            if hit.record.id not in core_ids
        ][: max(1, limit)]
        if not hits:
            return PersonalMemoryQueryResult()
        lines = [
            "## 【相关个人记忆】",
            "以下信息来自可治理的个人记忆，仅在与当前请求相关时使用；不能覆盖用户本轮表达。",
        ]
        used = sum(len(line) for line in lines)
        included: list[PersonalMemoryHit] = []
        for hit in hits:
            content = str(
                hit.record.data.get("content") or hit.record.summary or hit.record.title
            ).strip()
            line = f"- {content}（可信度 {round(hit.record.confidence * 100)}%）"
            if not content or used + len(line) > 1000:
                continue
            lines.append(line)
            included.append(hit)
            used += len(line)
        if not included:
            return PersonalMemoryQueryResult()
        return PersonalMemoryQueryResult(
            text_block="\n".join(lines),
            hits=tuple(included),
        )

    def optimize(self) -> LongTermMemorySyncResult:
        """Run deterministic lifecycle work without rewriting a second truth source."""

        expired = len(self.governance.personal_data.expire_due())
        return LongTermMemorySyncResult(expired=expired)

    def ingest_candidates(
        self,
        candidates: Sequence[Mapping[str, object]],
        *,
        source_ref: str,
        source: str = "conversation_consolidation",
    ) -> LongTermMemorySyncResult:
        created = unchanged = conflicts = skipped = 0
        active = self.governance.list_memories(limit=10000)

        for index, raw in enumerate(candidates):
            tag = str(raw.get("tag") or "").strip().lower()
            content = str(raw.get("content") or "").strip()
            if tag not in _TAG_KIND or not content:
                skipped += 1
                continue
            if self._find_exact(active, content) is not None:
                unchanged += 1
                continue

            kind = _TAG_KIND[tag]
            category = self._category_for_tag(tag)
            replaces = str(raw.get("replaces") or "").strip()
            replaced = (
                self._find_replaced(active, replaces) if tag == "correction" else None
            )
            record_key = replaced.record_key if replaced is not None else ""
            access_policy = (
                AccessPolicy.CONFIRM_WRITE
                if tag == "correction" and replaced is None
                else None
            )
            source_message_id = str(raw.get("source_message_id") or "").strip()
            candidate_ref = (
                f"{source_ref}#message:{source_message_id}"
                if source_message_id
                else f"{source_ref}#candidate:{index + 1}"
            )
            confidence = self._confidence(raw.get("confidence"), tag=tag)
            origin = str(raw.get("origin") or "").strip().lower()
            user_confirmed = bool(
                source_message_id
                and raw.get("_user_evidence_verified") is True
                and origin in {"explicit_user", "user_correction"}
            )
            actor = "user" if user_confirmed else "assistant"

            result = self.governance.propose(
                memory=MemoryData(
                    kind=kind,
                    content=content,
                    tags=[tag, "auto-extracted"],
                    category=category,
                    subject=str(raw.get("subject") or "").strip(),
                    predicate=str(raw.get("predicate") or "").strip(),
                    value=str(raw.get("value") or "").strip(),
                    scope=str(raw.get("scope") or "").strip(),
                    attributes=(
                        {str(key): value for key, value in raw["attributes"].items()}
                        if isinstance(raw.get("attributes"), Mapping)
                        else {}
                    ),
                ),
                summary=content,
                source=RecordSource(source, candidate_ref),
                record_key=record_key,
                confidence=confidence,
                data_category=category,
                access_policy=access_policy,
                valid_from=str(raw.get("valid_from") or "").strip() or None,
                expires_at=str(raw.get("expires_at") or "").strip() or None,
                actor=actor,
                user_confirmed=user_confirmed,
                explicit_replaces=bool(replaces),
            )
            if result.status in {"created", "superseded"}:
                created += 1
                if result.record is not None:
                    if result.record.supersedes_id:
                        active = [
                            record
                            for record in active
                            if record.id != result.record.supersedes_id
                        ]
                    active.append(result.record)
            elif result.status == "unchanged":
                unchanged += 1
            elif result.status == "conflict_pending":
                conflicts += 1
            else:
                skipped += 1

        return LongTermMemorySyncResult(
            created=created,
            unchanged=unchanged,
            conflicts=conflicts,
            skipped=skipped,
        )

    def _prompt_visible_records(self) -> list[PersonalRecord]:
        self.governance.personal_data.expire_due()
        return list(self._core_selector.select(self._readable_records()).records)

    def _readable_records(self) -> list[PersonalRecord]:
        readable = [
            record
            for record in self.governance.list_memories(limit=10000)
            if record.access_policy
            not in {AccessPolicy.CONFIRM_READ, AccessPolicy.OWNER_ONLY}
        ]
        by_key: dict[str, PersonalRecord] = {}
        for record in readable:
            current = by_key.get(record.record_key)
            if current is None or self._record_priority(record) > self._record_priority(
                current
            ):
                by_key[record.record_key] = record
        return list(by_key.values())

    @staticmethod
    def _render_records(
        records: Sequence[PersonalRecord],
    ) -> str:
        grouped: "OrderedDict[MemoryKind, list[str]]" = OrderedDict(
            (kind, []) for kind, _ in _KIND_LABELS
        )
        seen: set[str] = set()
        for record in records:
            content = str(
                record.data.get("content") or record.summary or record.title
            ).strip()
            normalized = _normalized(content)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            try:
                kind = MemoryKind(str(record.data.get("kind") or MemoryKind.FACT.value))
            except ValueError:
                kind = MemoryKind.FACT
            if kind == MemoryKind.EPISODE:
                kind = MemoryKind.HISTORICAL_EVENT
            if kind == MemoryKind.PROCEDURE:
                continue
            grouped.setdefault(kind, []).append(content)

        lines = ["# 用户长期记忆"]
        labels = dict(_KIND_LABELS)
        for kind, items in grouped.items():
            if not items:
                continue
            lines.extend(("", f"## {labels.get(kind, '其他记忆')}"))
            lines.extend(f"- {item}" for item in items)
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _category_for_tag(tag: str) -> DataCategory:
        if tag == "long_term_health":
            return DataCategory.HEALTH
        if tag == "relationship":
            return DataCategory.RELATIONSHIP
        return DataCategory.GENERAL

    @staticmethod
    def _confidence(value: object, *, tag: str) -> float:
        if not isinstance(value, (int, float, str)):
            return 0.75
        try:
            return max(0.0, min(1.0, float(value)))
        except ValueError:
            return 0.75

    @staticmethod
    def _find_exact(
        records: Sequence[PersonalRecord], content: str
    ) -> PersonalRecord | None:
        normalized = _normalized(content)
        return next(
            (
                record
                for record in records
                if _normalized(str(record.data.get("content") or record.summary))
                == normalized
            ),
            None,
        )

    @staticmethod
    def _find_replaced(
        records: Sequence[PersonalRecord],
        replaces: str,
    ) -> PersonalRecord | None:
        return (
            GovernedLongTermMemory._find_exact(records, replaces) if replaces else None
        )

    @staticmethod
    def _record_priority(record: PersonalRecord) -> tuple[int, int, int, float, str]:
        return (
            int(record.user_locked),
            int(record.last_confirmed_at is not None),
            int(record.source.source in {"user", "explicit_memory", "dashboard"}),
            record.confidence,
            record.updated_at,
        )


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())
