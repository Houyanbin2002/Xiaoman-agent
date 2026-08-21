from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_ORIGINS = {
    "explicit_user",
    "user_correction",
    "observed_execution",
    "inferred_pattern",
}
_MEMORY_TAG_ALIASES = {
    "health_long_term": "long_term_health",
    "agent_context": "project_context",
}
_MEMORY_TAGS = {
    "identity",
    "preference",
    "relationship",
    "long_term_health",
    "project_context",
    "correction",
}
_EXECUTION_KINDS = {
    "environment",
    "project_convention",
    "tool_lesson",
    "procedure",
    "decision",
    "capability",
}
_POLICY_EFFECTS = {
    "allow",
    "deny",
    "require_approval",
    "adjust_score",
    "defer",
    "limit_frequency",
}
_DAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}


def _confidence(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _text(value: object, *, limit: int = 1000) -> str:
    return str(value or "").strip()[:limit]


def _origin(value: object, *, default: str) -> str:
    candidate = _text(value, limit=32).lower()
    return candidate if candidate in _ORIGINS else default


def _refs(value: object, *fallbacks: str) -> tuple[str, ...]:
    values = value if isinstance(value, list) else []
    return tuple(
        dict.fromkeys(
            text for item in (*values, *fallbacks) if (text := _text(item, limit=256))
        )
    )


@dataclass(frozen=True)
class RecentActivityCandidate:
    summary: str
    importance: int = 0
    occurred_at: str = ""
    source_message_ids: tuple[str, ...] = ()

    @property
    def emotional_weight(self) -> int:
        return self.importance

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> RecentActivityCandidate | None:
        summary = _text(raw.get("summary") or raw.get("content"), limit=1600)
        if not summary:
            return None
        raw_weight = raw.get("importance", raw.get("emotional_weight"))
        try:
            weight = max(0, min(10, int(raw_weight or 0)))
        except (TypeError, ValueError):
            weight = 0
        return cls(
            summary=summary,
            importance=weight,
            occurred_at=_text(raw.get("occurred_at"), limit=80),
            source_message_ids=_refs(raw.get("source_message_ids")),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "importance": self.importance,
            "occurred_at": self.occurred_at,
            "source_message_ids": list(self.source_message_ids),
        }


@dataclass(frozen=True)
class MemoryCandidate:
    tag: str
    content: str
    confidence: float = 0.0
    origin: str = "explicit_user"
    evidence_refs: tuple[str, ...] = ()
    subject: str = ""
    predicate: str = ""
    value: str = ""
    scope: str = ""
    attributes: dict[str, object] = field(default_factory=dict)
    replaces: str = ""
    valid_from: str = ""
    expires_at: str = ""
    source_message_id: str = ""

    @property
    def extraction_confidence(self) -> float:
        return self.confidence

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> MemoryCandidate | None:
        tag = _text(raw.get("tag") or raw.get("memory_type"), limit=64).lower()
        tag = _MEMORY_TAG_ALIASES.get(tag, tag)
        content = _text(raw.get("content") or raw.get("summary"), limit=2000)
        if tag not in _MEMORY_TAGS or not content:
            return None
        source_message_id = _text(raw.get("source_message_id"), limit=256)
        raw_attributes = raw.get("attributes")
        attributes = (
            {str(key): value for key, value in raw_attributes.items()}
            if isinstance(raw_attributes, Mapping)
            else {}
        )
        default_origin = "user_correction" if tag == "correction" else "explicit_user"
        return cls(
            tag=tag,
            content=content,
            confidence=_confidence(raw.get("confidence")),
            origin=_origin(raw.get("origin"), default=default_origin),
            evidence_refs=_refs(raw.get("evidence_refs"), source_message_id),
            subject=_text(raw.get("subject"), limit=500),
            predicate=_text(raw.get("predicate"), limit=256),
            value=_text(raw.get("value"), limit=1000),
            scope=_text(raw.get("scope"), limit=500),
            attributes=attributes,
            replaces=_text(raw.get("replaces"), limit=1000),
            valid_from=_text(raw.get("valid_from"), limit=64),
            expires_at=_text(raw.get("expires_at"), limit=64),
            source_message_id=source_message_id,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "content": self.content,
            "confidence": self.confidence,
            "origin": self.origin,
            "evidence_refs": list(self.evidence_refs),
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "scope": self.scope,
            "attributes": dict(self.attributes),
            "replaces": self.replaces,
            "valid_from": self.valid_from,
            "expires_at": self.expires_at,
            "source_message_id": self.source_message_id,
        }


@dataclass(frozen=True)
class TaskEventCandidate:
    summary: str
    delivery_semantics: str
    operation: str = "upsert"
    confidence: float = 0.0
    origin: str = "explicit_user"
    evidence_refs: tuple[str, ...] = ()
    due_at: str = ""
    active_from: str = ""
    expires_at: str = ""
    source_message_id: str = ""
    related_summary: str = ""
    related_event_id: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> TaskEventCandidate | None:
        summary = _text(raw.get("summary") or raw.get("title"), limit=1000)
        semantics = _text(raw.get("delivery_semantics"), limit=32).lower()
        if not summary or semantics not in {
            "exact",
            "before_deadline",
            "opportunistic",
            "silent",
        }:
            return None
        operation = _text(raw.get("operation"), limit=16).lower() or "upsert"
        if operation not in {"upsert", "complete", "cancel"}:
            return None
        source_message_id = _text(raw.get("source_message_id"), limit=256)
        return cls(
            summary=summary,
            delivery_semantics=semantics,
            operation=operation,
            confidence=_confidence(raw.get("confidence")),
            origin=_origin(raw.get("origin"), default="explicit_user"),
            evidence_refs=_refs(raw.get("evidence_refs"), source_message_id),
            due_at=_text(raw.get("due_at"), limit=64),
            active_from=_text(raw.get("active_from"), limit=64),
            expires_at=_text(raw.get("expires_at"), limit=64),
            source_message_id=source_message_id,
            related_summary=_text(raw.get("related_summary"), limit=1000),
            related_event_id=_text(raw.get("related_event_id"), limit=256),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "delivery_semantics": self.delivery_semantics,
            "operation": self.operation,
            "confidence": self.confidence,
            "origin": self.origin,
            "evidence_refs": list(self.evidence_refs),
            "due_at": self.due_at,
            "active_from": self.active_from,
            "expires_at": self.expires_at,
            "source_message_id": self.source_message_id,
            "related_summary": self.related_summary,
            "related_event_id": self.related_event_id,
        }


@dataclass(frozen=True)
class AttentionObservationCandidate:
    type: str
    statement: str
    confidence: float = 0.0
    origin: str = "explicit_user"
    evidence_refs: tuple[str, ...] = ()
    source_message_id: str = ""
    attributes: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, object]
    ) -> AttentionObservationCandidate | None:
        kind = _text(raw.get("type") or raw.get("kind"), limit=64).lower()
        statement = _text(raw.get("statement") or raw.get("summary"), limit=1000)
        if kind not in {"opportunity", "policy"} or not statement:
            return None
        if kind == "opportunity" and not _valid_recurrence(raw):
            return None
        if kind == "policy" and not _valid_policy(raw):
            return None
        source_message_id = _text(raw.get("source_message_id"), limit=256)
        attributes = {
            str(key): value
            for key, value in raw.items()
            if key
            not in {
                "type",
                "kind",
                "statement",
                "summary",
                "confidence",
                "origin",
                "evidence_refs",
                "source_message_id",
                "_user_evidence_verified",
            }
        }
        return cls(
            type=kind,
            statement=statement,
            confidence=_confidence(raw.get("confidence")),
            origin=_origin(
                raw.get("origin"),
                default="explicit_user" if source_message_id else "inferred_pattern",
            ),
            evidence_refs=_refs(raw.get("evidence_refs"), source_message_id),
            source_message_id=source_message_id,
            attributes=attributes,
        )

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": self.type,
            "statement": self.statement,
            "confidence": self.confidence,
            "origin": self.origin,
            "evidence_refs": list(self.evidence_refs),
            **self.attributes,
        }
        if self.source_message_id:
            result["source_message_id"] = self.source_message_id
        return result


@dataclass(frozen=True)
class ExecutionMemoryCandidate:
    summary: str
    kind: str = "procedure"
    operation: str = "upsert"
    confidence: float = 0.0
    origin: str = "observed_execution"
    evidence_refs: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    outcome: str = "unknown"
    source_message_id: str = ""
    target_memory_id: str = ""
    target_summary: str = ""

    @property
    def tool_requirement(self) -> str:
        return self.required_tools[0] if self.required_tools else ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ExecutionMemoryCandidate | None:
        summary = _text(raw.get("summary") or raw.get("content"), limit=2000)
        if not summary:
            return None
        kind = _text(raw.get("kind"), limit=64).lower() or "procedure"
        if kind not in _EXECUTION_KINDS:
            return None
        operation = _text(raw.get("operation"), limit=24).lower() or "upsert"
        if operation not in {"upsert", "suspend", "supersede"}:
            return None
        outcome = _text(raw.get("outcome"), limit=16).lower() or "unknown"
        if outcome not in {"success", "failure", "unknown"}:
            outcome = "unknown"
        raw_steps = raw.get("steps")
        steps = (
            tuple(step for item in raw_steps[:12] if (step := _text(item, limit=600)))
            if isinstance(raw_steps, list)
            else ()
        )
        raw_tools = raw.get("required_tools")
        if not isinstance(raw_tools, list):
            legacy_tool = _text(raw.get("tool_requirement"), limit=120)
            raw_tools = [legacy_tool] if legacy_tool else []
        tools = tuple(
            dict.fromkeys(
                tool for item in raw_tools if (tool := _text(item, limit=120).lower())
            )
        )
        source_message_id = _text(raw.get("source_message_id"), limit=256)
        origin = _origin(
            raw.get("origin"),
            default="explicit_user" if source_message_id else "observed_execution",
        )
        return cls(
            summary=summary,
            kind=kind,
            operation=operation,
            confidence=_confidence(raw.get("confidence")),
            origin=origin,
            evidence_refs=_refs(raw.get("evidence_refs"), source_message_id),
            steps=steps or (summary,),
            required_tools=tools,
            outcome=outcome,
            source_message_id=source_message_id,
            target_memory_id=_text(raw.get("target_memory_id"), limit=256),
            target_summary=_text(raw.get("target_summary"), limit=1000),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "kind": self.kind,
            "operation": self.operation,
            "confidence": self.confidence,
            "origin": self.origin,
            "evidence_refs": list(self.evidence_refs),
            "steps": list(self.steps),
            "required_tools": list(self.required_tools),
            "outcome": self.outcome,
            "source_message_id": self.source_message_id,
            "target_memory_id": self.target_memory_id,
            "target_summary": self.target_summary,
        }


@dataclass(frozen=True)
class SemanticBatchPayload:
    recent_activity_entries: tuple[RecentActivityCandidate, ...] = ()
    memory_candidates: tuple[MemoryCandidate, ...] = ()
    task_events: tuple[TaskEventCandidate, ...] = ()
    attention_observations: tuple[AttentionObservationCandidate, ...] = ()
    execution_memories: tuple[ExecutionMemoryCandidate, ...] = ()

    @classmethod
    def empty(cls) -> SemanticBatchPayload:
        return cls()

    @classmethod
    def from_mapping(cls, raw: object) -> SemanticBatchPayload:
        payload = raw if isinstance(raw, Mapping) else {}
        recent_raw = payload.get("recent_activity_entries")
        execution_raw = payload.get("execution_memories")
        return cls(
            recent_activity_entries=_items(
                recent_raw, RecentActivityCandidate.from_mapping
            ),
            memory_candidates=_items(
                payload.get("memory_candidates"), MemoryCandidate.from_mapping
            ),
            task_events=_items(
                payload.get("task_events"), TaskEventCandidate.from_mapping
            ),
            attention_observations=_items(
                payload.get("attention_observations"),
                AttentionObservationCandidate.from_mapping,
            ),
            execution_memories=_items(
                execution_raw, ExecutionMemoryCandidate.from_mapping
            ),
        )

    def to_mapping(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "recent_activity_entries": [
                item.to_mapping() for item in self.recent_activity_entries
            ],
            "memory_candidates": [item.to_mapping() for item in self.memory_candidates],
            "task_events": [item.to_mapping() for item in self.task_events],
            "attention_observations": [
                item.to_mapping() for item in self.attention_observations
            ],
            "execution_memories": [
                item.to_mapping() for item in self.execution_memories
            ],
        }


def _items(raw: object, factory: Any) -> tuple[Any, ...]:
    if not isinstance(raw, list):
        return ()
    result: list[Any] = []
    dropped = 0
    for item in raw:
        if not isinstance(item, Mapping):
            dropped += 1
            continue
        normalized = factory(item)
        if normalized is None:
            dropped += 1
            continue
        result.append(normalized)
    if dropped:
        logger.info("semantic candidates dropped by schema: count=%d", dropped)
    return tuple(result)


def _valid_recurrence(raw: Mapping[str, object]) -> bool:
    recurrence = raw.get("recurrence")
    data = (
        dict(recurrence)
        if isinstance(recurrence, Mapping)
        else {
            "timezone": raw.get("timezone"),
            "days": raw.get("days"),
            "start": raw.get("start"),
            "end": raw.get("end"),
        }
    )
    days = data.get("days")
    return bool(
        _text(data.get("timezone"), limit=80)
        and isinstance(days, list)
        and days
        and all(str(day).lower() in _DAYS for day in days)
        and _valid_clock(data.get("start"))
        and _valid_clock(data.get("end"))
    )


def _valid_clock(value: object) -> bool:
    text = _text(value, limit=8)
    if len(text) not in {5, 8} or text[2] != ":":
        return False
    try:
        hour, minute = (int(part) for part in text[:5].split(":"))
    except ValueError:
        return False
    return 0 <= hour <= 23 and 0 <= minute <= 59


def _valid_policy(raw: Mapping[str, object]) -> bool:
    return bool(
        isinstance(raw.get("scope"), Mapping)
        and isinstance(raw.get("conditions"), Mapping)
        and _text(raw.get("effect"), limit=32).lower() in _POLICY_EFFECTS
    )


__all__ = [
    "AttentionObservationCandidate",
    "ExecutionMemoryCandidate",
    "MemoryCandidate",
    "RecentActivityCandidate",
    "SemanticBatchPayload",
    "TaskEventCandidate",
]
