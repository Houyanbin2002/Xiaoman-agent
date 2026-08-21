from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceRecord:
    id: str
    flow: str
    session_key: str
    title: str
    status: str
    parent_trace_id: str = ""
    started_at: str = ""
    updated_at: str = ""
    finished_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "flow": self.flow,
            "session_key": self.session_key,
            "title": self.title,
            "status": self.status,
            "parent_trace_id": self.parent_trace_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
            "event_count": self.event_count,
        }


@dataclass(frozen=True)
class TraceEvent:
    id: int
    trace_id: str
    span_id: str
    parent_span_id: str
    category: str
    name: str
    status: str
    started_at: str
    finished_at: str | None
    duration_ms: int | None
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "summary": self.summary,
            "payload": dict(self.payload),
        }
