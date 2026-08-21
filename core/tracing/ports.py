from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.tracing.models import TraceEvent, TraceRecord


@runtime_checkable
class TraceRecorder(Protocol):
    def start_trace(
        self,
        *,
        trace_id: str,
        flow: str,
        session_key: str,
        title: str,
        parent_trace_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord: ...

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord | None: ...

    def append_event(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str,
        category: str,
        name: str,
        status: str,
        started_at: str,
        finished_at: str | None,
        duration_ms: int | None,
        summary: str,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent: ...
