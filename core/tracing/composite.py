from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from core.tracing.models import TraceEvent, TraceRecord
from core.tracing.ports import TraceRecorder

logger = logging.getLogger(__name__)


class CompositeTraceRecorder:
    """Fan out traces without letting an optional backend break the main path."""

    def __init__(self, recorders: Sequence[TraceRecorder]) -> None:
        self._recorders = tuple(recorders)
        if not self._recorders:
            raise ValueError("at least one trace recorder is required")

    @property
    def recorders(self) -> tuple[TraceRecorder, ...]:
        return self._recorders

    def start_trace(
        self,
        *,
        trace_id: str,
        flow: str,
        session_key: str,
        title: str,
        parent_trace_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord:
        result: TraceRecord | None = None
        for recorder in self._recorders:
            try:
                current = recorder.start_trace(
                    trace_id=trace_id,
                    flow=flow,
                    session_key=session_key,
                    title=title,
                    parent_trace_id=parent_trace_id,
                    metadata=metadata,
                )
                result = result or current
            except Exception:
                logger.warning(
                    "trace backend start failed backend=%s trace_id=%s",
                    type(recorder).__name__,
                    trace_id,
                    exc_info=True,
                )
        if result is None:
            raise RuntimeError(f"all trace backends failed to start {trace_id}")
        return result

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord | None:
        result: TraceRecord | None = None
        succeeded = False
        for recorder in self._recorders:
            try:
                current = recorder.finish_trace(
                    trace_id,
                    status=status,
                    metadata=metadata,
                )
                succeeded = True
                result = result or current
            except Exception:
                logger.warning(
                    "trace backend finish failed backend=%s trace_id=%s",
                    type(recorder).__name__,
                    trace_id,
                    exc_info=True,
                )
        if not succeeded:
            raise RuntimeError(f"all trace backends failed to finish {trace_id}")
        return result

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
    ) -> TraceEvent:
        result: TraceEvent | None = None
        for recorder in self._recorders:
            try:
                current = recorder.append_event(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    category=category,
                    name=name,
                    status=status,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    summary=summary,
                    payload=payload,
                )
                result = result or current
            except Exception:
                logger.warning(
                    "trace backend event failed backend=%s trace_id=%s category=%s name=%s",
                    type(recorder).__name__,
                    trace_id,
                    category,
                    name,
                    exc_info=True,
                )
        if result is None:
            raise RuntimeError(f"all trace backends failed to append event for {trace_id}")
        return result

    def close(self) -> None:
        for recorder in reversed(self._recorders):
            close = getattr(recorder, "close", None)
            if not callable(close):
                close = getattr(recorder, "shutdown", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception:
                logger.warning(
                    "trace backend close failed backend=%s",
                    type(recorder).__name__,
                    exc_info=True,
                )
