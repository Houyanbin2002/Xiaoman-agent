from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from core.tracing.ports import TraceRecorder

logger = logging.getLogger(__name__)

_trace_id: ContextVar[str] = ContextVar("xiaoman_trace_id", default="")
_span_id: ContextVar[str] = ContextVar("xiaoman_span_id", default="")
_recorder: ContextVar[TraceRecorder | None] = ContextVar(
    "xiaoman_trace_recorder", default=None
)


def new_trace_id() -> str:
    return f"tr_{uuid.uuid4().hex}"


def new_span_id() -> str:
    return f"sp_{uuid.uuid4().hex[:20]}"


def current_trace_id() -> str:
    return _trace_id.get()


def current_span_id() -> str:
    return _span_id.get()


def current_trace_recorder() -> TraceRecorder | None:
    return _recorder.get()


@contextmanager
def bind_trace(
    recorder: TraceRecorder | None,
    trace_id: str,
    *,
    span_id: str = "",
) -> Iterator[None]:
    trace_token = _trace_id.set(trace_id)
    recorder_token = _recorder.set(recorder)
    span_token = _span_id.set(span_id)
    try:
        yield
    finally:
        _span_id.reset(span_token)
        _recorder.reset(recorder_token)
        _trace_id.reset(trace_token)


@contextmanager
def trace_root(
    recorder: TraceRecorder | None,
    *,
    trace_id: str,
    flow: str,
    session_key: str,
    title: str,
    parent_trace_id: str = "",
    metadata: dict[str, Any] | None = None,
    finish: bool = True,
) -> Iterator[str]:
    if recorder is not None:
        try:
            recorder.start_trace(
                trace_id=trace_id,
                flow=flow,
                session_key=session_key,
                title=title,
                parent_trace_id=parent_trace_id,
                metadata=metadata,
            )
        except Exception:
            logger.warning("trace start failed trace_id=%s", trace_id, exc_info=True)
    status = "completed"
    with bind_trace(recorder, trace_id):
        try:
            yield trace_id
        except BaseException:
            status = "interrupted" if _is_cancelled_exception() else "failed"
            raise
        finally:
            if recorder is not None and finish:
                try:
                    recorder.finish_trace(trace_id, status=status)
                except Exception:
                    logger.warning(
                        "trace finish failed trace_id=%s", trace_id, exc_info=True
                    )


def record_trace_event(
    *,
    category: str,
    name: str,
    summary: str,
    status: str = "completed",
    payload: dict[str, Any] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_ms: int | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
) -> None:
    recorder = current_trace_recorder()
    trace_id = current_trace_id()
    if recorder is None or not trace_id:
        return
    now = _utc_now()
    event_started_at = started_at
    if event_started_at is None and duration_ms is not None:
        event_started_at = (
            datetime.now(timezone.utc)
            - timedelta(milliseconds=max(0, duration_ms))
        ).isoformat()
    try:
        recorder.append_event(
            trace_id=trace_id,
            span_id=span_id or new_span_id(),
            parent_span_id=(
                current_span_id() if parent_span_id is None else parent_span_id
            ),
            category=category,
            name=name,
            status=status,
            started_at=event_started_at or now,
            finished_at=finished_at or now,
            duration_ms=duration_ms,
            summary=summary,
            payload=dict(payload or {}),
        )
    except Exception:
        # Observability must never turn a successful user task into a failure.
        logger.warning(
            "trace event failed trace_id=%s category=%s name=%s",
            trace_id,
            category,
            name,
            exc_info=True,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_cancelled_exception() -> bool:
    import sys

    exc = sys.exc_info()[1]
    return exc is not None and type(exc).__name__ == "CancelledError"
