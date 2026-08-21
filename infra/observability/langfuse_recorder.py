from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.config_models import LangfuseConfig
from core.tracing.models import TraceEvent, TraceRecord

logger = logging.getLogger(__name__)


@dataclass
class _LangfuseTraceState:
    trace_id: str
    root: Any
    flow: str
    session_key: str
    title: str
    parent_trace_id: str
    started_at: str
    metadata: dict[str, Any]
    spans: dict[str, Any] = field(default_factory=dict)
    event_count: int = 0


class LangfuseTraceRecorder:
    """Adapt Xiaoman's completed-event ledger to Langfuse observations.

    The SDK exports in the background. The recorder keeps only live trace handles and
    uses an isolated OpenTelemetry provider so unrelated framework spans are not sent
    to Langfuse (or counted against its quota).
    """

    def __init__(self, config: LangfuseConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._traces: dict[str, _LangfuseTraceState] = {}
        if client is None:
            from langfuse import Langfuse
            from opentelemetry.sdk.trace import TracerProvider

            client = Langfuse(
                public_key=config.public_key,
                secret_key=config.secret_key,
                base_url=config.base_url,
                environment=config.environment,
                sample_rate=config.sample_rate,
                flush_at=config.flush_at,
                flush_interval=config.flush_interval_seconds,
                debug=config.debug,
                tracer_provider=TracerProvider(),
            )
        self._client = client

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
        now = _utc_now()
        with self._lock:
            existing = self._traces.get(trace_id)
            if existing is not None:
                existing.metadata.update(metadata or {})
                return _record_from_state(trace_id, existing, status="running")

            langfuse_trace_id = self._client.create_trace_id(seed=trace_id)
            root = self._client.start_observation(
                trace_context={"trace_id": langfuse_trace_id},
                name=f"agent:{flow}",
                as_type="agent",
                input={"title": title} if self._config.capture_content else None,
                metadata={
                    "xiaoman_trace_id": trace_id,
                    "flow": flow,
                    "parent_trace_id": parent_trace_id,
                    **dict(metadata or {}),
                },
            )
            state = _LangfuseTraceState(
                trace_id=trace_id,
                root=root,
                flow=flow,
                session_key=session_key,
                title=title,
                parent_trace_id=parent_trace_id,
                started_at=now,
                metadata=dict(metadata or {}),
            )
            self._traces[trace_id] = state
            self._set_trace_attributes(root, state)
            return _record_from_state(trace_id, state, status="running")

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord | None:
        with self._lock:
            state = self._traces.pop(trace_id, None)
            if state is None:
                return None
            state.metadata.update(metadata or {})
            level = _level_for_status(status)
            state.root.update(
                output={"status": status},
                metadata={**state.metadata, "status": status},
                level=level,
                status_message=status if level != "DEFAULT" else None,
            )
            state.root.end()
            finished_at = _utc_now()
            return TraceRecord(
                id=trace_id,
                flow=state.flow,
                session_key=state.session_key,
                title=state.title,
                status=status,
                parent_trace_id=state.parent_trace_id,
                started_at=state.started_at,
                updated_at=finished_at,
                finished_at=finished_at,
                metadata=dict(state.metadata),
                event_count=state.event_count,
            )

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
        raw_payload = dict(payload or {})
        with self._lock:
            state = self._traces.get(trace_id)
            if state is None:
                raise KeyError(f"Langfuse trace is not active: {trace_id}")
            parent = state.spans.get(parent_span_id) or state.root
            observation_type = _observation_type(category)
            fields = self._event_fields(
                category=category,
                status=status,
                summary=summary,
                payload=raw_payload,
            )
            observation = self._start_child_at(
                parent=parent,
                state=state,
                name=f"{category}:{name}",
                as_type=observation_type,
                started_at=started_at,
                **fields,
            )
            end_ns = _end_time_ns(
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
            )
            observation.end(end_time=end_ns)
            state.spans[span_id] = observation
            state.event_count += 1

        return TraceEvent(
            id=state.event_count,
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
            payload=raw_payload,
        )

    def close(self) -> None:
        with self._lock:
            active = list(self._traces.items())
            self._traces.clear()
        for trace_id, state in active:
            try:
                state.root.update(
                    output={"status": "interrupted"},
                    level="WARNING",
                    status_message="runtime shutdown before trace completion",
                )
                state.root.end()
            except Exception:
                logger.warning(
                    "failed to close active Langfuse trace trace_id=%s",
                    trace_id,
                    exc_info=True,
                )
        self._client.shutdown()

    def _set_trace_attributes(self, observation: Any, state: _LangfuseTraceState) -> None:
        span = getattr(observation, "_otel_span", None)
        if span is None:
            return
        session_id = _ascii_identifier(state.session_key, prefix="session")
        span.set_attribute("langfuse.trace.name", f"xiaoman-{state.flow}")
        if session_id:
            span.set_attribute("session.id", session_id)
        span.set_attribute("langfuse.trace.tags", ["xiaoman", state.flow])
        span.set_attribute(
            "langfuse.trace.metadata",
            json.dumps(
                {
                    "xiaoman_trace_id": state.trace_id,
                    "session_key": state.session_key,
                    **state.metadata,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    def _start_child_at(
        self,
        *,
        parent: Any,
        state: _LangfuseTraceState,
        name: str,
        as_type: str,
        started_at: str,
        **fields: Any,
    ) -> Any:
        """Create a child with the event's original start time.

        Langfuse's public completed-observation adapter starts at call time. Xiaoman
        records an event after it finishes, so use the SDK's OTEL bridge to preserve
        latency. The public fallback keeps compatibility if the bridge changes in a
        later v4 patch.
        """

        parent_span = getattr(parent, "_otel_span", None)
        otel_tracer = getattr(self._client, "_otel_tracer", None)
        wrap = getattr(self._client, "_create_observation_from_otel_span", None)
        if parent_span is not None and otel_tracer is not None and callable(wrap):
            from opentelemetry import trace as otel_trace

            context = otel_trace.set_span_in_context(parent_span)
            otel_span = otel_tracer.start_span(
                name=name,
                context=context,
                start_time=_timestamp_ns(started_at),
            )
            observation = wrap(otel_span=otel_span, as_type=as_type, **fields)
        else:
            observation = parent.start_observation(
                name=name,
                as_type=as_type,
                **fields,
            )
        self._set_trace_attributes(observation, state)
        return observation

    def _event_fields(
        self,
        *,
        category: str,
        status: str,
        summary: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        content_limit = self._config.max_content_chars
        capture = self._config.capture_content
        event_input: Any = None
        event_output: Any = None
        if capture:
            if category == "model":
                event_input = payload.get("input")
                event_output = payload.get("output") or payload.get("error")
            elif category == "tool":
                event_input = payload.get("arguments")
                event_output = payload.get("result_preview") or payload.get("error")
            elif category == "memory":
                event_input = payload.get("query")
                event_output = {
                    "summary": summary,
                    "retrieved_count": payload.get("retrieved_count"),
                    "personal_memory_ids": payload.get("personal_memory_ids"),
                    "execution_memory_ids": payload.get("execution_memory_ids"),
                }
            else:
                event_output = summary

        usage_details = _usage_details(payload) if category == "model" else None
        excluded = {
            "input",
            "output",
            "arguments",
            "result_preview",
            "error",
            "usage_details",
        }
        metadata = {
            "xiaoman_status": status,
            "summary": summary,
            **{key: value for key, value in payload.items() if key not in excluded},
        }
        fields: dict[str, Any] = {
            "input": _bounded(event_input, content_limit),
            "output": _bounded(event_output, content_limit),
            "metadata": _bounded(metadata, content_limit),
            "level": _level_for_status(status),
            "status_message": summary if status not in {"completed", "success"} else None,
        }
        if category == "model":
            fields.update(
                model=str(payload.get("model") or ""),
                model_parameters=_model_parameters(payload),
                usage_details=usage_details,
            )
        return fields


def _record_from_state(
    trace_id: str,
    state: _LangfuseTraceState,
    *,
    status: str,
) -> TraceRecord:
    return TraceRecord(
        id=trace_id,
        flow=state.flow,
        session_key=state.session_key,
        title=state.title,
        status=status,
        parent_trace_id=state.parent_trace_id,
        started_at=state.started_at,
        updated_at=_utc_now(),
        metadata=dict(state.metadata),
        event_count=state.event_count,
    )


def _observation_type(category: str) -> str:
    return {
        "model": "generation",
        "tool": "tool",
        "memory": "retriever",
        "retrieval": "retriever",
        "workflow": "chain",
        "proactive": "agent",
    }.get(category, "span")


def _level_for_status(status: str) -> str:
    normalized = status.lower()
    if normalized in {"failed", "error"}:
        return "ERROR"
    if normalized in {"interrupted", "degraded", "warning", "skipped"}:
        return "WARNING"
    return "DEFAULT"


def _usage_details(payload: dict[str, Any]) -> dict[str, int] | None:
    raw = payload.get("usage_details")
    if isinstance(raw, dict):
        normalized = {
            str(key): int(value)
            for key, value in raw.items()
            if value is not None and _is_int_like(value)
        }
        return normalized or None

    prompt_total = _optional_int(
        payload.get("input_tokens", payload.get("cache_prompt_tokens"))
    )
    cached = _optional_int(payload.get("cache_hit_tokens")) or 0
    output = _optional_int(payload.get("output_tokens"))
    usage: dict[str, int] = {}
    if prompt_total is not None:
        usage["input"] = max(0, prompt_total - cached)
    if cached:
        usage["cache_read_input_tokens"] = cached
    if output is not None:
        usage["output"] = output
    return usage or None


def _model_parameters(payload: dict[str, Any]) -> dict[str, Any] | None:
    parameters = {
        "max_tokens": payload.get("max_tokens"),
        "tool_choice": payload.get("tool_choice"),
        "disable_thinking": payload.get("disable_thinking"),
    }
    result = {key: value for key, value in parameters.items() if value is not None}
    return result or None


def _bounded(value: Any, max_chars: int) -> Any:
    if value is None:
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        encoded = str(value)
    if len(encoded) <= max_chars:
        try:
            return json.loads(encoded)
        except (TypeError, ValueError):
            return encoded
    return {
        "truncated": True,
        "original_chars": len(encoded),
        "preview": encoded[:max_chars],
    }


def _timestamp_ns(value: str) -> int:
    dt = _parse_datetime(value)
    return int(dt.timestamp() * 1_000_000_000)


def _end_time_ns(
    *,
    started_at: str,
    finished_at: str | None,
    duration_ms: int | None,
) -> int:
    started = _parse_datetime(started_at)
    if duration_ms is not None:
        finished = started + timedelta(milliseconds=max(0, duration_ms))
    elif finished_at:
        finished = _parse_datetime(finished_at)
    else:
        finished = datetime.now(timezone.utc)
    if finished < started:
        finished = started
    return int(finished.timestamp() * 1_000_000_000)


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ascii_identifier(value: str, *, prefix: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 200 and text.isascii():
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
