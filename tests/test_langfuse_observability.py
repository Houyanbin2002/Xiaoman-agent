from __future__ import annotations

from datetime import datetime, timezone

from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from agent.config_models import LangfuseConfig
from core.tracing.composite import CompositeTraceRecorder
from infra.observability.langfuse_recorder import LangfuseTraceRecorder
from infra.persistence.trace_store import TraceStore


class _BrokenRemoteRecorder:
    def start_trace(self, **_kwargs):
        raise OSError("offline")

    def finish_trace(self, *_args, **_kwargs):
        raise OSError("offline")

    def append_event(self, **_kwargs):
        raise OSError("offline")


def test_composite_recorder_keeps_local_trace_when_remote_is_offline(tmp_path) -> None:
    local = TraceStore(tmp_path / "traces.db")
    recorder = CompositeTraceRecorder((local, _BrokenRemoteRecorder()))

    recorder.start_trace(
        trace_id="tr_offline",
        flow="passive",
        session_key="dashboard:user",
        title="offline fallback",
    )
    recorder.append_event(
        trace_id="tr_offline",
        span_id="sp_tool",
        parent_span_id="",
        category="tool",
        name="search",
        status="completed",
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        duration_ms=20,
        summary="done",
        payload={"arguments": {"q": "hello"}, "result_preview": "world"},
    )
    recorder.finish_trace("tr_offline", status="completed")

    assert local.require_trace("tr_offline").status == "completed"
    assert local.list_events("tr_offline")[0].name == "search"
    recorder.close()


def test_langfuse_recorder_maps_agent_generation_usage_and_duration() -> None:
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        span_exporter=exporter,
    )
    recorder = LangfuseTraceRecorder(
        LangfuseConfig(enabled=True, public_key="pk", secret_key="sk"),
        client=client,
    )
    started_at = datetime.now(timezone.utc).isoformat()

    recorder.start_trace(
        trace_id="tr_langfuse",
        flow="passive",
        session_key="dashboard:user",
        title="Langfuse mapping",
    )
    recorder.append_event(
        trace_id="tr_langfuse",
        span_id="sp_model",
        parent_span_id="",
        category="model",
        name="reasoning",
        status="completed",
        started_at=started_at,
        finished_at=None,
        duration_ms=250,
        summary="model replied",
        payload={
            "model": "test-model",
            "input": [{"role": "user", "content": "hello"}],
            "output": {"content": "world"},
            "input_tokens": 100,
            "output_tokens": 5,
            "cache_hit_tokens": 80,
        },
    )
    recorder.finish_trace("tr_langfuse", status="completed")
    recorder.close()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    root = spans["agent:passive"]
    generation = spans["model:reasoning"]
    assert root.attributes["langfuse.observation.type"] == "agent"
    assert root.attributes["session.id"] == "dashboard:user"
    assert generation.attributes["langfuse.observation.type"] == "generation"
    assert generation.attributes["langfuse.observation.model.name"] == "test-model"
    assert '"input": 20' in generation.attributes[
        "langfuse.observation.usage_details"
    ]
    assert '"cache_read_input_tokens": 80' in generation.attributes[
        "langfuse.observation.usage_details"
    ]
    assert 249_000_000 <= generation.end_time - generation.start_time <= 251_000_000
