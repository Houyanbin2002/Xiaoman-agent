from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.runtime.model_step import run_model_step
from bootstrap.dashboard_api.routes.traces import register_trace_routes
from core.llm import LLMResponse
from core.tracing import record_trace_event, trace_root
from infra.persistence.trace_store import TraceStore


class _Provider:
    async def chat(self, **_kwargs) -> LLMResponse:
        return LLMResponse(content="done")


class _BrokenRecorder:
    def start_trace(self, **_kwargs):
        raise OSError("storage unavailable")

    def finish_trace(self, *_args, **_kwargs):
        raise OSError("storage unavailable")

    def append_event(self, **_kwargs):
        raise OSError("storage unavailable")


def test_trace_root_persists_model_round_and_finishes(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.db")
    with trace_root(
        store,
        trace_id="tr_test",
        flow="passive",
        session_key="dashboard:user",
        title="测试任务",
    ):
        record_trace_event(category="memory", name="recall", summary="找到 2 条记忆")
        asyncio.run(
            run_model_step(
                _Provider(),
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                model="test-model",
                max_tokens=20,
                source="passive",
                iteration=1,
            )
        )

    trace = store.require_trace("tr_test")
    events = store.list_events("tr_test")
    assert trace.status == "completed"
    assert trace.event_count == 2
    assert [event.category for event in events] == ["memory", "model"]
    assert events[1].payload["model"] == "test-model"
    store.close()


def test_trace_api_filters_by_session_and_returns_replay(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.db")
    with trace_root(
        store,
        trace_id="tr_visible",
        flow="workflow",
        session_key="dashboard:user",
        title="整理资料",
    ):
        record_trace_event(
            category="workflow",
            name="collect",
            summary="资料收集完成",
        )
    with trace_root(
        store,
        trace_id="tr_other",
        flow="passive",
        session_key="qq:other",
        title="别的会话",
    ):
        pass

    app = FastAPI()
    register_trace_routes(app, store=store)
    with TestClient(app) as client:
        listing = client.get(
            "/api/dashboard/traces",
            params={"session_key": "dashboard:user"},
        )
        assert listing.status_code == 200
        assert [item["id"] for item in listing.json()["items"]] == ["tr_visible"]

        replay = client.get("/api/dashboard/traces/tr_visible")
        assert replay.status_code == 200
        assert replay.json()["trace"]["event_count"] == 1
        assert replay.json()["events"][0]["summary"] == "资料收集完成"
    store.close()


def test_trace_failure_never_breaks_user_execution() -> None:
    recorder = _BrokenRecorder()
    with trace_root(
        recorder,
        trace_id="tr_fail_open",
        flow="passive",
        session_key="dashboard:user",
        title="仍应完成",
    ):
        record_trace_event(category="turn", name="inbound", summary="收到请求")
