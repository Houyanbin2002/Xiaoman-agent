from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from bootstrap.dashboard_management import (
    DashboardRuntimeServices,
    register_dashboard_management,
)
from bootstrap.attention import build_attention_runtime
from agent.workflows.personal import PersonalRoutineService
from agent.workflows.runtime import WorkflowRuntime
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.marketplace import (
    MarketplaceInstaller,
    MarketplaceInstallResult,
    MarketplaceItem,
    MarketplaceService,
)
from core.personal.service import PersonalDataService
from core.personal.sources.service import ExternalSourceSyncService
from core.personal.today import PersonalTodayService
from core.personal.governance import MemoryGovernanceService
from core.personal.rhythm import PersonalRhythmService
from core.attention.events.models import (
    CanonicalEntity,
    CanonicalEvent,
    DeliverySemantics,
    EntityState,
    EventStatus,
)
from bus.events import OutboundMessage
from core.workflow.models import StepKind, StepSpec
from infra.persistence.memory_governance_store import MemoryGovernanceStore
from infra.persistence.personal_store import PersonalStore
from infra.persistence.external_source_store import ExternalSourceStore
from infra.persistence.workflow_store import WorkflowStore
from bootstrap.dashboard_management.routes import system as system_routes


class _EventBus:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def on(self, _event_type: type[Any], handler: Any) -> None:
        self.handlers.append(handler)


class _AgentLoop:
    active_turn_states: dict[str, Any] = {}

    async def process_direct(self, content: str, **_kwargs: Any) -> str:
        return f"小满收到：{content}"


class _BlockingAgentLoop:
    def __init__(self) -> None:
        self.task: Any = None

    async def process_direct(self, content: str, **_kwargs: Any) -> str:
        self.task = asyncio.current_task()
        await asyncio.Event().wait()
        return content

    def request_interrupt(self, session_key: str, **_kwargs: Any) -> Any:
        if self.task is None or self.task.done():
            return SimpleNamespace(status="idle", session_key=session_key)
        self.task.cancel()
        return SimpleNamespace(status="interrupted", session_key=session_key)


class _CapturingAgentLoop:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def process_direct(self, content: str, **kwargs: Any) -> str:
        self.calls.append((content, kwargs))
        return "附件已收到"


class _MarkItDownTool(Tool):
    name = "mcp_markitdown__convert_to_markdown"
    description = "Convert a document to Markdown"
    parameters = {
        "type": "object",
        "properties": {"uri": {"type": "string"}},
        "required": ["uri"],
    }

    async def execute(self, **kwargs: Any) -> str:
        assert str(kwargs["uri"]).startswith("file:")
        return "# 复习计划\n\n第一章：记忆系统"


class _Tools:
    def get_registered_names(self) -> list[str]:
        return ["memory_search", "task_create"]

    def get_documents(self) -> list[Any]:
        return []


class _McpRegistry:
    def snapshot(self) -> list[dict[str, Any]]:
        return []


class _Scheduler:
    def list_jobs(self) -> list[Any]:
        return []


class _PushTool:
    def __init__(self) -> None:
        self.channels: dict[str, Any] = {}

    def register_channel(self, channel: str, *, text: Any) -> None:
        self.channels[channel] = text


class _WorkflowPush:
    async def execute(self, **_kwargs: Any) -> str:
        return "ok"


class _PersonalRecordStub(Tool):
    name = "personal_record"
    description = "Store a personal record"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs)


class _MemoryAdmin:
    def describe(self) -> Any:
        return SimpleNamespace(name="akasha")

    def list_items_for_dashboard(self, **_kwargs: Any):
        return (
            [
                {
                    "id": "legacy-1",
                    "memory_type": "preference",
                    "summary": "Likes concise reminders",
                    "source_ref": "chat:legacy",
                }
            ],
            1,
        )


def _services(tmp_path: Path) -> DashboardRuntimeServices:
    config = SimpleNamespace(
        model="deepseek-v4-flash",
        light_model="",
        agent_model="deepseek-v4-flash",
        vl_model="",
        provider="openai",
        base_url="https://example.invalid/v1",
        light_base_url="",
        agent_base_url="",
        vl_base_url="",
        api_key="super-secret",
        light_api_key="",
        agent_api_key="",
        vl_api_key="",
        memory=SimpleNamespace(
            enabled=True,
            engine="akasha",
            embedding=SimpleNamespace(
                model="",
                api_key="",
                base_url="",
                output_dimensionality=None,
            ),
        ),
        channels=SimpleNamespace(
            socket="127.0.0.1:8765",
            telegram=None,
            qq=None,
        ),
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[llm.main]\nmodel = 'deepseek-v4-flash'\n", encoding="utf-8"
    )
    workflow_tools = ToolRegistry()
    workflow_tools.register(_PersonalRecordStub(), risk="write")
    workflow_runtime = WorkflowRuntime(
        store=WorkflowStore(tmp_path / "workflows.db"),
        agent_loop_provider=lambda: None,
        push_tool=_WorkflowPush(),  # type: ignore[arg-type]
        tool_registry=workflow_tools,
    )
    personal_data = PersonalDataService(PersonalStore(tmp_path / "personal.db"))
    memory_governance = MemoryGovernanceService(
        personal_data=personal_data,
        conflict_store=MemoryGovernanceStore(tmp_path / "personal.db"),
    )
    personal_rhythm = PersonalRhythmService(personal_data)
    attention_runtime = build_attention_runtime(tmp_path / "personal.db")
    external_store = ExternalSourceStore(tmp_path / "personal.db")
    external_sources = ExternalSourceSyncService(
        store=external_store,
        personal_data=personal_data,
        adapters={},
    )
    return DashboardRuntimeServices(
        config=config,
        config_path=config_path,
        agent_loop=_AgentLoop(),
        event_bus=_EventBus(),
        tools=_Tools(),
        mcp_registry=_McpRegistry(),
        scheduler=_Scheduler(),
        workflow_runtime=workflow_runtime,
        plugin_manager=None,
        push_tool=_PushTool(),
        workspace=tmp_path,
        personal_data=personal_data,
        personal_routines=PersonalRoutineService(workflow_runtime),
        memory_governance=memory_governance,
        memory_admin=_MemoryAdmin(),
        personal_rhythm=personal_rhythm,
        attention_runtime=attention_runtime,
        external_sources=external_sources,
        personal_today=PersonalTodayService(personal_data),
    )


def _close_services(services: DashboardRuntimeServices) -> None:
    services.external_sources.close()
    services.attention_runtime.close()
    services.memory_governance.close()
    services.personal_data.close()
    services.workflow_runtime.store.close()


class _MarketplaceProvider:
    def __init__(self, rows: list[MarketplaceItem]) -> None:
        self.rows = rows

    def search(self, query: str, limit: int) -> list[MarketplaceItem]:
        needle = query.casefold()
        return [
            row
            for row in self.rows
            if not needle or needle in f"{row.id} {row.name}".casefold()
        ][:limit]

    def get(self, item_id: str) -> MarketplaceItem | None:
        return next((row for row in self.rows if row.id == item_id), None)

    def refresh(self) -> list[MarketplaceItem]:
        return self.rows


def test_marketplace_routes_search_detail_and_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = MarketplaceItem(
        id="vendor/docs",
        kind="mcp",
        name="Docs",
        description="Docs",
        provider="vendor",
        source_url="https://example.com",
        install_mode="direct",
    )
    market = MarketplaceService(
        _MarketplaceProvider([]),
        _MarketplaceProvider([item]),
    )
    monkeypatch.setattr(
        system_routes,
        "default_marketplace_service",
        lambda _services: market,
    )
    install = AsyncMock(
        return_value=MarketplaceInstallResult(
            status="installed",
            item_id=item.id,
            kind="mcp",
            resource_name="docs",
        )
    )
    monkeypatch.setattr(MarketplaceInstaller, "install", install)
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)
    try:
        with TestClient(app) as client:
            search = client.get(
                "/api/dashboard/control/marketplace?kind=mcp&q=docs"
            )
            detail = client.get(
                "/api/dashboard/control/marketplace/items/vendor/docs?kind=mcp"
            )
            installed = client.post(
                "/api/dashboard/control/marketplace/install",
                json={"kind": "mcp", "item_id": "vendor/docs", "configuration": {}},
            )
            refreshed = client.post(
                "/api/dashboard/control/marketplace/refresh?kind=mcp"
            )
        assert search.status_code == 200
        assert search.json()["items"][0]["id"] == "vendor/docs"
        assert detail.status_code == 200
        assert detail.json()["name"] == "Docs"
        assert installed.status_code == 200
        assert installed.json()["status"] == "installed"
        assert refreshed.json() == {"refreshed": True, "kind": "mcp"}
        install.assert_awaited_once_with("mcp", "vendor/docs", {})
    finally:
        _close_services(services)


def test_extension_marketplace_ui_calls_real_backend_routes() -> None:
    source = Path(
        "frontend/dashboard/src/features/extensions/MarketplaceView.tsx"
    ).read_text(encoding="utf-8")
    shell = Path(
        "frontend/dashboard/src/features/extensions/ExtensionsView.tsx"
    ).read_text(encoding="utf-8")

    assert "/api/dashboard/control/marketplace" in source
    assert "/api/dashboard/control/marketplace/install" in source
    assert "安装" in source and "暂不支持" in source
    assert "已安装" in shell and "市场" in shell


def test_extension_marketplace_keeps_partial_results_when_one_source_fails() -> None:
    source = Path(
        "frontend/dashboard/src/features/extensions/MarketplaceView.tsx"
    ).read_text(encoding="utf-8")

    assert "Promise.allSettled" in source
    assert "部分市场来源暂时不可用" in source
    assert "setRows([])" in source


def test_attention_v2_patterns_policies_and_overview_are_manageable(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        created = client.post(
            "/api/dashboard/control/attention/patterns",
            json={
                "id": "pat_dashboard_commute",
                "scene": "commute",
                "timezone": "Asia/Shanghai",
                "days": ["mon", "tue", "wed", "thu", "fri"],
                "start": "08:00",
                "end": "08:25",
                "available_minutes": 20,
            },
        )
        policy = client.post(
            "/api/dashboard/control/attention/policies",
            json={
                "id": "pol_dashboard_quiet",
                "scope": {"action_type": "content"},
                "conditions": {"scene": ["work"]},
                "effect": "deny",
                "priority": 80,
            },
        )
        overview = client.get("/api/dashboard/control/attention/overview")
        patterns = client.get("/api/dashboard/control/attention/patterns")
        capabilities = client.get("/api/dashboard/control/attention/capabilities")
        observations = client.get("/api/dashboard/control/attention/observations")
        paused = client.patch(
            "/api/dashboard/control/attention/patterns/pat_dashboard_commute/status",
            json={"status": "suspended"},
        )
        resumed = client.patch(
            "/api/dashboard/control/attention/patterns/pat_dashboard_commute/status",
            json={"status": "active"},
        )

    assert created.status_code == 200
    assert created.json()["status"] == "active"
    assert policy.status_code == 200
    assert policy.json()["effect"] == "deny"
    assert overview.status_code == 200
    assert overview.json()["active_patterns"] == 1
    assert patterns.json()[0]["id"] == "pat_dashboard_commute"
    assert capabilities.json()[0]["id"] == "message.notify"
    assert observations.status_code == 200
    assert observations.json() == []
    assert paused.json()["status"] == "suspended"
    assert resumed.json()["status"] == "active"
    assert resumed.json()["user_locked"] is True
    _close_services(services)


def test_attention_dashboard_hides_silent_internal_events(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    runtime = services.attention_runtime
    assert runtime is not None
    now = datetime(2026, 7, 19, 9, 0, tzinfo=timezone.utc).isoformat()
    for suffix, semantics in (
        ("memory", DeliverySemantics.SILENT),
        ("deadline", DeliverySemantics.BEFORE_DEADLINE),
    ):
        entity = CanonicalEntity(
            id=f"entity:test:{suffix}",
            source_id="test",
            external_id=suffix,
            kind=suffix,
            title=suffix,
            state=EntityState.OPEN,
            source_version="1",
            payload_ref=suffix,
            updated_at=now,
        )
        runtime.store.upsert_entity(entity)
        runtime.store.upsert_event(
            CanonicalEvent(
                id=f"event:test:{suffix}",
                entity_id=entity.id,
                source_id="test",
                kind=suffix,
                occurred_at=now,
                due_at="",
                active_from="",
                expires_at="",
                urgency=0.5,
                confidence=0.9,
                delivery_semantics=semantics,
                dedupe_key=f"test:{suffix}",
                source_version="1",
                payload_ref=suffix,
                status=EventStatus.ACTIVE,
            )
        )
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        overview = client.get("/api/dashboard/control/attention/overview")
        events = client.get("/api/dashboard/control/attention/events")

    assert overview.status_code == 200
    assert overview.json()["active_events"] == 1
    assert [row["id"] for row in events.json()] == ["event:test:deadline"]
    _close_services(services)


def test_attention_runtime_target_can_be_saved_without_exposing_credentials(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    services.config.proactive = SimpleNamespace(
        enabled=False,
        default_channel="telegram",
        default_chat_id="",
    )
    services.config.plugins = {
        "qqbot": {"enabled": True, "allow_from": ["owner-openid"]}
    }
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        before = client.get("/api/dashboard/control/attention/overview")
        saved = client.patch(
            "/api/dashboard/control/attention/runtime",
            json={
                "enabled": True,
                "channel": "qqbot",
                "chat_id": "owner-openid",
            },
        )
        after = client.get("/api/dashboard/control/attention/overview")

    assert before.json()["available_targets"] == [
        {"channel": "qqbot", "chat_id": "owner-openid"}
    ]
    assert saved.json()["restart_required"] is True
    assert after.json()["runtime_enabled"] is True
    assert after.json()["target_configured"] is True
    assert "target" in services.config_path.read_text(encoding="utf-8")
    _close_services(services)


def test_control_overview_and_models_hide_secrets(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        overview = client.get("/api/dashboard/control/overview")
        models = client.get("/api/dashboard/control/models")
        tasks = client.get("/api/dashboard/control/tasks")
        legacy_workflows = client.get("/api/dashboard/control/workflows")

    assert overview.status_code == 200
    assert overview.json()["model"] == "deepseek-v4-flash"
    assert overview.json()["counts"]["tools"] == 2
    assert overview.json()["memory_status"] == "healthy"
    assert models.status_code == 200
    assert models.json()[0]["api_key_configured"] is True
    assert "super-secret" not in models.text
    assert tasks.status_code == 200
    assert tasks.json() == []
    assert legacy_workflows.status_code == 404
    assert "dashboard" in services.push_tool.channels
    _close_services(services)


def test_conversation_style_can_be_listed_and_changed_without_restart(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        before = client.get("/api/dashboard/control/conversation-styles")
        changed = client.patch(
            "/api/dashboard/control/conversation-styles",
            json={"style_id": "warm"},
        )
        rejected = client.patch(
            "/api/dashboard/control/conversation-styles",
            json={"style_id": "unknown"},
        )

    assert before.status_code == 200
    assert before.json()["active_style"] == "balanced"
    assert len(before.json()["styles"]) == 6
    assert changed.status_code == 200
    assert changed.json()["active_style"] == "warm"
    assert changed.json()["applies_from"] == "next_reply"
    assert rejected.status_code == 422
    assert services.conversation_styles.active_id == "warm"
    assert (tmp_path / "conversation_style.json").exists()
    _close_services(services)


def test_weixin_qr_login_saves_token_to_system_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            {"qrcode": "qr-token", "qrcode_img_content": "weixin://scan/value"},
            {
                "status": "confirmed",
                "ilink_bot_id": "bot-account",
                "bot_token": "secret-token",
                "ilink_user_id": "owner-user",
                "baseurl": "https://ilinkai.weixin.qq.com",
            },
        ]
    )

    async def _fake_weixin_get(_base_url: str, _endpoint: str) -> dict[str, Any]:
        return next(responses)

    saved = AsyncMock()
    monkeypatch.setattr(system_routes, "_weixin_get", _fake_weixin_get)
    monkeypatch.setattr(system_routes, "_qr_data_url", lambda value: f"data:{value}")
    monkeypatch.setattr(system_routes.LocalSecretStore, "set_bundle", saved)
    app = FastAPI()
    services = _services(tmp_path)
    services.config.plugins = {}
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        started = client.post("/api/dashboard/control/channels/weixin/qr")
        flow_id = started.json()["flow_id"]
        confirmed = client.get(f"/api/dashboard/control/channels/weixin/qr/{flow_id}")

    assert started.json()["image"] == "data:weixin://scan/value"
    assert confirmed.json()["status"] == "confirmed"
    saved.assert_awaited_once_with(
        "weixin",
        {
            "account_id": "bot-account",
            "token": "secret-token",
            "base_url": "https://ilinkai.weixin.qq.com",
        },
    )
    assert "secret-token" not in services.config_path.read_text(encoding="utf-8")
    assert services.config.plugins["weixin"]["account_id"] == "bot-account"
    assert services.config.plugins["weixin"]["allow_from"] == ["owner-user"]
    _close_services(services)


def test_channel_save_validates_credentials_and_refreshes_runtime_state(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    services.config.plugins = {}
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        incomplete = client.patch(
            "/api/dashboard/control/channels/qqbot",
            json={"app_id": "qq-app", "allow_from": ["owner"]},
        )
        saved = client.patch(
            "/api/dashboard/control/channels/qqbot",
            json={
                "app_id": "qq-app",
                "client_secret": "qq-secret",
                "allow_from": ["owner"],
            },
        )
        listed = client.get("/api/dashboard/control/channels")

    assert incomplete.status_code == 422
    assert "AppSecret" in incomplete.json()["detail"]
    assert saved.status_code == 200
    assert saved.json()["channel"]["configured"] is True
    assert saved.json()["channel"]["allow_from"] == ["owner"]
    assert services.config.plugins["qqbot"]["client_secret"] == "qq-secret"
    assert (
        next(row for row in listed.json() if row["id"] == "qqbot")["configured"] is True
    )
    assert "qq-secret" not in listed.text
    _close_services(services)


def test_gateway_restart_endpoint_reports_support_and_requests_restart(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    requested: list[bool] = []
    services.gateway_instance_id = "gateway-old"
    services.gateway_restart = lambda: requested.append(True)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        status = client.get("/api/dashboard/control/gateway/status")
        restarted = client.post("/api/dashboard/control/gateway/restart")

    assert status.json() == {
        "status": "running",
        "instance_id": "gateway-old",
        "restart_supported": True,
    }
    assert restarted.status_code == 202
    assert restarted.json() == {"accepted": True, "instance_id": "gateway-old"}
    assert requested == [True]
    _close_services(services)


def test_memory_model_settings_hide_secrets_and_persist_configuration(
    tmp_path: Path,
) -> None:
    import toml

    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        current = client.get("/api/dashboard/control/models")
        saved = client.patch(
            "/api/dashboard/control/models/memory",
            json={
                "provider": "dashscope",
                "model": "text-embedding-v4",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "output_dimensionality": 1024,
                "api_key": "embedding-secret",
            },
        )

    assert current.status_code == 200
    memory_row = next(item for item in current.json() if item["slot"] == "memory")
    assert memory_row["model"] == "text-embedding-v4"
    assert memory_row["engine"] == "akasha"
    assert memory_row["api_key_configured"] is True
    assert "super-secret" not in current.text
    assert saved.json() == {
        "saved": True,
        "hot_reloaded": False,
        "restart_required": True,
        "slot": "memory",
        "model": "text-embedding-v4",
    }
    persisted = toml.load(services.config_path)
    assert persisted["memory"]["enabled"] is True
    assert persisted["memory"]["engine"] == "akasha"
    assert persisted["memory"]["embedding"]["model"] == "text-embedding-v4"
    assert persisted["memory"]["embedding"]["output_dimensionality"] == 1024
    assert persisted["memory"]["embedding"]["api_key"] == "embedding-secret"
    assert "embedding-secret" not in saved.text
    _close_services(services)


def test_memory_connection_test_routes_are_removed(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        probe = client.post(
            "/api/dashboard/control/memory/settings/probe",
            json={
                "provider": "dashscope",
                "model": "text-embedding-v4",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "output_dimensionality": 1024,
                "api_key": "embedding-secret",
            },
        )

        health = client.get("/api/dashboard/control/memory/health")

    assert probe.status_code == 404
    assert health.status_code == 404
    _close_services(services)


def test_dashboard_chat_websocket_round_trip(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        with client.websocket_connect("/api/dashboard/chat/test") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"content": "你好"})
            status = websocket.receive_json()
            assert status["type"] == "status"
            assert status["status"] == "thinking"
            assert status["prompt"] == "你好"
            assert websocket.receive_json() == {
                "type": "final",
                "run_id": status["run_id"],
                "content": "小满收到：你好",
                "thinking": "",
            }
    _close_services(services)


def test_dashboard_generated_artifact_is_downloadable(tmp_path: Path) -> None:
    artifact = tmp_path / "reports" / "assistant-report.pdf"
    artifact.parent.mkdir()
    artifact.write_bytes(b"%PDF-generated")

    class _ArtifactAgentLoop:
        active_turn_states: dict[str, Any] = {}

        async def process_direct_outbound(
            self,
            _content: str,
            **_kwargs: Any,
        ) -> OutboundMessage:
            return OutboundMessage(
                channel="dashboard",
                chat_id="artifact-chat",
                content="报告已生成并附在本条消息中。",
                media=[str(artifact)],
            )

    app = FastAPI()
    services = _services(tmp_path)
    services.agent_loop = _ArtifactAgentLoop()
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/dashboard/chat/artifact-chat"
        ) as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"content": "制作报告并发送给我"})
            assert websocket.receive_json()["type"] == "status"
            final = websocket.receive_json()
            assert final["type"] == "final"
            assert final["artifacts"][0]["name"] == "assistant-report.pdf"
            response = client.get(final["artifacts"][0]["url"])
            assert response.status_code == 200
            assert response.content == b"%PDF-generated"

    _close_services(services)


def test_dashboard_chat_uploads_attachment_and_passes_media_to_agent(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    loop = _CapturingAgentLoop()
    services.agent_loop = loop
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/dashboard/chat/files/attachments",
            params={"filename": "复习计划.txt"},
            content="第一章：记忆系统".encode(),
            headers={"content-type": "text/plain"},
        )
        assert uploaded.status_code == 200
        attachment = uploaded.json()
        assert attachment["name"] == "复习计划.txt"

        with client.websocket_connect("/api/dashboard/chat/files") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json(
                {
                    "content": "总结附件",
                    "attachment_ids": [attachment["id"]],
                }
            )
            status = websocket.receive_json()
            assert status["type"] == "status"
            assert status["attachments"] == [attachment]
            assert websocket.receive_json()["content"] == "附件已收到"

    assert loop.calls[0][0] == "总结附件"
    media = loop.calls[0][1]["media"]
    assert len(media) == 1
    assert Path(media[0]).read_text(encoding="utf-8") == "第一章：记忆系统"
    _close_services(services)


def test_dashboard_chat_attachment_rejects_unsupported_files(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        response = client.post(
            "/api/dashboard/chat/files/attachments",
            params={"filename": "danger.exe"},
            content=b"not executable",
        )

    assert response.status_code == 400
    assert "暂不支持" in response.json()["detail"]
    _close_services(services)


def test_dashboard_chat_parses_binary_document_before_agent_turn(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    loop = _CapturingAgentLoop()
    tools = ToolRegistry()
    tools.register(_MarkItDownTool(), risk="read-only")
    services.agent_loop = loop
    services.tools = tools
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/dashboard/chat/documents/attachments",
            params={"filename": "复习计划.pdf"},
            content=b"%PDF-1.4 fake",
            headers={"content-type": "application/pdf"},
        )
        assert uploaded.status_code == 200
        attachment = uploaded.json()
        assert attachment["parsed"] is True

        with client.websocket_connect("/api/dashboard/chat/documents") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json(
                {"content": "总结附件", "attachment_ids": [attachment["id"]]}
            )
            assert websocket.receive_json()["type"] == "status"
            assert websocket.receive_json()["type"] == "final"

    media = loop.calls[0][1]["media"]
    assert len(media) == 1
    assert Path(media[0]).name.endswith(".pdf.md")
    assert "第一章：记忆系统" in Path(media[0]).read_text(encoding="utf-8")
    _close_services(services)


def test_dashboard_chat_explains_missing_document_parser(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/dashboard/chat/documents/attachments",
            params={"filename": "复习计划.pdf"},
            content=b"%PDF-1.4 fake",
            headers={"content-type": "application/pdf"},
        )

    assert uploaded.status_code == 503
    assert "文档解析服务未连接" in uploaded.json()["detail"]
    _close_services(services)


def test_dashboard_chat_run_survives_reconnect_and_can_stop(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    services.agent_loop = _BlockingAgentLoop()
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        with client.websocket_connect("/api/dashboard/chat/reconnect") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"content": "执行一个长任务", "request_id": "run-1"})
            assert websocket.receive_json()["run_id"] == "run-1"

        active = client.get("/api/dashboard/chat/runs").json()["items"]
        assert active[0]["chat_id"] == "reconnect"
        assert active[0]["prompt"] == "执行一个长任务"

        with client.websocket_connect("/api/dashboard/chat/reconnect") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            restored = websocket.receive_json()
            assert restored["type"] == "status"
            assert restored["run_id"] == "run-1"
            websocket.send_json({"type": "stop"})
            stopping = websocket.receive_json()
            assert stopping["status"] == "stopping"
            cancelled = websocket.receive_json()
            assert cancelled["type"] == "cancelled"
            assert cancelled["run_id"] == "run-1"

    _close_services(services)


def test_dashboard_chat_run_can_stop_from_conversation_list(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    services.agent_loop = _BlockingAgentLoop()
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        with client.websocket_connect("/api/dashboard/chat/sidebar-stop") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"content": "执行后台任务", "request_id": "run-2"})
            assert websocket.receive_json()["run_id"] == "run-2"

        response = client.post(
            "/api/dashboard/chat/runs/stop",
            params={"session_key": "dashboard:sidebar-stop"},
        )
        assert response.status_code == 200
        assert response.json() == {"stopped": True, "status": "interrupted"}

        for _ in range(20):
            if not client.get("/api/dashboard/chat/runs").json()["items"]:
                break
        assert client.get("/api/dashboard/chat/runs").json()["items"] == []

    _close_services(services)


def test_dashboard_chat_rejects_cross_site_websocket(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as raised:
            with client.websocket_connect(
                "/api/dashboard/chat/test",
                headers={"origin": "https://evil.example"},
            ):
                pass

    assert raised.value.code == 1008
    _close_services(services)


def test_personal_data_and_routine_routes_use_shared_runtime(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        created = client.post(
            "/api/dashboard/control/personal/records",
            json={
                "entity_type": "commitment",
                "title": "Finish report",
                "summary": "Weekly report",
                "data": {"priority": "high"},
            },
        )
        records = client.get(
            "/api/dashboard/control/personal/records?entity_type=commitment"
        )
        routine = client.post(
            "/api/dashboard/control/personal/routines",
            json={
                "routine": "morning_brief",
                "local_date": "2026-07-10",
                "chat_id": "owner",
            },
        )
        task_id = routine.json()["task"]["id"]
        active_delete = client.delete(f"/api/dashboard/control/tasks/{task_id}")
        cancelled = client.post(
            f"/api/dashboard/control/tasks/{task_id}/cancel",
            json={"note": "dashboard stop"},
        )
        deleted = client.delete(f"/api/dashboard/control/tasks/{task_id}")

    assert created.status_code == 200
    assert records.json()[0]["id"] == created.json()["id"]
    assert routine.status_code == 200
    assert routine.json()["task"]["status"] == "running"
    assert active_delete.status_code == 409
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert deleted.json() == {"deleted": True, "id": task_id}
    assert services.workflow_runtime.store.list_workflows() == []
    _close_services(services)


def test_external_source_and_today_routes_share_canonical_records(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        source = client.post(
            "/api/dashboard/control/sources",
            json={
                "provider": "notion",
                "server_name": "notion",
                "name": "每日待办",
                "resource_url": "collection://tasks",
                "entity_type": "commitment",
                "mapping": {"title": "任务", "status": "状态"},
                "poll_interval_minutes": 15,
            },
        )
        listed = client.get("/api/dashboard/control/sources")
        client.post(
            "/api/dashboard/control/personal/records",
            json={
                "entity_type": "commitment",
                "title": "今天完成",
                "summary": "dashboard item",
                "data": {
                    "state": "open",
                    "due_at": "2026-07-16T09:00:00+08:00",
                },
            },
        )
        today = client.get(
            "/api/dashboard/control/personal/today",
            params={"local_date": "2026-07-16", "timezone": "Asia/Shanghai"},
        )

    assert source.status_code == 200
    assert listed.json()[0]["id"] == source.json()["id"]
    assert today.status_code == 200
    assert today.json()["records"][0]["title"] == "今天完成"
    assert today.json()["counts"] == {"commitment": 1}
    _close_services(services)


def test_task_center_can_finish_waiting_approval_and_user_input(tmp_path: Path) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    runtime = services.workflow_runtime
    approval = runtime.create_workflow(
        name="Confirm reminder",
        goal="Wait for an explicit decision",
        steps=[
            StepSpec(
                id="approve",
                title="确认提醒",
                description="是否继续？",
                kind=StepKind.APPROVAL,
            )
        ],
        session_key="dashboard:owner",
        channel="dashboard",
        chat_id="owner",
    )
    question = runtime.create_workflow(
        name="Ask preference",
        goal="Wait for an answer",
        steps=[
            StepSpec(
                id="answer",
                title="补充偏好",
                description="希望几点提醒？",
                kind=StepKind.WAIT_USER,
            )
        ],
        session_key="dashboard:owner",
        channel="dashboard",
        chat_id="owner",
    )
    runtime.store.prepare_human_steps()
    register_dashboard_management(app, services)

    with TestClient(app) as client:
        rows = client.get("/api/dashboard/control/tasks")
        approval_result = client.post(
            f"/api/dashboard/control/tasks/{approval.id}/approval",
            json={"step_id": "approve", "approved": True, "note": "dashboard approval"},
        )
        response_result = client.post(
            f"/api/dashboard/control/tasks/{question.id}/respond",
            json={"step_id": "answer", "response": "每天早上九点"},
        )

    assert rows.status_code == 200
    by_id = {item["id"]: item for item in rows.json()}
    assert by_id[approval.id]["waiting_actions"] == [
        {
            "id": "approve",
            "title": "确认提醒",
            "description": "是否继续？",
            "kind": "approval",
        }
    ]
    assert by_id[question.id]["waiting_actions"][0]["kind"] == "wait_user"
    assert approval_result.status_code == 200
    assert approval_result.json()["status"] == "succeeded"
    assert response_result.status_code == 200
    assert response_result.json()["status"] == "succeeded"
    _close_services(services)


def test_rhythm_routes_share_context_recommendations_and_followups(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)
    now = datetime.now(timezone.utc)
    base = "/api/dashboard/control/rhythm"

    with TestClient(app) as client:
        scene = client.post(
            f"{base}/scene",
            json={"scene": "home", "duration_minutes": 120},
        )
        focus = client.post(
            f"{base}/focus",
            json={"minutes": 25, "label": "写测试"},
        )
        task = client.post(
            f"{base}/records/commitment",
            json={
                "title": "Small task",
                "summary": "Finish a small task",
                "data": {
                    "state": "open",
                    "estimated_minutes": 20,
                    "energy": "medium",
                    "contexts": ["home"],
                },
            },
        )
        recommendations = client.get(f"{base}/recommendations?minutes=30")
        follow_up = client.post(
            f"{base}/follow-ups",
            json={
                "title": "Weekly check",
                "message": "Review this week's progress",
                "reason": "Keep the goal moving",
                "trigger_type": "interval",
                "interval_minutes": 10080,
                "next_trigger_at": (now + timedelta(days=7)).isoformat(),
                "enabled": True,
            },
        )
        report = client.post(
            f"{base}/reports",
            json={"period": "week", "persist": True},
        )
        stopped = client.delete(f"{base}/focus")
        overview = client.get(f"{base}/overview")

    assert scene.status_code == 200
    assert scene.json()["context"]["scene"] == "home"
    assert focus.json()["context"]["focus_active"] is True
    assert task.status_code == 200
    assert recommendations.json()["recommendations"][0]["title"] == "Small task"
    assert follow_up.json()["data"]["enabled"] is True
    assert report.json()["record_id"]
    assert stopped.json()["stopped"] == 1
    assert overview.json()["counts"]["proactive_intent"] == 1
    _close_services(services)


def test_memory_governance_routes_detect_resolve_export_and_delete(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)
    base = "/api/dashboard/control/memory-governance"

    with TestClient(app) as client:
        created = client.post(
            f"{base}/memories",
            json={
                "kind": "preference",
                "content": "Prefers morning workouts",
                "summary": "Workout time preference",
                "record_key": "preference:workout-time",
            },
        )
        conflict = client.post(
            f"{base}/memories",
            json={
                "kind": "preference",
                "content": "Prefers evening workouts",
                "summary": "Workout time preference",
                "record_key": "preference:workout-time",
            },
        )
        conflict_id = conflict.json()["conflict"]["id"]
        pending = client.get(f"{base}/conflicts")
        resolved = client.post(
            f"{base}/conflicts/{conflict_id}/resolve",
            json={"action": "accept_candidate", "note": "Confirmed in UI"},
        )
        record_id = resolved.json()["record"]["id"]
        locked = client.patch(
            f"/api/dashboard/control/personal/records/{record_id}",
            json={"user_locked": True, "reason": "protect preference"},
        )
        exported = client.get(f"{base}/export")
        deleted = client.delete(f"{base}/memories/{record_id}?hard=true")

    assert created.json()["status"] == "created"
    assert conflict.json()["status"] == "conflict_pending"
    assert len(pending.json()) == 1
    assert resolved.json()["status"] == "accepted_candidate"
    assert locked.json()["user_locked"] is True
    assert exported.json()["format"] == "xiaoman-memory-governance-v1"
    assert "attachment" in exported.headers["content-disposition"]
    assert deleted.json() == {"deleted": True, "hard": True}
    _close_services(services)


def test_memory_knowledge_graph_exposes_user_facing_semantic_relations(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    services = _services(tmp_path)
    register_dashboard_management(app, services)
    base = "/api/dashboard/control/memory-governance"

    with TestClient(app) as client:
        created = client.post(
            f"{base}/memories",
            json={
                "kind": "relationship",
                "content": "小林是用户的研究生同学",
                "summary": "与小林的关系",
                "record_key": "relationship:xiaolin",
                "subject": "我",
                "predicate": "研究生同学",
                "value": "小林",
            },
        )
        graph = client.get(f"{base}/graph")

    assert created.status_code == 200
    assert graph.status_code == 200
    payload = graph.json()
    assert payload["center_id"] == "person:self"
    assert {node["label"] for node in payload["nodes"]} >= {"我", "小林"}
    assert {
        (edge["source"], edge["label"], edge["target"])
        for edge in payload["edges"]
    } == {("person:self", "研究生同学", "entity:小林")}
    xiaolin = next(node for node in payload["nodes"] if node["label"] == "小林")
    assert xiaolin["memory_ids"]
    assert xiaolin["kind"] == "relationship"
    _close_services(services)


def test_capability_presentation_declares_brand_and_source_states() -> None:
    source = Path(
        "frontend/dashboard/src/features/extensions/capabilityPresentation.ts"
    ).read_text(encoding="utf-8")

    for token in ("notion", "gmail", "obsidian", "markitdown"):
        assert token in source.lower()
    for token in ('"active"', '"paused"', '"none"'):
        assert token in source


def test_capability_logos_are_local_and_have_a_fallback() -> None:
    component = Path(
        "frontend/dashboard/src/features/extensions/CapabilityLogo.tsx"
    ).read_text(encoding="utf-8")
    assert "onError" in component
    assert "Cable" in component

    asset_dir = Path("frontend/dashboard/public/assets/brands")
    for name in ("notion.svg", "gmail.svg", "obsidian.svg", "microsoft.svg"):
        content = (asset_dir / name).read_text(encoding="utf-8")
        assert content.startswith("<svg")
        assert len(content) > 100


def test_unified_extensions_route_preserves_skill_and_mcp_hashes() -> None:
    main = Path("frontend/dashboard/src/main.tsx").read_text(encoding="utf-8")
    settings = Path(
        "frontend/dashboard/src/features/settings/SettingsHubView.tsx"
    ).read_text(encoding="utf-8")
    shell = Path(
        "frontend/dashboard/src/features/extensions/ExtensionsView.tsx"
    ).read_text(encoding="utf-8")

    assert 'case "skills":' in main
    assert 'case "mcp": return <ExtensionsView' in main
    assert 'title: "扩展能力"' in main
    assert "查看技能" not in settings
    assert 'role="tablist"' in shell
    assert 'aria-selected={activeTab === "skills"}' in shell
    assert 'aria-selected={activeTab === "mcp"}' in shell


def test_skill_capability_cards_keep_real_install_actions() -> None:
    source = Path(
        "frontend/dashboard/src/features/skills/SkillsView.tsx"
    ).read_text(encoding="utf-8")

    assert "interface SkillsViewProps" in source
    assert 'className="capability-grid extension-capability-grid' in source
    assert 'className="capability-card extension-capability-card' in source
    assert '"/api/dashboard/control/skills/install"' in source
    assert 'method: "DELETE"' in source
    assert "缺少依赖" in source


def test_mcp_capability_cards_expose_real_actions_in_detail_drawer() -> None:
    cards = Path(
        "frontend/dashboard/src/features/mcp/McpCards.tsx"
    ).read_text(encoding="utf-8")
    view = Path(
        "frontend/dashboard/src/features/mcp/McpView.tsx"
    ).read_text(encoding="utf-8")

    assert "CapabilityLogo" in cards
    assert "sourceStateForServer" in cards
    assert 'role="dialog"' in cards
    assert 'aria-modal="true"' in cards
    assert "信号源" in cards
    for callback in (
        "onAuthorize",
        "onCreateSource",
        "onSetSourceEnabled",
        "onSyncSource",
        "onDeleteSource",
        "onRemove",
    ):
        assert callback in cards
    assert "McpConnectionCard" in view
    assert "McpDetailDrawer" in view
    assert "/api/dashboard/control/mcp/" in view
    assert '"/api/dashboard/control/sources"' in view


def test_extension_card_footer_resets_global_tag_row_spacing() -> None:
    styles = Path(
        "frontend/dashboard/src/features/extensions/extensions.css"
    ).read_text(encoding="utf-8")

    assert (
        ".desktop-app .extension-card-footer .tag-row {"
        " min-width: 0; margin-top: 0;"
    ) in styles


def test_extension_positive_badges_use_scoped_theme_colors() -> None:
    styles = Path(
        "frontend/dashboard/src/features/extensions/extensions.css"
    ).read_text(encoding="utf-8")

    assert ".desktop-app .extensions-view .badge-green" in styles
    assert "#ecebff" in styles
    assert "#5148a7" in styles
