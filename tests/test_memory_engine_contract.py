from __future__ import annotations
from typing import Any, cast

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import SemanticBatchPayload
from agent.config_models import Config, MemoryConfig
from agent.tools.registry import ToolRegistry
from bootstrap.memory import build_memory_runtime
from plugins.default_memory.engine import DefaultMemoryEngine
from core.memory.engine import (
    EngineProfile,
    MemoryCapability,
    MemoryMutation,
    MemoryQuery,
    MemoryQueryFilters,
    MemoryScope,
)
from core.memory.execution import build_execution_state
from memory2.store import MemoryStore2
from core.memory.plugin import MemoryPluginRuntime


def _make_default_engine(
    *,
    config=None,
    provider=None,
    retriever=None,
    execution_retriever=None,
    memorizer=None,
    tagger=None,
    event_publisher=None,
):
    engine = DefaultMemoryEngine.__new__(DefaultMemoryEngine)
    engine._config = config or SimpleNamespace(model="lm")
    engine._workspace = Path(".")
    engine._provider = provider
    engine._light_provider = None
    engine._light_model = ""
    engine._v2_store = None
    engine._embedder = None
    engine._memorizer = memorizer
    engine._retriever = retriever
    engine._execution_retriever = execution_retriever
    engine._event_bus = event_publisher
    engine.closeables = []
    engine._wire_memory2_events()
    return engine


async def _drain_maintenance(maintenance: object) -> None:
    for _ in range(5):
        tasks = list(getattr(maintenance, "_maintenance_tasks").values())
        if not tasks:
            return
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)


async def test_default_memory_engine_retrieve_maps_hits_and_text_block():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "m1",
                    "summary": "记住用户偏好中文回复",
                    "score": 0.88,
                    "source_ref": "cli:1@seed",
                    "memory_type": "preference",
                    "extra_json": {"origin": "test"},
                }
            ]
        ),
        build_injection_block=lambda items: ("注入块", ["m1"]),
    )
    engine = _make_default_engine(retriever=cast(Any, retriever))

    result = await engine.query(
        MemoryQuery(
            text="中文回复",
            intent="context",
            scope=MemoryScope(channel="cli", chat_id="1"),
            filters=MemoryQueryFilters(
                kinds=("preference",),
                hints={"require_scope_match": True},
            ),
            limit=3,
        )
    )

    assert result.text_block == "注入块"
    assert len(result.records) == 1
    assert result.records[0].id == "m1"
    assert result.records[0].injected is True
    assert result.records[0].engine_kind == "default"
    assert result.records[0].kind == "preference"
    assert result.trace["profile"] == EngineProfile.RICH_MEMORY_ENGINE.value


async def test_default_memory_engine_consumes_shared_semantic_batch() -> None:
    event_bus = EventBus()
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=[{"id": "old-rule", "score": 0.9}])
    )
    memorizer = SimpleNamespace(
        save_from_consolidation=AsyncMock(),
        save_item=AsyncMock(return_value="new:rule-1"),
    )
    _make_default_engine(
        retriever=cast(Any, retriever),
        memorizer=cast(Any, memorizer),
        event_publisher=event_bus,
    )
    event = ConversationSemanticBatchCommitted(
        batch_id="semantic_1",
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        analysis_version="conversation-v3",
        message_ids=("cli:1:0", "cli:1:1"),
        user_message_ids=("cli:1:0",),
        end_seq=1,
        context_consolidate_through=-1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "recent_activity_entries": [
                    {
                        "summary": "用户纠正了旧流程",
                        "importance": 2,
                        "source_message_ids": ["cli:1:0"],
                    }
                ],
                "execution_memories": [
                    {
                        "summary": "回答工具数量前先读取真实工具清单",
                        "kind": "procedure",
                        "operation": "upsert",
                        "confidence": 0.98,
                        "origin": "explicit_user",
                        "source_message_id": "cli:1:0",
                        "evidence_refs": ["cli:1:0"],
                        "required_tools": ["list_tools"],
                        "steps": ["读取当前工具清单", "根据清单给出精确数量"],
                    }
                ],
            }
        ),
    )

    await event_bus.emit(event)

    memorizer.save_from_consolidation.assert_awaited_once()
    memorizer.save_item.assert_awaited_once()
    rule_write = memorizer.save_item.await_args.kwargs
    assert rule_write["summary"] == "回答工具数量前先读取真实工具清单"
    assert rule_write["memory_type"] == "procedure"
    assert rule_write["extra"]["tool_requirement"] == "list_tools"
    assert rule_write["extra"]["steps"] == [
        "读取当前工具清单",
        "根据清单给出精确数量",
    ]
    assert rule_write["extra"]["authority"] == "user"
    assert rule_write["extra"]["lifecycle_status"] == "active"
    assert rule_write["execution_verified"] is False
    assert rule_write["source_ref"].startswith("semantic_1#r:")
    await event_bus.aclose()


async def test_default_memory_engine_separates_personal_and_execution_retrieval():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "p1",
                    "summary": "用户偏好上午处理重要任务",
                    "score": 0.82,
                    "source_ref": "cli:1@seed",
                    "memory_type": "preference",
                    "extra_json": {},
                }
            ]
        ),
        build_injection_block=lambda items: ("个人记忆块", ["p1"]),
    )
    execution_retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "x1",
                    "summary": "启动前先运行迁移",
                    "score": 0.9,
                    "source_ref": "task:1",
                    "memory_type": "procedure",
                    "extra_json": {"execution": {"verification_status": "verified"}},
                }
            ]
        ),
        build_injection_block=lambda items: ("执行经验块", ["x1"]),
    )
    engine = _make_default_engine(
        retriever=cast(Any, retriever),
        execution_retriever=cast(Any, execution_retriever),
    )

    result = await engine.query(
        MemoryQuery(
            text="启动项目",
            intent="context",
            context={"execution": {"project_id": "xiaoman-agent"}},
        )
    )

    assert result.text_block == "个人记忆块\n\n执行经验块"
    assert [record.id for record in result.records if record.injected] == ["p1", "x1"]
    assert retriever.retrieve.await_args.kwargs["memory_types"] == [
        "preference",
        "profile",
        "event",
    ]
    execution_retriever.retrieve.assert_awaited_once()


async def test_default_memory_engine_retrieve_keeps_raw_items_and_mode_trace():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "e1",
                    "summary": "用户昨天提过 FitBit",
                    "score": 0.81,
                    "source_ref": "telegram:1@seed",
                    "memory_type": "event",
                    "extra_json": {"origin": "test"},
                }
            ]
        ),
        build_injection_block=lambda items: ("历史块", ["e1"]),
    )
    engine = _make_default_engine(retriever=cast(Any, retriever))

    result = await engine.query(
        MemoryQuery(
            text="Fitbit 型号",
            intent="context",
            scope=MemoryScope(session_key="telegram:1"),
            filters=MemoryQueryFilters(
                kinds=("event",),
                hints={"require_scope_match": True},
            ),
            limit=2,
        )
    )

    assert result.text_block == "历史块"
    assert result.trace["intent"] == "context"
    raw = cast(dict[str, object], result.raw)
    raw_items = cast(list[object], raw["items"])
    assert cast(dict[str, object], raw_items[0])["id"] == "e1"
    assert result.records[0].id == "e1"
    assert result.records[0].injected is True


async def test_default_memory_engine_interest_preserves_read_only_effect():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "p1",
                    "summary": "用户偏好中文回复",
                    "score": 0.8,
                    "source_ref": "telegram:1@seed",
                    "memory_type": "preference",
                    "extra_json": {},
                }
            ]
        ),
        build_injection_block=lambda items: ("", []),
    )
    engine = _make_default_engine(retriever=cast(Any, retriever))

    result = await engine.query(
        MemoryQuery(
            text="中文回复",
            intent="interest",
            effect="read_only",
            scope=MemoryScope(session_key="telegram:1"),
            limit=2,
        )
    )

    assert result.trace["intent"] == "interest"
    assert result.trace["effect"] == "read_only"
    assert result.records[0].id == "p1"
    retriever.retrieve.assert_awaited_once()


async def test_default_memory_engine_retrieve_falls_back_to_session_scope():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=[]),
        build_injection_block=lambda items: ("", []),
    )
    engine = _make_default_engine(retriever=cast(Any, retriever))

    await engine.query(
        MemoryQuery(
            text="作用域测试",
            intent="context",
            scope=MemoryScope(session_key="telegram:test_user"),
            filters=MemoryQueryFilters(hints={"require_scope_match": True}),
        )
    )

    kwargs = retriever.retrieve.await_args.kwargs
    assert kwargs["scope_channel"] == "telegram"
    assert kwargs["scope_chat_id"] == "test_user"
    assert kwargs["require_scope_match"] is True
    assert "keyword_only_enabled" not in kwargs


async def test_default_engine_keeps_history_injected_ids():
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                {
                    "id": "e1",
                    "summary": "用户昨天提过 FitBit",
                    "score": 0.81,
                    "source_ref": "telegram:1@seed",
                    "memory_type": "event",
                    "extra_json": {"origin": "engine"},
                }
            ]
        ),
        build_injection_block=lambda items: (
            "## 【相关历史】\n- 用户昨天提过 FitBit",
            ["e1"],
        ),
    )
    engine = _make_default_engine(retriever=cast(Any, retriever))

    history_result = await engine.query(
        MemoryQuery(
            text="Fitbit 型号",
            intent="context",
            scope=MemoryScope(
                session_key="telegram:1", channel="telegram", chat_id="1"
            ),
            filters=MemoryQueryFilters(
                kinds=("event",),
                hints={"require_scope_match": True},
            ),
            limit=8,
        )
    )

    assert "用户昨天提过 FitBit" in history_result.text_block
    assert [record.id for record in history_result.records if record.injected] == ["e1"]


async def test_execution_memory_feedback_only_uses_matching_tool_outcome(
    tmp_path: Path,
):
    event_bus = EventBus()
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        event_publisher=event_bus,
    )
    store = MemoryStore2(tmp_path / "memory2.db", vec_dim=4)
    result = store.upsert_item(
        memory_type="procedure",
        summary="执行命令时使用 shell",
        embedding=None,
        source_ref="task:seed",
        extra={"tool_requirement": "shell"},
    )
    item_id = result.split(":", 1)[1]
    store.execution.upsert(
        build_execution_state(
            item_id=item_id,
            source_ref="task:seed",
            metadata={"tool_requirement": "shell"},
        )
    )
    engine._v2_store = store
    engine._wire_memory2_events()

    await event_bus.fanout(
        TurnCommitted(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            input_message="列出文件",
            persisted_user_message="列出文件",
            assistant_response="完成",
            tools_used=["shell"],
            tool_chain_raw=[
                {
                    "text": "",
                    "calls": [{"name": "shell", "status": "success", "result": "ok"}],
                }
            ],
            extra={
                "memory_retrieval": {
                    "execution_memory_ids": [item_id],
                    "used_execution_memory_ids": [item_id],
                }
            },
        )
    )

    state = store.execution.get(item_id)
    assert state is not None
    assert state.success_count == 1
    assert state.failure_count == 0
    store.close()
    await event_bus.aclose()


async def test_default_memory_engine_remember_uses_memorizer():
    memorizer = SimpleNamespace(
        save_item_with_supersede=AsyncMock(return_value="new:memu-1")
    )
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        memorizer=cast(Any, memorizer),
    )

    result = await engine.mutate(
        MemoryMutation(
            kind="remember",
            summary="以后用中文回复",
            memory_kind="preference",
            scope=MemoryScope(session_key="cli:1", channel="cli", chat_id="1"),
        )
    )

    assert result.item_id == "memu-1"
    assert result.status == "new"
    memorizer.save_item_with_supersede.assert_awaited_once()


async def test_default_memory_engine_remember_merged_keeps_target_id_alive():
    memorizer = SimpleNamespace(
        save_item_with_supersede=AsyncMock(return_value="merged:memu-1")
    )
    engine = _make_default_engine(
        retriever=cast(Any, SimpleNamespace()),
        memorizer=cast(Any, memorizer),
    )

    result = await engine.mutate(
        MemoryMutation(
            kind="remember",
            summary="以后用中文回复",
            memory_kind="preference",
            scope=MemoryScope(session_key="cli:1", channel="cli", chat_id="1"),
        )
    )

    assert result.item_id == "memu-1"
    assert result.status == "merged"
    assert result.affected_ids == []


def test_default_memory_engine_descriptor_does_not_claim_direct_message_ingest():
    descriptor = DefaultMemoryEngine.DESCRIPTOR

    assert descriptor.profile == EngineProfile.RICH_MEMORY_ENGINE
    assert MemoryCapability.INGEST_MESSAGES not in descriptor.capabilities
    assert MemoryCapability.INGEST_TEXT not in descriptor.capabilities


def test_build_memory_runtime_uses_memory_plugin(monkeypatch, tmp_path: Path):
    import bootstrap.memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "register_memory_meta_tools",
        lambda *args, **kwargs: None,
    )

    captured: dict[str, object] = {}

    class _CustomEngine:
        def describe(self):
            return SimpleNamespace(name="custom")

    class _CustomPlugin:
        plugin_id = "custom"

        def build(self, deps):
            captured["deps"] = deps
            return MemoryPluginRuntime(engine=cast(Any, _CustomEngine()))

    monkeypatch.setattr(
        "bootstrap.wiring.resolve_memory_plugin",
        lambda name: _CustomPlugin(),
    )

    runtime = build_memory_runtime(
        config=Config(
            provider="test",
            model="gpt-test",
            api_key="k",
            system_prompt="hi",
            memory=MemoryConfig(enabled=True, engine="custom"),
        ),
        workspace=tmp_path,
        tools=ToolRegistry(),
        provider=cast(Any, SimpleNamespace()),
        light_provider=None,
        http_resources=cast(Any, SimpleNamespace(external_default=SimpleNamespace())),
    )

    assert runtime.engine is not None
    assert runtime.engine.describe().name == "custom"
    deps = captured["deps"]
    assert deps.config.model == "gpt-test"
    assert deps.workspace == tmp_path
    assert deps.http_resources is not None


def test_build_memory_runtime_exposes_default_memory_engine(
    monkeypatch,
    tmp_path: Path,
):
    import bootstrap.memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "register_memory_meta_tools",
        lambda *args, **kwargs: None,
    )

    class _MemoryStore:
        def __init__(self, workspace):
            self.workspace = workspace

    class _SkillsLoader:
        def __init__(self, workspace):
            self.workspace = workspace

        def list_skill_records(self, filter_unavailable=False):
            return [SimpleNamespace(name="demo")]

    class _WriteFileTool:
        pass

    class _EditFileTool:
        pass

    class _Store2:
        def __init__(self, db_path):
            self.db_path = db_path

        def close(self):
            return None

    class _Embedder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            return None

    class _Memorizer:
        def __init__(self, store, embedder):
            self.store = store
            self.embedder = embedder

    class _Retriever:
        def __init__(self, store, embedder, **kwargs):
            self.store = store
            self.embedder = embedder
            self.kwargs = kwargs

    monkeypatch.setattr("agent.memory.MemoryStore", _MemoryStore)
    monkeypatch.setattr("agent.skills.SkillsLoader", _SkillsLoader)
    monkeypatch.setattr("agent.tools.filesystem.WriteFileTool", _WriteFileTool)
    monkeypatch.setattr("agent.tools.filesystem.EditFileTool", _EditFileTool)
    monkeypatch.setattr("memory2.store.MemoryStore2", _Store2)
    monkeypatch.setattr("memory2.embedder.Embedder", _Embedder)
    monkeypatch.setattr("memory2.memorizer.Memorizer", _Memorizer)
    monkeypatch.setattr("memory2.retriever.Retriever", _Retriever)

    runtime = build_memory_runtime(
        config=Config(
            provider="test",
            model="gpt-test",
            api_key="k",
            system_prompt="hi",
            memory=MemoryConfig(enabled=True, engine="default"),
        ),
        workspace=tmp_path,
        tools=ToolRegistry(),
        provider=cast(Any, SimpleNamespace()),
        light_provider=None,
        http_resources=cast(Any, SimpleNamespace(external_default=SimpleNamespace())),
    )

    assert runtime.engine is not None
    assert runtime.engine.describe().name == "default"
    assert (
        MemoryCapability.SEMANTICS_RICH_MEMORY in runtime.engine.describe().capabilities
    )
