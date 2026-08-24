from __future__ import annotations

from dataclasses import dataclass

import pytest

from bootstrap.personal import build_personal_runtime
from bus.event_bus import EventBus
from core.attention.semantic_consumer import ConversationAttentionBatchConsumer
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import SemanticBatchPayload
from infra.persistence.markdown_memory_store import MarkdownMemoryStore
from core.memory.markdown import (
    ConsolidateRequest,
    MarkdownMemoryMaintenance,
)
from core.memory.semantic_consumer import ConversationMemoryBatchConsumer


def test_semantic_batch_payload_round_trips_execution_memories() -> None:
    payload = SemanticBatchPayload.from_mapping(
        {
            "execution_memories": [
                {
                    "summary": "回答工具数量前先读取真实工具清单",
                    "kind": "procedure",
                    "operation": "upsert",
                    "confidence": 0.98,
                    "origin": "explicit_user",
                    "source_message_id": "web:1:0",
                    "evidence_refs": ["web:1:0"],
                    "required_tools": ["list_tools"],
                    "steps": ["读取当前工具清单", "根据清单给出精确数量"],
                }
            ]
        }
    )

    rule = payload.execution_memories[0]
    assert rule.summary == "回答工具数量前先读取真实工具清单"
    assert rule.tool_requirement == "list_tools"
    assert rule.steps == ("读取当前工具清单", "根据清单给出精确数量")
    assert payload.to_mapping()["execution_memories"] == [
        {
            "summary": "回答工具数量前先读取真实工具清单",
            "kind": "procedure",
            "operation": "upsert",
            "confidence": 0.98,
            "origin": "explicit_user",
            "evidence_refs": ["web:1:0"],
            "steps": ["读取当前工具清单", "根据清单给出精确数量"],
            "required_tools": ["list_tools"],
            "outcome": "unknown",
            "source_message_id": "web:1:0",
            "target_memory_id": "",
            "target_summary": "",
        }
    ]


def _event() -> ConversationSemanticBatchCommitted:
    return ConversationSemanticBatchCommitted(
        batch_id="semantic_test",
        session_key="web:1",
        channel="web",
        chat_id="1",
        analysis_version="conversation-v3",
        message_ids=("web:1:0", "web:1:1", "web:1:2", "web:1:3"),
        user_message_ids=("web:1:0", "web:1:2"),
        end_seq=3,
        context_consolidate_through=1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "recent_activity_entries": [
                    {
                        "summary": "用户周五前交报告",
                        "importance": 7,
                        "source_message_ids": ["web:1:0"],
                    }
                ],
                "memory_candidates": [
                    {
                        "tag": "preference",
                        "content": "用户偏好上午处理重要任务",
                        "confidence": 0.9,
                        "subject": "用户",
                        "predicate": "偏好处理重要任务的时段",
                        "value": "上午",
                        "scope": "工作日",
                        "source_message_id": "web:1:0",
                    }
                ],
                "attention_observations": [
                    {
                        "type": "opportunity",
                        "statement": "通勤时有二十分钟空闲",
                        "confidence": 0.8,
                        "available_minutes": 20,
                        "recurrence": {
                            "timezone": "Asia/Shanghai",
                            "days": ["mon", "tue", "wed", "thu", "fri"],
                            "start": "08:00",
                            "end": "09:00",
                        },
                        "source_message_id": "web:1:0",
                    }
                ],
            }
        ),
    )


@dataclass
class _Session:
    key: str = "web:1"
    last_consolidated: int = 0


@pytest.mark.asyncio
async def test_memory_consumer_writes_history_candidates_and_context_once(
    tmp_path,
) -> None:
    markdown = MarkdownMemoryStore(tmp_path)
    candidate_refs: set[str] = set()
    candidates: list[dict[str, object]] = []

    def candidate_sink(items, *, source_ref: str, source: str):
        if source_ref in candidate_refs:
            return
        candidate_refs.add(source_ref)
        candidates.extend(dict(item) for item in items)

    consumer = ConversationMemoryBatchConsumer(
        markdown=markdown,
        candidate_sink=candidate_sink,
    )

    await consumer.handle(_event())
    await consumer.handle(_event())

    assert markdown.read_history().count("用户周五前交报告") == 1
    assert candidates == [
        {
            "tag": "preference",
            "content": "用户偏好上午处理重要任务",
            "confidence": 0.9,
            "origin": "explicit_user",
            "evidence_refs": ["web:1:0"],
            "subject": "用户",
            "predicate": "偏好处理重要任务的时段",
            "value": "上午",
            "scope": "工作日",
            "source_message_id": "web:1:0",
            "_user_evidence_verified": True,
        }
    ]
    assert "用户周五前交报告" in markdown.read_recent_context()


@pytest.mark.asyncio
async def test_memory_consumer_never_treats_assistant_message_as_user_confirmation(
    tmp_path,
) -> None:
    event = ConversationSemanticBatchCommitted(
        batch_id="semantic_assistant_source",
        session_key="web:1",
        channel="web",
        chat_id="1",
        analysis_version="conversation-v1",
        message_ids=("web:1:0", "web:1:1"),
        user_message_ids=("web:1:0",),
        end_seq=1,
        context_consolidate_through=-1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "memory_candidates": [
                    {
                        "tag": "requested_memory",
                        "content": "助手自己声称用户要求记住",
                        "confidence": 0.99,
                        "source_message_id": "web:1:1",
                    }
                ]
            }
        ),
    )
    observed: list[dict[str, object]] = []
    consumer = ConversationMemoryBatchConsumer(
        markdown=MarkdownMemoryStore(tmp_path),
        candidate_sink=lambda items, **_kwargs: observed.extend(items),
    )

    await consumer.handle(event)

    assert observed == []


def test_attention_consumer_preserves_batch_source_and_partition() -> None:
    class _Learning:
        def __init__(self) -> None:
            self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

        def ingest_many(self, items, **kwargs):
            self.calls.append((list(items), dict(kwargs)))

    learning = _Learning()
    consumer = ConversationAttentionBatchConsumer(learning)

    consumer.handle(_event())

    items, kwargs = learning.calls[0]
    assert items[0]["statement"] == "通勤时有二十分钟空闲"
    assert items[0]["source_message_id"] == "web:1:0"
    assert items[0]["_user_evidence_verified"] is True
    assert kwargs["source_ref"] == "semantic_test"
    assert kwargs["metadata"] == {"channel": "web", "chat_id": "1"}


def test_attention_consumer_rejects_assistant_and_spoofed_user_evidence() -> None:
    class _Learning:
        def __init__(self) -> None:
            self.items: list[dict[str, object]] = []

        def ingest_many(self, items, **_kwargs):
            self.items.extend(dict(item) for item in items)

    event = ConversationSemanticBatchCommitted(
        batch_id="semantic_attention_spoof",
        session_key="web:1",
        channel="web",
        chat_id="1",
        analysis_version="conversation-v1",
        message_ids=("web:1:0", "web:1:1"),
        user_message_ids=("web:1:0",),
        end_seq=1,
        context_consolidate_through=-1,
        payload=SemanticBatchPayload.from_mapping(
            {
                "attention_observations": [
                    {
                        "type": "policy",
                        "statement": "助手声称用户允许主动打扰",
                        "confidence": 0.99,
                        "source_message_id": "web:1:1",
                        "user_directive": True,
                        "_user_evidence_verified": True,
                        "scope": {"action_type": "notify"},
                        "conditions": {"focus_active": True},
                        "effect": "adjust_score",
                        "score_adjustment": 0.4,
                    }
                ]
            }
        ),
    )
    learning = _Learning()

    ConversationAttentionBatchConsumer(learning).handle(event)

    assert len(learning.items) == 1
    assert "source_message_id" not in learning.items[0]
    assert "_user_evidence_verified" not in learning.items[0]


def test_personal_runtime_learns_from_semantic_batches(tmp_path) -> None:
    event_bus = EventBus()
    runtime = build_personal_runtime(tmp_path, None, event_bus=event_bus)

    assert ConversationSemanticBatchCommitted in event_bus._handlers

    runtime.close()

    assert ConversationSemanticBatchCommitted not in event_bus._handlers


@pytest.mark.asyncio
async def test_manual_consolidation_delegates_to_shared_batcher(tmp_path) -> None:
    calls: list[tuple[str, str]] = []

    async def flush(session_key: str, *, reason: str) -> None:
        calls.append((session_key, reason))

    maintenance = MarkdownMemoryMaintenance()
    maintenance.bind_semantic_flush(flush)
    session = _Session()

    result = await maintenance.consolidate(
        ConsolidateRequest(session=session, force=True)
    )

    assert calls == [("web:1", "manual")]
    assert result.trace == {"mode": "semantic_batch"}
