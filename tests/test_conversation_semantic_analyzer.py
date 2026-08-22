from __future__ import annotations

import json

import pytest

from core.conversation_semantics.analyzer import ConversationSemanticAnalyzer
from core.conversation_semantics.prompt import (
    SEMANTIC_SYSTEM_PROMPT,
    build_semantic_batch_prompt,
)
from core.llm import LLMResponse


class _Provider:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.call_count = 0
        self.calls: list[dict[str, object]] = []

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.call_count += 1
        self.calls.append(dict(kwargs))
        return LLMResponse(content=json.dumps(self.payload, ensure_ascii=False))


MESSAGES = [
    {
        "id": "web:1:0",
        "seq": 0,
        "role": "user",
        "content": "我周五前要交报告，重要任务尽量放上午。",
    },
    {
        "id": "web:1:1",
        "seq": 1,
        "role": "assistant",
        "content": "记下了。",
    },
]


@pytest.mark.asyncio
async def test_analyzer_calls_model_once_and_normalizes_all_partitions() -> None:
    provider = _Provider(
        {
            "recent_activity_entries": [
                {
                    "summary": "用户本周五前交报告",
                    "importance": 7,
                    "source_message_ids": ["web:1:0"],
                }
            ],
            "memory_candidates": [
                {
                    "content": "用户偏好上午处理重要任务",
                    "tag": "preference",
                    "confidence": 0.9,
                    "subject": "用户",
                    "predicate": "偏好处理重要任务的时段",
                    "value": "上午",
                    "scope": "工作日",
                    "attributes": {"importance": "high"},
                    "source_message_id": "web:1:0",
                }
            ],
            "task_events": [
                {
                    "summary": "周五前交报告",
                    "delivery_semantics": "before_deadline",
                    "confidence": 0.95,
                    "source_message_id": "web:1:0",
                }
            ],
            "attention_observations": [
                {
                    "statement": "通勤时有二十分钟空闲",
                    "type": "opportunity",
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
            "execution_memories": [],
            "ignored_partition": [{"unsafe": True}],
        }
    )

    result = await ConversationSemanticAnalyzer(provider, "light").analyze(MESSAGES)

    assert provider.call_count == 1
    assert result.recent_activity_entries[0].summary == "用户本周五前交报告"
    assert result.memory_candidates[0].tag == "preference"
    assert result.memory_candidates[0].subject == "用户"
    assert result.memory_candidates[0].predicate == "偏好处理重要任务的时段"
    assert result.memory_candidates[0].value == "上午"
    assert result.memory_candidates[0].scope == "工作日"
    assert result.memory_candidates[0].attributes == {"importance": "high"}
    assert result.memory_candidates[0].source_message_id == "web:1:0"
    assert result.task_events[0].delivery_semantics == "before_deadline"
    assert result.task_events[0].operation == "upsert"
    assert result.attention_observations[0].attributes["available_minutes"] == 20
    assert result.attention_observations[0].source_message_id == "web:1:0"
    assert provider.calls[0]["messages"][0]["role"] == "system"
    assert provider.calls[0]["messages"][1]["role"] == "user"
    assert provider.calls[0]["tools"] == []
    assert provider.calls[0]["disable_thinking"] is True


@pytest.mark.asyncio
async def test_analyzer_returns_empty_payload_for_non_object_json() -> None:
    provider = _Provider(["ordinary answer"])

    result = await ConversationSemanticAnalyzer(provider, "light").analyze(MESSAGES)

    assert provider.call_count == 1
    assert result == result.empty()


@pytest.mark.asyncio
async def test_analyzer_preserves_explicit_directives_when_model_omits_them() -> None:
    provider = _Provider({"memory_candidates": [], "attention_observations": []})
    messages = [
        {
            "id": "web:explicit:style",
            "role": "user",
            "content": "我喜欢先给结论，再给简短步骤。",
        },
        {
            "id": "web:explicit:channel",
            "role": "user",
            "content": "涉及工作任务时优先在当前对话里回复，不要发到外部群。",
        },
    ]

    result = await ConversationSemanticAnalyzer(provider, "light").analyze(messages)

    assert provider.call_count == 1
    assert {item.attributes["preference_key"] for item in result.memory_candidates} == {
        "response_style",
        "communication_channel",
    }
    assert all(item.confidence == 0.98 for item in result.memory_candidates)
    assert all(item.origin == "explicit_user" for item in result.memory_candidates)


@pytest.mark.asyncio
async def test_analyzer_extracts_explicit_correction_with_replacement_evidence() -> None:
    provider = _Provider(
        {
            "memory_candidates": [
                {
                    "tag": "correction",
                    "content": "以后代码示例改为 Python",
                    "confidence": 0.99,
                    "origin": "user_correction",
                    "source_message_id": "web:explicit:correction",
                    "evidence_refs": ["web:explicit:correction"],
                    "subject": "用户",
                    "predicate": "代码示例语言",
                    "value": "Python",
                    "replaces": "JavaScript",
                    "attributes": {"preference_key": "code_language"},
                }
            ]
        }
    )
    messages = [
        {
            "id": "web:explicit:correction",
            "role": "user",
            "content": "之前记的 JavaScript 偏好作废，以后改为 Python。",
        }
    ]

    result = await ConversationSemanticAnalyzer(provider, "light").analyze(messages)

    assert len(result.memory_candidates) == 1
    candidate = result.memory_candidates[0]
    assert candidate.tag == "correction"
    assert candidate.value == "Python"
    assert candidate.replaces == "JavaScript"
    assert candidate.source_message_id == "web:explicit:correction"


@pytest.mark.asyncio
async def test_explicit_fallback_replaces_same_slot_model_candidate() -> None:
    provider = _Provider(
        {
            "memory_candidates": [
                {
                    "tag": "correction",
                    "content": "用户只允许内部沟通",
                    "confidence": 0.98,
                    "origin": "user_correction",
                    "source_message_id": "web:channel:0",
                    "evidence_refs": ["web:channel:0"],
                    "subject": "用户",
                    "predicate": "沟通渠道",
                    "value": "work_content_internal_only",
                    "replaces": "external_group_allowed",
                    "attributes": {"preference_key": "communication_channel"},
                }
            ]
        }
    )
    messages = [
        {
            "id": "web:channel:0",
            "role": "user",
            "content": "之前允许发群里的规则取消，工作内容只在当前会话处理。",
        }
    ]

    result = await ConversationSemanticAnalyzer(provider, "light").analyze(messages)

    assert len(result.memory_candidates) == 1
    assert result.memory_candidates[0].value == "当前会话"
    assert result.memory_candidates[0].replaces == "群里"


def test_semantic_prompt_uses_generic_memory_and_execution_boundaries() -> None:
    prompt = build_semantic_batch_prompt(MESSAGES)

    assert ConversationSemanticAnalyzer.ANALYSIS_VERSION == "conversation-v3"
    assert "conversation_evidence" in prompt
    assert "tool_chain" not in prompt
    assert "recent_activity_entries" in SEMANTIC_SYSTEM_PROMPT
    assert "普通单次成功不保存" in SEMANTIC_SYSTEM_PROMPT
    assert "密码" in SEMANTIC_SYSTEM_PROMPT
    assert "Obsidian" not in prompt
    assert "Notion" not in prompt
