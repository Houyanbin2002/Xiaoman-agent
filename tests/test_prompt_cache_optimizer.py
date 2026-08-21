from __future__ import annotations

from agent.runtime.prompt_cache import (
    PromptCacheConfig,
    PromptCacheOptimizer,
    tool_schema_fingerprint,
)


def _tool_round(index: int, result: str) -> list[dict]:
    call_id = f"call-{index}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": f"tool_{index}", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def test_cache_breakpoint_compacts_only_tool_results_before_recent_rounds() -> None:
    messages: list[dict] = [
        {"role": "system", "content": "stable-system"},
        {"role": "user", "content": "do work"},
    ]
    for index in range(5):
        messages.extend(_tool_round(index, f"result-{index}-" + "x" * 1200))

    optimizer = PromptCacheOptimizer(
        PromptCacheConfig(
            keep_recent_tool_rounds=3,
            cold_tool_result_chars=500,
            recent_tool_result_chars=5000,
        )
    )
    view = optimizer.prepare_model_messages(messages)

    raw_tool_results = [m["content"] for m in messages if m["role"] == "tool"]
    model_tool_results = [m["content"] for m in view.messages if m["role"] == "tool"]
    assert "tier=cold" in model_tool_results[0]
    assert "tool=tool_0" in model_tool_results[0]
    assert "tier=cold" in model_tool_results[1]
    assert model_tool_results[2:] == raw_tool_results[2:]
    assert raw_tool_results[0].startswith("result-0-")
    assert view.plan.compacted_tool_messages == 2
    assert view.plan.protected_tool_rounds == 3
    assert view.plan.chars_saved > 0


def test_cache_breakpoint_transform_is_deterministic_and_lossless() -> None:
    messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "task"},
        *_tool_round(0, "a" * 3000),
        *_tool_round(1, "b" * 3000),
        *_tool_round(2, "c" * 3000),
        *_tool_round(3, "d" * 3000),
    ]
    optimizer = PromptCacheOptimizer(
        PromptCacheConfig(
            keep_recent_tool_rounds=2,
            cold_tool_result_chars=600,
            recent_tool_result_chars=4000,
        )
    )

    first = optimizer.prepare_model_messages(messages)
    second = optimizer.prepare_model_messages(messages)

    assert first.messages == second.messages
    assert first.plan == second.plan
    assert messages[3]["content"] == "a" * 3000
    assert len(first.messages) == len(messages)


def test_cache_breakpoint_caps_pathological_recent_result_without_touching_state() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *_tool_round(0, "head-" + "z" * 5000 + "-tail"),
    ]
    optimizer = PromptCacheOptimizer(
        PromptCacheConfig(
            keep_recent_tool_rounds=3,
            cold_tool_result_chars=500,
            recent_tool_result_chars=1000,
        )
    )

    view = optimizer.prepare_model_messages(messages)

    assert view.plan.compacted_tool_messages == 0
    assert view.plan.capped_recent_tool_messages == 1
    assert "tier=recent-capped" in view.messages[-1]["content"]
    assert view.messages[-1]["content"].endswith("-tail")
    assert len(messages[-1]["content"]) > 5000


def test_cache_breakpoint_collapses_cold_tool_artifact_blocks() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *_tool_round(0, "ok"),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "tool artifact"},
                {"type": "text", "text": "x" * 3000},
            ],
        },
        *_tool_round(1, "ok"),
        *_tool_round(2, "ok"),
        *_tool_round(3, "ok"),
    ]
    optimizer = PromptCacheOptimizer(
        PromptCacheConfig(
            keep_recent_tool_rounds=3,
            cold_tool_result_chars=500,
            recent_tool_result_chars=2000,
        )
    )

    view = optimizer.prepare_model_messages(messages)

    assert view.plan.compacted_artifact_messages == 1
    assert "historical_tool_artifact" in view.messages[3]["content"]
    assert isinstance(messages[3]["content"], list)


def test_tool_schema_fingerprint_ignores_mapping_key_order() -> None:
    left = [{"type": "function", "function": {"name": "read", "strict": True}}]
    right = [{"function": {"strict": True, "name": "read"}, "type": "function"}]

    assert tool_schema_fingerprint(left) == tool_schema_fingerprint(right)
