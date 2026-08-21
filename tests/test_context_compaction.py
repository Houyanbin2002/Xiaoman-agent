from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.core.types import ContextRenderResult, ContextRequest, ReasonerResult
from agent.looping.ports import LLMConfig
from agent.runtime.context_compaction import (
    ContextCompactionConfig,
    ContextSummaryState,
    SUMMARY_METADATA_KEY,
    select_compaction_boundary,
)
from core.llm import LLMResponse
from session.manager import Session


def _message(role: str, content: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _turn_message(content: str = "继续") -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc),
        metadata={},
    )


def _reasoner(
    *,
    light_chat: AsyncMock,
    save_async: AsyncMock,
) -> DefaultReasoner:
    def render(request: ContextRequest, **_: object) -> ContextRenderResult:
        return ContextRenderResult(
            system_prompt="stable-system",
            messages=[
                {"role": "system", "content": "stable-system"},
                *request.history,
                {"role": "user", "content": request.current_message},
            ],
        )

    return DefaultReasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=SimpleNamespace(chat=AsyncMock()),
                light_provider=SimpleNamespace(chat=light_chat),
            ),
        ),
        llm_config=LLMConfig(
            model="main",
            light_model="light",
            max_iterations=4,
            max_tokens=512,
        ),
        tools=cast(
            Any,
            SimpleNamespace(
                get_always_on_names=lambda: set(),
                get_schemas=lambda names=None: [],
                get_tool=lambda name: None,
            ),
        ),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=2_000,
        context=cast(Any, SimpleNamespace(render=render)),
        session_manager=cast(Any, SimpleNamespace(save_async=save_async)),
        context_compaction_config=ContextCompactionConfig(
            enabled=True,
            trigger_tokens=8_000,
            target_tokens=7_000,
            keep_recent_tokens=4_000,
            summary_max_tokens=256,
            chunk_tokens=4_000,
            max_history_messages=2_000,
        ),
    )


def test_boundary_only_splits_before_a_user_turn() -> None:
    messages = [
        _message("user", "first"),
        {
            **_message("assistant", "used a tool"),
            "tool_chain": [
                {"calls": [{"call_id": "c1", "name": "shell", "result": "ok"}]}
            ],
        },
        _message("user", "second"),
        _message("assistant", "done"),
    ]

    boundary = select_compaction_boundary(
        messages,
        start_index=0,
        keep_recent_tokens=10_000,
        protect_recent_tool_rounds=0,
    )

    assert boundary is not None
    assert boundary.end_index == 2
    assert [item["role"] for item in boundary.cold_messages] == [
        "user",
        "assistant",
    ]
    assert boundary.cold_messages[1]["tool_chain"][0]["calls"][0]["name"] == "shell"
    assert boundary.recent_messages[0]["role"] == "user"


def test_boundary_keeps_recent_tool_turns_in_cache_hot_suffix() -> None:
    messages: list[dict[str, Any]] = []
    for index in range(4):
        messages.extend(
            [
                _message("user", f"request-{index}"),
                {
                    **_message("assistant", f"reply-{index}"),
                    "tool_chain": [
                        {
                            "calls": [
                                {
                                    "call_id": f"c{index}",
                                    "name": "shell",
                                    "result": "x" * 2_000,
                                }
                            ]
                        }
                    ],
                },
            ]
        )

    boundary = select_compaction_boundary(
        messages,
        start_index=0,
        keep_recent_tokens=10,
        protect_recent_tool_rounds=3,
    )

    assert boundary is not None
    assert boundary.end_index == 2
    assert boundary.recent_messages[0]["content"] == "request-1"


def test_watermark_compaction_persists_epoch_and_keeps_summary_prefix_stable() -> None:
    light_chat = AsyncMock(
        return_value=LLMResponse(content="保留目标、约束和未完成步骤。")
    )
    save_async = AsyncMock()
    reasoner = _reasoner(light_chat=light_chat, save_async=save_async)
    reasoner.run = AsyncMock(
        return_value=ReasonerResult(
            reply="完成",
            metadata={"tools_used": [], "tool_chain": []},
        )
    )
    session = Session(
        key="cli:1",
        messages=[
            _message("user", "A" * 5_000),
            _message("assistant", "B" * 5_000),
            _message("user", "C" * 5_000),
            _message("assistant", "D" * 5_000),
            _message("user", "E" * 5_000),
            _message("assistant", "F" * 5_000),
        ],
    )

    first = asyncio.run(reasoner.run_turn(msg=_turn_message(), session=session))

    state = ContextSummaryState.from_metadata(session.metadata)
    assert first.reply == "完成"
    assert state is not None
    assert state.epoch == 1
    assert state.summarized_through == session.last_consolidated
    assert session.messages[0]["content"] == "A" * 5_000
    assert save_async.await_count == 1
    assert light_chat.await_count >= 1
    first_model_messages = reasoner.run.await_args.args[0]
    first_summary_frame = first_model_messages[1]
    assert "conversation_summary" in str(first_summary_frame["content"])
    assert "epoch=1" in str(first_summary_frame["content"])

    # No new persisted history: the summary epoch and its cache prefix remain
    # byte-for-byte stable on the next turn.
    reasoner.run.reset_mock()
    light_chat.reset_mock()
    second = asyncio.run(reasoner.run_turn(msg=_turn_message(), session=session))
    second_model_messages = reasoner.run.await_args.args[0]

    assert second.reply == "完成"
    assert light_chat.await_count == 0
    assert second_model_messages[1] == first_summary_frame
    assert session.metadata[SUMMARY_METADATA_KEY]["epoch"] == 1


def test_summary_commit_failure_does_not_advance_cursor() -> None:
    light_chat = AsyncMock(return_value=LLMResponse(content="摘要"))
    save_async = AsyncMock(side_effect=OSError("disk unavailable"))
    reasoner = _reasoner(light_chat=light_chat, save_async=save_async)
    reasoner.run = AsyncMock()
    session = Session(
        key="cli:1",
        messages=[
            _message("user", "A" * 8_000),
            _message("assistant", "B" * 8_000),
            _message("user", "C" * 8_000),
            _message("assistant", "D" * 8_000),
        ],
    )

    result = asyncio.run(reasoner.run_turn(msg=_turn_message(), session=session))

    assert "原对话没有丢失" in str(result.reply)
    assert session.last_consolidated == 0
    assert SUMMARY_METADATA_KEY not in session.metadata
    reasoner.run.assert_not_awaited()


def test_enabled_compaction_does_not_apply_legacy_message_count_window() -> None:
    light_chat = AsyncMock()
    reasoner = _reasoner(light_chat=light_chat, save_async=AsyncMock())
    reasoner._context_compaction = ContextCompactionConfig(  # noqa: SLF001
        enabled=True,
        trigger_tokens=200_000,
        target_tokens=100_000,
        keep_recent_tokens=40_000,
    ).normalized()
    reasoner.run = AsyncMock(
        return_value=ReasonerResult(
            reply="ok",
            metadata={"tools_used": [], "tool_chain": []},
        )
    )
    session = Session(
        key="cli:many",
        messages=[
            _message("user" if index % 2 == 0 else "assistant", str(index))
            for index in range(2_100)
        ],
    )

    result = asyncio.run(reasoner.run_turn(msg=_turn_message(), session=session))
    model_messages = reasoner.run.await_args.args[0]

    assert result.reply == "ok"
    assert len(model_messages) == 2_102  # system + complete history + current user
    assert light_chat.await_count == 0


def test_compaction_config_has_hysteresis() -> None:
    config = ContextCompactionConfig(
        trigger_tokens=200_000,
        target_tokens=100_000,
        keep_recent_tokens=40_000,
    ).normalized()

    assert config.keep_recent_tokens < config.target_tokens < config.trigger_tokens
