from typing import Any, cast
import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import ToolDiscoveryState
from agent.core.runtime_support import LLMServices
from agent.core.types import ContextRenderResult
from agent.core.types import ContextRequest
from agent.core.types import ReasonerResult
from agent.looping.ports import LLMConfig
from core.llm import ContentSafetyError, ContextLengthError


def _stub_turn_injection_context(
    *, turn_injection_prompt: str | None = None
) -> dict[str, str]:
    if not turn_injection_prompt:
        return {}
    return {"turn_injection": turn_injection_prompt}


def _msg():
    return SimpleNamespace(
        content="hello",
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime.now(timezone.utc),
    )


def _session():
    return SimpleNamespace(
        key="s:1",
        messages=[{"role": "user", "content": str(i)} for i in range(6)],
        get_history=lambda max_messages: [{"role": "user", "content": str(i)} for i in range(6)],
        last_consolidated=3,
    )


def _make_reasoner(*, discovery: ToolDiscoveryState, tool_search_enabled: bool):
    def _render(request: ContextRequest, **kwargs: object) -> ContextRenderResult:
        messages = list(request.history) + [{"role": "user"}]
        return ContextRenderResult(
            system_prompt="",
            turn_injection_context=_stub_turn_injection_context(
                turn_injection_prompt=request.turn_injection_prompt
            ),
            messages=messages,
            debug_breakdown=[],
        )

    return DefaultReasoner(
        llm=cast(Any, LLMServices(provider=SimpleNamespace(chat=AsyncMock()), light_provider=SimpleNamespace())),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=256),
        tools=cast(Any, SimpleNamespace(get_always_on_names=lambda: {"always"}, get_schemas=lambda names=None: [], get_tool=lambda name: None)),
        discovery=discovery,
        tool_search_enabled=tool_search_enabled,
        memory_window=10,
        context=cast(Any, SimpleNamespace(
            render=_render,
        )),
        session_manager=cast(Any, SimpleNamespace(save_async=AsyncMock())),
    )


def test_reasoner_run_turn_does_not_rewrite_context_after_safety_rejection():
    discovery = ToolDiscoveryState()
    discovery._unlocked = {"s:1": OrderedDict({"old": None})}
    reasoner = _make_reasoner(discovery=discovery, tool_search_enabled=True)
    reasoner.run = AsyncMock(
        side_effect=[
            ContentSafetyError("blocked"),
            ReasonerResult(
                reply="ok",
                metadata={"tools_used": ["tool_search", "x"], "tool_chain": []},
            ),
        ]
    )

    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=cast(Any, _session())))

    assert "安全审查" in str(result.reply)
    assert result.tools_used == []
    assert result.tool_chain == []
    assert result.thinking is None
    assert result.context_retry["selected_plan"] is None
    assert "x" not in discovery._unlocked["s:1"]
    assert reasoner.run.await_count == 1


def test_reasoner_run_turn_context_length_all_fail_returns_fallback():
    reasoner = _make_reasoner(discovery=ToolDiscoveryState(), tool_search_enabled=False)
    reasoner.run = AsyncMock(side_effect=[ContextLengthError("long")] * 7)

    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=cast(Any, _session())))
    assert "上下文摘要后仍超过模型限制" in str(result.reply)
    assert result.tools_used == []
    assert result.tool_chain == []
    assert reasoner.run.await_count == 1


def test_reasoner_run_turn_context_length_never_drops_prompt_sections():
    calls: list[dict] = []

    def _render(request: ContextRequest, **kwargs: object) -> ContextRenderResult:
        calls.append(
            {
                "history_len": len(request.history),
                "disabled_sections": set(request.disabled_sections or set()),
            }
        )
        return ContextRenderResult(
            system_prompt="",
            turn_injection_context=_stub_turn_injection_context(
                turn_injection_prompt=request.turn_injection_prompt
            ),
            messages=list(request.history) + [{"role": "user"}],
            debug_breakdown=[],
        )

    discovery = ToolDiscoveryState()
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=SimpleNamespace(chat=AsyncMock()), light_provider=SimpleNamespace())),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=256),
        tools=cast(Any, SimpleNamespace(get_always_on_names=lambda: {"always"}, get_schemas=lambda names=None: [], get_tool=lambda name: None)),
        discovery=discovery,
        tool_search_enabled=False,
        memory_window=10,
        context=cast(Any, SimpleNamespace(
            render=_render,
        )),
        session_manager=cast(Any, SimpleNamespace(save_async=AsyncMock())),
    )
    reasoner.run = AsyncMock(
        side_effect=[
            ContextLengthError("long"),
            ReasonerResult(
                reply="ok",
                metadata={"tools_used": [], "tool_chain": []},
            ),
        ]
    )

    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=cast(Any, _session())))
    assert "上下文摘要后仍超过模型限制" in str(result.reply)
    assert result.tools_used == []
    assert result.tool_chain == []
    assert calls[0]["history_len"] == 6
    assert calls[0]["disabled_sections"] == set()
    assert len(calls) == 1
    assert reasoner.run.await_count == 1
