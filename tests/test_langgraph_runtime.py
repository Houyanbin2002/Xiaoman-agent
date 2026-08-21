from __future__ import annotations

from typing import TypedDict, cast

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.runtime.langgraph_runtime import LangGraphRuntime


class _State(TypedDict):
    value: str


def _approval_graph(runtime: LangGraphRuntime):
    async def approval(_state: _State) -> dict[str, str]:
        answer = interrupt({"question": "continue?"})
        return {"value": str(answer)}

    builder = StateGraph(_State)
    builder.add_node("approval", approval)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder, runtime


@pytest.mark.asyncio
async def test_sqlite_checkpointer_resumes_interrupt_after_runtime_restart(tmp_path):
    checkpoint_path = tmp_path / "checkpoints.db"
    config = {"configurable": {"thread_id": "durable-thread"}}

    first_runtime = LangGraphRuntime(checkpoint_path)
    first_builder, _ = _approval_graph(first_runtime)
    first_graph = first_builder.compile(
        checkpointer=await first_runtime.checkpointer(),
        store=first_runtime.store,
    )
    first = await first_graph.ainvoke(_State(value=""), config, durability="sync")
    assert "__interrupt__" in first
    assert (await first_graph.aget_state(config)).next == ("approval",)
    await first_runtime.aclose()

    second_runtime = LangGraphRuntime(checkpoint_path)
    second_builder, _ = _approval_graph(second_runtime)
    second_graph = second_builder.compile(
        checkpointer=await second_runtime.checkpointer(),
        store=second_runtime.store,
    )
    resumed = cast(
        _State,
        await second_graph.ainvoke(
            Command(resume="yes"),
            config,
            durability="sync",
        ),
    )
    assert resumed["value"] == "yes"
    assert (await second_graph.aget_state(config)).next == ()
    await second_runtime.aclose()
