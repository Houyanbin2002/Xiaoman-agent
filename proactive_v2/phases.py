from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import asyncio
import uuid
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from agent.lifecycle.phase import inspect_phase, topo_sort_modules
from proactive_v2.frame import ProactiveFrame
from agent.runtime.langgraph_runtime import LangGraphRuntime


PROACTIVE_PHASES: tuple[str, ...] = (
    "proactive.tick",
    "proactive.gate",
    "proactive.source",
    "proactive.drift",
    "proactive.prompt",
    "proactive.judge",
    "proactive.resolve",
    "proactive.deliver",
    "proactive.schedule",
)


class ProactiveGraphState(TypedDict):
    run_id: str
    session_key: str
    started_at: str
    base_score: float | None


class ProactivePhaseRunner:
    """Compile proactive phases into a checkpointed LangGraph."""

    def __init__(
        self,
        modules: Iterable[object],
        *,
        graph_runtime: LangGraphRuntime | None = None,
    ) -> None:
        grouped: dict[str, list[object]] = defaultdict(list)
        for module in modules:
            phase = getattr(module, "phase", None)
            if not isinstance(phase, str) or not phase:
                raise RuntimeError(f"Proactive 模块缺少 phase 声明: {type(module).__name__}")
            if phase not in PROACTIVE_PHASES:
                raise RuntimeError(f"未知 proactive phase: {phase}")
            grouped[phase].append(module)
        self._modules_by_phase = {
            phase: topo_sort_modules(grouped.get(phase, []))
            for phase in PROACTIVE_PHASES
        }
        self._graph_runtime = graph_runtime or LangGraphRuntime()
        self._graph: Any | None = None
        self._compile_lock = asyncio.Lock()
        self._frames: dict[str, ProactiveFrame] = {}

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        graph = await self._compiled_graph()
        session_key = str(getattr(frame.input, "session_key", "test"))
        started_at = getattr(frame.input, "started_at", None)
        tick_key = (
            started_at.isoformat()
            if hasattr(started_at, "isoformat")
            else uuid.uuid4().hex
        )
        thread_id = f"proactive:{session_key}:{tick_key}"
        run_id = uuid.uuid4().hex
        self._frames[run_id] = frame
        try:
            _ = cast(
                ProactiveGraphState,
                await graph.ainvoke(
                    ProactiveGraphState(
                        run_id=run_id,
                        session_key=session_key,
                        started_at=str(tick_key),
                        base_score=None,
                    ),
                    {
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": max(30, len(PROACTIVE_PHASES) * 3),
                    },
                    durability="sync",
                ),
            )
            return self._frames[run_id]
        finally:
            self._frames.pop(run_id, None)

    async def _compiled_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        async with self._compile_lock:
            if self._graph is not None:
                return self._graph
            builder = StateGraph(ProactiveGraphState)
            previous = START
            for phase in PROACTIVE_PHASES:
                node_name = phase.replace(".", "_")
                builder.add_node(node_name, self._phase_node(phase))
                builder.add_edge(previous, node_name)
                previous = node_name
            builder.add_edge(previous, END)
            self._graph = builder.compile(
                checkpointer=await self._graph_runtime.checkpointer(),
                store=self._graph_runtime.store,
            )
            return self._graph

    def _phase_node(self, phase: str):
        async def _run(state: ProactiveGraphState) -> dict[str, float | None]:
            frame = self._frames.get(state["run_id"])
            if frame is None:
                raise RuntimeError("proactive tick transient frame is unavailable")
            for module in self._modules_by_phase[phase]:
                runner = getattr(module, "run")
                frame = await runner(frame)
            self._frames[state["run_id"]] = frame
            return {
                "base_score": (
                    frame.output.base_score if frame.output is not None else None
                )
            }

        return _run

    async def aclose(self) -> None:
        await self._graph_runtime.aclose()

    def inspect(self) -> str:
        sections: list[str] = []
        for phase in PROACTIVE_PHASES:
            modules = self._modules_by_phase[phase]
            if not modules:
                continue
            sections.append(f"[{phase}]\n{inspect_phase(modules)}")
        return "\n\n".join(sections)

    @property
    def modules_by_phase(self) -> dict[str, list[object]]:
        return {phase: list(modules) for phase, modules in self._modules_by_phase.items()}
