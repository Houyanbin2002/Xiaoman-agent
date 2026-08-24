from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

import agent.core.passive_support as support
from agent.capabilities import CapabilityCatalog, CapabilityRouter
from agent.core.runtime_support import ToolDiscoveryState
from agent.core.contracts import ContextStore, Reasoner
from agent.core.types import (
    ContextBundle,
    LLMToolCall,
    ReasonerResult,
)
from agent.prompting import is_context_frame
from core.llm import ContentSafetyError, ContextLengthError
from core.tracing import record_trace_event
from agent.retrieval.protocol import RetrievalRequest, RetrievalResult
from agent.runtime.execution_policy import (
    AgentExecutionPolicy,
    PASSIVE_EXECUTION_POLICY,
    ProgressSummary,
)
from agent.runtime.execution_guard import ExecutionGuard, ExecutionGuardConfig
from agent.runtime.context_compaction import (
    ContextCompactionConfig,
    ContextSummaryState,
    build_summary_state,
    chunk_summary_evidence,
    estimate_tokens,
    render_summary_evidence,
    select_compaction_boundary,
    summary_source_digest,
    write_summary_state,
)
from agent.runtime.langgraph_agent import LangGraphAgentExecutor
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.runtime.model_step import run_model_step
from agent.runtime.prompt_cache import PromptCacheConfig, PromptCacheOptimizer
from agent.tool_hooks import ToolExecutor
from agent.turns.outbound import OutboundDispatch, OutboundPort
from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import (
    ToolCallCompleted,
    ToolCallStarted,
)
from agent.lifecycle.phase import Phase
from agent.lifecycle.phases.after_reasoning import (
    AfterReasoningFrame,
    default_after_reasoning_modules,
)
from agent.lifecycle.phases.after_step import AfterStepFrame, default_after_step_modules
from agent.lifecycle.phases.after_turn import AfterTurnFrame, default_after_turn_modules
from agent.lifecycle.phases.before_reasoning import (
    BeforeReasoningFrame,
    default_before_reasoning_modules,
)
from agent.lifecycle.phases.before_step import BeforeStepFrame, default_before_step_modules
from agent.lifecycle.phases.before_turn import (
    BeforeTurnFrame,
    MemoryConsolidator,
    default_before_turn_modules,
)
from agent.lifecycle.phases.prompt_render import (
    PromptRenderFrame,
    default_prompt_render_modules,
)
from agent.lifecycle.types import (
    AfterReasoningInput,
    AfterReasoningResult,
    AfterStepCtx,
    BeforeReasoningCtx,
    BeforeReasoningInput,
    BeforeStepCtx,
    BeforeStepInput,
    BeforeTurnCtx,
    PromptRenderInput,
    PromptRenderResult,
    TurnSnapshot,
    TurnState,
)

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.runtime_support import SessionLike, TurnRunResult
    from agent.looping.ports import LLMConfig, LLMServices, SessionServices
    from agent.retrieval.protocol import MemoryRetrievalPipeline
    from agent.tool_hooks.base import ToolHook
    from agent.tools.registry import ToolRegistry
    from session.manager import SessionManager
from core.common.diagnostic_log import diagnostic_context, diagnostic_line

# 1. 统一通过模块 logger 记录关键分支，供排障和回归测试抓取。
logger = logging.getLogger(__name__)

# 被动链路核心入口，负责串起 lifecycle 模块链与 reasoner。
#
# ┌─ inbound
# │  └─ AgentCore.process
# │     └─ PassiveTurnPipeline.run
# │        ├─ BeforeTurn
# │        │  └─ session acquire + ContextStore.prepare + EventBus.emit
# │        ├─ BeforeReasoning
# │        │  └─ tool context sync + EventBus.emit + prompt warmup
# │        ├─ Reasoner.run_turn
# │        │  ├─ PromptRender
# │        │  │  └─ ContextBuilder.render + plugin prompt modules
# │        │  └─ Reasoner.run
# │        │     ├─ BeforeStep
# │        │     │  └─ token estimate + EventBus.emit + hint injection
# │        │     └─ AfterStep
# │        │        └─ EventBus.fanout
# │        ├─ AfterReasoning
# │        │  └─ parse + EventBus.emit + persist + outbound build
# │        └─ AfterTurn
# │           └─ TurnCommitted fanout + AfterTurn fanout + dispatch
# └─ done

# ── 被动 turn 内联常量 ──────────────────────────────────────────
_SUMMARY_MAX_TOKENS = 512
_CONTEXT_SUMMARY_SYSTEM_PROMPT = """你是对话上下文压缩器。请把证据压缩成可供同一个 Agent 继续工作的中文状态摘要。
必须保留：用户目标、明确决定与纠正、偏好和硬约束、关键实体/路径/配置值、已完成工作、工具关键结果与失败原因、未完成事项和下一步。
证据里以键值、清单或状态字段表达的继续执行信息，必须逐项保留其键名与对应值，不得只保留其中一部分。
必须区分用户陈述、Agent 推断和工具证据；不得编造，不得省略仍然有效的约束。
删除寒暄、重复解释、冗长工具原文和已经被新结论取代的信息。只输出摘要正文，不要 JSON。"""


def _turn_log_id(key: str, msg: InboundMessage) -> str:
    raw = f"{key}|{msg.timestamp.isoformat()}|{msg.content[:80]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def _disabled_tools_from_msg(msg: object) -> set[str]:
    metadata: object = getattr(msg, "metadata", None)
    if not isinstance(metadata, dict):
        return set()
    raw = metadata.get("disabled_tools")
    if isinstance(raw, str):
        return {raw} if raw else set()
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw if str(item)}
    return set()


def _permission_mode_from_msg(msg: object) -> str:
    metadata: object = getattr(msg, "metadata", None)
    if not isinstance(metadata, dict):
        return "full_access"
    return str(metadata.get("permission_mode") or "full_access")


class _NoopOutboundPort:
    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        return False


@dataclass
class AgentCoreDeps:
    session: "SessionServices"
    context_store: "ContextStore"
    context: "ContextBuilder"
    tools: "ToolRegistry"
    reasoner: "Reasoner"
    event_bus: "EventBus | None" = None
    outbound_port: "OutboundPort | None" = None
    history_window: int = 500
    memory_consolidator: MemoryConsolidator | None = None
    before_turn_plugin_modules: list[object] | None = None
    before_reasoning_plugin_modules: list[object] | None = None
    before_step_plugin_modules: list[object] | None = None
    after_step_plugin_modules: list[object] | None = None
    after_reasoning_plugin_modules: list[object] | None = None
    after_turn_plugin_modules: list[object] | None = None


class AgentCore:
    """
    ┌──────────────────────────────────────┐
    │ AgentCore                            │
    ├──────────────────────────────────────┤
    │ 1. 持有 PassiveTurnPipeline          │
    │ 2. 委托 pipeline 处理被动消息        │
    └──────────────────────────────────────┘
    """

    def __init__(self, deps: AgentCoreDeps) -> None:
        self._passive_pipeline = PassiveTurnPipeline(deps)

    @property
    def pipeline(self) -> "PassiveTurnPipeline":
        return self._passive_pipeline

    def add_before_turn_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._passive_pipeline.add_before_turn_plugin_modules(modules)

    def add_before_reasoning_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._passive_pipeline.add_before_reasoning_plugin_modules(modules)

    def add_after_reasoning_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._passive_pipeline.add_after_reasoning_plugin_modules(modules)

    def add_after_turn_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._passive_pipeline.add_after_turn_plugin_modules(modules)

    async def process(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        return await self._passive_pipeline.run(
            msg,
            key,
            dispatch_outbound=dispatch_outbound,
        )


class PassiveTurnPipeline:
    """
    ┌──────────────────────────────────────┐
    │ PassiveTurnPipeline                  │
    ├──────────────────────────────────────┤
    │ 1. BeforeTurn（会话准备）             │
    │ 2. BeforeReasoning                   │
    │ 3. 执行 reasoner（含 BeforeStep/AfterStep）│
    │ 4. AfterReasoning（parse + 持久化 + 构建出站消息）│
    │ 5. AfterTurn（TurnCommitted + dispatch） │
    │ 6. 返回出站消息                      │
    └──────────────────────────────────────┘
    """

    def __init__(self, deps: AgentCoreDeps) -> None:
        self._session = deps.session
        self._context_store = deps.context_store
        self._context = deps.context
        self._tools = deps.tools
        self._reasoner = deps.reasoner
        add_before_step = getattr(self._reasoner, "add_before_step_plugin_modules", None)
        if add_before_step is not None:
            add_before_step(list(deps.before_step_plugin_modules or []))
        add_after_step = getattr(self._reasoner, "add_after_step_plugin_modules", None)
        if add_after_step is not None:
            add_after_step(list(deps.after_step_plugin_modules or []))
        self._outbound_port = deps.outbound_port or _NoopOutboundPort()
        self._history_window = deps.history_window
        self._memory_consolidator = deps.memory_consolidator
        self._before_turn_plugin_modules = list(deps.before_turn_plugin_modules or [])
        self._before_reasoning_plugin_modules = list(
            deps.before_reasoning_plugin_modules or []
        )
        self._after_reasoning_plugin_modules = list(
            deps.after_reasoning_plugin_modules or []
        )
        self._after_turn_plugin_modules = list(deps.after_turn_plugin_modules or [])
        bus = deps.event_bus or EventBus()
        self._bus = bus

        self._before_turn = self._build_before_turn_phase()
        self._before_reasoning = self._build_before_reasoning_phase()
        self._after_reasoning = self._build_after_reasoning_phase()
        self._after_turn = self._build_after_turn_phase()

    def add_before_turn_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._before_turn_plugin_modules.extend(modules)
        self._before_turn = self._build_before_turn_phase()

    def add_before_reasoning_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._before_reasoning_plugin_modules.extend(modules)
        self._before_reasoning = self._build_before_reasoning_phase()

    def add_after_reasoning_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._after_reasoning_plugin_modules.extend(modules)
        self._after_reasoning = self._build_after_reasoning_phase()

    def add_after_turn_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._after_turn_plugin_modules.extend(modules)
        self._after_turn = self._build_after_turn_phase()

    def _build_before_turn_phase(self) -> Phase[TurnState, BeforeTurnCtx, BeforeTurnFrame]:
        return Phase(
            default_before_turn_modules(
                self._bus,
                self._session.session_manager,
                self._context_store,
                keep_count=self._history_window,
                consolidator=self._memory_consolidator,
                plugin_modules=cast("list[Any]", self._before_turn_plugin_modules),
            ),
            frame_factory=BeforeTurnFrame,
        )

    def _build_before_reasoning_phase(
        self,
    ) -> Phase[BeforeReasoningInput, BeforeReasoningCtx, BeforeReasoningFrame]:
        return Phase(
            default_before_reasoning_modules(
                self._bus,
                self._tools,
                self._session.session_manager,
                self._context,
                plugin_modules=cast("list[Any]", self._before_reasoning_plugin_modules),
            ),
            frame_factory=BeforeReasoningFrame,
        )

    def _build_after_reasoning_phase(
        self,
    ) -> Phase[AfterReasoningInput, AfterReasoningResult, AfterReasoningFrame]:
        return Phase(
            default_after_reasoning_modules(
                self._bus,
                self._session,
                workspace=getattr(self._context, "workspace", None),
                plugin_modules=cast("list[Any]", self._after_reasoning_plugin_modules),
            ),
            frame_factory=AfterReasoningFrame,
        )

    def _build_after_turn_phase(
        self,
    ) -> Phase[TurnSnapshot, OutboundMessage, AfterTurnFrame]:
        return Phase(
            default_after_turn_modules(
                self._bus,
                self._outbound_port,
                self._context,
                self._history_window,
                plugin_modules=cast("list[Any]", self._after_turn_plugin_modules),
            ),
            frame_factory=AfterTurnFrame,
        )

    # 核心方法：处理一条普通被动消息，并提交最终出站结果。
    async def run(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        started = time.perf_counter()
        turn_id = _turn_log_id(key, msg)
        state = TurnState(
            msg=msg,
            session_key=key,
            dispatch_outbound=dispatch_outbound,
        )
        with diagnostic_context(session=key, flow="passive", turn=turn_id):
            logger.info(
                diagnostic_line(
                    "PassiveTurnPipeline.run",
                    event="start",
                    flow="passive",
                    phase="before_turn",
                    session=key,
                    turn=turn_id,
                    action="run",
                )
            )
            # try/except 只包前置模块链和 reasoning：在派发前兜底并返回错误提示。
            try:
                # Phase 1: BeforeTurn 模块链（会话、上下文、BeforeTurn 事件）。
                with diagnostic_context(phase="before_turn"):
                    before_turn = await self._before_turn.run(state)
                # TurnState 存内部默认 metadata；BeforeTurnCtx 存插件导出，同名 key 以后者覆盖。
                state.extra_metadata.update(before_turn.extra_metadata)
                if before_turn.abort:
                    logger.info(
                        diagnostic_line(
                            "PassiveTurnPipeline.run",
                            event="gate_exit",
                            flow="passive",
                            phase="before_turn",
                            session=key,
                            turn=turn_id,
                            action="abort",
                            reason="before_turn_abort",
                            duration_ms=int((time.perf_counter() - started) * 1000),
                        )
                    )
                    return await self._control_outbound(
                        state,
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=before_turn.abort_reply,
                        ),
                    )
                logger.info(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="end",
                        flow="passive",
                        phase="before_turn",
                        session=key,
                        turn=turn_id,
                        action="continue",
                        duration_ms=int((time.perf_counter() - started) * 1000),
                    )
                )

                # Phase 2: BeforeReasoning 模块链（工具上下文、BeforeReasoning 事件、prompt warmup）。
                with diagnostic_context(phase="before_reasoning"):
                    before_reasoning = await self._before_reasoning.run(
                        BeforeReasoningInput(state=state, before_turn=before_turn)
                    )
                if before_reasoning.abort:
                    logger.info(
                        diagnostic_line(
                            "PassiveTurnPipeline.run",
                            event="gate_exit",
                            flow="passive",
                            phase="before_reasoning",
                            session=key,
                            turn=turn_id,
                            action="abort",
                            reason="before_reasoning_abort",
                            duration_ms=int((time.perf_counter() - started) * 1000),
                        )
                    )
                    return await self._control_outbound(
                        state,
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=before_reasoning.abort_reply,
                        ),
                    )
                logger.info(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="end",
                        flow="passive",
                        phase="before_reasoning",
                        session=key,
                        turn=turn_id,
                        action="continue",
                        counts=f"skills:{len(before_reasoning.skill_names)},hints:{len(before_reasoning.extra_hints)}",
                        duration_ms=int((time.perf_counter() - started) * 1000),
                    )
                )

                # Phase 3-4: Reasoning（BeforeStep/AfterStep 模块链在 Reasoner 内部执行）。
                session = state.session
                if session is None:
                    raise RuntimeError("Passive turn requires TurnState.session")
                with diagnostic_context(phase="reasoner"):
                    turn_result = await self._reasoner.run_turn(
                        msg=msg,
                        skill_names=list(before_reasoning.skill_names) or None,
                        session=session,
                        base_history=None,
                        retrieved_memory_block=before_reasoning.retrieved_memory_block,
                        extra_hints=list(before_reasoning.extra_hints) or None,
                    )
                logger.info(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="end",
                        flow="passive",
                        phase="reasoner",
                        session=key,
                        turn=turn_id,
                        action="continue",
                        duration_ms=int((time.perf_counter() - started) * 1000),
                    )
                )
            except Exception as exc:
                record_trace_event(
                    category="turn",
                    name="failure",
                    summary="主对话链路执行失败",
                    status="failed",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    payload={
                        "phase": "reasoner",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:300],
                    },
                )
                logger.exception(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="phase_error",
                        flow="passive",
                        phase="reasoner",
                        session=key,
                        turn=turn_id,
                        action="fail",
                        reason="provider_error",
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        error_type=type(exc).__name__,
                        note=str(exc)[:160],
                    )
                )
                return await self._control_outbound(
                    state,
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="处理消息时出错，请稍后再试。",
                    ),
                )

            try:
                # Phase 5: AfterReasoning 模块链（parse、AfterReasoning 事件、持久化、出站消息）。
                with diagnostic_context(phase="after_reasoning"):
                    after_reasoning = await self._after_reasoning.run(
                        AfterReasoningInput(state=state, turn_result=turn_result)
                    )
            except Exception as exc:
                logger.exception(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="phase_error",
                        flow="passive",
                        phase="after_reasoning",
                        session=key,
                        turn=turn_id,
                        action="fail",
                        reason="invalid_output",
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        error_type=type(exc).__name__,
                        note=str(exc)[:160],
                    )
                )
                raise
            logger.info(
                diagnostic_line(
                    "PassiveTurnPipeline.run",
                    event="end",
                    flow="passive",
                    phase="after_reasoning",
                    session=key,
                    turn=turn_id,
                    action="continue",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            )

            try:
                # Phase 6: AfterTurn 模块链（TurnCommitted fanout、AfterTurn fanout、dispatch）。
                with diagnostic_context(phase="after_turn"):
                    outbound = await self._after_turn.run(
                        TurnSnapshot(
                            state=state,
                            outbound=after_reasoning.outbound,
                            ctx=after_reasoning.ctx,
                        )
                    )
            except Exception as exc:
                logger.exception(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="phase_error",
                        flow="passive",
                        phase="after_turn",
                        session=key,
                        turn=turn_id,
                        action="fail",
                        reason="write_error",
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        error_type=type(exc).__name__,
                        note=str(exc)[:160],
                    )
                )
                raise
            logger.info(
                diagnostic_line(
                    "PassiveTurnPipeline.run",
                    event="end",
                    flow="passive",
                    phase="after_turn",
                    session=key,
                    turn=turn_id,
                    action="done",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            return outbound

    # abort / 错误路径的统一 dispatch helper，只有 dispatch_outbound=True 时才发送。
    async def _control_outbound(
        self,
        state: TurnState,
        outbound: OutboundMessage,
    ) -> OutboundMessage:
        await self._persist_control_turn(state, outbound)
        if state.dispatch_outbound:
            _ = await self._outbound_port.dispatch(
                OutboundDispatch(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    content=outbound.content,
                    thinking=outbound.thinking,
                    metadata=outbound.metadata,
                    media=outbound.media,
                )
            )
        return outbound

    async def _persist_control_turn(
        self,
        state: TurnState,
        outbound: OutboundMessage,
    ) -> None:
        """Keep abort/error turns visible after refresh without masking recovery."""

        session = state.session
        manager = self._session.session_manager
        append_messages = getattr(manager, "append_messages", None)
        if session is None or not callable(append_messages):
            return
        start = len(getattr(session, "messages", []))
        trace_id = str((state.msg.metadata or {}).get("trace_id") or "").strip()
        trace_kwargs = {"trace_id": trace_id} if trace_id else {}
        try:
            if not bool((state.msg.metadata or {}).get("omit_user_turn")):
                session.add_message(
                    "user",
                    state.msg.content,
                    media=state.msg.media if state.msg.media else None,
                    **trace_kwargs,
                )
            session.add_message(
                "assistant",
                outbound.content,
                control_reply=True,
                **trace_kwargs,
            )
            persist_result = append_messages(session, session.messages[start:])
            if inspect.isawaitable(persist_result):
                await persist_result
        except Exception:
            messages = getattr(session, "messages", None)
            if isinstance(messages, list):
                del messages[start:]
            logger.warning("failed to persist control reply", exc_info=True)


class DefaultContextStore(ContextStore):
    def __init__(
        self,
        *,
        retrieval: "MemoryRetrievalPipeline",
        context: "ContextBuilder",
        history_window: int = 500,
    ) -> None:
        self._retrieval = retrieval
        self._context = context
        self._history_window = max(1, int(history_window))

    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        # 1. 先读取 session history，并转换成 retrieval pipeline 需要的结构。
        # Retrieval only needs the recent conversational frame.  Keeping this
        # bounded also lets a turn continue safely when background memory
        # consolidation is delayed or temporarily unavailable.
        raw_history = list(session.get_history(max_messages=self._history_window))
        history_messages = support.to_history_messages(raw_history)

        # 2. 系统轮次可显式跳过预检索，避免污染检索诊断和激活状态。
        if bool((msg.metadata or {}).get("skip_memory_retrieval")):
            retrieval_result = RetrievalResult(block="", trace=None)
        else:
            retrieval_result = await self._retrieval.retrieve(
                RetrievalRequest(
                    message=msg.content,
                    session_key=session_key,
                    channel=msg.context_channel,
                    chat_id=msg.context_chat_id,
                    history=history_messages,
                    session_metadata=(
                        session.metadata if isinstance(session.metadata, dict) else {}
                    ),
                    timestamp=msg.timestamp,
                )
            )

        # 3. 最后补齐 ContextBundle，把主链正式字段直接收进显式合同。
        skill_names = [
            record.name
            for record in self._context.skills.list_skill_records(
                filter_unavailable=False
            )
        ]
        skill_mentions = support.collect_skill_mentions(
            msg.content,
            skill_names,
        )
        return ContextBundle(
            history=support.to_chat_messages(raw_history),
            memory_blocks=[retrieval_result.block] if retrieval_result.block else [],
            skill_mentions=skill_mentions,
            retrieved_memory_block=retrieval_result.block or "",
            retrieval_trace_raw=(
                retrieval_result.trace.raw
                if retrieval_result.trace is not None
                else None
            ),
            retrieval_metadata=dict(retrieval_result.metadata or {}),
            history_messages=history_messages,
        )

class AgentExecutionKernel(Reasoner):
    """唯一的 Agent 执行内核；主 Agent/SubAgent 仅注入不同状态与策略。"""

    def __init__(
        self,
        llm: "LLMServices",
        llm_config: "LLMConfig",
        tools: "ToolRegistry",
        discovery: ToolDiscoveryState,
        *,
        tool_search_enabled: bool,
        memory_window: int,
        context: "ContextBuilder | None" = None,
        session_manager: "SessionManager | None" = None,
        event_bus: "EventBus | None" = None,
        execution_policy: AgentExecutionPolicy = PASSIVE_EXECUTION_POLICY,
        graph_runtime: LangGraphRuntime | None = None,
        prompt_cache_config: PromptCacheConfig | None = None,
        context_compaction_config: ContextCompactionConfig | None = None,
        execution_guard_config: ExecutionGuardConfig | None = None,
    ) -> None:
        self._llm = llm
        self._llm_config = llm_config
        self._tools = tools
        self._discovery = discovery
        self._tool_search_enabled = tool_search_enabled
        self._memory_window = memory_window
        self._context = context
        self._capability_router = CapabilityRouter(
            CapabilityCatalog(
                tools,
                getattr(context, "skills", None) if context is not None else None,
            )
        )
        self._session_manager = session_manager
        self._event_bus = event_bus
        self._execution_policy = execution_policy
        self._prompt_cache = PromptCacheOptimizer(prompt_cache_config)
        self._context_compaction = (
            context_compaction_config or ContextCompactionConfig(enabled=False)
        ).normalized()
        self._execution_guard = ExecutionGuard(execution_guard_config)
        self._graph_runtime = graph_runtime or LangGraphRuntime()
        self._graph_executor = LangGraphAgentExecutor(self, self._graph_runtime)
        self._prompt_render_plugin_modules: list[object] = []
        self._before_step_plugin_modules: list[object] = []
        self._after_step_plugin_modules: list[object] = []
        self._tool_executor = ToolExecutor([])
        self._stream_sink_factory: Callable[
            [object], Callable[[dict[str, str] | str], Awaitable[None]] | None
        ] | None = None
        bus = event_bus or EventBus()
        self._bus = bus
        self._before_step = self._build_before_step_phase()
        self._after_step = self._build_after_step_phase()
        self._prompt_render: Phase[
            PromptRenderInput,
            PromptRenderResult,
            PromptRenderFrame,
        ] | None = (
            self._build_prompt_render_phase(context)
            if context is not None
            else None
        )

    def add_tool_hooks(self, hooks: list["ToolHook"]) -> None:
        self._tool_executor.add_hooks(hooks)

    def add_prompt_render_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._prompt_render_plugin_modules.extend(modules)
        if self._context is not None:
            self._prompt_render = self._build_prompt_render_phase(self._context)

    def add_before_step_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._before_step_plugin_modules.extend(modules)
        self._before_step = self._build_before_step_phase()

    def add_after_step_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._after_step_plugin_modules.extend(modules)
        self._after_step = self._build_after_step_phase()

    def _build_before_step_phase(
        self,
    ) -> Phase[BeforeStepInput, BeforeStepCtx, BeforeStepFrame]:
        return Phase(
            default_before_step_modules(
                self._bus,
                plugin_modules=cast("list[Any]", self._before_step_plugin_modules),
            ),
            frame_factory=BeforeStepFrame,
        )

    def _build_after_step_phase(self) -> Phase[AfterStepCtx, AfterStepCtx, AfterStepFrame]:
        return Phase(
            default_after_step_modules(
                self._bus,
                plugin_modules=cast("list[Any]", self._after_step_plugin_modules),
            ),
            frame_factory=AfterStepFrame,
        )

    def _build_prompt_render_phase(
        self,
        context: "ContextBuilder",
    ) -> Phase[PromptRenderInput, PromptRenderResult, PromptRenderFrame]:
        return Phase(
            default_prompt_render_modules(
                self._bus,
                context,
                plugin_modules=cast("list[Any]", self._prompt_render_plugin_modules),
            ),
            frame_factory=PromptRenderFrame,
        )

    async def render_prompt(
        self,
        input: PromptRenderInput,
    ) -> PromptRenderResult:
        if self._context is None:
            raise RuntimeError("AgentExecutionKernel.render_prompt requires context")
        if self._prompt_render is None:
            self._prompt_render = self._build_prompt_render_phase(self._context)
        return await self._prompt_render.run(input)

    def set_stream_sink_factory(
        self,
        factory: Callable[
            [object], Callable[[dict[str, str] | str], Awaitable[None]] | None
        ]
        | None,
    ) -> None:
        self._stream_sink_factory = factory

    async def run_turn(
        self,
        *,
        msg: InboundMessage,
        session: "SessionLike",
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
        extra_hints: list[str] | None = None,
    ) -> "TurnRunResult":
        from agent.core.runtime_support import TurnRunResult

        if self._context is None or self._session_manager is None:
            raise RuntimeError("AgentExecutionKernel.run_turn requires context and session_manager")
        if self._prompt_render is None:
            self._prompt_render = self._build_prompt_render_phase(self._context)

        # 1. 先准备 trace、history 和 preload 工具集合。
        retry_attempts: list[dict[str, object]] = []
        retry_trace: dict[str, object] = {
            "attempts": retry_attempts,
            "selected_plan": None,
            "trimmed_sections": [],
            "compaction": {},
        }
        summary_state: ContextSummaryState | None = None
        if base_history is None and self._context_compaction.enabled:
            summary_state = ContextSummaryState.from_metadata(
                getattr(session, "metadata", None)
            )
            if summary_state is not None:
                # The durable summary state is authoritative after restart.
                session.last_consolidated = summary_state.summarized_through
            elif int(getattr(session, "last_consolidated", 0) or 0) > 0:
                # Legacy semantic extraction could advance this cursor without
                # producing a context summary.  Replay the durable transcript
                # once instead of silently losing those messages.
                session.last_consolidated = 0
                await self._session_manager.save_async(cast(Any, session))

        def load_history() -> list[dict]:
            history_window = self._memory_window
            if base_history is None and self._context_compaction.enabled:
                # Token watermarks, not a message-count window, own the model
                # context.  Include every unsummarized durable turn so no old
                # messages disappear before the summary transaction commits.
                history_window = max(
                    history_window,
                    len(getattr(session, "messages", ()) or ()) + 1,
                )
            raw = (
                list(base_history)
                if base_history is not None
                else get_history_since_consolidated(session, history_window)
            )
            if base_history is None and summary_state is not None:
                return self._history_with_summary(summary_state, raw)
            return raw

        source_history = load_history()
        disabled_tools = _disabled_tools_from_msg(msg)
        permission_mode = _permission_mode_from_msg(msg)
        preloaded: set[str] | None = None
        preloaded_order: list[str] = []
        capability_route = self._capability_router.route(
            msg.content,
            explicit_skills=skill_names,
            visible_tools=self._tools.get_always_on_names() | disabled_tools,
            max_tools=3 if self._tool_search_enabled else 0,
        )
        skill_names = list(capability_route.active_skills)
        if self._tool_search_enabled:
            preloaded_order = self._discovery.get_preloaded_ordered(session.key)
            preloaded_order = list(
                dict.fromkeys(
                    [*preloaded_order, *capability_route.preloaded_tools]
                )
            )
            preloaded = set(preloaded_order)
            logger.info(
                "[capability_router] preloaded=%s active_skills=%s",
                preloaded_order if preloaded_order else "[]",
                skill_names,
            )
        stream_sink = (
            self._stream_sink_factory(msg) if self._stream_sink_factory is not None else None
        )

        turn_injection_prompt = build_turn_injection_prompt(
            tools=self._tools,
            tool_search_enabled=self._tool_search_enabled,
            visible_names=(
                (preloaded or set()) | disabled_tools
                if self._tool_search_enabled
                else None
            ),
        )
        route_prompt = capability_route.prompt()
        if route_prompt:
            turn_injection_prompt = "\n\n".join(
                part
                for part in (route_prompt, turn_injection_prompt.strip())
                if part
            )

        async def render_current_history() -> tuple[list[dict], int]:
            prompt_render = await self.render_prompt(
                PromptRenderInput(
                    session_key=session.key,
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=msg.content,
                    media=msg.media if msg.media else None,
                    timestamp=msg.timestamp,
                    history=source_history,
                    skill_names=skill_names,
                    retrieved_memory_block=retrieved_memory_block,
                    disabled_sections=set(),
                    turn_injection_prompt=turn_injection_prompt,
                    extra_hints=extra_hints,
                )
            )
            messages = list(prompt_render.messages)
            return messages, self._estimate_complete_input_tokens(
                messages,
                preloaded=preloaded,
                disabled_tools=disabled_tools,
            )

        # 2. 在模型调用前按 token 软水位压缩。摘要按 epoch 固定，之后的
        # turn 只追加原文，因此不会在每一轮破坏 Prompt Cache 前缀。
        compacted_this_turn = False
        initial_messages, estimated_tokens = await render_current_history()
        retry_trace["estimated_input_tokens"] = estimated_tokens
        if (
            base_history is None
            and self._context_compaction.enabled
            and estimated_tokens >= self._context_compaction.trigger_tokens
        ):
            try:
                summary_state = await self._compact_session_history(
                    session,
                    previous=summary_state,
                    reason="token_watermark",
                    estimated_input_tokens=estimated_tokens,
                    keep_recent_tokens=self._compaction_recent_budget(
                        estimated_input_tokens=estimated_tokens,
                        history=source_history,
                    ),
                )
            except Exception as exc:
                logger.warning("上下文分代摘要失败，保持原游标不变: %s", exc)
                return TurnRunResult(
                    reply="上下文摘要暂时失败，原对话没有丢失，请稍后重试。",
                    context_retry=retry_trace,
                )
            if summary_state is None:
                return TurnRunResult(
                    reply="当前消息或固定上下文过长，已经没有可安全摘要的完整历史回合。",
                    context_retry=retry_trace,
                )
            compacted_this_turn = True
            source_history = load_history()
            initial_messages, estimated_tokens = await render_current_history()
            retry_trace["estimated_input_tokens_after_compaction"] = estimated_tokens
            retry_trace["compaction"] = summary_state.to_metadata()

        # 3. 正常只调用一次主模型。若 provider 的真实 tokenizer 与估算有偏差，
        # 只允许再执行一次同样的摘要机制，不再删 prompt 区块或清空历史。
        while True:
            plan_name = (
                f"summary_epoch_{summary_state.epoch}"
                if summary_state is not None
                else "full"
            )
            retry_attempts.append(
                {
                    "name": plan_name,
                    "history_messages": len(source_history),
                    "estimated_input_tokens": estimated_tokens,
                    "disabled_sections": [],
                }
            )
            llm_user_content, llm_context_frame = extract_model_facing_turn(
                initial_messages
            )
            try:
                result = await self.run(
                    initial_messages,
                    request_time=msg.timestamp,
                    preloaded_tools=preloaded,
                    preloaded_tool_order=preloaded_order,
                    preflight_injected=True,
                    on_content_delta=stream_sink,
                    tool_event_session_key=session.key,
                    tool_event_channel=msg.channel,
                    tool_event_chat_id=msg.chat_id,
                    request_text=msg.content,
                    permission_mode=permission_mode,
                    disabled_tools=disabled_tools,
                    resume_from_checkpoint=bool(
                        (getattr(msg, "metadata", None) or {}).get(
                            "resumed_from_interrupt"
                        )
                    ),
                )
            except ContentSafetyError:
                logger.warning("安全拦截：不改变上下文或重写 prompt")
                return TurnRunResult(
                    reply="你的消息触发了安全审查，无法处理。",
                    context_retry=retry_trace,
                )
            except ContextLengthError:
                if (
                    base_history is None
                    and self._context_compaction.enabled
                    and not compacted_this_turn
                ):
                    try:
                        summary_state = await self._compact_session_history(
                            session,
                            previous=summary_state,
                            reason="provider_context_length",
                            estimated_input_tokens=estimated_tokens,
                            keep_recent_tokens=self._compaction_recent_budget(
                                estimated_input_tokens=estimated_tokens,
                                history=source_history,
                            ),
                        )
                    except Exception as exc:
                        logger.warning("provider 超限后的上下文摘要失败: %s", exc)
                        return TurnRunResult(
                            reply="上下文摘要暂时失败，原对话没有丢失，请稍后重试。",
                            context_retry=retry_trace,
                        )
                    if summary_state is not None:
                        compacted_this_turn = True
                        source_history = load_history()
                        initial_messages, estimated_tokens = await render_current_history()
                        retry_trace["estimated_input_tokens_after_compaction"] = estimated_tokens
                        retry_trace["compaction"] = summary_state.to_metadata()
                        continue
                logger.warning("上下文超长：摘要后仍超限，不执行删除式降级")
                return TurnRunResult(
                    reply="上下文摘要后仍超过模型限制，请检查模型上下文配置或缩短当前消息。",
                    context_retry=retry_trace,
                )
            except asyncio.TimeoutError:
                logger.warning("LLM 流响应超时，远端连接中断")
                return TurnRunResult(
                    reply="模型流响应中断，请刷新对话重试。",
                    context_retry=retry_trace,
                )

            tools_used = list(result.metadata.get("tools_used") or [])
            tools_unlocked = list(result.metadata.get("tools_unlocked") or [])
            tool_chain = list(result.metadata.get("tool_chain") or [])
            if self._tool_search_enabled and (tools_used or tools_unlocked):
                self._discovery.update(
                    session.key,
                    [*tools_unlocked, *tools_used],
                    self._tools.get_always_on_names(),
                )
            retry_trace["selected_plan"] = plan_name
            if isinstance(llm_user_content, (str, list)):
                retry_trace["llm_user_content"] = llm_user_content
            if isinstance(llm_context_frame, str) and llm_context_frame.strip():
                retry_trace["llm_context_frame"] = llm_context_frame
            retry_trace["react_stats"] = dict(result.metadata.get("react_stats") or {})
            return TurnRunResult(
                reply=result.reply,
                tools_used=tools_used,
                tool_chain=tool_chain,
                thinking=result.thinking,
                streamed=result.streamed,
                context_retry=retry_trace,
            )

    async def run(
        self,
        initial_messages: list[dict],
        *,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preloaded_tool_order: list[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
        request_text: str = "",
        permission_mode: str = "full_access",
        disabled_tools: set[str] | None = None,
        resume_from_checkpoint: bool = False,
    ) -> ReasonerResult:
        del preflight_injected
        return await self._graph_executor.run(
            cast("list[dict[str, Any]]", initial_messages),
            request_time=request_time,
            preloaded_tools=preloaded_tools,
            preloaded_tool_order=preloaded_tool_order,
            on_content_delta=on_content_delta,
            tool_event_session_key=tool_event_session_key,
            tool_event_channel=tool_event_channel,
            tool_event_chat_id=tool_event_chat_id,
            request_text=request_text,
            permission_mode=permission_mode,
            disabled_tools=disabled_tools,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    async def _observe_tool_call_started(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        iteration: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        if self._event_bus is None or not session_key:
            return
        await self._event_bus.observe(
            ToolCallStarted(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                iteration=iteration,
                call_id=call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
            )
        )

    async def _observe_tool_call_completed(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        iteration: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        final_arguments: dict[str, Any],
        status: str,
        result_preview: str,
    ) -> None:
        if self._event_bus is None or not session_key:
            return
        await self._event_bus.observe(
            ToolCallCompleted(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                iteration=iteration,
                call_id=call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                final_arguments=dict(final_arguments),
                status=status,
                result_preview=result_preview,
            )
        )

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict],
        *,
        reason: str,
        iteration: int,
        tools_used: list[str],
    ) -> ProgressSummary:
        # 1. 先构造收尾总结 prompt。
        summary_prompt = self._execution_policy.build_summary_prompt(
            reason=reason,
            iteration=iteration,
            tools_used=tools_used,
        )

        # 2. 先尝试让模型给一段中文收尾总结。
        try:
            cache_view = self._prompt_cache.prepare_model_messages(
                messages,
                keep_recent_tool_rounds=self._execution_policy.recent_tool_rounds,
            )
            response = await asyncio.wait_for(
                run_model_step(
                    self._llm.provider,
                    messages=cache_view.messages + [
                        support.build_context_hint_message(
                            "summary_request",
                            summary_prompt,
                        )
                    ],
                    tools=[],
                    model=self._llm_config.model,
                    max_tokens=min(_SUMMARY_MAX_TOKENS, self._llm_config.max_tokens),
                    source=self._execution_policy.source,
                    iteration=iteration + 1,
                    purpose="incomplete_summary",
                    cache_metadata=cache_view.plan.to_metadata(),
                ),
                timeout=self._execution_guard.config.model_call_timeout_seconds,
            )
            text = (response.content or "").strip()
            if text:
                return ProgressSummary(text=text)
        except Exception as exc:
            logger.warning("生成预算收尾总结失败: %s", exc)

        # 3. 模型收尾失败时，返回固定兜底文案。
        return ProgressSummary(
            text=self._execution_policy.fallback_summary(
                reason=reason,
                iteration=iteration,
                tools_used=tools_used,
            ),
            used_fallback=True,
        )

    def _build_result(
        self,
        *,
        reply: str,
        tools_used: list[str],
        tool_chain: list[dict[str, Any]],
        visible_names: set[str] | None,
        thinking: str | None,
        streamed: bool,
        react_input_samples: list[int],
        cache_prompt_tokens: int,
        cache_hit_tokens: int,
        cache_seen: bool,
        cache_plan: dict[str, Any] | None = None,
        tools_unlocked: list[str] | None = None,
        exit_reason: str = "completed",
        execution_guard_state: dict[str, Any] | None = None,
    ) -> ReasonerResult:
        # 1. 先把 tool_chain 扁平化成 invocations。
        invocations: list[LLMToolCall] = []
        for group in tool_chain:
            for call in group.get("calls") or []:
                args = call.get("arguments")
                invocations.append(
                    LLMToolCall(
                        id=str(call.get("call_id", "") or ""),
                        name=str(call.get("name", "") or ""),
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        # 2. 再把运行时元数据统一塞进 metadata。
        react_stats: dict[str, object] = {
            "iteration_count": len(react_input_samples),
            "turn_input_sum_tokens": sum(react_input_samples),
            "turn_input_peak_tokens": max(react_input_samples, default=0),
            "final_call_input_tokens": react_input_samples[-1] if react_input_samples else 0,
        }
        if cache_seen:
            react_stats["cache_prompt_tokens"] = cache_prompt_tokens
            react_stats["cache_hit_tokens"] = cache_hit_tokens
            hit_rate = (
                cache_hit_tokens / cache_prompt_tokens
                if cache_prompt_tokens > 0
                else 0.0
            )
            logger.info(
                "[KV缓存] 本轮 prompt_tokens=%d hit_tokens=%d hit_rate=%.2f%%",
                cache_prompt_tokens,
                cache_hit_tokens,
                hit_rate * 100,
            )
        if cache_plan:
            react_stats["prompt_cache_plan"] = dict(cache_plan)
        if execution_guard_state:
            react_stats["execution_guard"] = {
                key: value
                for key, value in execution_guard_state.items()
                if key != "recent_rounds"
            }
        metadata = {
            "tools_used": list(tools_used),
            "tools_unlocked": list(tools_unlocked or []),
            "tool_chain": list(tool_chain),
            "visible_names": set(visible_names) if visible_names is not None else None,
            "react_stats": react_stats,
            "exit_reason": exit_reason,
        }

        # 3. 最后返回标准 ReasonerResult。
        return ReasonerResult(
            reply=reply,
            invocations=invocations,
            thinking=thinking,
            streamed=streamed,
            metadata=metadata,
        )

    def _estimate_complete_input_tokens(
        self,
        messages: list[dict],
        *,
        preloaded: set[str] | None,
        disabled_tools: set[str],
    ) -> int:
        """Estimate the cache-aware model view plus the visible tool schemas."""

        visible_names: set[str] | None = None
        if self._tool_search_enabled:
            visible_names = (
                self._tools.get_always_on_names() | (preloaded or set())
            ) - disabled_tools
        try:
            schemas = self._tools.get_schemas(visible_names)
        except TypeError:
            schemas = self._tools.get_schemas(names=visible_names)
        cache_view = self._prompt_cache.prepare_model_messages(
            messages,
            keep_recent_tool_rounds=self._execution_policy.recent_tool_rounds,
        )
        return estimate_tokens(cache_view.messages) + estimate_tokens(schemas)

    @staticmethod
    def _history_with_summary(
        state: ContextSummaryState,
        recent_history: list[dict],
    ) -> list[dict]:
        summary_message = support.build_context_hint_message(
            "conversation_summary",
            (
                f"[固定对话摘要 epoch={state.epoch} summarized_through="
                f"{state.summarized_through}]\n{state.summary}\n\n"
                "摘要之后的消息是近期原文；冲突时以较新的用户原文为准。"
            ),
        )
        return [summary_message, *recent_history]

    async def _compact_session_history(
        self,
        session: "SessionLike",
        *,
        previous: ContextSummaryState | None,
        reason: str,
        estimated_input_tokens: int,
        keep_recent_tokens: int,
    ) -> ContextSummaryState | None:
        config = self._context_compaction
        raw_messages = list(getattr(session, "messages", ()) or ())
        start_index = (
            previous.summarized_through
            if previous is not None
            else max(0, int(getattr(session, "last_consolidated", 0) or 0))
        )
        boundary = select_compaction_boundary(
            raw_messages,
            start_index=start_index,
            keep_recent_tokens=keep_recent_tokens,
            protect_recent_tool_rounds=self._prompt_cache.config.keep_recent_tool_rounds,
        )
        if boundary is None:
            return None

        prior_summary = previous.summary if previous is not None else ""
        digest = summary_source_digest(prior_summary, boundary.cold_messages)
        summary = await self._generate_context_summary(
            prior_summary=prior_summary,
            messages=list(boundary.cold_messages),
            epoch=(previous.epoch + 1 if previous is not None else 1),
        )
        if not summary.strip():
            raise RuntimeError("context summary model returned empty content")
        state = build_summary_state(
            summary=summary,
            summarized_through=boundary.end_index,
            previous_epoch=(previous.epoch if previous is not None else 0),
            source_digest=digest,
        )

        old_metadata = getattr(session, "metadata", {})
        old_cursor = int(getattr(session, "last_consolidated", 0) or 0)
        session.metadata = write_summary_state(old_metadata, state)
        session.last_consolidated = state.summarized_through
        try:
            await self._session_manager.save_async(cast(Any, session))
        except Exception:
            # Never advance the in-memory cursor unless the durable commit
            # succeeds.  The raw transcript remains the source of truth.
            session.metadata = old_metadata
            session.last_consolidated = old_cursor
            raise

        record_trace_event(
            category="context",
            name="summary_compaction",
            summary=f"上下文摘要 epoch {state.epoch} 已持久化",
            payload={
                "session_key": session.key,
                "reason": reason,
                "estimated_input_tokens": estimated_input_tokens,
                "trigger_tokens": config.trigger_tokens,
                "target_tokens": config.target_tokens,
                "keep_recent_tokens": keep_recent_tokens,
                "cold_tokens": boundary.cold_tokens,
                "recent_tokens": boundary.recent_tokens,
                "start_index": boundary.start_index,
                "summarized_through": boundary.end_index,
                "epoch": state.epoch,
                "source_digest": state.source_digest,
            },
        )
        logger.info(
            "context summary committed session=%s epoch=%d through=%d cold~=%d recent~=%d reason=%s",
            session.key,
            state.epoch,
            state.summarized_through,
            boundary.cold_tokens,
            boundary.recent_tokens,
            reason,
        )
        return state

    def _compaction_recent_budget(
        self,
        *,
        estimated_input_tokens: int,
        history: list[dict],
    ) -> int:
        """Choose a suffix budget that also aims at the configured target."""

        history_tokens = estimate_tokens(history) if history else 0
        fixed_tokens = max(0, estimated_input_tokens - history_tokens)
        target_available = (
            self._context_compaction.target_tokens
            - fixed_tokens
            - self._context_compaction.summary_max_tokens
        )
        return min(
            self._context_compaction.keep_recent_tokens,
            max(2_000, target_available),
        )

    async def _generate_context_summary(
        self,
        *,
        prior_summary: str,
        messages: list[dict[str, Any]],
        epoch: int,
    ) -> str:
        evidence = render_summary_evidence(messages)
        chunks = chunk_summary_evidence(
            evidence,
            chunk_tokens=self._context_compaction.chunk_tokens,
        )
        if not chunks:
            raise RuntimeError("no context evidence available for summarization")

        partials: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            partials.append(
                await self._summarize_context_text(
                    (
                        f"这是 epoch {epoch} 的第 {index}/{len(chunks)} 个旧对话证据分块。\n"
                        "请提取跨回合继续工作所需的全部有效状态。\n\n"
                        + chunk
                    ),
                    purpose="context_summary_map",
                    iteration=index,
                )
            )

        merge_items = []
        if prior_summary.strip():
            merge_items.append("[上一代摘要]\n" + prior_summary.strip())
        merge_items.extend(
            f"[本轮分块摘要 {index}]\n{text}"
            for index, text in enumerate(partials, start=1)
        )

        # Recursively reduce when a very large session creates many map
        # summaries.  Every model call remains below the configured chunk size.
        round_index = 0
        while len(merge_items) > 1 or prior_summary.strip():
            round_index += 1
            packed = "\n\n".join(merge_items)
            groups = chunk_summary_evidence(
                packed,
                chunk_tokens=self._context_compaction.chunk_tokens,
            )
            reduced = [
                await self._summarize_context_text(
                    (
                        f"合并 epoch {epoch} 的摘要材料（reduce round {round_index}）。\n"
                        "保留仍有效信息并消除重复、冲突和已被替代的旧结论。\n\n"
                        + group
                    ),
                    purpose="context_summary_reduce",
                    iteration=round_index * 1000 + index,
                )
                for index, group in enumerate(groups, start=1)
            ]
            merge_items = reduced
            prior_summary = ""
            if len(merge_items) == 1:
                return merge_items[0]
        return merge_items[0]

    async def _summarize_context_text(
        self,
        text: str,
        *,
        purpose: str,
        iteration: int,
    ) -> str:
        provider = self._llm.light_provider or self._llm.provider
        model = self._llm_config.light_model or self._llm_config.model
        response = await run_model_step(
            provider,
            messages=[
                {"role": "system", "content": _CONTEXT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tools=[],
            model=model,
            max_tokens=min(
                self._context_compaction.summary_max_tokens,
                self._llm_config.max_tokens,
            ),
            disable_thinking=True,
            source=self._execution_policy.source,
            iteration=iteration,
            purpose=purpose,
        )
        content = (response.content or "").strip()
        if not content:
            raise RuntimeError(f"{purpose} returned empty content")
        return content

class DefaultReasoner(AgentExecutionKernel):
    """主 Agent 适配器；执行控制流由 AgentExecutionKernel 统一提供。"""


# ── 模块级辅助函数 ──────────────────────────────────────────────



def get_history_since_consolidated(
    session: "SessionLike",
    memory_window: int,
) -> list[dict]:
    try:
        return session.get_history(
            max_messages=memory_window,
            start_index=session.last_consolidated,
        )
    except TypeError:
        return session.get_history(max_messages=memory_window)


def extract_model_facing_turn(
    messages: list[dict],
) -> tuple[object | None, str | None]:
    if not messages:
        return None, None
    user_content = (
        messages[-1].get("content")
        if messages[-1].get("role") == "user"
        else None
    )
    if len(messages) < 2:
        return user_content, None
    frame = messages[-2]
    frame_content = frame.get("content")
    if isinstance(frame_content, str) and is_context_frame(frame_content):
        return user_content, frame_content
    return user_content, None


def build_turn_injection_prompt(
    *,
    tools: "ToolRegistry",
    tool_search_enabled: bool,
    visible_names: set[str] | None,
) -> str:
    if not tool_search_enabled:
        return ""
    return build_deferred_tools_hint(tools, visible=visible_names)


def build_deferred_tools_hint(
    tools: "ToolRegistry",
    visible: set[str] | None = None,
) -> str:
    get_deferred_names = getattr(tools, "get_deferred_names", None)
    if not callable(get_deferred_names):
        return ""
    deferred_raw = get_deferred_names(visible=visible)
    if not isinstance(deferred_raw, dict):
        return ""
    counts: dict[str, int] = {}
    for source_type, raw_group in deferred_raw.items():
        if isinstance(raw_group, list):
            count = sum(1 for name in raw_group if isinstance(name, str))
        elif isinstance(raw_group, dict):
            count = sum(
                sum(1 for name in names if isinstance(name, str))
                for names in raw_group.values()
                if isinstance(names, list)
            )
        else:
            count = 0
        if count:
            counts[str(source_type)] = count

    if not counts:
        return ""

    labels = {
        "builtin": "内置",
        "mcp": "MCP",
        "plugin": "插件",
        "peer": "Agent",
    }
    summary = "，".join(
        f"{labels.get(source, source)} {count} 个"
        for source, count in sorted(counts.items())
    )
    total = sum(counts.values())
    lines: list[str] = [
        f"【未加载能力：共 {total} 个（{summary}）】",
    ]
    lines.append(
        "系统只展示与本轮相关的少量能力。已知名称可用 "
        "tool_search(query=\"select:名称\")；否则直接描述目标功能搜索。"
    )
    return "\n".join(lines) + "\n\n"
