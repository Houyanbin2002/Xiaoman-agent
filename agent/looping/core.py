import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeAlias, cast
from weakref import WeakValueDictionary

from core.error_context import current_session_key
from core.tracing import current_trace_id, new_trace_id, record_trace_event, trace_root
from agent.context import ContextBuilder
from agent.core.passive_turn import (
    AgentCore,
    AgentCoreDeps,
    DefaultContextStore,
    DefaultReasoner,
)
from agent.looping.interrupt import InterruptResult, TurnInterruptState
from agent.core.runner import CoreRunner, CoreRunnerDeps
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMServices,
    MemoryServices,
    SessionServices,
)
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.turns.outbound import BusOutboundPort

# Re-export for backward-compat: existing callers import these from core.py
__all__ = [
    "AgentLoop",
]
from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    TurnStarted,
)
from bus.processing import ProcessingState

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine
    from core.memory.markdown import MemoryProfileApi
    from agent.tool_hooks.base import ToolHook

logger = logging.getLogger("agent.loop")
_MANUAL_CONSOLIDATION_TIMEOUT_SECONDS = 30.0

StreamDelta: TypeAlias = dict[str, str] | str
StreamSink: TypeAlias = Callable[[StreamDelta], Awaitable[None]]
StreamSinkFactory: TypeAlias = Callable[[object], StreamSink | None]
StreamSupportPolicy: TypeAlias = Callable[[str], bool]


def _is_positive_int(value: str) -> bool:
    try:
        return int(value) > 0
    except ValueError:
        return False


def _is_nonempty(value: str) -> bool:
    return bool(value)


_STREAM_SUPPORT_POLICIES: dict[str, StreamSupportPolicy] = {
    "telegram": _is_positive_int,
    # 飞书私聊渠道：chat_id 形如 oc_xxx，全程支持流式预览（卡片 PATCH 消费 StreamDeltaReady）。
    "feishu": _is_nonempty,
    "dashboard": _is_nonempty,
}


def _supports_stream_events(channel: str, chat_id: str) -> bool:
    policy = _STREAM_SUPPORT_POLICIES.get(channel)
    return bool(policy is not None and policy(chat_id))


def _suppresses_stream_events(msg: object) -> bool:
    metadata: object = getattr(msg, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    typed = cast(dict[str, object], metadata)
    return bool(typed.get("suppress_stream_events"))


class AgentLoop:
    """
    主循环：从 MessageBus 消费 InboundMessage，
    驱动 LLM + 工具调用，将结果发回 MessageBus。
    对话历史按 session_key 独立维护，格式为 OpenAI messages。
    """

    def __init__(
        self,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
    ) -> None:
        # 1. 先挂基础运行时对象和配置。
        self._llm_config = config.llm
        self.bus = deps.bus
        self.tools = deps.tools
        self.memory_window = config.memory.window
        self._running = False
        self._processing_state = deps.processing_state
        self._event_bus = deps.event_bus or EventBus()
        self._trace_recorder = deps.trace_recorder
        self._graph_runtime = deps.graph_runtime or LangGraphRuntime()
        self._passive_runtime_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._inbound_capacity = asyncio.Semaphore(4)
        self._inbound_workers: set[asyncio.Task[None]] = set()

        # ── 中断控制面（纯内存态） ──
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._active_turn_states: dict[str, TurnInterruptState] = {}
        self._interrupt_states: dict[str, TurnInterruptState] = {}

        # 2. 再解析 memory runtime 入口。
        memory_engine = self._resolve_memory_runtime(deps)
        markdown_memory = self._resolve_markdown_runtime(deps)
        self._tool_search_enabled = bool(config.llm.tool_search_enabled)
        self._memory_engine = memory_engine
        self._markdown_memory = markdown_memory
        memory_profile = (
            deps.memory_runtime.profile
            if deps.memory_runtime is not None
            else cast("MemoryProfileApi", self._memory_engine)
        )
        self._context = deps.context or ContextBuilder(
            deps.workspace,
            memory=memory_profile,
            multimodal=config.llm.multimodal,
            vl_available=config.llm.vl_available,
        )
        self._llm_services = deps.llm_services or LLMServices(
            provider=deps.provider,
            light_provider=deps.light_provider or deps.provider,
        )
        self._session_services = deps.session_services or SessionServices(
            session_manager=deps.session_manager,
            presence=deps.presence,
        )

        # 3. 最后把 passive chain 装起来。
        self._assemble_passive_runtime(
            deps=deps,
            config=config,
        )
        self._configure_stream_events()

    def set_stream_sink_factory(self, factory: StreamSinkFactory | None) -> None:
        setter = getattr(self._reasoner, "set_stream_sink_factory", None)
        if callable(setter):
            _ = setter(self._wrap_stream_sink_factory(factory))

    def _configure_stream_events(self) -> None:
        setter = getattr(self._reasoner, "set_stream_sink_factory", None)
        if callable(setter):
            _ = setter(self._build_stream_event_sink)

    def _wrap_stream_sink_factory(
        self,
        factory: StreamSinkFactory | None,
    ) -> StreamSinkFactory | None:
        if factory is None:
            return None

        def _build(msg: object) -> StreamSink | None:
            if _suppresses_stream_events(msg):
                return None
            downstream = factory(msg)
            channel = str(getattr(msg, "channel", ""))
            chat_id = str(getattr(msg, "chat_id", ""))
            session_key = str(getattr(msg, "session_key", f"{channel}:{chat_id}"))
            if downstream is None:
                return None

            async def _push(delta: StreamDelta) -> None:
                if isinstance(delta, str):
                    payload = {"content_delta": delta}
                else:
                    payload = delta
                content_delta = payload.get("content_delta")
                if isinstance(content_delta, str) and content_delta:
                    self._append_partial_reply(session_key, content_delta)
                thinking_delta = payload.get("thinking_delta")
                if isinstance(thinking_delta, str) and thinking_delta:
                    self._append_partial_thinking(session_key, thinking_delta)
                await downstream(payload)

            return _push

        return _build

    def _build_stream_event_sink(self, msg: object) -> StreamSink | None:
        channel = str(getattr(msg, "channel", ""))
        chat_id = str(getattr(msg, "chat_id", ""))
        if _suppresses_stream_events(msg):
            return None
        if not _supports_stream_events(channel, chat_id):
            return None
        session_key = str(getattr(msg, "session_key", f"{channel}:{chat_id}"))

        async def _push(delta: StreamDelta) -> None:
            if isinstance(delta, str):
                payload = {"content_delta": delta}
            else:
                payload = delta
            content_delta = payload.get("content_delta")
            if isinstance(content_delta, str) and content_delta:
                self._append_partial_reply(session_key, content_delta)
            thinking_delta = payload.get("thinking_delta")
            if isinstance(thinking_delta, str) and thinking_delta:
                self._append_partial_thinking(session_key, thinking_delta)
            await self._event_bus.observe(
                StreamDeltaReady(
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                    content_delta=(
                        content_delta if isinstance(content_delta, str) else ""
                    ),
                    thinking_delta=(
                        thinking_delta if isinstance(thinking_delta, str) else ""
                    ),
                )
            )

        return _push

    def _append_partial_reply(self, session_key: str, delta: str) -> None:
        state = self._active_turn_states.get(session_key)
        if state is None or not delta:
            return
        state.partial_reply += delta

    def _append_partial_thinking(self, session_key: str, delta: str) -> None:
        state = self._active_turn_states.get(session_key)
        if state is None or not delta:
            return
        state.partial_thinking = (state.partial_thinking or "") + delta

    def _resolve_memory_runtime(
        self,
        deps: AgentLoopDeps,
    ) -> "MemoryEngine":
        if deps.memory_runtime is not None:
            return deps.memory_runtime.engine
        if deps.memory_services is not None and deps.memory_services.engine is not None:
            return deps.memory_services.engine
        raise ValueError("AgentLoop requires memory_runtime.engine")

    def _resolve_markdown_runtime(
        self,
        deps: AgentLoopDeps,
    ):
        if deps.memory_runtime is not None:
            return deps.memory_runtime.markdown
        return None

    def _assemble_passive_runtime(
        self,
        *,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
    ) -> None:
        # 1. 先组基础 service ports。
        llm_svc = self._llm_services
        memory_svc = deps.memory_services or MemoryServices(
            engine=getattr(deps.memory_runtime, "engine", None),
            runtime=deps.memory_runtime,
        )
        session_svc = self._session_services
        # 2. 组执行层。
        self._tool_discovery = deps.tool_discovery or ToolDiscoveryState()
        history_limit = (
            config.context_compaction.max_history_messages
            if config.context_compaction.enabled
            else config.memory.keep_count
        )
        self._reasoner = deps.reasoner or DefaultReasoner(
            llm=llm_svc,
            llm_config=config.llm,
            tools=deps.tools,
            discovery=self._tool_discovery,
            tool_search_enabled=self._tool_search_enabled,
            memory_window=history_limit,
            context_compaction_config=config.context_compaction,
            prompt_cache_config=config.prompt_cache,
            execution_guard_config=config.execution_guard,
            context=self._context,
            session_manager=self.session_manager,
            event_bus=self._event_bus,
            graph_runtime=self._graph_runtime,
        )

        # 3. 最后串 passive prepare / execute / commit 主链。
        retrieval_pipeline = deps.retrieval_pipeline or DefaultMemoryRetrievalPipeline(
            memory=memory_svc,
            workspace=deps.workspace,
        )
        passive_context_store = DefaultContextStore(
            retrieval=retrieval_pipeline,
            context=self._context,
            history_window=history_limit,
        )
        agent_core = AgentCore(
            AgentCoreDeps(
                session=session_svc,
                context_store=passive_context_store,
                context=self._context,
                tools=deps.tools,
                reasoner=self._reasoner,
                event_bus=self._event_bus,
                outbound_port=BusOutboundPort(self.bus),
                history_window=history_limit,
                memory_consolidator=self,
            )
        )
        self._agent_core = agent_core
        self._core_runner = deps.core_runner or CoreRunner(
            CoreRunnerDeps(agent_core=agent_core)
        )

    @property
    def light_model(self) -> str:
        # 1. 兼容外部读取 loop.light_model，真实值统一来自 llm 配置。
        return self._llm_config.light_model or self._llm_config.model

    @property
    def context(self) -> ContextBuilder:
        # 1. 兼容外部读取 loop.context，真实值统一来自私有 context 依赖。
        return self._context

    @property
    def light_provider(self):
        # 1. 兼容外部读取 loop.light_provider，真实值统一来自 llm services。
        return self._llm_services.light_provider

    @property
    def session_manager(self):
        # 1. 兼容外部读取 loop.session_manager，真实值统一来自 session services。
        return self._session_services.session_manager

    @light_model.setter
    def light_model(self, value: str) -> None:
        # 1. 兼容初始化期和少量外部覆写，统一回写到 llm 配置。
        self._llm_config.light_model = value

    def reconfigure_models(
        self,
        *,
        provider: object | None = None,
        light_provider: object | None = None,
        model: str | None = None,
        light_model: str | None = None,
    ) -> None:
        """Apply model/provider changes to future turns without rebuilding the loop."""

        if provider is not None:
            self._llm_services.provider = cast(Any, provider)
        if light_provider is not None:
            self._llm_services.light_provider = cast(Any, light_provider)
        if model is not None:
            self._llm_config.model = model
        if light_model is not None:
            self._llm_config.light_model = light_model

    @property
    def max_iterations(self) -> int:
        # 1. 兼容外部读取 loop.max_iterations，真实值统一来自 llm 配置。
        return int(self._llm_config.max_iterations)

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        # 1. 兼容测试或外部直接改 loop.max_iterations，真实执行也同步生效。
        self._llm_config.max_iterations = int(value)

    async def run(self) -> None:
        self._running = True
        logger.info(f"AgentLoop 启动  max_iter={self.max_iterations}")
        try:
            while self._running:
                try:
                    item = await asyncio.wait_for(
                        self.bus.consume_inbound(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                worker = asyncio.create_task(self._run_inbound_turn(item))
                self._inbound_workers.add(worker)
                worker.add_done_callback(self._inbound_workers.discard)
        finally:
            if self._inbound_workers:
                await asyncio.gather(
                    *tuple(self._inbound_workers), return_exceptions=True
                )

    async def _run_inbound_turn(self, item: InboundMessage) -> None:
        """并行处理不同会话，同时保证同一会话严格按到达顺序串行。"""
        key = item.session_key
        lock = self._passive_runtime_locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                async with self._inbound_capacity:
                    task = asyncio.current_task()
                    if task is not None:
                        self._active_tasks[key] = task
                    self._active_turn_states[key] = self._build_initial_turn_state(
                        item, key
                    )
                    try:
                        await self._process(item)
                    except asyncio.CancelledError:
                        logger.info(f"Turn cancelled for {key}")
                    except Exception as exc:
                        logger.error(f"处理消息出错: {exc}", exc_info=True)
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                channel=item.channel,
                                chat_id=item.chat_id,
                                content=f"出错：{exc}",
                            )
                        )
                    finally:
                        if self._active_tasks.get(key) is task:
                            self._active_tasks.pop(key, None)
                            self._active_turn_states.pop(key, None)
        finally:
            await self.bus.complete_inbound(item)

    @property
    def processing_state(self) -> ProcessingState | None:
        return self._processing_state

    @property
    def active_turn_states(self) -> dict[str, TurnInterruptState]:
        return self._active_turn_states

    def stop(self) -> None:
        self._running = False
        logger.info("AgentLoop 停止")

    async def aclose(self) -> None:
        """Release the durable LangGraph checkpoint connection."""
        await self._graph_runtime.aclose()

    def add_tool_hooks(self, hooks: list["ToolHook"]) -> None:
        self._reasoner.add_tool_hooks(hooks)

    def add_before_turn_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._agent_core.add_before_turn_plugin_modules(modules)

    def add_before_reasoning_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._agent_core.add_before_reasoning_plugin_modules(modules)

    def add_after_reasoning_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._agent_core.add_after_reasoning_plugin_modules(modules)

    def add_after_turn_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._agent_core.add_after_turn_plugin_modules(modules)

    def add_prompt_render_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._reasoner.add_prompt_render_plugin_modules(modules)

    def add_before_step_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._reasoner.add_before_step_plugin_modules(modules)

    def add_after_step_plugin_modules(
        self,
        modules: list[object],
    ) -> None:
        self._reasoner.add_after_step_plugin_modules(modules)

    # ── 中断控制面 ────────────────────────────────────────────────

    def request_interrupt(
        self,
        session_key: str,
        sender: str = "",
        command: str = "/stop",
    ) -> InterruptResult:
        """Channel 层调用的中断入口，不走 MessageBus。"""
        task = self._active_tasks.get(session_key)
        if task is None or task.done():
            return InterruptResult(
                status="idle",
                session_key=session_key,
                message="当前没有正在执行的任务。",
            )

        # 保存中断态（纯内存，不落库）
        active_state = self._active_turn_states.get(session_key)
        if active_state is None:
            active_state = TurnInterruptState(
                session_key=session_key,
                original_user_message="",
            )
        self._interrupt_states[session_key] = replace(
            active_state,
            interrupted_by=command,
            interrupted_at=time.monotonic(),
        )
        task.cancel()
        logger.info(
            f"Turn interrupted  session_key={session_key}  "
            f"sender={sender}  command={command}"
        )
        return InterruptResult(
            status="interrupted",
            session_key=session_key,
            message="本轮已中断。你可以继续补充要求，我会接着这件事处理。",
        )

    def _get_interrupt_state(self, session_key: str) -> TurnInterruptState | None:
        """读取中断态（含 TTL 过期检查），不提前消费。"""
        state = self._interrupt_states.get(session_key)
        if state is None:
            return None
        if state.expired:
            logger.info(f"Interrupt state expired for {session_key}, discarding")
            self._interrupt_states.pop(session_key, None)
            return None
        return state

    def _build_initial_turn_state(
        self,
        item: InboundMessage,
        key: str,
    ) -> TurnInterruptState:
        return TurnInterruptState(
            session_key=key,
            original_user_message=item.content,
            original_metadata=dict(item.metadata or {}),
        )

    async def _resume_interrupted_message(
        self,
        msg: InboundMessage,
        key: str,
    ) -> tuple[InboundMessage, bool]:
        interrupted = self._get_interrupt_state(key)
        if interrupted is None:
            return msg, False

        # 2. 有中断态时，补一段结构化历史；当前用户消息保持原文。
        await self._persist_interrupted_turn_marker(key, interrupted)
        resumed = InboundMessage(
            channel=msg.channel,
            sender=msg.sender,
            chat_id=msg.chat_id,
            content=msg.content,
            timestamp=msg.timestamp,
            media=msg.media,
            metadata={**(msg.metadata or {}), "resumed_from_interrupt": True},
        )
        logger.info(f"Resuming interrupted turn for {key}")
        self._active_turn_states[key] = TurnInterruptState(
            session_key=key,
            original_user_message=msg.content,
            original_metadata=dict(resumed.metadata or {}),
        )
        return resumed, True

    async def _persist_interrupted_turn_marker(
        self,
        key: str,
        state: TurnInterruptState,
        *,
        preserve_partial: bool = False,
    ) -> None:
        if not state.original_user_message.strip():
            return
        session = self.session_manager.get_or_create(key)
        start = len(getattr(session, "messages", []))
        session.add_message(
            "user",
            state.original_user_message,
        )
        tool_chain = (
            cast(list[dict[str, Any]], list(state.tool_chain_partial))
            if state.tool_chain_partial
            else None
        )
        assistant_content = (
            state.partial_reply.strip()
            if preserve_partial and state.partial_reply.strip()
            else "[interrupted]"
        )
        assistant_extra: dict[str, Any] = {
            "tools_used": list(state.tools_used) if state.tools_used else None,
            "tool_chain": tool_chain,
        }
        if preserve_partial:
            assistant_extra["interrupted"] = True
            if state.partial_thinking:
                assistant_extra["reasoning_content"] = state.partial_thinking
        session.add_message("assistant", assistant_content, **assistant_extra)
        await self.session_manager.append_messages(session, session.messages[start:])

    async def _observe_turn_started(
        self,
        msg: InboundMessage,
        key: str,
    ) -> None:
        # 1. 对外发布被动 turn 开始事件，具体副作用由 observer 决定。
        await self._event_bus.observe(
            TurnStarted(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=msg.content,
                timestamp=msg.timestamp,
            )
        )

    # ── 被动 turn 处理 ────────────────────────────────────────────

    async def _process(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        started = time.time()
        key = session_key or msg.session_key
        busy_key = busy_session_key or key
        # 给本 turn task 打上 session 归属，供 observe 全局错误采集关联。
        _ = current_session_key.set(key)

        inherited_trace_id = str((msg.metadata or {}).get("trace_id") or "").strip()
        trace_id = inherited_trace_id or new_trace_id()
        msg.metadata["trace_id"] = trace_id
        trace_flow = str((msg.metadata or {}).get("trace_flow") or "passive")
        trace_title = str(
            (msg.metadata or {}).get("trace_title") or msg.content
        ).strip()

        with trace_root(
            self._trace_recorder,
            trace_id=trace_id,
            flow=trace_flow,
            session_key=key,
            title=trace_title[:180] or "未命名任务",
            parent_trace_id=str((msg.metadata or {}).get("parent_trace_id") or ""),
            metadata={"channel": msg.channel, "chat_id": msg.chat_id},
            finish=not bool(inherited_trace_id),
        ):
            record_trace_event(
                category="turn",
                name="inbound",
                summary=(
                    "收到用户请求" if trace_flow == "passive" else "开始执行任务步骤"
                ),
                payload={
                    "channel": msg.channel,
                    "chat_id": msg.chat_id,
                    "content_preview": msg.content[:500],
                },
            )
            # 1. 先处理可能存在的续跑态，并发布 turn started。
            msg, resumed_from_interrupt = await self._resume_interrupted_message(
                msg, key
            )
            await self._observe_turn_started(msg, key)
            content = msg.content
            preview = content[:60] + "..." if len(content) > 60 else content
            logger.info(f"Processing message from {msg.channel}: {preview}")

            # 2. 再进入 busy 状态并执行核心处理。
            if self._processing_state:
                self._processing_state.enter(busy_key)
            try:
                outbound = await self._core_runner.process(
                    msg,
                    key,
                    dispatch_outbound=dispatch_outbound,
                )
                if resumed_from_interrupt:
                    self._interrupt_states.pop(key, None)
                record_trace_event(
                    category="turn",
                    name="outbound",
                    summary="回复已生成",
                    payload={
                        "response_chars": len(outbound.content or ""),
                        "streamed": bool(outbound.metadata.get("streamed_reply")),
                    },
                )
                return outbound
            finally:
                # 3. 最后无论成功失败都直接释放 busy 状态。
                if self._processing_state:
                    self._processing_state.exit(busy_key)
                _ = started

    async def _process_with_runtime_admission(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        busy_session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        key = session_key or msg.session_key
        lock = self._passive_runtime_locks.setdefault(key, asyncio.Lock())
        if lock.locked():
            logger.info("[runtime_admission] 等待相同会话 session=%s", key)
        async with lock:
            return await self._process(
                msg,
                session_key=session_key,
                busy_session_key=busy_session_key,
                dispatch_outbound=dispatch_outbound,
            )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        busy_session_key: str | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        omit_user_turn: bool = False,
        skip_post_memory: bool = False,
        skip_memory_retrieval: bool = False,
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
        permission_mode: str = "full_access",
        media: list[str] | None = None,
        trace_id: str = "",
        trace_flow: str = "workflow",
        trace_title: str = "",
    ) -> str:
        response = await self.process_direct_outbound(
            content,
            session_key=session_key,
            busy_session_key=busy_session_key,
            channel=channel,
            chat_id=chat_id,
            omit_user_turn=omit_user_turn,
            skip_post_memory=skip_post_memory,
            skip_memory_retrieval=skip_memory_retrieval,
            stream_events=stream_events,
            disabled_tools=disabled_tools,
            permission_mode=permission_mode,
            media=media,
            trace_id=trace_id,
            trace_flow=trace_flow,
            trace_title=trace_title,
        )
        return response.content if response else ""

    async def process_direct_outbound(
        self,
        content: str,
        session_key: str = "cli:direct",
        busy_session_key: str | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        omit_user_turn: bool = False,
        skip_post_memory: bool = False,
        skip_memory_retrieval: bool = False,
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
        permission_mode: str = "full_access",
        media: list[str] | None = None,
        trace_id: str = "",
        trace_flow: str = "workflow",
        trace_title: str = "",
    ) -> OutboundMessage:
        metadata: dict[str, object] = {}
        trace_id = trace_id or current_trace_id()
        if omit_user_turn:
            metadata["omit_user_turn"] = True
        if skip_post_memory:
            metadata["skip_post_memory"] = True
        if skip_memory_retrieval:
            metadata["skip_memory_retrieval"] = True
        if not stream_events:
            metadata["suppress_stream_events"] = True
        if disabled_tools:
            metadata["disabled_tools"] = list(disabled_tools)
        if permission_mode != "full_access":
            metadata["permission_mode"] = permission_mode
        if trace_id:
            metadata["trace_id"] = trace_id
            metadata["trace_flow"] = trace_flow
        if trace_title:
            metadata["trace_title"] = trace_title
        msg = InboundMessage(
            channel=channel,
            sender="user",
            chat_id=chat_id,
            content=content,
            media=list(media or ()),
            metadata=metadata,
        )
        # Direct callers (dashboard, workflow runtime, scheduler) bypass run(), so
        # register them here as first-class turns for the shared interrupt plane.
        if not hasattr(self, "_active_tasks"):
            self._active_tasks = {}
        if not hasattr(self, "_active_turn_states"):
            self._active_turn_states = {}
        if not hasattr(self, "_interrupt_states"):
            self._interrupt_states = {}
        task = asyncio.current_task()
        active = self._active_tasks.get(session_key)
        if active is not None and active is not task and not active.done():
            raise RuntimeError("该会话已有正在执行的任务")
        if task is not None:
            self._active_tasks[session_key] = task
        self._active_turn_states[session_key] = self._build_initial_turn_state(
            msg,
            session_key,
        )
        try:
            response = await self._process_with_runtime_admission(
                msg,
                session_key=session_key,
                busy_session_key=busy_session_key,
                dispatch_outbound=False,
            )
            return response
        except asyncio.CancelledError:
            interrupted = self._interrupt_states.pop(session_key, None)
            interrupted = interrupted or self._active_turn_states.get(session_key)
            if interrupted is not None:
                await self._persist_interrupted_turn_marker(
                    session_key,
                    interrupted,
                    preserve_partial=True,
                )
            raise
        finally:
            if self._active_tasks.get(session_key) is task:
                self._active_tasks.pop(session_key, None)
                self._active_turn_states.pop(session_key, None)

    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
        force: bool = False,
    ) -> bool:
        from core.memory.markdown import ConsolidateRequest

        session = self.session_manager.get_or_create(session_key)
        if self._markdown_memory is None:
            raise RuntimeError("markdown memory runtime unavailable")
        maintenance = self._markdown_memory.maintenance
        try:
            result = await asyncio.wait_for(
                maintenance.consolidate(
                    ConsolidateRequest(
                        session=session,
                        archive_all=archive_all,
                        force=force,
                    )
                ),
                timeout=_MANUAL_CONSOLIDATION_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise TimeoutError("memory consolidation busy") from exc
        if result.trace.get("mode") in {"markdown", "semantic_batch"}:
            await self.session_manager.save_async(session)
            return True
        return False


# ── 模块级辅助 ────────────────────────────────────────────────────
