from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.config_models import Config, WiringConfig
from agent.context import ContextBuilder
from agent.peer_agent.process_manager import PeerProcessManager
from agent.peer_agent.poller import PeerAgentPoller
from agent.looping.core import AgentLoop
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    LLMServices,
    MemoryConfig,
    MemoryServices,
    SessionServices,
)
from agent.mcp.registry import McpServerRegistry
from agent.permissions import (
    PermissionClassifier,
    PermissionGuardHook,
    PermissionService,
)
from core.llm import LLMProvider
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.scheduler import SchedulerService
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.workflows.runtime import WorkflowRuntime
from bootstrap.core_runtime import CoreRuntime
from bootstrap.personal import build_personal_runtime, register_personal_tools
from bootstrap.observability import build_trace_recorder
from bootstrap.toolsets.meta import (
    build_readonly_tools,
)
from bootstrap.toolsets.peer import build_peer_agent_resources
from bootstrap.toolsets.protocol import ToolsetDeps
from bootstrap.toolsets.protocol import ToolsetProvider
from bootstrap.toolsets.schedule import build_scheduler
from bootstrap.wiring import (
    wire_turn_lifecycle,
    resolve_context_factory,
    resolve_memory_toolset_provider,
    resolve_toolset_provider,
)
from agent.lifecycle.facade import TurnLifecycle
from agent.plugins.jobs import ProviderPluginLlmService
from bootstrap.providers import build_providers, build_vl_provider
from bus.event_bus import EventBus
from bus.processing import ProcessingState
from bus.queue import MessageBus
from core.memory.governed import GovernedLongTermMemory
from core.memory.runtime import MemoryRuntime
from core.memory.semantic_consumer import ConversationMemoryBatchConsumer
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.runtime import build_conversation_semantics_runtime
from core.net.http import SharedHttpResources
from proactive_v2.presence import PresenceStore
from session.manager import SessionManager
from infra.persistence.trace_store import TraceStore
from core.tracing.ports import TraceRecorder


@dataclass(frozen=True)
class ToolRuntimeAssembly:
    registry: ToolRegistry
    push_tool: MessagePushTool
    scheduler: SchedulerService
    mcp_registry: McpServerRegistry
    memory_runtime: MemoryRuntime
    peer_process_manager: PeerProcessManager | None
    peer_poller: PeerAgentPoller | None
    workflow_runtime: WorkflowRuntime | None


def _ordered_toolset_providers(
    providers: list[tuple[str, ToolsetProvider]],
) -> list[tuple[str, ToolsetProvider]]:
    """Stable topological order for optional toolset-to-toolset dependencies."""
    names = [name for name, _ in providers]
    if len(set(names)) != len(names):
        raise ValueError("wiring.toolsets 不能包含重复项")

    configured = set(names)
    completed: set[str] = set()
    pending = list(providers)
    ordered: list[tuple[str, ToolsetProvider]] = []
    while pending:
        for index, (name, provider) in enumerate(pending):
            run_after = set(getattr(provider, "run_after", ())) & configured
            if run_after <= completed:
                ordered.append((name, provider))
                completed.add(name)
                pending.pop(index)
                break
        else:
            unresolved = ", ".join(name for name, _ in pending)
            raise ValueError(f"toolset 依赖存在循环: {unresolved}")
    return ordered


def build_registered_tools(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
    *,
    bus: MessageBus,
    provider,
    light_provider,
    agent_provider=None,
    vl_provider=None,
    session_store=None,
    tools: ToolRegistry | None = None,
    event_publisher=None,
    agent_loop_provider: Callable[[], Any] | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> ToolRuntimeAssembly:
    from session.store import SessionStore

    # ── 第一阶段：建服务（依赖无顺序陷阱）────────────────────────────────────
    wiring = getattr(config, "wiring", WiringConfig())
    tools = tools or ToolRegistry()
    multimodal = getattr(config, "multimodal", True)
    vl_available = (not multimodal) and bool(getattr(config, "vl_model", ""))
    readonly_tools = build_readonly_tools(
        http_resources, multimodal=multimodal, vl_available=vl_available
    )
    store = session_store or SessionStore(workspace / "sessions.db")
    push_tool = MessagePushTool(chat_lane=bus.chat_lane)
    memory_result = resolve_memory_toolset_provider(wiring.memory).register(
        tools,
        ToolsetDeps(
            config=config,
            workspace=workspace,
            provider=provider,
            light_provider=light_provider,
            http_resources=http_resources,
            event_publisher=event_publisher,
            trace_recorder=trace_recorder,
        ),
    )
    memory_runtime = memory_result.extras["memory_runtime"]
    scheduler = build_scheduler(
        workspace,
        push_tool,
        agent_loop_provider=agent_loop_provider,
        event_publisher=(
            event_publisher.enqueue
            if event_publisher is not None
            else None
        ),
    )
    peer_process_manager, peer_poller = build_peer_agent_resources(
        config, bus, http_resources
    )

    # ── 第二阶段：注册工具（所有服务已就绪）──────────────────────────────────
    mcp_registry = None
    toolset_extras: dict[str, Any] = {}
    providers = [
        (
            name,
            resolve_toolset_provider(
                name,
                readonly_tools=readonly_tools if name == "meta_common" else None,
            ),
        )
        for name in wiring.toolsets
    ]
    for name, provider_obj in _ordered_toolset_providers(providers):
        result = provider_obj.register(
            tools,
            ToolsetDeps(
                config=config,
                workspace=workspace,
                session_store=store,
                push_tool=push_tool,
                http_resources=http_resources,
                provider=provider,
                light_provider=light_provider,
                agent_provider=agent_provider,
                vl_provider=vl_provider,
                vl_model=getattr(config, "vl_model", ""),
                bus=bus,
                memory_engine=memory_runtime.engine,
                scheduler=scheduler,
                event_publisher=event_publisher,
                agent_loop_provider=agent_loop_provider,
                task_executor=toolset_extras.get("task_executor"),
                trace_recorder=trace_recorder,
            ),
        )
        toolset_extras.update(result.extras)
        maybe_mcp = result.extras.get("mcp_registry")
        if maybe_mcp is not None:
            mcp_registry = maybe_mcp
    if mcp_registry is None:
        from agent.mcp.registry import McpServerRegistry

        mcp_registry = McpServerRegistry(
            config_path=workspace / "mcp_servers.json",
            tool_registry=tools,
        )

    workflow_runtime = toolset_extras.get("workflow_runtime")
    return ToolRuntimeAssembly(
        registry=tools,
        push_tool=push_tool,
        scheduler=scheduler,
        mcp_registry=mcp_registry,
        memory_runtime=memory_runtime,
        peer_process_manager=peer_process_manager,
        peer_poller=peer_poller,
        workflow_runtime=(
            workflow_runtime if isinstance(workflow_runtime, WorkflowRuntime) else None
        ),
    )


def _build_loop_deps(
    *,
    config: Config,
    workspace: Path,
    bus: MessageBus,
    provider: LLMProvider,
    light_provider: LLMProvider | None,
    tools: ToolRegistry,
    session_manager: SessionManager,
    presence: PresenceStore,
    processing_state: ProcessingState,
    event_bus: EventBus,
    memory_runtime: MemoryRuntime,
    trace_recorder: TraceRecorder | None = None,
) -> AgentLoopDeps:
    wiring = getattr(config, "wiring", WiringConfig())
    context = resolve_context_factory(wiring.context)(
        workspace,
        memory_runtime.profile,
    )
    if isinstance(context, ContextBuilder):
        context.set_media_capabilities(
            multimodal=bool(getattr(config, "multimodal", True)),
            vl_available=bool(getattr(config, "vl_model", "")),
        )
    memory_engine = memory_runtime.engine
    light = light_provider or provider
    llm_services = LLMServices(provider=provider, light_provider=light)
    memory_services = MemoryServices(engine=memory_engine, runtime=memory_runtime)
    session_services = SessionServices(
        session_manager=session_manager, presence=presence
    )
    retrieval_pipeline = DefaultMemoryRetrievalPipeline(
        memory=memory_services,
        workspace=workspace,
    )

    return AgentLoopDeps(
        bus=bus,
        event_bus=event_bus,
        provider=provider,
        tools=tools,
        session_manager=session_manager,
        workspace=workspace,
        presence=presence,
        light_provider=light_provider,
        processing_state=processing_state,
        memory_runtime=memory_runtime,
        retrieval_pipeline=retrieval_pipeline,
        trace_recorder=trace_recorder,
        context=context,
        llm_services=llm_services,
        memory_services=memory_services,
        session_services=session_services,
        graph_runtime=LangGraphRuntime(workspace / "langgraph-checkpoints.db"),
    )


def build_core_runtime(
    config: Config,
    workspace: Path,
    http_resources: SharedHttpResources,
) -> CoreRuntime:
    bus = MessageBus()
    event_bus = EventBus()
    trace_store = TraceStore(workspace / "traces.db")
    trace_recorder = build_trace_recorder(trace_store, config.observability)
    provider, light_provider, agent_provider = build_providers(config)
    vl_provider = build_vl_provider(config)
    # 主模型负责小满的主对话与工具决策；Agent 模型只服务委派子任务。
    loop_provider = provider
    loop_model = config.model
    session_manager = SessionManager(workspace)
    loop_ref: dict[str, AgentLoop] = {}
    tool_runtime = build_registered_tools(
        config,
        workspace,
        http_resources,
        bus=bus,
        provider=provider,
        light_provider=light_provider,
        agent_provider=agent_provider,
        vl_provider=vl_provider,
        session_store=session_manager._store,
        event_publisher=event_bus,
        agent_loop_provider=lambda: loop_ref.get("loop"),
        trace_recorder=trace_recorder,
    )
    tools = tool_runtime.registry
    push_tool = tool_runtime.push_tool
    scheduler = tool_runtime.scheduler
    mcp_registry = tool_runtime.mcp_registry
    memory_runtime = tool_runtime.memory_runtime
    peer_pm = tool_runtime.peer_process_manager
    peer_poller = tool_runtime.peer_poller
    presence = PresenceStore(session_manager._store)
    processing_state = ProcessingState()
    loop_deps = _build_loop_deps(
        config=config,
        workspace=workspace,
        bus=bus,
        provider=loop_provider,
        light_provider=light_provider,
        tools=tools,
        session_manager=session_manager,
        presence=presence,
        processing_state=processing_state,
        event_bus=event_bus,
        memory_runtime=memory_runtime,
        trace_recorder=trace_recorder,
    )
    loop = AgentLoop(
        loop_deps,
        AgentLoopConfig(
            llm=LLMConfig(
                model=loop_model,
                light_model=config.light_model,
                max_iterations=config.max_iterations,
                max_tokens=config.max_tokens,
                tool_search_enabled=config.tool_search_enabled,
                multimodal=bool(getattr(config, "multimodal", True)),
                vl_available=bool(getattr(config, "vl_model", "")),
            ),
            memory=MemoryConfig(
                window=config.memory_window,
            ),
            context_compaction=config.context_compaction,
            prompt_cache=config.prompt_cache,
        ),
    )
    loop_ref["loop"] = loop
    permission_service = PermissionService()
    loop.add_tool_hooks(
        [
            PermissionGuardHook(
                classifier=PermissionClassifier(workspace),
                service=permission_service,
                risk_resolver=tools.get_risk,
            )
        ]
    )
    workflow_runtime = tool_runtime.workflow_runtime
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(event_bus),
        active_turn_states=loop.active_turn_states,
    )

    from agent.plugins.manager import PluginManager as _PluginManager

    plugin_manager = _PluginManager(
        plugin_dirs=_resolve_plugin_dirs(workspace),
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
        session_manager=session_manager,
        memory_engine=memory_runtime.engine,
        llm=ProviderPluginLlmService(
            provider=provider,
            model=config.model,
            max_tokens=config.max_tokens,
        ),
        plugin_configs=config.plugins,
        installed_cache_root=_resolve_installed_plugin_cache_root(),
    )
    personal = build_personal_runtime(
        workspace,
        workflow_runtime,
        event_bus=event_bus,
        tools=tools,
        scheduler=scheduler,
        default_channel=config.proactive.default_channel,
        default_chat_id=config.proactive.default_chat_id,
    )
    governed_long_term = GovernedLongTermMemory(
        governance=personal.governance,
    )
    governed_long_term.optimize()
    memory_runtime.bind_canonical_long_term_memory(governed_long_term)
    conversation_memory_consumer = ConversationMemoryBatchConsumer(
        markdown=memory_runtime.markdown.store,
        candidate_sink=governed_long_term.ingest_candidates,
        get_session=session_manager.get_or_create,
        save_session=session_manager.save_async,
    )
    conversation_memory_unsubscribe = event_bus.on(
        ConversationSemanticBatchCommitted,
        conversation_memory_consumer.handle,
    )
    conversation_semantics = build_conversation_semantics_runtime(
        config=config.conversation_semantics,
        workspace=workspace,
        provider=light_provider or provider,
        model=config.light_model or config.model,
        session_store=session_manager._store,
        event_bus=event_bus,
        # 语义提取有自己的 durable cursor，不再按消息条数推进模型上下文游标。
        # 上下文游标只由 token 水位摘要器在摘要持久化成功后推进。
        keep_messages=(
            config.context_compaction.max_history_messages
            if config.context_compaction.enabled
            else max(2, config.memory_window // 2)
        ),
    )
    if conversation_semantics is not None:
        memory_runtime.markdown.maintenance.bind_semantic_flush(
            conversation_semantics.batcher.flush
        )
    register_personal_tools(tools, personal)

    return CoreRuntime(
        config=config,
        workspace=workspace,
        http_resources=http_resources,
        loop=loop,
        bus=bus,
        event_bus=event_bus,
        tools=tools,
        push_tool=push_tool,
        session_manager=session_manager,
        scheduler=scheduler,
        provider=provider,
        light_provider=light_provider,
        agent_provider=agent_provider,
        vl_provider=vl_provider,
        mcp_registry=mcp_registry,
        memory_runtime=memory_runtime,
        permission_service=permission_service,
        personal=personal,
        presence=presence,
        peer_process_manager=peer_pm,
        peer_poller=peer_poller,
        workflow_runtime=workflow_runtime,
        plugin_manager=plugin_manager,
        trace_store=trace_store,
        trace_recorder=trace_recorder,
        conversation_semantics=conversation_semantics,
        conversation_memory_unsubscribe=conversation_memory_unsubscribe,
    )


def _resolve_plugin_dirs(workspace: Path) -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    return [project_root / "plugins"]


def _resolve_installed_plugin_cache_root() -> Path:
    return Path.home() / ".xiaoman-plugin" / "cache"
