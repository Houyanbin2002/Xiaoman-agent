from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from agent.config_models import Config
from agent.looping.core import AgentLoop
from agent.permissions import PermissionService
from agent.mcp.registry import McpServerRegistry
from agent.peer_agent.poller import PeerAgentPoller
from agent.peer_agent.process_manager import PeerProcessManager
from agent.peer_agent.registry import PeerAgentRegistry
from core.llm import LLMProvider
from agent.scheduler import SchedulerService
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import BusOutboundPort
from agent.workflows.runtime import WorkflowRuntime
from bootstrap.plugin_runtime import activate_plugins_for_core
from bootstrap.personal import PersonalRuntime
from bus.event_bus import EventBus
from bus.queue import MessageBus
from core.memory.runtime import MemoryRuntime
from core.conversation_semantics.runtime import ConversationSemanticsRuntime
from core.net.http import SharedHttpResources
from proactive_v2.presence import PresenceStore
from session.manager import SessionManager
from infra.persistence.trace_store import TraceStore
from core.tracing.ports import TraceRecorder

if TYPE_CHECKING:
    from agent.plugins.manager import PluginManager


@dataclass
class CoreRuntime:
    config: Config
    http_resources: SharedHttpResources
    loop: AgentLoop
    bus: MessageBus
    event_bus: EventBus
    tools: ToolRegistry
    push_tool: MessagePushTool
    session_manager: SessionManager
    scheduler: SchedulerService
    provider: LLMProvider
    light_provider: LLMProvider | None
    mcp_registry: McpServerRegistry
    memory_runtime: MemoryRuntime
    presence: PresenceStore
    peer_process_manager: PeerProcessManager | None
    peer_poller: PeerAgentPoller | None
    workflow_runtime: WorkflowRuntime | None = None
    permission_service: PermissionService | None = None
    personal: PersonalRuntime | None = None
    agent_provider: LLMProvider | None = None
    vl_provider: LLMProvider | None = None
    plugin_manager: "PluginManager | None" = None
    workspace: Path | None = None
    trace_store: TraceStore | None = None
    trace_recorder: TraceRecorder | None = None
    conversation_semantics: ConversationSemanticsRuntime | None = None
    conversation_memory_unsubscribe: Callable[[], None] | None = None

    async def start(self) -> None:
        bind_running_loop = getattr(self.event_bus, "bind_running_loop", None)
        if callable(bind_running_loop):
            bind_running_loop()
        self.mcp_registry.start_connect_all_background()

        if (
            self.peer_poller is not None
            and self.peer_process_manager is not None
            and self.config.peer_agents
        ):
            peer_registry = PeerAgentRegistry(
                process_manager=self.peer_process_manager,
                poller=self.peer_poller,
                requester=self.http_resources.local_service,
            )
            peer_tools = await peer_registry.discover_all(self.config.peer_agents)
            for t in peer_tools:
                self.tools.register(
                    t,
                    always_on=False,
                    risk="external-side-effect",
                )
            self.peer_poller.start()
        await activate_plugins_for_core(
            plugin_manager=self.plugin_manager,
            loop=self.loop,
            tools=self.tools,
            mcp_registry=self.mcp_registry,
            workspace=self.workspace,
            memory_runtime=self.memory_runtime,
        )
        if self.conversation_semantics is not None:
            await self.conversation_semantics.start()

    async def inspect_modules(self) -> str:
        if self.plugin_manager is not None:
            await self.plugin_manager.load_all()

        from agent.lifecycle.phase import inspect_phase
        from agent.lifecycle.phases.after_reasoning import (
            default_after_reasoning_modules,
        )
        from agent.lifecycle.phases.after_step import default_after_step_modules
        from agent.lifecycle.phases.after_turn import default_after_turn_modules
        from agent.lifecycle.phases.before_reasoning import (
            default_before_reasoning_modules,
        )
        from agent.lifecycle.phases.before_step import default_before_step_modules
        from agent.lifecycle.phases.before_turn import default_before_turn_modules
        from agent.lifecycle.phases.prompt_render import default_prompt_render_modules

        manager = self.plugin_manager
        before_turn_modules = manager.before_turn_modules if manager is not None else []
        before_reasoning_modules = (
            manager.before_reasoning_modules if manager is not None else []
        )
        prompt_render_modules = (
            manager.prompt_render_modules if manager is not None else []
        )
        before_step_modules = manager.before_step_modules if manager is not None else []
        after_step_modules = manager.after_step_modules if manager is not None else []
        after_reasoning_modules = (
            manager.after_reasoning_modules if manager is not None else []
        )
        after_turn_modules = manager.after_turn_modules if manager is not None else []

        agent_core = cast(Any, getattr(self.loop, "_agent_core"))
        pipeline = agent_core.pipeline
        reasoner = getattr(self.loop, "_reasoner", None)
        context = getattr(reasoner, "_context", None)

        phases = [
            (
                "before_turn",
                default_before_turn_modules(
                    self.event_bus,
                    self.session_manager,
                    cast(Any, getattr(pipeline, "_context_store", None)),
                    plugin_modules=cast(Any, before_turn_modules),
                ),
            ),
            (
                "before_reasoning",
                default_before_reasoning_modules(
                    self.event_bus,
                    self.tools,
                    self.session_manager,
                    cast(Any, context),
                    plugin_modules=cast(Any, before_reasoning_modules),
                ),
            ),
            (
                "prompt_render",
                default_prompt_render_modules(
                    self.event_bus,
                    cast(Any, context),
                    plugin_modules=cast(Any, prompt_render_modules),
                ),
            ),
            (
                "before_step",
                default_before_step_modules(
                    self.event_bus,
                    plugin_modules=cast(Any, before_step_modules),
                ),
            ),
            (
                "after_step",
                default_after_step_modules(
                    self.event_bus,
                    plugin_modules=cast(Any, after_step_modules),
                ),
            ),
            (
                "after_reasoning",
                default_after_reasoning_modules(
                    self.event_bus,
                    cast(Any, getattr(pipeline, "_session", None)),
                    workspace=getattr(context, "workspace", None),
                    plugin_modules=cast(Any, after_reasoning_modules),
                ),
            ),
            (
                "after_turn",
                default_after_turn_modules(
                    self.event_bus,
                    cast(
                        Any,
                        getattr(
                            pipeline,
                            "_outbound_port",
                            BusOutboundPort(self.bus),
                        ),
                    ),
                    cast(Any, context),
                    cast(int, getattr(pipeline, "_history_window", 500)),
                    plugin_modules=cast(Any, after_turn_modules),
                ),
            ),
        ]

        parts: list[str] = []
        for phase_name, modules in phases:
            parts.append("=" * 60)
            parts.append(phase_name)
            parts.append("=" * 60)
            parts.append(inspect_phase(modules))
        return "\n".join(parts)

    async def stop(self) -> None:
        if self.conversation_semantics is not None:
            await self.conversation_semantics.aclose()
        if self.conversation_memory_unsubscribe is not None:
            self.conversation_memory_unsubscribe()
            self.conversation_memory_unsubscribe = None
        if self.workflow_runtime is not None:
            await self.workflow_runtime.aclose()
        close_loop = getattr(self.loop, "aclose", None)
        if callable(close_loop):
            await close_loop()
        if self.plugin_manager is not None:
            await self.plugin_manager.terminate_all()
        await self.mcp_registry.shutdown()
        await self.event_bus.aclose()
        if self.peer_poller is not None:
            await self.peer_poller.stop()
        if self.peer_process_manager is not None:
            await self.peer_process_manager.shutdown_all()
        if self.personal is not None:
            self.personal.close()
        if self.trace_recorder is not None and self.trace_recorder is not self.trace_store:
            close = getattr(self.trace_recorder, "close", None)
            if callable(close):
                close()
        elif self.trace_store is not None:
            self.trace_store.close()
