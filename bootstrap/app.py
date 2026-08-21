from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.config_models import Config
from bootstrap.channel_host import ChannelHost
from bootstrap.channels import start_channels
from bootstrap.dashboard_api import build_dashboard_server
from bootstrap.dashboard_management import DashboardRuntimeServices
from bootstrap.proactive import build_memory_optimizer_task, build_proactive_runtime
from bootstrap.runtime_models import RuntimeModelService
from bootstrap.tools import CoreRuntime, build_core_runtime
from bus.event_bus import EventBus
from agent.plugins.jobs import PluginJobRuntime
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _run_cleanup_steps(*steps: tuple[str, Callable[[], Awaitable[None]]]) -> None:
    first_error: Exception | None = None
    for name, step in steps:
        try:
            await step()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            logger.warning("shutdown step failed: %s: %s", name, exc)
    if first_error is not None:
        raise first_error


async def _noop_async() -> None:
    return None


def _stop_plugin_jobs(
    runtime: PluginJobRuntime | None,
) -> Callable[[], Awaitable[None]]:
    async def stop() -> None:
        if runtime is not None:
            runtime.stop()

    return stop


def _stop_workflow_runtime(
    runtime: Any | None,
    task: asyncio.Task[None] | None,
) -> Callable[[], Awaitable[None]]:
    async def stop() -> None:
        if runtime is not None:
            await runtime.aclose()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    return stop


class AppRuntime:
    def __init__(
        self,
        config: Config,
        workspace: Path,
        *,
        config_path: str | Path = "config.toml",
        dashboard_host: str = "127.0.0.1",
        dashboard_port: int = 2236,
        dashboard_allow_remote: bool = False,
        gateway_restart: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.config_path = Path(config_path).expanduser().resolve(strict=False)
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        self.dashboard_allow_remote = dashboard_allow_remote
        self.gateway_restart = gateway_restart
        self.http_resources = SharedHttpResources()
        self.ipc = None
        self.channel_host: ChannelHost | None = None
        self.core: CoreRuntime | None = None
        self.agent_loop = None
        self.bus = None
        self.event_bus: EventBus | None = None
        self.tools = None
        self.push_tool = None
        self.session_manager = None
        self.scheduler = None
        self.workflow_runtime = None
        self.workflow_task: asyncio.Task[None] | None = None
        self.provider = None
        self.light_provider = None
        self.mcp_registry = None
        self.memory_runtime = None
        self.permission_service = None
        self.presence = None
        self.proactive_loop = None
        self.peer_process_manager = None
        self.peer_poller = None
        self.dashboard_server = None
        self.dashboard_task: asyncio.Task[None] | None = None
        self.plugin_job_runtime: PluginJobRuntime | None = None
        self.tasks: list[Awaitable[None]] = []
        self._memory_optimizer = None
        self.model_runtime: RuntimeModelService | None = None
        self._shutdown = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        configure_default_shared_http_resources(self.http_resources)
        try:
            await self._start_core_runtime()
            plugin_manager = getattr(self.core, "plugin_manager", None)
            await self._start_channels(plugin_manager)
            self._start_workflow_runtime()
            self._register_core_tasks()
            self._register_plugin_job_tasks(plugin_manager)
            self._register_memory_optimizer()
            self._register_proactive_tasks(plugin_manager)
            if self.model_runtime is not None:
                self.model_runtime.bind_background_services(
                    proactive_loop=self.proactive_loop,
                    memory_optimizer=self._memory_optimizer,
                )
            self._start_dashboard()
            self._started = True
        except Exception:
            await self.shutdown()
            raise

    async def _start_core_runtime(self) -> None:
        self.core = build_core_runtime(
            self.config,
            self.workspace,
            self.http_resources,
        )
        self.agent_loop = self.core.loop
        self.bus = self.core.bus
        self.event_bus = self.core.event_bus
        self.tools = self.core.tools
        self.push_tool = self.core.push_tool
        self.session_manager = self.core.session_manager
        self.scheduler = self.core.scheduler
        self.workflow_runtime = getattr(self.core, "workflow_runtime", None)
        self.provider = self.core.provider
        self.light_provider = self.core.light_provider
        self.mcp_registry = self.core.mcp_registry
        self.mcp_registry.set_oauth_callback_base_url(
            f"http://127.0.0.1:{self.dashboard_port}"
        )
        self.memory_runtime = self.core.memory_runtime
        self.permission_service = getattr(self.core, "permission_service", None)
        self.presence = self.core.presence
        self.peer_process_manager = self.core.peer_process_manager
        self.peer_poller = self.core.peer_poller
        await self.core.start()
        self.model_runtime = RuntimeModelService(
            config=self.config,
            main_provider=self.core.provider,
            light_provider=self.core.light_provider,
            agent_provider=getattr(self.core, "agent_provider", None),
            vl_provider=getattr(self.core, "vl_provider", None),
            agent_loop=self.core.loop,
            memory_runtime=self.core.memory_runtime,
            plugin_manager=getattr(self.core, "plugin_manager", None),
            workflow_runtime=getattr(self.core, "workflow_runtime", None),
            tools=self.core.tools,
            workspace=self.workspace,
            http_resources=self.http_resources,
            core_runtime=self.core,
        )

    async def _start_channels(self, plugin_manager: Any | None) -> None:
        assert self.bus is not None
        assert self.session_manager is not None
        assert self.push_tool is not None
        assert self.event_bus is not None
        assert self.agent_loop is not None
        self.ipc, self.channel_host = await start_channels(
            self.config,
            bus=self.bus,
            session_manager=self.session_manager,
            push_tool=self.push_tool,
            http_resources=self.http_resources,
            event_bus=self.event_bus,
            bot_commands=(
                plugin_manager.telegram_bot_commands if plugin_manager else None
            ),
            interrupt_controller=self.agent_loop,
            plugin_channels=plugin_manager.channels if plugin_manager else None,
        )
        await self.channel_host.start_all()

    def _register_core_tasks(self) -> None:
        assert self.agent_loop is not None
        assert self.bus is not None
        assert self.scheduler is not None
        self.tasks = [
            self.agent_loop.run(),
            self.bus.dispatch_outbound(),
            self.scheduler.run(),
        ]
        personal = getattr(self.core, "personal", None)
        if personal is not None:
            self.tasks.append(personal.external_sources.run())

    def _start_workflow_runtime(self) -> None:
        if self.workflow_runtime is None:
            return
        self.workflow_task = asyncio.create_task(
            self.workflow_runtime.run(),
            name="workflow_runtime",
        )

    def _register_plugin_job_tasks(self, plugin_manager: Any | None) -> None:
        plugin_jobs = plugin_manager.jobs if plugin_manager else []
        if not plugin_jobs:
            return
        llm = plugin_manager.llm
        if llm is None:
            return
        assert self.event_bus is not None
        self.plugin_job_runtime = PluginJobRuntime(
            event_bus=self.event_bus,
            llm=llm,
            jobs=plugin_jobs,
        )
        self.tasks.append(self.plugin_job_runtime.run())

    def _register_memory_optimizer(self) -> None:
        assert self.provider is not None
        assert self.memory_runtime is not None
        optimizer_tasks, self._memory_optimizer = build_memory_optimizer_task(
            self.config,
            canonical=getattr(self.memory_runtime, "long_term", None),
        )
        self.tasks.extend(optimizer_tasks)

    def _start_dashboard(self) -> None:
        assert self.agent_loop is not None
        assert self.event_bus is not None
        assert self.tools is not None
        assert self.mcp_registry is not None
        assert self.scheduler is not None
        assert self.push_tool is not None
        assert self.memory_runtime is not None
        personal = getattr(self.core, "personal", None)
        runtime_services = DashboardRuntimeServices(
            config=self.config,
            config_path=self.config_path,
            agent_loop=self.agent_loop,
            event_bus=self.event_bus,
            tools=self.tools,
            mcp_registry=self.mcp_registry,
            scheduler=self.scheduler,
            workflow_runtime=self.workflow_runtime,
            plugin_manager=getattr(self.core, "plugin_manager", None),
            push_tool=self.push_tool,
            workspace=self.workspace,
            personal_data=personal.data if personal is not None else None,
            personal_automation=(personal.automation if personal is not None else None),
            personal_routines=personal.routines if personal is not None else None,
            memory_governance=(personal.governance if personal is not None else None),
            memory_admin=self.memory_runtime.engine,
            permission_service=self.permission_service,
            attention_runtime=(personal.attention if personal is not None else None),
            personal_rhythm=personal.rhythm if personal is not None else None,
            runtime_models=self.model_runtime,
            external_sources=(
                personal.external_sources if personal is not None else None
            ),
            personal_today=(personal.today if personal is not None else None),
            gateway_restart=self.gateway_restart,
        )
        self.dashboard_server = build_dashboard_server(
            workspace=self.workspace,
            host=self.dashboard_host,
            port=self.dashboard_port,
            allow_remote=self.dashboard_allow_remote,
            manual_consolidator=self.agent_loop,
            manual_memory_optimizer=self._memory_optimizer,
            memory_admin=self.memory_runtime.engine,
            runtime_services=runtime_services,
        )
        self.dashboard_task = asyncio.create_task(
            self.dashboard_server.serve(),
            name="dashboard_server",
        )

    def _register_proactive_tasks(self, plugin_manager: Any | None) -> None:
        assert self.session_manager is not None
        assert self.provider is not None
        assert self.push_tool is not None
        assert self.memory_runtime is not None
        assert self.presence is not None
        assert self.agent_loop is not None
        assert self.event_bus is not None
        personal = getattr(self.core, "personal", None)
        proactive_tasks, self.proactive_loop = build_proactive_runtime(
            self.config,
            self.workspace,
            session_manager=self.session_manager,
            provider=self.provider,
            push_tool=self.push_tool,
            memory_store=self.memory_runtime,
            presence=self.presence,
            agent_loop=self.agent_loop,
            event_bus=self.event_bus,
            tool_hooks=list(plugin_manager.tool_hooks) if plugin_manager else None,
            proactive_modules=(
                list(plugin_manager.proactive_modules) if plugin_manager else None
            ),
            personal_source=(
                personal.attention_source if personal is not None else None
            ),
            trace_recorder=getattr(self.core, "trace_recorder", None),
        )
        self.tasks.extend(proactive_tasks)
        if self.proactive_loop is not None:
            if personal is not None:
                personal.attention_wakes.bind_tick(self.proactive_loop.run_event_tick)
                self.tasks.append(personal.attention_wakes.run())
            assert self.ipc is not None
            self.ipc.set_proactive_loop(self.proactive_loop)

    async def run(self) -> None:
        service_tasks: list[asyncio.Future[None]] = []
        try:
            await self.start()
            service_tasks = [asyncio.ensure_future(item) for item in self.tasks]
            if not service_tasks:
                return
            completed, _ = await asyncio.wait(
                service_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            results = await asyncio.gather(*completed, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        finally:
            for task in service_tasks:
                if not task.done():
                    task.cancel()
            if service_tasks:
                await asyncio.gather(*service_tasks, return_exceptions=True)
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if self.dashboard_server is not None:
                self.dashboard_server.should_exit = True
            if self.dashboard_task is not None:
                try:
                    await self.dashboard_task
                except asyncio.CancelledError:
                    pass
            await _run_cleanup_steps(
                (
                    "workflow_runtime.stop",
                    _stop_workflow_runtime(self.workflow_runtime, self.workflow_task),
                ),
                (
                    "plugin_jobs.stop",
                    _stop_plugin_jobs(self.plugin_job_runtime),
                ),
                ("core.stop", self.core.stop if self.core else _noop_async),
                ("ipc.stop", self.ipc.stop if self.ipc else _noop_async),
                (
                    "channels.stop",
                    self.channel_host.stop_all if self.channel_host else _noop_async,
                ),
                (
                    "memory_runtime.aclose",
                    self.memory_runtime.aclose if self.memory_runtime else _noop_async,
                ),
                (
                    "model_runtime.aclose",
                    self.model_runtime.aclose if self.model_runtime else _noop_async,
                ),
                ("http_resources.aclose", self.http_resources.aclose),
            )
        finally:
            clear_default_shared_http_resources(self.http_resources)


def build_app_runtime(
    config: Config,
    workspace: Path | None = None,
    *,
    config_path: str | Path = "config.toml",
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 2236,
    dashboard_allow_remote: bool = False,
    gateway_restart: Callable[[], None] | None = None,
) -> AppRuntime:
    return AppRuntime(
        config,
        workspace or (Path.home() / ".xiaoman" / "workspace"),
        config_path=config_path,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        dashboard_allow_remote=dashboard_allow_remote,
        gateway_restart=gateway_restart,
    )
