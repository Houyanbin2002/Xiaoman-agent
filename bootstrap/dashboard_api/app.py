from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from bootstrap.dashboard_api.contracts import (
    ManualConsolidator,
    ManualMemoryOptimizer,
)
from bootstrap.dashboard_api.plugins import (
    close_dashboard_value,
    install_plugin_dashboards,
)
from bootstrap.dashboard_api.proactive_reader import ProactiveDashboardReader
from bootstrap.dashboard_api.routes.memory import register_memory_routes
from bootstrap.dashboard_api.routes.proactive import register_proactive_routes
from bootstrap.dashboard_api.routes.root import register_root_route
from bootstrap.dashboard_api.routes.sessions import register_session_routes
from bootstrap.dashboard_api.routes.traces import register_trace_routes
from bootstrap.dashboard_management import (
    DashboardRuntimeServices,
    register_dashboard_management,
)
from core.memory.engine import MemoryAdminApi
from proactive_v2.state import ProactiveStateStore
from infra.persistence.trace_store import TraceStore
from session.store import SessionStore


def create_dashboard_app(
    workspace: Path,
    *,
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_admin: MemoryAdminApi,
    runtime_services: DashboardRuntimeServices | None = None,
) -> FastAPI:
    workspace.mkdir(parents=True, exist_ok=True)
    store = SessionStore(workspace / "sessions.db")
    trace_store = TraceStore(workspace / "traces.db")
    proactive_reader: ProactiveDashboardReader | None = None
    plugin_closeables: list[object] = []
    project_root = Path(__file__).resolve().parents[2]
    static_dir = project_root / "static" / "dashboard"

    def get_proactive_reader() -> ProactiveDashboardReader:
        nonlocal proactive_reader
        if proactive_reader is None:
            ProactiveStateStore(workspace / "proactive.db").close()
            proactive_reader = ProactiveDashboardReader(workspace / "proactive.db")
        return proactive_reader

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            store.close()
            trace_store.close()
            close_dashboard_value(memory_admin)
            for closeable in reversed(plugin_closeables):
                close_dashboard_value(closeable)
            if proactive_reader is not None:
                proactive_reader.close()

    app = FastAPI(title="Xiaoman Dashboard API", lifespan=lifespan)
    app.state.memory_admin = memory_admin
    register_dashboard_management(app, runtime_services)

    # The Vite output is absent on fresh clones and CI. Mounting with
    # check_dir=False keeps API startup independent from the frontend build.
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="dashboard-assets",
    )

    plugin_closeables.extend(
        install_plugin_dashboards(
            app,
            project_root=project_root,
            workspace=workspace,
        )
    )
    register_root_route(app, static_dir=static_dir)
    register_session_routes(
        app,
        store=store,
        manual_consolidator=manual_consolidator,
    )
    register_trace_routes(app, store=trace_store)
    register_memory_routes(
        app,
        memory_admin=memory_admin,
        manual_memory_optimizer=manual_memory_optimizer,
    )
    register_proactive_routes(app, get_reader=get_proactive_reader)
    return app
