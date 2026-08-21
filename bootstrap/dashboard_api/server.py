from __future__ import annotations

from pathlib import Path

import uvicorn

from bootstrap.dashboard_api.app import create_dashboard_app
from bootstrap.dashboard_api.contracts import (
    ManualConsolidator,
    ManualMemoryOptimizer,
)
from bootstrap.dashboard_api.logging import install_dashboard_access_log_filter
from bootstrap.dashboard_api.security import _validate_dashboard_binding
from bootstrap.dashboard_management import DashboardRuntimeServices
from core.memory.engine import MemoryAdminApi


def run_dashboard_api(
    *,
    workspace: Path,
    host: str = "127.0.0.1",
    port: int = 2236,
    allow_remote: bool = False,
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_admin: MemoryAdminApi,
    runtime_services: DashboardRuntimeServices | None = None,
) -> None:
    server = uvicorn.Server(
        _build_dashboard_uvicorn_config(
            workspace=workspace,
            host=host,
            port=port,
            allow_remote=allow_remote,
            manual_consolidator=manual_consolidator,
            manual_memory_optimizer=manual_memory_optimizer,
            memory_admin=memory_admin,
            runtime_services=runtime_services,
        )
    )
    server.run()


def build_dashboard_server(
    *,
    workspace: Path,
    host: str = "127.0.0.1",
    port: int = 2236,
    allow_remote: bool = False,
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_admin: MemoryAdminApi,
    runtime_services: DashboardRuntimeServices | None = None,
) -> uvicorn.Server:
    config = _build_dashboard_uvicorn_config(
        workspace=workspace,
        host=host,
        port=port,
        allow_remote=allow_remote,
        manual_consolidator=manual_consolidator,
        manual_memory_optimizer=manual_memory_optimizer,
        memory_admin=memory_admin,
        runtime_services=runtime_services,
    )
    return uvicorn.Server(config)


def _build_dashboard_uvicorn_config(
    *,
    workspace: Path,
    host: str,
    port: int,
    allow_remote: bool = False,
    manual_consolidator: ManualConsolidator | None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_admin: MemoryAdminApi,
    runtime_services: DashboardRuntimeServices | None = None,
) -> uvicorn.Config:
    _validate_dashboard_binding(host, allow_remote=allow_remote)
    config = uvicorn.Config(
        create_dashboard_app(
            workspace,
            manual_consolidator=manual_consolidator,
            manual_memory_optimizer=manual_memory_optimizer,
            memory_admin=memory_admin,
            runtime_services=runtime_services,
        ),
        host=host,
        port=port,
        log_level="info",
    )
    install_dashboard_access_log_filter()
    return config
