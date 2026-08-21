"""Composable Dashboard API package.

The package re-exports the original module's supported import surface while
keeping HTTP routes, plugin integration, storage readers, and server setup in
separate cohesive modules.
"""

from bootstrap.dashboard_api.app import create_dashboard_app
from bootstrap.dashboard_api.contracts import (
    ManualConsolidator,
    ManualMemoryOptimizer,
    MemoryBatchDeletePayload,
    MemoryUpdatePayload,
    MessageBatchDeletePayload,
    MessageUpdatePayload,
    SessionBatchDeletePayload,
    SessionConsolidatePayload,
    SessionUpdatePayload,
)
from bootstrap.dashboard_api.proactive_reader import ProactiveDashboardReader
from bootstrap.dashboard_api.security import (
    _is_loopback_dashboard_host,
    _validate_dashboard_binding,
)
from bootstrap.dashboard_api.server import (
    _build_dashboard_uvicorn_config,
    build_dashboard_server,
    run_dashboard_api,
)

__all__ = [
    "ManualConsolidator",
    "ManualMemoryOptimizer",
    "MemoryBatchDeletePayload",
    "MemoryUpdatePayload",
    "MessageBatchDeletePayload",
    "MessageUpdatePayload",
    "ProactiveDashboardReader",
    "SessionBatchDeletePayload",
    "SessionConsolidatePayload",
    "SessionUpdatePayload",
    "_build_dashboard_uvicorn_config",
    "_is_loopback_dashboard_host",
    "_validate_dashboard_binding",
    "build_dashboard_server",
    "create_dashboard_app",
    "run_dashboard_api",
]
