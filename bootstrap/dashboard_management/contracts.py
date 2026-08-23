from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


@dataclass
class DashboardRuntimeServices:
    """Runtime dependencies shared by dashboard management route groups."""

    config: Any
    config_path: Path
    agent_loop: Any
    event_bus: Any
    tools: Any
    mcp_registry: Any
    scheduler: Any
    workflow_runtime: Any | None
    plugin_manager: Any | None
    push_tool: Any | None
    workspace: Path
    personal_data: Any | None = None
    personal_automation: Any | None = None
    personal_routines: Any | None = None
    memory_governance: Any | None = None
    memory_admin: Any | None = None
    permission_service: Any | None = None
    attention_runtime: Any | None = None
    personal_rhythm: Any | None = None
    runtime_models: Any | None = None
    external_sources: Any | None = None
    personal_today: Any | None = None
    conversation_styles: Any | None = None
    gateway_restart: Callable[[], None] | None = None
    gateway_instance_id: str = field(default_factory=lambda: uuid4().hex)
