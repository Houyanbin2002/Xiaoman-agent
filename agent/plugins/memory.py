from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from core.memory.engine import MemoryAdminApi, MemoryEngine

if TYPE_CHECKING:
    from agent.config_models import Config
    from core.llm import LLMProvider
    from bus.event_bus import EventBus
    from core.memory.markdown import MarkdownMemoryRuntime
    from core.net.http import SharedHttpResources


@dataclass(frozen=True)
class MemoryPluginBuildDeps:
    config: "Config"
    workspace: Path
    provider: "LLMProvider"
    light_provider: "LLMProvider | None"
    http_resources: "SharedHttpResources"
    event_publisher: "EventBus | None"
    markdown: "MarkdownMemoryRuntime"


@dataclass
class MemoryPluginRuntime:
    engine: MemoryEngine
    closeables: list[object] = field(default_factory=list[object])
    admin: MemoryAdminApi | None = None


@runtime_checkable
class MemoryPlugin(Protocol):
    plugin_id: str

    def build(
        self,
        deps: MemoryPluginBuildDeps,
    ) -> MemoryPluginRuntime: ...


