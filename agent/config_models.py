from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proactive_v2.config import ProactiveConfig
from agent.runtime.context_compaction import ContextCompactionConfig
from agent.runtime.prompt_cache import PromptCacheConfig


@dataclass
class TelegramChannelConfig:
    token: str
    allow_from: list[str] = field(default_factory=list)
    channel_name: str = "telegram"


@dataclass
class ChannelsConfig:
    telegram: TelegramChannelConfig | None = None
    socket: str = "/tmp/xiaoman.sock"
    cli_session_key: str = ""


@dataclass
class MemoryEmbeddingConfig:
    model: str = "text-embedding-v3"
    api_key: str = ""
    base_url: str = ""
    output_dimensionality: int | None = None


@dataclass
class MemoryConfig:
    enabled: bool = True
    engine: str = "akasha"
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)


@dataclass
class ConversationSemanticsConfig:
    enabled: bool = True
    idle_seconds: int = 480
    max_turns: int = 8
    analysis_version: str = "conversation-v3"


@dataclass
class LangfuseConfig:
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    base_url: str = "https://cloud.langfuse.com"
    environment: str = "development"
    sample_rate: float = 1.0
    flush_at: int = 15
    flush_interval_seconds: float = 1.0
    capture_content: bool = True
    max_content_chars: int = 60000
    debug: bool = False


@dataclass
class ObservabilityConfig:
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)


@dataclass
class FitbitIntegrationConfig:
    enabled: bool = False


@dataclass
class PeerAgentConfig:
    name: str
    base_url: str
    launcher: list[str]  # 拉起命令，如 ["uv", "run", "python", "-m", "app.a2a_server"]
    cwd: str | None = None  # 子进程工作目录，None 表示继承父进程
    description: str = ""  # 工具描述，用于 LLM 路由；服务器在线时会被 AgentCard 覆盖
    health_path: str = "/health"
    startup_timeout_s: int = 30
    shutdown_timeout_s: int = 10


@dataclass
class WiringConfig:
    context: str = "default"
    memory: str = "default"
    toolsets: list[str] = field(
        default_factory=lambda: [
            "meta_common",
            "task_executor",
            "schedule",
            "mcp",
            "workflow",
        ]
    )


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    system_prompt: str
    max_tokens: int = 8192
    max_iterations: int = 10
    memory_window: int = 40
    base_url: str | None = None
    extra_body: dict = field(default_factory=dict)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    memory_optimizer_enabled: bool = True
    memory_optimizer_interval_seconds: int = 64800
    light_model: str = ""
    light_provider: str = ""
    light_api_key: str = ""
    light_base_url: str = ""
    agent_model: str = ""
    agent_provider: str = ""
    agent_api_key: str = ""
    agent_base_url: str = ""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    conversation_semantics: ConversationSemanticsConfig = field(
        default_factory=ConversationSemanticsConfig
    )
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    fitbit: FitbitIntegrationConfig = field(default_factory=FitbitIntegrationConfig)
    multimodal: bool = True
    vl_model: str = ""
    vl_provider: str = ""
    vl_api_key: str = ""
    vl_base_url: str = ""
    tool_search_enabled: bool = False
    context_compaction: ContextCompactionConfig = field(
        default_factory=ContextCompactionConfig
    )
    prompt_cache: PromptCacheConfig = field(default_factory=PromptCacheConfig)
    dev_mode: bool = False
    peer_agents: list[PeerAgentConfig] = field(default_factory=list)
    wiring: WiringConfig = field(default_factory=WiringConfig)
    plugins: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config.toml") -> Config:
        from importlib import import_module

        return import_module("agent.config").load_config(path)


__all__ = [
    "ChannelsConfig",
    "Config",
    "ConversationSemanticsConfig",
    "ContextCompactionConfig",
    "FitbitIntegrationConfig",
    "LangfuseConfig",
    "MemoryConfig",
    "MemoryEmbeddingConfig",
    "ObservabilityConfig",
    "PeerAgentConfig",
    "PromptCacheConfig",
    "TelegramChannelConfig",
    "WiringConfig",
]
