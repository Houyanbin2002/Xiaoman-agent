"""
配置加载模块
从 config.toml 读取配置，支持 ${ENV_VAR} 格式的环境变量插值。
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
import zlib
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from agent.config_models import (
    ChannelsConfig,
    Config,
    ConversationSemanticsConfig,
    ContextCompactionConfig,
    ExecutionGuardConfig,
    FitbitIntegrationConfig,
    LangfuseConfig,
    MemoryConfig,
    MemoryEmbeddingConfig,
    ObservabilityConfig,
    PeerAgentConfig,
    PromptCacheConfig,
    TelegramChannelConfig,
    WiringConfig,
)
from proactive_v2.config import ProactiveConfig
from proactive_v2.config_loader import ProactiveConfigError, load_proactive_config

_PRESETS: dict[str, str] = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}

# CLI channel 默认 Unix socket 路径
DEFAULT_SOCKET = "127.0.0.1:8765" if os.name == "nt" else "/tmp/xiaoman.sock"


def _normalize_cli_socket_endpoint(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return DEFAULT_SOCKET
    if os.name != "nt":
        return text
    host, sep, port = text.rpartition(":")
    if sep and host:
        try:
            int(port)
            return text
        except ValueError:
            pass
    port_seed = zlib.crc32(text.encode("utf-8")) % 20000
    return f"127.0.0.1:{20000 + port_seed}"


def _validated_timezone(tz_name: str, *, enabled: bool) -> str:
    """仅当 anyaction_enabled=True 时校验时区合法性，无效则启动时 fail-fast。"""
    if not enabled:
        return tz_name
    try:
        ZoneInfo(tz_name)
        return tz_name
    except Exception:
        raise ValueError(
            f"proactive.anyaction_timezone 无效: {tz_name!r}，"
            "请使用 IANA 格式，如 'Asia/Shanghai'"
        )


def load_config(path: str | Path = "config.toml") -> Config:
    data = _load_config_data(path)

    llm = _as_dict(data.get("llm"))
    llm_main = _as_dict(llm.get("main"))
    llm_fast = _as_dict(llm.get("fast"))
    llm_agent = _as_dict(llm.get("agent"))
    llm_vl = _as_dict(llm.get("vl"))
    agent_cfg = _as_dict(data.get("agent"))
    agent_context = _as_dict(agent_cfg.get("context"))
    context_compaction = _as_dict(agent_context.get("compaction"))
    prompt_cache = _as_dict(agent_context.get("cache"))
    execution_guard = _as_dict(agent_cfg.get("guard"))
    agent_tools = _as_dict(agent_cfg.get("tools"))
    agent_maintenance = _as_dict(agent_cfg.get("maintenance"))
    provider = str(llm.get("provider") or data["provider"])
    channels = _load_channels_config(data)
    proactive = _load_proactive_config(data)
    memory = _load_memory_config(data)
    conversation_semantics = _load_conversation_semantics_config(data)
    observability = _load_observability_config(data)
    peer_agents = _load_peer_agents_config(data)
    fitbit = _load_fitbit_config(data)
    wiring = _load_wiring_config(data)
    plugins = _load_plugins_config(data)

    return Config(
        provider=provider,
        model=str(llm_main.get("model") or data["model"]),
        api_key=_resolve(str(llm_main.get("api_key") or data.get("api_key", ""))),
        system_prompt=str(
            agent_cfg.get("system_prompt")
            or data.get("system_prompt", "You are a helpful assistant.")
        ),
        max_tokens=int(agent_cfg.get("max_tokens", data.get("max_tokens", 8192))),
        max_iterations=int(
            agent_cfg.get("max_iterations", data.get("max_iterations", 10))
        ),
        memory_window=int(
            agent_context.get("memory_window", data.get("memory_window", 40))
        ),
        base_url=str(
            llm_main.get("base_url")
            or data.get("base_url")
            or _PRESETS.get(provider)
            or ""
        ),
        extra_body=_load_extra_body(data),
        channels=channels,
        proactive=proactive,
        memory_optimizer_enabled=bool(
            agent_maintenance.get(
                "memory_optimizer_enabled",
                data.get("memory_optimizer_enabled", True),
            )
        ),
        memory_optimizer_interval_seconds=int(
            agent_maintenance.get(
                "memory_optimizer_interval_seconds",
                data.get("memory_optimizer_interval_seconds", 64800),
            )
        ),
        light_model=str(llm_fast.get("model") or data.get("light_model", "")),
        light_provider=str(llm_fast.get("provider") or provider),
        light_api_key=_resolve(
            str(llm_fast.get("api_key") or data.get("light_api_key", ""))
        ),
        light_base_url=str(llm_fast.get("base_url") or data.get("light_base_url", "")),
        agent_model=str(llm_agent.get("model") or data.get("agent_model", "")),
        agent_provider=str(llm_agent.get("provider") or provider),
        agent_api_key=_resolve(
            str(llm_agent.get("api_key") or data.get("agent_api_key", ""))
        ),
        agent_base_url=str(llm_agent.get("base_url") or data.get("agent_base_url", "")),
        memory=memory,
        conversation_semantics=conversation_semantics,
        observability=observability,
        fitbit=fitbit,
        tool_search_enabled=bool(
            agent_tools.get("search_enabled", data.get("tool_search_enabled", False))
        ),
        context_compaction=ContextCompactionConfig(
            enabled=bool(context_compaction.get("enabled", True)),
            trigger_tokens=int(context_compaction.get("trigger_tokens", 200_000)),
            target_tokens=int(context_compaction.get("target_tokens", 100_000)),
            keep_recent_tokens=int(
                context_compaction.get("keep_recent_tokens", 40_000)
            ),
            summary_max_tokens=int(context_compaction.get("summary_max_tokens", 4_096)),
            chunk_tokens=int(context_compaction.get("chunk_tokens", 24_000)),
            max_history_messages=int(
                context_compaction.get("max_history_messages", 2_000)
            ),
        ).normalized(),
        prompt_cache=PromptCacheConfig(
            enabled=bool(prompt_cache.get("enabled", True)),
            keep_recent_tool_rounds=max(
                1, int(prompt_cache.get("keep_recent_tool_rounds", 3))
            ),
            cold_tool_result_chars=max(
                400, int(prompt_cache.get("cold_tool_result_chars", 1800))
            ),
            recent_tool_result_chars=max(
                800, int(prompt_cache.get("recent_tool_result_chars", 24000))
            ),
        ).normalized(),
        execution_guard=ExecutionGuardConfig(
            enabled=bool(execution_guard.get("enabled", True)),
            window_rounds=int(execution_guard.get("window_rounds", 6)),
            same_signature_warn=int(execution_guard.get("same_signature_warn", 2)),
            same_signature_stop=int(execution_guard.get("same_signature_stop", 3)),
            no_progress_rounds=int(execution_guard.get("no_progress_rounds", 4)),
            max_tool_calls=int(execution_guard.get("max_tool_calls", 12)),
            soft_timeout_seconds=float(
                execution_guard.get("soft_timeout_seconds", 600)
            ),
            hard_timeout_seconds=float(
                execution_guard.get("hard_timeout_seconds", 3900)
            ),
            model_call_timeout_seconds=float(
                execution_guard.get("model_call_timeout_seconds", 180)
            ),
            tool_timeout_seconds=float(
                execution_guard.get("tool_timeout_seconds", 300)
            ),
            side_effect_tool_timeout_seconds=float(
                execution_guard.get("side_effect_tool_timeout_seconds", 300)
            ),
            blocking_tool_timeout_seconds=float(
                execution_guard.get("blocking_tool_timeout_seconds", 3600)
            ),
            context_soft_tokens=int(
                execution_guard.get("context_soft_tokens", 120_000)
            ),
            context_hard_tokens=int(
                execution_guard.get("context_hard_tokens", 160_000)
            ),
            max_tool_result_chars=int(
                execution_guard.get("max_tool_result_chars", 12_000)
            ),
            max_tool_round_chars=int(
                execution_guard.get("max_tool_round_chars", 24_000)
            ),
            max_turn_tool_result_chars=int(
                execution_guard.get("max_turn_tool_result_chars", 60_000)
            ),
            subagent_max_iterations=int(
                execution_guard.get("subagent_max_iterations", 10)
            ),
            subagent_timeout_seconds=float(
                execution_guard.get("subagent_timeout_seconds", 3900)
            ),
            subagent_result_chars=int(
                execution_guard.get("subagent_result_chars", 12_000)
            ),
            workflow_max_concurrency=int(
                execution_guard.get("workflow_max_concurrency", 2)
            ),
            workflow_step_timeout_seconds=float(
                execution_guard.get("workflow_step_timeout_seconds", 4200)
            ),
            workflow_max_subagent_steps=int(
                execution_guard.get("workflow_max_subagent_steps", 4)
            ),
        ).normalized(),
        dev_mode=bool(
            agent_cfg.get(
                "dev_mode",
                agent_cfg.get(
                    "dev_model",
                    data.get("dev_mode", data.get("dev_model", False)),
                ),
            )
        ),
        multimodal=bool(llm_main.get("multimodal", True)),
        vl_model=str(llm_vl.get("model") or data.get("vl_model", "")),
        vl_provider=str(llm_vl.get("provider") or provider),
        vl_api_key=_resolve(str(llm_vl.get("api_key") or data.get("vl_api_key", ""))),
        vl_base_url=str(llm_vl.get("base_url") or data.get("vl_base_url", "")),
        peer_agents=peer_agents,
        wiring=wiring,
        plugins=plugins,
    )


def _load_channels_config(data: dict) -> ChannelsConfig:
    channels_data = data.get("channels", {})

    telegram = None
    if tg := channels_data.get("telegram"):
        token = _normalize_optional_config_text(_resolve(str(tg.get("token", ""))))
        if bool(tg.get("enabled", True)) and token:
            telegram = TelegramChannelConfig(
                token=token,
                allow_from=[
                    str(u) for u in tg.get("allow_from", tg.get("allowFrom", []))
                ],
                channel_name=str(tg.get("channel_name", "telegram")),
            )

    cli_data = _as_dict(channels_data.get("cli"))
    socket_value = channels_data.get("socket") or cli_data.get("socket", DEFAULT_SOCKET)
    cli_session_key = str(cli_data.get("session_key") or "").strip()
    cli_channel = str(cli_data.get("channel") or "").strip()
    cli_chat_id = str(cli_data.get("chat_id") or "").strip()
    if not cli_session_key and cli_channel and cli_chat_id:
        cli_session_key = f"{cli_channel}:{cli_chat_id}"
    channels = ChannelsConfig(
        telegram=telegram,
        socket=_normalize_cli_socket_endpoint(socket_value),
        cli_session_key=cli_session_key,
    )
    channels.socket = _normalize_cli_socket_endpoint(channels.socket)
    return channels


def _load_proactive_config(data: dict) -> ProactiveConfig:
    proactive = ProactiveConfig()
    if p := data.get("proactive"):
        try:
            proactive = load_proactive_config(p)
        except ProactiveConfigError as e:
            print(f"❌ Proactive 配置错误: {e}", file=sys.stderr)
            sys.exit(1)
    return proactive


def _load_memory_config(data: dict) -> MemoryConfig:
    memory = _as_dict(data.get("memory"))
    embedding = _as_dict(memory.get("embedding"))
    raw_output_dimensionality = embedding.get("output_dimensionality")
    output_dimensionality = (
        int(raw_output_dimensionality)
        if raw_output_dimensionality not in (None, "")
        else None
    )
    if output_dimensionality is not None and output_dimensionality <= 0:
        raise ValueError("memory.embedding.output_dimensionality 必须大于 0")
    return MemoryConfig(
        enabled=bool(memory.get("enabled", True)),
        engine=str(memory.get("engine", "") or "akasha"),
        embedding=MemoryEmbeddingConfig(
            model=str(embedding.get("model", "text-embedding-v3")),
            api_key=_resolve(str(embedding.get("api_key", ""))),
            base_url=str(embedding.get("base_url", "")),
            output_dimensionality=output_dimensionality,
        ),
    )


def _load_conversation_semantics_config(data: dict) -> ConversationSemanticsConfig:
    raw = _as_dict(data.get("conversation_semantics"))
    return ConversationSemanticsConfig(
        enabled=bool(raw.get("enabled", True)),
        idle_seconds=max(30, int(raw.get("idle_seconds", 480))),
        max_turns=max(1, int(raw.get("max_turns", 8))),
        analysis_version=str(raw.get("analysis_version") or "conversation-v3"),
    )


def _load_observability_config(data: dict) -> ObservabilityConfig:
    observability = _as_dict(data.get("observability"))
    raw = _as_dict(observability.get("langfuse"))
    public_key = _normalize_optional_config_text(
        _resolve(
            str(raw.get("public_key") or os.environ.get("LANGFUSE_PUBLIC_KEY", ""))
        )
    )
    secret_key = _normalize_optional_config_text(
        _resolve(
            str(raw.get("secret_key") or os.environ.get("LANGFUSE_SECRET_KEY", ""))
        )
    )
    base_url = _normalize_optional_config_text(
        _resolve(
            str(
                raw.get("base_url")
                or raw.get("host")
                or os.environ.get("LANGFUSE_BASE_URL")
                or os.environ.get("LANGFUSE_HOST")
                or "https://cloud.langfuse.com"
            )
        )
    )
    sample_rate = min(1.0, max(0.0, float(raw.get("sample_rate", 1.0))))
    return ObservabilityConfig(
        langfuse=LangfuseConfig(
            enabled=bool(raw.get("enabled", False)),
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url or "https://cloud.langfuse.com",
            environment=str(raw.get("environment") or "development"),
            sample_rate=sample_rate,
            flush_at=max(1, int(raw.get("flush_at", 15))),
            flush_interval_seconds=max(
                0.1, float(raw.get("flush_interval_seconds", 1.0))
            ),
            capture_content=bool(raw.get("capture_content", True)),
            max_content_chars=max(1000, int(raw.get("max_content_chars", 60000))),
            debug=bool(raw.get("debug", False)),
        )
    )


def _load_peer_agents_config(data: dict) -> list[PeerAgentConfig]:
    integrations = _as_dict(data.get("integrations"))
    peer_agents = integrations.get("peer_agents", data.get("peer_agents", []))
    return [
        PeerAgentConfig(
            name=pa["name"],
            base_url=pa["base_url"],
            launcher=pa["launcher"],
            cwd=pa.get("cwd"),
            description=pa.get("description", ""),
            health_path=pa.get("health_path", "/health"),
            startup_timeout_s=int(pa.get("startup_timeout_s", 30)),
            shutdown_timeout_s=int(pa.get("shutdown_timeout_s", 10)),
        )
        for pa in peer_agents
    ]


def _load_fitbit_config(data: dict) -> FitbitIntegrationConfig:
    integrations = _as_dict(data.get("integrations"))
    fitbit = _as_dict(integrations.get("fitbit"))
    return FitbitIntegrationConfig(
        enabled=bool(fitbit.get("enabled", False)),
    )


def _load_wiring_config(data: dict) -> WiringConfig:
    agent_cfg = _as_dict(data.get("agent"))
    raw = _as_dict(agent_cfg.get("wiring")) or data.get("wiring", {}) or {}
    toolsets = raw.get(
        "toolsets",
        ["meta_common", "task_executor", "schedule", "mcp", "workflow"],
    )
    if not isinstance(toolsets, list):
        toolsets = ["meta_common", "task_executor", "schedule", "mcp", "workflow"]
    return WiringConfig(
        context=str(raw.get("context", "default") or "default"),
        memory=str(raw.get("memory", "default") or "default"),
        toolsets=[str(name) for name in toolsets if str(name).strip()],
    )


def _load_plugins_config(data: dict) -> dict[str, dict[str, Any]]:
    plugins_data = _as_dict(data.get("plugins"))
    plugins: dict[str, dict[str, Any]] = {}
    for name, value in plugins_data.items():
        if isinstance(name, str) and isinstance(value, dict):
            plugins[name] = cast(dict[str, Any], _resolve_config_value(value))

    return plugins


def _load_extra_body(data: dict) -> dict:
    llm = _as_dict(data.get("llm"))
    llm_main = _as_dict(llm.get("main"))
    extra_body = dict(data.get("extra_body", {}))
    thinking = llm_main.get("thinking")
    if isinstance(thinking, dict):
        extra_body["thinking"] = thinking
    if "enable_thinking" in llm_main:
        extra_body["enable_thinking"] = bool(llm_main.get("enable_thinking"))
    if "reasoning_effort" in llm_main:
        effort = str(llm_main.get("reasoning_effort") or "").strip()
        if effort:
            extra_body["reasoning_effort"] = effort
    return extra_body


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _resolve_config_value(value: object) -> object:
    if isinstance(value, str):
        return _resolve(value)
    if isinstance(value, list):
        return [_resolve_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _resolve_config_value(item) for key, item in value.items()}
    return value


def _resolve(value: str) -> str:
    resolved = re.sub(
        r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value
    )
    # 若仍是未展开的占位符，尝试从 workspace/memory/<VAR_NAME> 文件读取
    m = re.fullmatch(r"\$\{(\w+)\}", resolved)
    if m:
        key_file = Path.home() / ".xiaoman" / "workspace" / "memory" / m.group(1)
        if key_file.exists():
            resolved = key_file.read_text(encoding="utf-8").strip()
    return resolved


def _normalize_optional_config_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\$\{(\w+)\}", text):
        return ""
    return text


def _load_config_data(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"主配置仅支持 TOML: {path.suffix}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "ChannelsConfig",
    "Config",
    "ContextCompactionConfig",
    "DEFAULT_SOCKET",
    "MemoryConfig",
    "MemoryEmbeddingConfig",
    "ObservabilityConfig",
    "TelegramChannelConfig",
    "_validated_timezone",
    "load_config",
]
