from __future__ import annotations

import logging

from agent.config_models import ObservabilityConfig
from core.tracing.composite import CompositeTraceRecorder
from core.tracing.ports import TraceRecorder
from infra.observability.langfuse_recorder import LangfuseTraceRecorder

logger = logging.getLogger(__name__)


def build_trace_recorder(
    local_recorder: TraceRecorder,
    config: ObservabilityConfig,
) -> TraceRecorder:
    langfuse = config.langfuse
    if not langfuse.enabled:
        return local_recorder
    if not langfuse.public_key or not langfuse.secret_key:
        logger.warning(
            "Langfuse 已启用但未配置 LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY；"
            "本次仅记录本地 TraceStore"
        )
        return local_recorder
    try:
        remote = LangfuseTraceRecorder(langfuse)
    except Exception:
        logger.warning("Langfuse 初始化失败；本次仅记录本地 TraceStore", exc_info=True)
        return local_recorder
    logger.info(
        "Langfuse observability enabled base_url=%s environment=%s sample_rate=%.2f",
        langfuse.base_url,
        langfuse.environment,
        langfuse.sample_rate,
    )
    return CompositeTraceRecorder((local_recorder, remote))


__all__ = ["build_trace_recorder"]
