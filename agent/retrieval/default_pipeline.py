from __future__ import annotations

import asyncio
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.core.types import RetrievalTrace
from agent.looping.ports import MemoryServices
from agent.retrieval.protocol import (
    MemoryRetrievalPipeline,
    RetrievalRequest,
    RetrievalResult,
)
from core.memory.engine import (
    MemoryQuery,
    MemoryQueryFilters,
    MemoryQueryResult,
    MemoryScope,
)
from core.memory.personal_retrieval import PersonalMemoryQueryResult
from core.tracing import record_trace_event

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE_TIMEOUT_S = 3.0


class DefaultMemoryRetrievalPipeline(MemoryRetrievalPipeline):
    def __init__(
        self,
        memory: MemoryServices,
        workspace: Path | None = None,
        *,
        source_timeout_s: float = _DEFAULT_SOURCE_TIMEOUT_S,
    ) -> None:
        self._memory = memory
        self._workspace = workspace
        self._source_timeout_s = max(0.05, float(source_timeout_s))

    # 被动预检索入口：只转换请求形状，检索语义统一交给 MemoryEngine。
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        # 1. 两类来源都不可用时，主链继续无记忆回复。
        if self._memory.engine is None and self._memory.runtime is None:
            record_trace_event(
                category="memory",
                name="recall",
                summary="未启用可用的记忆来源",
                status="skipped",
                started_at=started_at,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            return RetrievalResult(block="", trace=None)

        # 2. 语义/执行引擎与 governed personal store 独立检索，再统一组装。
        # Engine memory and governed personal memory are independent sources.
        # Run them concurrently and isolate their latency/failures so memory
        # enrichment can never take the main conversation path down.
        (result, engine_error), (personal, personal_error) = await asyncio.gather(
            self._retrieve_engine(request),
            self._retrieve_personal(request),
        )
        personal_block = personal.text_block if personal is not None else ""
        personal_count = len(personal.hits) if personal is not None else 0
        execution_ids = [
            record.id
            for record in result.records
            if record.injected and record.kind == "procedure" and record.id
        ]
        personal_ids = [hit.record.id for hit in personal.hits] if personal else []
        block = "\n\n".join(
            part for part in (personal_block, result.text_block) if part.strip()
        )
        trace = _build_retrieval_trace(result)
        if personal_count:
            trace = trace or RetrievalTrace()
            trace.injected_count += personal_count

        # 3. 只返回主链需要注入的文本块和可观测 trace。
        retrieval = RetrievalResult(
            block=block,
            trace=trace,
            metadata={
                "personal_memory_count": personal_count,
                "personal_memory_ids": personal_ids,
                "execution_memory_ids": execution_ids,
            },
        )
        total_count = personal_count + sum(
            1 for record in result.records if record.injected
        )
        record_trace_event(
            category="memory",
            name="recall",
            summary=(
                f"找到 {total_count} 条相关记忆"
                if total_count
                else "没有找到需要带入本轮的记忆"
            ),
            status=(
                "degraded" if engine_error or personal_error else "completed"
            ),
            started_at=started_at,
            duration_ms=int((time.perf_counter() - started) * 1000),
            payload={
                "retrieved_count": total_count,
                "personal_memory_ids": personal_ids,
                "execution_memory_ids": execution_ids,
                "gate_type": trace.gate_type if trace else None,
                "route_decision": trace.route_decision if trace else None,
                "rewritten_query": trace.rewritten_query if trace else None,
                "degraded_sources": [
                    source
                    for source, error in (
                        ("engine", engine_error),
                        ("personal", personal_error),
                    )
                    if error
                ],
            },
        )
        return retrieval

    async def _retrieve_engine(
        self,
        request: RetrievalRequest,
    ) -> tuple[MemoryQueryResult, str]:
        engine = self._memory.engine
        if engine is None:
            return MemoryQueryResult(), ""
        query = MemoryQuery(
            text=request.message,
            intent="context",
            scope=MemoryScope(
                session_key=request.session_key,
                channel=request.channel,
                chat_id=request.chat_id,
            ),
            context={
                "history": request.history,
                "session_metadata": request.session_metadata,
                "execution": self._execution_context(),
            },
            filters=MemoryQueryFilters(hints=dict(request.extra or {})),
            timestamp=request.timestamp,
        )
        try:
            return (
                await asyncio.wait_for(
                    engine.query(query),
                    timeout=self._source_timeout_s,
                ),
                "",
            )
        except TimeoutError:
            logger.warning("memory engine recall timed out; continuing without it")
            return MemoryQueryResult(), "timeout"
        except Exception as exc:
            logger.warning(
                "memory engine recall failed; continuing without it: %s",
                exc,
            )
            return MemoryQueryResult(), type(exc).__name__

    async def _retrieve_personal(
        self,
        request: RetrievalRequest,
    ) -> tuple[PersonalMemoryQueryResult | None, str]:
        runtime = self._memory.runtime
        if runtime is None:
            return None, ""
        try:
            return (
                await asyncio.wait_for(
                    runtime.retrieve_personal_memory_async(
                        request.message,
                        limit=4,
                    ),
                    timeout=self._source_timeout_s,
                ),
                "",
            )
        except TimeoutError:
            logger.warning("personal memory recall timed out; continuing without it")
            return None, "timeout"
        except Exception as exc:
            logger.warning(
                "personal memory recall failed; continuing without it: %s",
                exc,
            )
            return None, type(exc).__name__

    def _execution_context(self) -> dict[str, object]:
        if self._workspace is None:
            return {"platform": platform.system()}
        resolved = self._workspace.expanduser().resolve()
        return {
            "workspace_id": str(resolved),
            "project_id": resolved.name,
            "platform": platform.system(),
        }


# 把 engine trace 收窄成 agent loop 认识的检索 trace。
def _build_retrieval_trace(
    result: MemoryQueryResult,
) -> RetrievalTrace | None:
    if not result.trace and not result.records and not result.text_block:
        return None
    return RetrievalTrace(
        gate_type=str(result.trace.get("gate_type") or "") or None,
        route_decision=str(result.trace.get("route_decision") or "") or None,
        rewritten_query=str(result.raw.get("rewritten_query") or "") or None,
        injected_count=sum(1 for record in result.records if record.injected),
        raw=result.raw.get("retrieval_event"),
    )
