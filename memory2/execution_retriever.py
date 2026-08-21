from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from core.memory.execution import (
    ExecutionContext,
    ExecutionMemoryState,
    ExecutionVerificationStatus,
    execution_context_from_mapping,
    execution_rank_score,
    execution_reliability_score,
)
from memory2.execution_store import ExecutionMemoryRepository
from memory2.retriever import Retriever


class ExecutionMemoryRetriever:
    def __init__(
        self,
        *,
        retriever: Retriever,
        repository: ExecutionMemoryRepository,
        inject_max_chars: int = 1200,
    ) -> None:
        self._retriever = retriever
        self._repository = repository
        self._inject_max_chars = max(300, int(inject_max_chars))

    async def retrieve(
        self,
        query: str,
        *,
        context: ExecutionContext | Mapping[str, object] | None = None,
        top_k: int = 5,
        aux_queries: list[str] | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, object]]:
        runtime_context = (
            context
            if isinstance(context, ExecutionContext)
            else execution_context_from_mapping(context)
        )
        candidates = await self._retriever.retrieve(
            query,
            memory_types=["procedure"],
            top_k=max(12, top_k * 3),
            aux_queries=aux_queries,
            hotness_alpha=0.0,
        )
        ranked: list[dict[str, object]] = []
        for candidate in candidates:
            item_id = str(candidate.get("id") or "")
            state = self._repository.get(item_id)
            if state is None:
                continue
            semantic_score = _number(candidate.get("score"))
            score = execution_rank_score(
                semantic_score=semantic_score,
                state=state,
                context=runtime_context,
                now=now,
            )
            if score <= 0.0:
                continue
            item = dict(candidate)
            extra = item.get("extra_json")
            signals = dict(extra) if isinstance(extra, dict) else {}
            signals["execution"] = _state_signals(state, now=now)
            item["extra_json"] = signals
            item["score"] = round(score, 4)
            item["execution_reliability"] = round(
                execution_reliability_score(state, now=now),
                4,
            )
            ranked.append(item)
        ranked.sort(key=lambda item: _number(item.get("score")), reverse=True)
        return ranked[: max(1, int(top_k))]

    def build_injection_block(
        self,
        items: list[dict[str, object]],
    ) -> tuple[str, list[str]]:
        if not items:
            return "", []
        lines = [
            "## 【Agent 执行经验】",
            "以下内容是有适用范围的历史执行经验；环境或版本不一致时必须重新验证，不能当作用户事实。",
            "只有真正采用某条经验时，才在内部推理中写入 "
            '<used-execution-memory id="对应ref"/>；不要在最终答复展示该标记。',
        ]
        injected: list[str] = []
        total = sum(len(line) for line in lines) + 2
        for item in items:
            item_id = str(item.get("id") or "")
            summary = " ".join(str(item.get("summary") or "").split())
            execution = _execution_signals(item)
            label = _verification_label(str(execution.get("verification_status") or ""))
            scope = str(execution.get("scope_label") or "全局")
            line = f"- [ref={item_id} · {label} · {scope}] {summary}"
            if not summary or total + len(line) + 1 > self._inject_max_chars:
                continue
            lines.append(line)
            total += len(line) + 1
            if item_id:
                injected.append(item_id)
        if not injected:
            return "", []
        return "\n".join(lines), injected


def _state_signals(
    state: ExecutionMemoryState,
    *,
    now: datetime | None,
) -> dict[str, object]:
    return {
        "kind": state.kind.value,
        "verification_status": state.verification_status.value,
        "success_count": state.success_count,
        "failure_count": state.failure_count,
        "last_verified_at": (
            state.last_verified_at.isoformat() if state.last_verified_at else ""
        ),
        "expires_at": state.expires_at.isoformat() if state.expires_at else "",
        "reliability": round(execution_reliability_score(state, now=now), 4),
        "scope_kind": state.scope.kind.value,
        "scope_label": _scope_label(state),
        "workspace_id": state.scope.workspace_id,
        "project_id": state.scope.project_id,
        "tool_name": state.scope.tool_name,
        "plugin_name": state.scope.plugin_name,
        "platform": state.scope.platform,
        "environment_fingerprint": state.scope.environment_fingerprint,
        "version_key": state.scope.version_key,
        "version_value": state.scope.version_value,
    }


def _scope_label(state: ExecutionMemoryState) -> str:
    scope = state.scope
    value = (
        scope.tool_name
        or scope.plugin_name
        or scope.project_id
        or scope.workspace_id
        or scope.kind.value
    )
    return f"{scope.kind.value}:{value}" if value != scope.kind.value else value


def _execution_signals(item: Mapping[str, object]) -> dict[str, object]:
    raw_extra = item.get("extra_json")
    if not isinstance(raw_extra, Mapping):
        return {}
    raw_execution = raw_extra.get("execution")
    return dict(raw_execution) if isinstance(raw_execution, Mapping) else {}


def _verification_label(status: str) -> str:
    return {
        ExecutionVerificationStatus.VERIFIED.value: "已验证",
        ExecutionVerificationStatus.CANDIDATE.value: "候选",
        ExecutionVerificationStatus.STALE.value: "需复核",
    }.get(status, "不可用")


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
