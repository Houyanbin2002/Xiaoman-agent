from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import platform
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from agent.config_models import Config
from core.llm import LLMProvider, LLMResponse
from bus.events_lifecycle import TurnCommitted
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import ExecutionMemoryCandidate
from core.memory.engine import (
    EngineProfile,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryMutation,
    MemoryMutationResult,
    MemoryQuery,
    MemoryQueryResult,
    MemoryRecord,
    MemoryToolProfile,
    MemoryToolSpec,
)
from core.memory.execution import ExecutionMemoryState
from core.memory.utils import (
    evidence_from_source_ref,
    resolve_memory_scope,
    should_require_scope_match,
)
from core.net.http import SharedHttpResources
from memory2.embedder import Embedder
from memory2.execution_retriever import ExecutionMemoryRetriever
from memory2.memorizer import Memorizer
from memory2.query_builder import build_procedure_queries
from memory2.retriever import Retriever
from memory2.rule_schema import build_procedure_rule_schema
from memory2.store import VEC_DIM, MemoryStore2
from plugins.default_memory.config import DefaultMemoryConfig, resolve_memory_db_path

if TYPE_CHECKING:
    from bus.event_bus import EventBus

logger = logging.getLogger("plugins.default_memory.engine")

_HYPOTHESIS_MAX_TOKENS = 80
_HYPOTHESIS_TIMEOUT_S = 3.0
_VECTOR_SCORE_THRESHOLD = 0.35
_VECTOR_TOP_K = 15
_ChatCall = Callable[..., Awaitable[LLMResponse]]


def _build_entry_source_ref(base_source_ref: str, entry: str) -> str:
    text = (entry or "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else "empty"
    return f"{base_source_ref}#h:{digest}"


def _build_rule_source_ref(base_source_ref: str, summary: str) -> str:
    text = (summary or "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else "empty"
    return f"{base_source_ref}#r:{digest}"


def _source_ref_message_ids(source_ref: str) -> list[str]:
    raw = str(source_ref or "").strip()
    if not raw:
        return []
    base = raw.split("#", 1)[0].strip()
    if not base.startswith("["):
        return []
    try:
        loaded: object = json.loads(base)
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    values: list[str] = []
    for item in cast(list[object], loaded):
        text = str(item).strip()
        if text:
            values.append(text)
    return values


def _exact_execution_target_id(
    store: MemoryStore2,
    candidate: ExecutionMemoryCandidate,
) -> str:
    """Resolve invalidation by exact id or one exact normalized summary only."""

    target_id = candidate.target_memory_id.strip()
    if target_id and store.execution.get(target_id) is not None:
        return target_id
    target_summary = " ".join(candidate.target_summary.split()).casefold()
    if not target_summary:
        return ""
    matches = [
        str(row.get("id") or "")
        for row in store.execution.list(include_inactive=True, limit=5000)
        if " ".join(str(row.get("summary") or "").split()).casefold() == target_summary
    ]
    return matches[0] if len(matches) == 1 else ""


def _undo_store_by_message_sources(
    store: MemoryStore2,
    message_ids: list[str],
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    clean_ids = [str(item).strip() for item in message_ids if str(item).strip()]
    if not clean_ids:
        return {"affected_ids": [], "restored_ids": [], "rollback_source_ids": []}
    target_ids = set(clean_ids)
    with store._lock:
        rows = store._db.execute("""
            SELECT id, source_ref
            FROM memory_items
            WHERE COALESCE(source_ref, '') != ''
            """).fetchall()
        affected_ids: set[str] = set()
        rollback_source_ids: set[str] = set()
        for item_id, source_ref in rows:
            source = str(source_ref or "").strip()
            base_ids = _source_ref_message_ids(source)
            if source in target_ids:
                affected_ids.add(str(item_id))
                rollback_source_ids.add(source)
                continue
            if base_ids and target_ids.intersection(base_ids):
                affected_ids.add(str(item_id))
                rollback_source_ids.update(base_ids)

        if affected_ids and not dry_run:
            now = datetime.now().astimezone().isoformat()
            store._db.executemany(
                "UPDATE memory_items SET status='superseded', updated_at=? WHERE id=?",
                [(now, item_id) for item_id in sorted(affected_ids)],
            )
        restored_ids = _restore_replacements_for_undo(
            store,
            affected_ids,
            dry_run=dry_run,
        )
        if not dry_run:
            store._db.commit()
    return {
        "affected_ids": sorted(affected_ids),
        "restored_ids": sorted(restored_ids),
        "rollback_source_ids": sorted(rollback_source_ids),
    }


def _restore_replacements_for_undo(
    store: MemoryStore2,
    affected_ids: set[str],
    *,
    dry_run: bool = False,
) -> set[str]:
    if not affected_ids:
        return set()
    sorted_affected = sorted(affected_ids)
    placeholders = ",".join("?" for _ in sorted_affected)
    rows = store._db.execute(
        f"""
        SELECT DISTINCT old_item_id
        FROM memory_replacements
        WHERE new_item_id IN ({placeholders})
        """,
        tuple(sorted_affected),
    ).fetchall()
    old_ids = {str(row[0]) for row in rows if str(row[0]).strip()}
    restored: set[str] = set()
    now = datetime.now().astimezone().isoformat()
    for old_id in sorted(old_ids):
        active_replacement = store._db.execute(
            """
            SELECT 1
            FROM memory_replacements r
            JOIN memory_items m ON m.id = r.new_item_id
            WHERE r.old_item_id = ?
              AND r.new_item_id NOT IN ({})
              AND m.status = 'active'
            LIMIT 1
            """.format(placeholders),
            tuple([old_id, *sorted_affected]),
        ).fetchone()
        if active_replacement is not None:
            continue
        if dry_run:
            old_row = store._db.execute(
                "SELECT 1 FROM memory_items WHERE id=? AND status='superseded'",
                (old_id,),
            ).fetchone()
            if old_row is not None:
                restored.add(old_id)
            continue
        cur = store._db.execute(
            "UPDATE memory_items SET status='active', updated_at=? WHERE id=? AND status='superseded'",
            (now, old_id),
        )
        if cur.rowcount:
            restored.add(old_id)
    return restored


def _default_memory_tool_profile() -> MemoryToolProfile:
    return MemoryToolProfile(
        recall=MemoryToolSpec(
            description=(
                "检索长期记忆中的事实、偏好、流程与历史事件线索。"
                "query 写成陈述句；intent=answer 做主题检索，intent=timeline 做时间线回顾。"
                "返回的是记忆摘要和 evidence，回答依赖原文细节时继续用 fetch_messages 取证。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查找的记忆主题，推荐写成陈述句",
                    },
                    "intent": {
                        "type": "string",
                        "enum": ["answer", "timeline"],
                        "description": "answer=主题检索；timeline=按 time_filter 列出历史事件",
                        "default": "answer",
                    },
                    "memory_kind": {
                        "type": "string",
                        "enum": ["event", "profile", "preference", "procedure", ""],
                        "description": "限定记忆类型，留空表示不限",
                        "default": "",
                    },
                    "time_filter": {
                        "type": "string",
                        "description": "today / yesterday / recent_3d / recent_7d / recent_30d / YYYY-MM-DD / YYYY-MM-DD~YYYY-MM-DD",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回条数",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
            search_hint="记得 以前 历史 做过什么 有没有 重构 记忆查询",
        ),
        memorize=MemoryToolSpec(
            description=(
                "将用户明确要求长期保留的信息写入记忆。"
                "memory_kind 可选 event/profile/preference/procedure，engine 会自行校正分类。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "一句话描述要记住的内容",
                    },
                    "memory_kind": {
                        "type": "string",
                        "enum": ["procedure", "preference", "event", "profile", ""],
                        "description": "记忆类型，留空由 engine 决定",
                        "default": "",
                    },
                    "tool_requirement": {
                        "type": "string",
                        "description": "该规则要求必须调用的工具名（可选）",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "执行步骤（可选）",
                    },
                    "subject": {
                        "type": "string",
                        "description": "个人事实主体，例如“用户”；可可靠结构化时填写",
                    },
                    "predicate": {
                        "type": "string",
                        "description": "稳定关系名，例如“偏好锻炼时间”",
                    },
                    "value": {
                        "type": "string",
                        "description": "关系当前值，例如“晚上”",
                    },
                    "scope": {
                        "type": "string",
                        "description": "事实适用情境或地点；无则留空",
                    },
                    "attributes": {
                        "type": "object",
                        "description": "不改变事实身份的补充属性",
                        "additionalProperties": True,
                    },
                    "replaces": {
                        "type": "string",
                        "description": "用户明确纠正时，被替代的旧说法",
                    },
                    "valid_from": {
                        "type": "string",
                        "description": "事实开始有效的带时区 ISO 时间（可选）",
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "临时事实到期的带时区 ISO 时间（可选）",
                    },
                },
                "required": ["summary"],
            },
            risk="write",
        ),
        forget=MemoryToolSpec(
            description="将已确认错误的记忆条目标记为失效。",
            parameters={
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要失效的 memory item id 列表",
                    }
                },
                "required": ["ids"],
            },
            risk="write",
            search_hint="记错了 删除记忆 撤销错误记忆 失效记忆",
        ),
    )


class DefaultMemoryEngine:
    DESCRIPTOR = MemoryEngineDescriptor(
        name="default",
        profile=EngineProfile.RICH_MEMORY_ENGINE,
        capabilities=frozenset(
            {
                MemoryCapability.RETRIEVE_SEMANTIC,
                MemoryCapability.RETRIEVE_CONTEXT_BLOCK,
                MemoryCapability.RETRIEVE_STRUCTURED_HITS,
                MemoryCapability.MANAGE_HISTORY,
                MemoryCapability.MANAGE_UPDATE,
                MemoryCapability.MANAGE_DELETE,
                MemoryCapability.SEMANTICS_RICH_MEMORY,
            }
        ),
        notes={"owner": "plugins.default_memory.engine"},
    )

    def __init__(
        self,
        *,
        config: Config,
        default_config: DefaultMemoryConfig,
        workspace: Path,
        provider: LLMProvider,
        light_provider: LLMProvider | None = None,
        http_resources: SharedHttpResources,
        event_publisher: "EventBus | None" = None,
        enable_conversation_ingest: bool = True,
    ) -> None:
        self._config = config
        self._default_config = default_config
        self._workspace = workspace
        self._provider = provider
        self._light_provider = light_provider or provider
        self._light_model = config.light_model or config.model
        self._v2_store: MemoryStore2 | None = None
        self._embedder: Embedder | None = None
        self._memorizer: Memorizer | None = None
        self._retriever: Retriever | None = None
        self._execution_retriever: ExecutionMemoryRetriever | None = None
        self._event_bus = event_publisher
        self._embedding_requester = http_resources.external_default
        self.closeables: list[object] = []

        db_path = resolve_memory_db_path(
            workspace=workspace,
            default_config=default_config,
        )
        embedding = config.memory.embedding
        retrieval = default_config.retrieval
        self._v2_store = MemoryStore2(
            db_path,
            vec_dim=embedding.output_dimensionality or VEC_DIM,
        )
        self._embedder = Embedder(
            base_url=embedding.base_url
            or config.light_base_url
            or config.base_url
            or "",
            api_key=embedding.api_key or config.light_api_key or config.api_key,
            model=embedding.model,
            output_dimensionality=embedding.output_dimensionality,
            requester=self._embedding_requester,
        )
        self._memorizer = Memorizer(self._v2_store, self._embedder)
        self._retriever = Retriever(
            self._v2_store,
            self._embedder,
            top_k=retrieval.top_k_history,
            score_threshold=retrieval.score_threshold,
            score_thresholds={
                "procedure": retrieval.thresholds.procedure,
                "preference": retrieval.thresholds.preference,
                "event": retrieval.thresholds.event,
                "profile": retrieval.thresholds.profile,
            },
            relative_delta=retrieval.relative_delta,
            inject_max_chars=retrieval.inject.max_chars,
            inject_max_forced=retrieval.inject.forced,
            inject_max_procedure_preference=retrieval.inject.procedure_preference,
            inject_max_event_profile=retrieval.inject.event_profile,
            inject_line_max=retrieval.inject.line_max,
            procedure_guard_enabled=retrieval.procedure_guard_enabled,
            hotness_alpha=0.20,
        )
        self._execution_retriever = ExecutionMemoryRetriever(
            retriever=self._retriever,
            repository=self._v2_store.execution,
            inject_max_chars=retrieval.inject.max_chars,
        )
        _ = enable_conversation_ingest  # compatibility; subscriptions are domain-specific now
        self._wire_memory2_events()
        self.closeables = [self._v2_store, self._embedder]

    @classmethod
    def ensure_workspace_storage(
        cls,
        *,
        default_config: DefaultMemoryConfig,
        workspace: Path,
    ) -> None:
        db_path = resolve_memory_db_path(
            workspace=workspace,
            default_config=default_config,
        )
        store = MemoryStore2(db_path)
        store.close()

    @property
    def embedding_provider(self) -> Embedder:
        if self._embedder is None:
            raise RuntimeError("embedding provider unavailable")
        return self._embedder

    def reconfigure_embedding(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        output_dimensionality: int | None,
    ) -> None:
        embedding = self._config.memory.embedding
        embedding.model = model
        embedding.base_url = base_url
        embedding.api_key = api_key
        embedding.output_dimensionality = output_dimensionality
        previous = self._embedder
        self._embedder = Embedder(
            base_url=base_url,
            api_key=api_key,
            model=model,
            output_dimensionality=output_dimensionality,
            requester=self._embedding_requester,
        )
        self._rebuild_retrieval_stack()
        self.closeables = [
            self._embedder if item is previous else item for item in self.closeables
        ]

    def _rebuild_retrieval_stack(self) -> None:
        if self._v2_store is None or self._embedder is None:
            raise RuntimeError("memory retrieval stack unavailable")
        retrieval = self._default_config.retrieval
        self._memorizer = Memorizer(self._v2_store, self._embedder)
        self._retriever = Retriever(
            self._v2_store,
            self._embedder,
            top_k=retrieval.top_k_history,
            score_threshold=retrieval.score_threshold,
            score_thresholds={
                "procedure": retrieval.thresholds.procedure,
                "preference": retrieval.thresholds.preference,
                "event": retrieval.thresholds.event,
                "profile": retrieval.thresholds.profile,
            },
            relative_delta=retrieval.relative_delta,
            inject_max_chars=retrieval.inject.max_chars,
            inject_max_forced=retrieval.inject.forced,
            inject_max_procedure_preference=retrieval.inject.procedure_preference,
            inject_max_event_profile=retrieval.inject.event_profile,
            inject_line_max=retrieval.inject.line_max,
            procedure_guard_enabled=retrieval.procedure_guard_enabled,
            hotness_alpha=0.20,
        )
        self._execution_retriever = ExecutionMemoryRetriever(
            retriever=self._retriever,
            repository=self._v2_store.execution,
            inject_max_chars=retrieval.inject.max_chars,
        )

    def _wire_memory2_events(self) -> None:
        if self._event_bus is None:
            return
        self._event_bus.on(
            ConversationSemanticBatchCommitted,
            self._on_semantic_batch_committed,
        )
        if self._v2_store is not None:
            self._event_bus.on(TurnCommitted, self._on_execution_feedback)

    async def _on_execution_feedback(self, event: TurnCommitted) -> None:
        store = self._v2_store
        if store is None:
            return
        retrieval = (event.extra or {}).get("memory_retrieval")
        if not isinstance(retrieval, Mapping):
            return
        raw_ids = retrieval.get("used_execution_memory_ids")
        if not isinstance(raw_ids, list):
            return
        calls = [
            call
            for group in event.tool_chain_raw
            if isinstance(group, dict)
            for call in group.get("calls", [])
            if isinstance(call, dict)
        ]
        if not calls:
            return
        evidence_ref = f"{event.session_key}@{event.timestamp.isoformat() if event.timestamp else 'turn'}"
        for item_id in dict.fromkeys(
            str(item) for item in raw_ids if str(item).strip()
        ):
            state = store.execution.get(item_id)
            item = store.get_item_for_dashboard(item_id)
            if state is None or item is None:
                continue
            required_tools = _execution_required_tools(state, item)
            if not required_tools:
                continue
            matching = [
                call
                for call in calls
                if any(
                    _tool_name_matches(str(call.get("name") or ""), required)
                    for required in required_tools
                )
            ]
            if not matching:
                continue
            statuses = {str(call.get("status") or "").lower() for call in matching}
            if "error" in statuses:
                store.execution.record_outcome(
                    item_id,
                    success=False,
                    evidence_ref=evidence_ref,
                )
                continue
            if (
                statuses
                and statuses <= {"success"}
                and all(
                    any(
                        _tool_name_matches(str(call.get("name") or ""), required)
                        and str(call.get("status") or "").lower() == "success"
                        for call in matching
                    )
                    for required in required_tools
                )
            ):
                store.execution.record_outcome(
                    item_id,
                    success=True,
                    evidence_ref=evidence_ref,
                )

    async def _on_semantic_batch_committed(
        self,
        event: ConversationSemanticBatchCommitted,
    ) -> None:
        save_coros = [
            self._save_from_consolidation(
                history_entry=entry.summary,
                behavior_updates=[],
                source_ref=_build_entry_source_ref(event.batch_id, entry.summary),
                scope_channel=event.channel,
                scope_chat_id=event.chat_id,
                emotional_weight=entry.emotional_weight,
            )
            for entry in event.payload.recent_activity_entries
        ]
        if save_coros:
            await asyncio.gather(*save_coros)
        for candidate in event.payload.execution_memories:
            await self._ingest_execution_candidate(event, candidate)

    async def _ingest_execution_candidate(
        self,
        event: ConversationSemanticBatchCommitted,
        candidate: ExecutionMemoryCandidate,
    ) -> None:
        store = self._v2_store
        user_ids = set(event.user_message_ids) & set(event.message_ids)
        episode_ids = set(event.execution_episode_ids)
        explicit = candidate.origin in {"explicit_user", "user_correction"}
        if explicit:
            if (
                candidate.source_message_id not in user_ids
                or candidate.confidence < 0.75
            ):
                logger.info("execution candidate rejected: unverified user authority")
                return
            authority = "user"
            lifecycle = "active"
            user_locked = True
        else:
            if (
                not episode_ids.intersection(candidate.evidence_refs)
                or candidate.confidence < 0.70
            ):
                logger.info("execution candidate rejected: unverified episode evidence")
                return
            authority = "learned"
            lifecycle = "proposed" if candidate.outcome == "success" else "candidate"
            user_locked = False

        if candidate.operation in {"suspend", "supersede"}:
            if store is None or not explicit:
                logger.info("execution invalidation rejected: exact target required")
                return
            target_id = _exact_execution_target_id(store, candidate)
            if not target_id:
                logger.info("execution invalidation rejected: exact target not found")
                return
            if candidate.operation == "supersede":
                store.execution.mark_superseded([target_id])
            else:
                store.execution.suspend([target_id])
            return

        required_tools = list(candidate.required_tools)
        if not explicit:
            observed_tools = set(event.execution_tool_names)
            required_tools = [tool for tool in required_tools if tool in observed_tools]
        source_ref = _build_rule_source_ref(event.batch_id, candidate.summary)
        result = await self._remember(
            MemoryMutation(
                kind="remember",
                summary=candidate.summary,
                memory_kind="procedure",
                source_ref=source_ref,
                user_confirmed=explicit,
                metadata={
                    "tool_requirement": required_tools[0] if required_tools else "",
                    "required_tools": required_tools,
                    "steps": list(candidate.steps) or [candidate.summary],
                    "execution_kind": candidate.kind,
                    "execution_scope": {
                        "kind": "workspace",
                        "workspace_id": str(self._workspace.resolve()),
                        "project_id": self._workspace.name,
                        "platform": platform.system(),
                    },
                    "authority": authority,
                    "lifecycle_status": lifecycle,
                    "user_locked": user_locked,
                    "extraction_confidence": candidate.confidence,
                    "execution_verified": candidate.outcome == "success"
                    and not explicit,
                    "evidence_refs": list(candidate.evidence_refs),
                },
            )
        )
        if not result.item_id or explicit or store is None:
            return
        if candidate.outcome in {"success", "failure"}:
            store.execution.record_outcome(
                result.item_id,
                success=candidate.outcome == "success",
                evidence_ref=next(iter(candidate.evidence_refs), source_ref),
            )

    def tool_profile(self) -> MemoryToolProfile:
        return _default_memory_tool_profile()

    async def query(
        self,
        request: MemoryQuery,
    ) -> MemoryQueryResult:
        if self._retriever is None:
            return MemoryQueryResult(raw={"items": []})
        if request.intent == "timeline":
            return self._query_timeline(request)
        if request.intent == "interest":
            return await self._query_interest(request)
        if request.intent in {"procedure", "execution"}:
            return await self._query_execution(request)
        if request.intent == "context":
            return await self._query_context(request)
        return await self._query_answer(request)

    async def _query_context(self, request: MemoryQuery) -> MemoryQueryResult:
        retriever = self._retriever
        if retriever is None:
            return MemoryQueryResult(raw={"items": []})
        scope = resolve_memory_scope(request.scope)
        queries = self._resolve_queries(request)
        execution_retriever = getattr(self, "_execution_retriever", None)
        memory_types = self._resolve_memory_types(request)
        if execution_retriever is not None and memory_types is None:
            memory_types = ["preference", "profile", "event"]
        items = await self._retrieve_related(
            request.text,
            memory_types=memory_types,
            top_k=request.limit,
            scope_channel=scope.channel or None,
            scope_chat_id=scope.chat_id or None,
            require_scope_match=bool(
                request.filters.hints.get("require_scope_match", False)
            ),
            aux_queries=queries[1:],
            time_start=request.filters.time_start,
            time_end=request.filters.time_end,
        )
        text_block, injected_ids = retriever.build_injection_block(items)
        execution_items: list[dict[str, object]] = []
        execution_block = ""
        execution_ids: list[str] = []
        if execution_retriever is not None:
            execution_items = await execution_retriever.retrieve(
                request.text,
                context=_execution_context(request),
                top_k=min(5, max(1, request.limit)),
                aux_queries=queries[1:],
                now=request.timestamp,
            )
            execution_block, execution_ids = execution_retriever.build_injection_block(
                execution_items
            )
        combined_block = "\n\n".join(
            part for part in (text_block, execution_block) if part.strip()
        )
        all_items = [*items, *execution_items]
        all_injected_ids = [*injected_ids, *execution_ids]
        records = [
            self._build_record(item, injected_ids=all_injected_ids)
            for item in all_items
        ]
        return MemoryQueryResult(
            text_block=combined_block,
            records=records,
            trace={
                "engine": self.DESCRIPTOR.name,
                "profile": self.DESCRIPTOR.profile.value,
                "intent": request.intent,
                "effect": request.effect,
            },
            raw={
                "items": items,
                "execution_items": execution_items,
            },
        )

    async def _query_execution(self, request: MemoryQuery) -> MemoryQueryResult:
        execution_retriever = getattr(self, "_execution_retriever", None)
        if execution_retriever is None:
            return MemoryQueryResult(raw={"items": []})
        queries = self._resolve_queries(request)
        items = await execution_retriever.retrieve(
            request.text,
            context=_execution_context(request),
            top_k=min(5, max(1, request.limit)),
            aux_queries=queries[1:],
            now=request.timestamp,
        )
        text_block, injected_ids = execution_retriever.build_injection_block(items)
        return MemoryQueryResult(
            text_block=text_block,
            records=[
                self._build_record(item, injected_ids=injected_ids) for item in items
            ],
            trace={
                "engine": self.DESCRIPTOR.name,
                "profile": self.DESCRIPTOR.profile.value,
                "intent": "execution",
                "effect": request.effect,
            },
            raw={"items": items},
        )

    # 对话语义摄入由共享批处理器统一负责，避免每个消费者重复调用模型。
    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        return MemoryIngestResult(
            accepted=False,
            summary="conversation semantics are owned by the shared batcher",
            raw={
                "engine": self.DESCRIPTOR.name,
                "reason": "semantic_batcher_owned",
                "source_kind": request.source_kind,
            },
        )

    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        if request.kind == "forget":
            return await self._forget(request)
        return await self._remember(request)

    # 显式记忆写入入口，供 memorize 工具和内部迁移代码复用。
    async def _remember(self, request: MemoryMutation) -> MemoryMutationResult:
        # 1. procedure 必须有执行条件，否则降级为 preference。
        if self._memorizer is None:
            raise RuntimeError("memorizer unavailable")

        raw_steps = request.metadata.get("steps")
        steps = (
            [str(step) for step in cast(list[object], raw_steps)]
            if isinstance(raw_steps, list)
            else None
        )
        memory_type = _coerce_memory_type(
            request.memory_kind,
            str(request.metadata.get("tool_requirement") or ""),
            steps,
        )
        extra: dict[str, object] = {
            "tool_requirement": request.metadata.get("tool_requirement"),
            "steps": list(steps or []),
        }
        for key in (
            "authority",
            "lifecycle_status",
            "user_locked",
            "extraction_confidence",
            "evidence_refs",
        ):
            if key in request.metadata:
                extra[key] = request.metadata[key]
        if isinstance(request.metadata.get("execution_scope"), Mapping):
            extra["execution_scope"] = dict(
                cast(Mapping[str, object], request.metadata["execution_scope"])
            )
        if request.metadata.get("execution_kind"):
            extra["execution_kind"] = str(request.metadata["execution_kind"])
        if memory_type == "procedure":
            extra["rule_schema"] = build_procedure_rule_schema(
                summary=request.summary,
                tool_requirement=str(request.metadata.get("tool_requirement") or "")
                or None,
                steps=list(steps or []),
            )
            raw_required = request.metadata.get("required_tools")
            required_tools = (
                [
                    str(tool).strip().lower()
                    for tool in raw_required
                    if str(tool).strip()
                ]
                if isinstance(raw_required, list)
                else []
            )
            if required_tools:
                extra["rule_schema"] = {
                    **cast(dict[str, object], extra["rule_schema"]),
                    "required_tools": list(dict.fromkeys(required_tools)),
                }
            extra["trigger_tags"] = {
                "tools": list(dict.fromkeys(required_tools)),
                "skills": [],
                "keywords": [],
                "scope": "tool_triggered" if required_tools else "global",
            }

        # 2. 执行经验只按内容哈希去重，避免学习候选凭相似度淘汰旧规则；
        #    非执行类型保留原有的显式 supersede 行为。
        save = (
            self._memorizer.save_item
            if memory_type == "procedure"
            else self._memorizer.save_item_with_supersede
        )
        result = await save(
            summary=request.summary,
            memory_type=memory_type,
            extra=extra,
            source_ref=request.source_ref or "memorize_tool",
            execution_verified=bool(request.metadata.get("execution_verified")),
        )
        write_status, actual_id = _split_write_result(result)
        return MemoryMutationResult(
            accepted=bool(actual_id),
            item_id=actual_id,
            actual_kind=memory_type,
            status=write_status,
        )

    # 显式遗忘入口：只把条目标成 superseded，不物理删除。
    async def _forget(self, request: MemoryMutation) -> MemoryMutationResult:
        # 1. 先按 id 去重并读取现存条目。
        store = self._require_v2_store()
        clean_ids = _dedupe_ids(list(request.ids))
        items = store.get_items_by_ids(clean_ids)
        found_ids = [str(item.get("id") or "") for item in items if item.get("id")]

        # 2. 只失效能确认存在的条目，缺失 id 返回给调用方展示。
        if found_ids:
            store.mark_superseded_batch(found_ids)
        return MemoryMutationResult(
            accepted=bool(found_ids),
            status="superseded",
            affected_ids=found_ids,
            missing_ids=[
                item_id for item_id in clean_ids if item_id not in set(found_ids)
            ],
            items=[
                {
                    "id": item.get("id"),
                    "memory_type": item.get("memory_type"),
                    "summary": item.get("summary"),
                }
                for item in items
            ],
        )

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR

    def reinforce_items_batch(self, ids: list[str]) -> None:
        if self._memorizer is not None:
            self._memorizer.reinforce_items_batch(ids)

    def list_execution_memories(
        self,
        *,
        include_inactive: bool = False,
        limit: int = 500,
    ) -> list[dict[str, object]]:
        return self._require_v2_store().execution.list(
            include_inactive=include_inactive,
            limit=limit,
        )

    def record_execution_outcome(
        self,
        item_id: str,
        *,
        success: bool,
        evidence_ref: str = "",
        verified_at: datetime | None = None,
    ) -> ExecutionMemoryState:
        return self._require_v2_store().execution.record_outcome(
            item_id,
            success=success,
            evidence_ref=evidence_ref,
            verified_at=verified_at,
        )

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        store = self._v2_store
        return (
            store.keyword_match_procedures(action_tokens) if store is not None else []
        )

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        store = self._v2_store
        if store is None:
            return []
        return store.list_events_by_time_range(time_start, time_end, limit=limit)

    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        store = self._require_v2_store()
        return store.list_items_for_dashboard(
            q=q,
            memory_type=memory_type,
            status=status,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            has_embedding=has_embedding,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        return self._require_v2_store().get_item_for_dashboard(
            item_id,
            include_embedding=include_embedding,
        )

    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None:
        return self._require_v2_store().update_item_for_dashboard(
            item_id,
            status=status,
            extra_json=extra_json,
            source_ref=source_ref,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    def delete_item(self, item_id: str) -> bool:
        return self._require_v2_store().delete_item(item_id)

    def delete_items_batch(self, ids: list[str]) -> int:
        return self._require_v2_store().delete_items_batch(ids)

    def undo_by_message_sources(
        self,
        message_ids: list[str],
        *,
        dry_run: bool = False,
    ) -> dict[str, object]:
        return _undo_store_by_message_sources(
            self._require_v2_store(),
            message_ids,
            dry_run=dry_run,
        )

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        return self._require_v2_store().find_similar_items_for_dashboard(
            item_id,
            top_k=top_k,
            memory_type=memory_type,
            score_threshold=score_threshold,
            include_superseded=include_superseded,
        )

    async def _save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict[str, object]],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        emotional_weight: int = 0,
    ) -> None:
        if self._memorizer is None:
            return
        await self._memorizer.save_from_consolidation(
            history_entry=history_entry,
            behavior_updates=behavior_updates,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            emotional_weight=emotional_weight,
        )

    async def _save_item_with_supersede(
        self,
        summary: str,
        memory_type: str,
        extra: dict[str, object],
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        if self._memorizer is None:
            return ""
        return await self._memorizer.save_item_with_supersede(
            summary=summary,
            memory_type=memory_type,
            extra=extra,
            source_ref=source_ref,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    async def _query_answer(
        self,
        request: MemoryQuery,
    ) -> MemoryQueryResult:
        hyp1_task = asyncio.create_task(
            self._gen_hypothesis(request.text, style="event")
        )
        hyp2_task = asyncio.create_task(
            self._gen_hypothesis(request.text, style="general")
        )
        hyp1, hyp2 = await asyncio.gather(hyp1_task, hyp2_task)
        aux_queries = [text for text in (hyp1, hyp2) if text]
        scope = resolve_memory_scope(request.scope)
        types = self._resolve_memory_types(request)
        hits = await self._retrieve_related(
            request.text,
            memory_types=types,
            top_k=max(request.limit, _VECTOR_TOP_K),
            scope_channel=scope.channel or None,
            scope_chat_id=scope.chat_id or None,
            require_scope_match=should_require_scope_match(request, scope),
            aux_queries=aux_queries,
            score_threshold=_VECTOR_SCORE_THRESHOLD,
            time_start=request.filters.time_start,
            time_end=request.filters.time_end,
            keyword_enabled=True,
        )
        sliced = list(hits)[: request.limit]
        return MemoryQueryResult(
            records=[
                self._build_record(item) for item in sliced if isinstance(item, dict)
            ],
            trace={
                "source": self.DESCRIPTOR.name,
                "intent": request.intent,
                "effect": request.effect,
                "hit_count": len(sliced),
                "hyde_hypotheses": aux_queries,
            },
            raw={"items": sliced},
        )

    def _query_timeline(
        self,
        request: MemoryQuery,
    ) -> MemoryQueryResult:
        if request.filters.time_start is None or request.filters.time_end is None:
            return MemoryQueryResult(
                trace={
                    "source": self.DESCRIPTOR.name,
                    "intent": "timeline_missing_time",
                    "effect": request.effect,
                }
            )
        hits = self.list_events_by_time_range(
            request.filters.time_start,
            request.filters.time_end,
            limit=request.limit,
        )
        return MemoryQueryResult(
            records=[
                self._build_record(item) for item in hits if isinstance(item, dict)
            ],
            trace={
                "source": self.DESCRIPTOR.name,
                "intent": "timeline",
                "effect": request.effect,
                "hit_count": len(hits),
            },
            raw={"items": list(hits)},
        )

    async def _query_interest(
        self,
        request: MemoryQuery,
    ) -> MemoryQueryResult:
        scope = resolve_memory_scope(request.scope)
        hits = await self._retrieve_related(
            request.text,
            memory_types=["preference", "profile"],
            top_k=request.limit,
            scope_channel=scope.channel or None,
            scope_chat_id=scope.chat_id or None,
            require_scope_match=should_require_scope_match(request, scope),
        )
        records = [self._build_record(item) for item in hits if isinstance(item, dict)]
        texts = [record.summary for record in records]
        return MemoryQueryResult(
            text_block="\n---\n".join(texts),
            records=records,
            trace={
                "source": self.DESCRIPTOR.name,
                "intent": "interest",
                "effect": request.effect,
            },
            raw={"items": list(hits)},
        )

    async def _retrieve_related(
        self,
        query: str,
        *,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
        aux_queries: list[str] | None = None,
        score_threshold: float | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        keyword_enabled: bool = True,
    ) -> list[dict[str, object]]:
        retriever = self._retriever
        if retriever is None:
            return []
        return cast(
            list[dict[str, object]],
            await retriever.retrieve(
                query,
                memory_types=memory_types,
                top_k=top_k,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
                require_scope_match=require_scope_match,
                aux_queries=aux_queries,
                score_threshold=score_threshold,
                time_start=time_start,
                time_end=time_end,
                keyword_enabled=keyword_enabled,
            ),
        )

    async def _gen_hypothesis(self, query: str, style: str) -> str | None:
        prompt = _explicit_hypothesis_prompt(query, style)
        try:
            chat = cast(_ChatCall, getattr(self._light_provider, "chat"))
            resp = await asyncio.wait_for(
                chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._light_model,
                    max_tokens=_HYPOTHESIS_MAX_TOKENS,
                ),
                timeout=_HYPOTHESIS_TIMEOUT_S,
            )
            text = (resp.content or "").strip()
            return text if text else None
        except Exception as e:
            logger.debug("explicit retrieval hypothesis failed: %s", e)
            return None

    def _require_v2_store(self) -> MemoryStore2:
        if self._v2_store is None:
            raise RuntimeError("memory v2 store unavailable")
        return self._v2_store

    @classmethod
    def _build_record(
        cls,
        item: dict[str, object],
        *,
        injected_ids: list[str] | None = None,
    ) -> MemoryRecord:
        extra = item.get("extra_json")
        signals = (
            dict(cast(dict[str, object], extra)) if isinstance(extra, dict) else {}
        )
        memory_kind = str(item.get("memory_type", "") or "")
        item_id = str(item.get("id", "") or "")
        source_ref = str(item.get("source_ref", "") or "")
        raw_score = item.get("score", 0.0)
        score = raw_score if isinstance(raw_score, int | float) else 0.0
        return MemoryRecord(
            id=item_id,
            kind=memory_kind,
            summary=str(item.get("summary", "") or ""),
            score=float(score),
            engine_kind=cls.DESCRIPTOR.name,
            evidence=evidence_from_source_ref(source_ref),
            signals=signals,
            injected=item_id in set(injected_ids or []),
        )

    @staticmethod
    def _resolve_memory_types(
        request: MemoryQuery,
    ) -> list[str] | None:
        if request.filters.kinds:
            return [str(item) for item in request.filters.kinds if str(item).strip()]
        if request.intent in {"procedure", "execution"}:
            return ["procedure", "preference"]
        return None

    @staticmethod
    def _resolve_queries(request: MemoryQuery) -> list[str]:
        raw_queries = request.filters.hints.get("queries")
        if isinstance(raw_queries, list):
            queries = [str(item).strip() for item in raw_queries if str(item).strip()]
            if queries:
                return queries
        if request.intent in {"procedure", "execution"}:
            return build_procedure_queries(request.text)
        return [request.text]


def _execution_context(request: MemoryQuery) -> dict[str, object]:
    raw = request.context.get("execution")
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    raw_tools = payload.get("tools")
    tools = (
        [str(item) for item in raw_tools]
        if isinstance(raw_tools, (list, tuple, set, frozenset))
        else [str(raw_tools)] if isinstance(raw_tools, str) else []
    )
    tools.extend(
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,80}", request.text)
    )
    payload["tools"] = list(dict.fromkeys(item for item in tools if item.strip()))
    return payload


def _execution_required_tools(
    state: ExecutionMemoryState,
    item: Mapping[str, object],
) -> tuple[str, ...]:
    required: list[str] = []
    if state.scope.tool_name:
        required.append(state.scope.tool_name)
    extra = item.get("extra_json")
    if isinstance(extra, Mapping):
        schema = extra.get("rule_schema")
        if isinstance(schema, Mapping):
            raw_required = schema.get("required_tools")
            if isinstance(raw_required, list):
                required.extend(str(tool) for tool in raw_required)
    return tuple(
        dict.fromkeys(tool.strip().lower() for tool in required if tool.strip())
    )


def _tool_name_matches(actual: str, required: str) -> bool:
    actual_name = actual.strip().lower()
    required_name = required.strip().lower()
    if not actual_name or not required_name:
        return False
    if actual_name == required_name or actual_name.startswith(f"{required_name}__"):
        return True
    return required_name in re.split(r"[^a-z0-9]+", actual_name)


def _coerce_memory_type(
    memory_type: str,
    tool_requirement: str | None,
    steps: list[str] | None,
) -> str:
    if memory_type != "procedure":
        return memory_type
    if tool_requirement and tool_requirement.strip():
        return memory_type
    if steps and any(str(step).strip() for step in steps):
        return memory_type
    return "preference"


def _split_write_result(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if ":" not in raw:
        return "new", raw
    status, item_id = raw.split(":", 1)
    return status or "new", item_id


def _dedupe_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        item_id = str(raw or "").strip()
        if item_id and item_id not in seen:
            seen.add(item_id)
            out.append(item_id)
    return out


def _keep_count(window: int) -> int:
    aligned_window = max(6, ((max(1, window) + 5) // 6) * 6)
    return aligned_window // 2


def _explicit_hypothesis_prompt(query: str, style: str) -> str:
    if style == "event":
        return (
            "你是个人助手的记忆系统。根据用户提问，生成一条带具体时间的假想记忆条目，"
            "格式如 '[2026-03-08] 用户...'\n"
            "规则：第三人称、简洁事实陈述、只输出那一条文本\n\n"
            f"用户提问：{query}\n假想记忆条目："
        )
    return (
        "你是个人助手的记忆系统。根据用户提问，生成一条假想记忆条目。\n"
        "规则：始终生成肯定式、第三人称（'用户…'）、简洁事实陈述、只输出那一条文本\n\n"
        f"用户提问：{query}\n假想记忆条目："
    )
