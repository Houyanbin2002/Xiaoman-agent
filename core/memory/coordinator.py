from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
import inspect
from typing import Protocol

from core.memory.engine import (
    EngineProfile,
    EvidenceRef,
    ExecutionMemoryAdminApi,
    MemoryCapability,
    MemoryEngine,
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
from core.memory.personal_retrieval import PersonalMemoryQueryResult
from core.memory.personal_semantic import PersonalSemanticRecallService
from core.personal.models import PersonalRecord


class GovernedPersonalMemorySource(Protocol):
    def search_personal_memory(
        self,
        query: str,
        *,
        limit: int = 6,
        context_tags: set[str] | None = None,
        now: datetime | None = None,
        semantic_scores: Mapping[str, float] | None = None,
    ) -> PersonalMemoryQueryResult: ...

    def personal_memory_records(self) -> list[PersonalRecord]: ...

    def remember_explicit(
        self,
        summary: str,
        *,
        memory_kind: str = "",
        source_ref: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[str, PersonalRecord | None]: ...

    def forget_explicit(
        self,
        ids: tuple[str, ...],
    ) -> tuple[list[str], list[str], list[dict[str, object]]]: ...


class CompositeMemoryEngine:
    """Coordinate personal, execution and episodic memory without mixing ownership.

    ``structured`` owns execution experience, ``episodic`` owns the rebuildable
    conversation index, and the governed personal source is bound after personal.db
    has been assembled by the application composition root.
    """

    DESCRIPTOR = MemoryEngineDescriptor(
        name="xiaoman",
        profile=EngineProfile.RICH_MEMORY_ENGINE,
        capabilities=frozenset(
            {
                MemoryCapability.INGEST_MESSAGES,
                MemoryCapability.RETRIEVE_SEMANTIC,
                MemoryCapability.RETRIEVE_CONTEXT_BLOCK,
                MemoryCapability.RETRIEVE_STRUCTURED_HITS,
                MemoryCapability.MANAGE_HISTORY,
                MemoryCapability.MANAGE_UPDATE,
                MemoryCapability.MANAGE_DELETE,
                MemoryCapability.ENRICH_GRAPH_RELATIONS,
                MemoryCapability.SEMANTICS_RICH_MEMORY,
            }
        ),
        notes={
            "owner": "core.memory.coordinator",
            "configured_as": "akasha",
            "components": ["governed_personal", "execution", "akasha_episodic"],
            "truth": {
                "personal": "personal.db",
                "execution": "memory2.db/execution_memory_state",
                "episodic": "sessions.db/messages",
            },
        },
    )

    def __init__(
        self,
        *,
        structured: MemoryEngine,
        episodic: MemoryEngine,
        personal_semantic: PersonalSemanticRecallService | None = None,
    ) -> None:
        self._structured = structured
        self._episodic = episodic
        self._personal: GovernedPersonalMemorySource | None = None
        self._personal_semantic = personal_semantic

    def bind_personal_memory(self, source: GovernedPersonalMemorySource) -> None:
        self._personal = source

    def reconfigure_embedding(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        output_dimensionality: int | None,
    ) -> None:
        for engine in (self._structured, self._episodic):
            reconfigure = getattr(engine, "reconfigure_embedding", None)
            if callable(reconfigure):
                reconfigure(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    output_dimensionality=output_dimensionality,
                )
        semantic = self._personal_semantic
        provider = getattr(self._structured, "embedding_provider", None)
        if semantic is not None and provider is not None:
            semantic.reconfigure(
                embedder=provider,
                model=f"{model.strip() or 'default'}:{output_dimensionality or 'native'}",
            )

    async def aclose(self) -> None:
        closeables = [
            self._personal_semantic,
            *getattr(self._structured, "closeables", []),
            *getattr(self._episodic, "closeables", []),
        ]
        seen: set[int] = set()
        for closeable in reversed([item for item in closeables if item is not None]):
            if id(closeable) in seen:
                continue
            seen.add(id(closeable))
            aclose = getattr(closeable, "aclose", None)
            close = getattr(closeable, "close", None)
            result = (
                aclose() if callable(aclose) else close() if callable(close) else None
            )
            if inspect.isawaitable(result):
                await result

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        # Raw conversations belong to the episodic index. Structured memory does
        # not receive a second copy of the same turn.
        return await self._episodic.ingest(request)

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        if request.intent in {"procedure", "execution"}:
            return await self._structured.query(replace(request, intent="execution"))

        if request.intent == "context":
            episodic, execution = await asyncio.gather(
                self._episodic.query(request),
                self._structured.query(
                    replace(request, intent="execution", limit=min(request.limit, 6))
                ),
            )
            return _merge_results(
                episodic,
                execution,
                trace={"engine": self.DESCRIPTOR.name, "intent": "context"},
            )

        episodic = await self._episodic.query(request)
        personal = await self._search_personal(request)
        personal_records = _personal_records(personal)
        return MemoryQueryResult(
            text_block=episodic.text_block,
            records=[*personal_records, *episodic.records],
            trace={
                "engine": self.DESCRIPTOR.name,
                "intent": request.intent,
                "personal_count": len(personal_records),
                "episodic": episodic.trace,
            },
            raw={
                "personal_ids": [item.id for item in personal_records],
                "episodic": episodic.raw,
            },
        )

    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        if request.kind == "forget":
            return await self._forget(request)
        if not request.user_confirmed:
            return MemoryMutationResult(
                accepted=False,
                actual_kind=request.memory_kind or "requested",
                status="explicit_confirmation_required",
            )
        if _is_execution_write(request):
            return await self._structured.mutate(request)
        if self._personal is None:
            return MemoryMutationResult(
                accepted=False,
                actual_kind=request.memory_kind or "requested",
                status="personal_memory_unavailable",
            )
        status, record = self._personal.remember_explicit(
            request.summary,
            memory_kind=request.memory_kind,
            source_ref=request.source_ref,
            metadata=dict(request.metadata),
        )
        return MemoryMutationResult(
            accepted=record is not None,
            item_id=str(getattr(record, "id", "")),
            actual_kind=str(
                getattr(record, "data", {}).get(
                    "kind", request.memory_kind or "requested"
                )
                if record is not None
                else request.memory_kind or "requested"
            ),
            status=status,
            items=[record.to_dict()] if record is not None else [],
        )

    async def _forget(self, request: MemoryMutation) -> MemoryMutationResult:
        remaining = list(request.ids)
        affected: list[str] = []
        items: list[dict[str, object]] = []
        if self._personal is not None:
            personal_affected, remaining, personal_items = (
                self._personal.forget_explicit(tuple(remaining))
            )
            affected.extend(personal_affected)
            items.extend(personal_items)
        structured = MemoryMutationResult(accepted=False, missing_ids=remaining)
        if remaining:
            structured = await self._structured.mutate(
                replace(request, ids=tuple(remaining))
            )
            affected.extend(structured.affected_ids)
            items.extend(structured.items)
        return MemoryMutationResult(
            accepted=bool(affected),
            status="forgotten" if affected else "not_found",
            affected_ids=affected,
            missing_ids=structured.missing_ids,
            items=items,
        )

    def reinforce_items_batch(self, ids: list[str]) -> None:
        self._structured.reinforce_items_batch(ids)
        self._episodic.reinforce_items_batch(ids)

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR

    def tool_profile(self) -> MemoryToolProfile:
        structured = self._structured.tool_profile()
        episodic = self._episodic.tool_profile()
        recall_parameters = (
            structured.recall.parameters
            if structured.recall is not None
            else episodic.recall.parameters
            if episodic.recall is not None
            else {}
        )
        recall = MemoryToolSpec(
            description=(
                "统一检索小满的个人记忆与可回源的历史对话。"
                "流程和工具经验使用 memory_kind=procedure；依赖原文细节时继续调用 "
                "fetch_messages(source_ref)。"
            ),
            parameters=recall_parameters,
            search_hint="记得 以前 历史 偏好 个人信息 做过什么 工具经验",
        )
        memorize = structured.memorize
        if memorize is not None:
            memorize = replace(
                memorize,
                description=(
                    "仅在当前用户明确要求记住、纠正记忆或给出跨会话长期指令时立即写入；"
                    "普通事实和偏好披露由后台提炼，不得调用本工具。带工具要求或步骤的 "
                    "procedure 会进入独立的 Agent 执行经验库。"
                ),
            )
        forget = structured.forget
        if forget is not None:
            forget = replace(
                forget,
                description=(
                    "仅用于用户明确要求忘记、删除或清除一条记忆的隐私删除。"
                    "用户说旧偏好作废并给出新偏好属于纠错，禁止调用本工具；"
                    "该情况由回合后的后台语义批次和冲突治理自动建立新旧版本血缘。"
                ),
            )
        tools = _dedupe_tools((*structured.tools, *episodic.tools))
        return MemoryToolProfile(
            recall=recall,
            memorize=memorize,
            forget=forget,
            tools=tools,
        )

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        return self._structured.keyword_match_procedures(action_tokens)

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        return self._structured.list_events_by_time_range(
            time_start,
            time_end,
            limit=limit,
        )

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
        return self._episodic.list_items_for_dashboard(
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
        item = self._episodic.get_item_for_dashboard(
            item_id,
            include_embedding=include_embedding,
        )
        if item is not None:
            return item
        return self._structured.get_item_for_dashboard(
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
        if self._episodic.get_item_for_dashboard(item_id) is not None:
            return self._episodic.update_item_for_dashboard(
                item_id,
                status=status,
                extra_json=extra_json,
                source_ref=source_ref,
                happened_at=happened_at,
                emotional_weight=emotional_weight,
            )
        return self._structured.update_item_for_dashboard(
            item_id,
            status=status,
            extra_json=extra_json,
            source_ref=source_ref,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    def delete_item(self, item_id: str) -> bool:
        if self._episodic.get_item_for_dashboard(item_id) is not None:
            return self._episodic.delete_item(item_id)
        return self._structured.delete_item(item_id)

    def delete_items_batch(self, ids: list[str]) -> int:
        episodic_ids = [
            item_id
            for item_id in ids
            if self._episodic.get_item_for_dashboard(item_id) is not None
        ]
        structured_ids = [item_id for item_id in ids if item_id not in episodic_ids]
        return self._episodic.delete_items_batch(
            episodic_ids
        ) + self._structured.delete_items_batch(structured_ids)

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        if self._episodic.get_item_for_dashboard(item_id) is not None:
            return self._episodic.find_similar_items_for_dashboard(
                item_id,
                top_k=top_k,
                memory_type=memory_type,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
            )
        return self._structured.find_similar_items_for_dashboard(
            item_id,
            top_k=top_k,
            memory_type=memory_type,
            score_threshold=score_threshold,
            include_superseded=include_superseded,
        )

    def list_execution_memories(
        self,
        *,
        include_inactive: bool = False,
        limit: int = 500,
    ) -> list[dict[str, object]]:
        if not isinstance(self._structured, ExecutionMemoryAdminApi):
            return []
        return self._structured.list_execution_memories(
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
        if not isinstance(self._structured, ExecutionMemoryAdminApi):
            raise RuntimeError("execution memory is unavailable")
        return self._structured.record_execution_outcome(
            item_id,
            success=success,
            evidence_ref=evidence_ref,
            verified_at=verified_at,
        )

    async def query_personal_memory(
        self,
        query: str,
        *,
        limit: int = 6,
        now: datetime | None = None,
    ) -> PersonalMemoryQueryResult:
        source = self._personal
        if source is None:
            return PersonalMemoryQueryResult()
        semantic = self._personal_semantic
        if semantic is not None:
            return await semantic.search(source, query, limit=limit, now=now)
        return source.search_personal_memory(query, limit=limit, now=now)

    async def _search_personal(
        self,
        request: MemoryQuery,
    ) -> PersonalMemoryQueryResult:
        if self._personal is None:
            return PersonalMemoryQueryResult()
        kinds = {item.casefold() for item in request.filters.kinds}
        if kinds and kinds <= {"procedure", "execution"}:
            return PersonalMemoryQueryResult()
        return await self.query_personal_memory(
            request.text,
            limit=request.limit,
            now=request.timestamp,
        )


def _is_execution_write(request: MemoryMutation) -> bool:
    if request.memory_kind.strip().casefold() in {"procedure", "execution"}:
        return True
    return bool(
        request.metadata.get("tool_requirement") or request.metadata.get("steps")
    )


def _personal_records(result: PersonalMemoryQueryResult) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for hit in result.hits:
        record = hit.record
        summary = str(
            record.data.get("content") or record.summary or record.title
        ).strip()
        source_ref = record.source.source_ref
        records.append(
            MemoryRecord(
                id=record.id,
                kind=str(record.data.get("kind") or "personal"),
                summary=summary,
                score=hit.score,
                engine_kind="governed_personal",
                evidence=[
                    EvidenceRef(
                        kind="external",
                        refs=[source_ref] if source_ref else [],
                        resolver="personal",
                        source_ref=source_ref,
                        metadata={"source": record.source.source},
                    )
                ],
                signals={
                    "keyword": hit.signals.keyword,
                    "semantic": hit.signals.semantic,
                    "entity_context": hit.signals.entity_context,
                    "confidence": record.confidence,
                },
            )
        )
    return records


def _merge_results(
    *results: MemoryQueryResult,
    trace: dict[str, object],
) -> MemoryQueryResult:
    return MemoryQueryResult(
        text_block="\n\n".join(
            result.text_block.strip() for result in results if result.text_block.strip()
        ),
        records=[record for result in results for record in result.records],
        trace={
            **trace,
            "components": [result.trace for result in results],
            "hit_count": sum(len(result.records) for result in results),
        },
        raw={"components": [result.raw for result in results]},
    )


def _dedupe_tools(tools: tuple[MemoryToolSpec, ...]) -> tuple[MemoryToolSpec, ...]:
    by_name: dict[str, MemoryToolSpec] = {}
    for tool in tools:
        if tool.name:
            _ = by_name.setdefault(tool.name, tool)
    return tuple(by_name.values())
