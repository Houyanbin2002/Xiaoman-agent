from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent.looping.ports import MemoryServices
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import RetrievalRequest
from core.memory.personal_core import PersonalCoreMemorySelector
from core.memory.engine import MemoryQueryResult
from core.memory.personal_retrieval import (
    GovernedPersonalMemoryRetriever,
    PersonalMemoryHit,
    PersonalMemoryQueryResult,
    PersonalRecallSignals,
)
from core.memory.personal_semantic import PersonalSemanticRecallService
from core.personal.models import (
    AccessPolicy,
    DataCategory,
    MemoryKind,
    PersonalEntityType,
    PersonalRecord,
    RecordSource,
    RecordStatus,
    SensitivityLevel,
)
from infra.persistence.personal_memory_vector_store import PersonalMemoryVectorStore

NOW = datetime(2026, 7, 14, tzinfo=timezone.utc)


def _record(
    item_id: str,
    content: str,
    kind: MemoryKind,
    *,
    confidence: float = 0.8,
    locked: bool = False,
    updated_days_ago: int = 0,
    expires_at: datetime | None = None,
    access_policy: AccessPolicy = AccessPolicy.STANDARD,
) -> PersonalRecord:
    updated = NOW - timedelta(days=updated_days_ago)
    return PersonalRecord(
        id=item_id,
        entity_type=PersonalEntityType.MEMORY,
        record_key=f"memory:{item_id}",
        title=content,
        summary=content,
        data={"kind": kind.value, "content": content, "tags": []},
        source=RecordSource("user", f"turn:{item_id}"),
        confidence=confidence,
        sensitivity=SensitivityLevel.PERSONAL,
        data_category=DataCategory.GENERAL,
        access_policy=access_policy,
        status=RecordStatus.ACTIVE,
        valid_from=None,
        expires_at=expires_at.isoformat() if expires_at else None,
        last_confirmed_at=NOW.isoformat() if locked else None,
        user_locked=locked,
        allow_auto_update=not locked,
        supersedes_id=None,
        revision=1,
        created_at=updated.isoformat(),
        updated_at=updated.isoformat(),
    )


def test_core_selector_keeps_stable_personal_memory_and_excludes_other_layers() -> None:
    records = [
        _record("locked", "不喜欢临时改变计划", MemoryKind.REQUESTED, locked=True),
        _record("fact", "重要任务尽量安排在上午", MemoryKind.PREFERENCE),
        _record("event", "上周完成了项目迁移", MemoryKind.HISTORICAL_EVENT),
        _record("temp", "最近有些疲惫", MemoryKind.TEMPORARY_STATE),
        _record("procedure", "启动项目前运行迁移", MemoryKind.PROCEDURE),
    ]

    selection = PersonalCoreMemorySelector(max_chars=600).select(records, now=NOW)

    assert [record.id for record in selection.records] == ["locked", "fact"]


def test_core_selector_respects_budget_without_deleting_full_records() -> None:
    records = [
        _record(
            f"memory-{index}",
            f"用户稳定偏好 {index} " + "内容" * 80,
            MemoryKind.PREFERENCE,
        )
        for index in range(10)
    ]

    selection = PersonalCoreMemorySelector(max_chars=600).select(records, now=NOW)

    assert 0 < len(selection.records) < len(records)
    assert selection.dropped_count > 0


def test_personal_retrieval_does_not_decay_stable_preference() -> None:
    old_preference = _record(
        "preference",
        "重要任务尽量安排在上午",
        MemoryKind.PREFERENCE,
        updated_days_ago=600,
    )
    retriever = GovernedPersonalMemoryRetriever()

    hits = retriever.retrieve(
        [old_preference],
        query="上午安排重要任务",
        now=NOW,
    )

    assert hits
    assert hits[0].record.id == "preference"
    assert hits[0].score > 0.2


def test_personal_retrieval_excludes_expired_restricted_and_procedure_records() -> None:
    records = [
        _record(
            "expired",
            "最近正在准备考试",
            MemoryKind.TEMPORARY_STATE,
            expires_at=NOW - timedelta(hours=1),
        ),
        _record(
            "restricted",
            "用户的私密账号信息",
            MemoryKind.FACT,
            access_policy=AccessPolicy.CONFIRM_READ,
        ),
        _record(
            "procedure",
            "操作 Notion 前刷新 token",
            MemoryKind.PROCEDURE,
        ),
    ]

    hits = GovernedPersonalMemoryRetriever().retrieve(
        records,
        query="Notion 考试 账号",
        semantic_scores={item.id: 1.0 for item in records},
        now=NOW,
    )

    assert hits == []


async def test_default_pipeline_combines_governed_personal_and_engine_memory() -> None:
    record = _record("personal", "用户上午适合重要任务", MemoryKind.PREFERENCE)

    class _Engine:
        async def query(self, request):
            return MemoryQueryResult(text_block="执行记忆块")

    async def retrieve_personal_memory_async(query, limit=4):
        return PersonalMemoryQueryResult(
            text_block="个人记忆块",
            hits=(
                PersonalMemoryHit(
                    record=record,
                    score=0.8,
                    signals=PersonalRecallSignals(keyword=1.0),
                ),
            ),
        )

    runtime = SimpleNamespace(
        retrieve_personal_memory_async=retrieve_personal_memory_async,
    )
    pipeline = DefaultMemoryRetrievalPipeline(
        MemoryServices(engine=_Engine(), runtime=runtime),  # type: ignore[arg-type]
    )

    result = await pipeline.retrieve(
        RetrievalRequest(
            message="上午安排什么",
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            history=[],
            session_metadata={},
        )
    )

    assert result.block == "个人记忆块\n\n执行记忆块"
    assert result.trace is not None
    assert result.trace.injected_count == 1
    assert result.metadata["personal_memory_count"] == 1


async def test_default_pipeline_degrades_slow_engine_without_blocking_personal_memory() -> None:
    record = _record("personal", "用户喜欢直接说明结论", MemoryKind.PREFERENCE)

    class _SlowEngine:
        async def query(self, request):
            await asyncio.sleep(1)
            return MemoryQueryResult(text_block="不应等到这里")

    async def retrieve_personal_memory_async(query, limit=4):
        return PersonalMemoryQueryResult(
            text_block="个人记忆块",
            hits=(
                PersonalMemoryHit(
                    record=record,
                    score=0.8,
                    signals=PersonalRecallSignals(keyword=1.0),
                ),
            ),
        )

    pipeline = DefaultMemoryRetrievalPipeline(
        MemoryServices(
            engine=_SlowEngine(),
            runtime=SimpleNamespace(
                retrieve_personal_memory_async=retrieve_personal_memory_async,
            ),
        ),  # type: ignore[arg-type]
        source_timeout_s=0.02,
    )

    started = time.perf_counter()
    result = await pipeline.retrieve(
        RetrievalRequest(
            message="你好",
            session_key="dashboard:1",
            channel="dashboard",
            chat_id="1",
            history=[],
            session_metadata={},
        )
    )

    assert time.perf_counter() - started < 0.2
    assert result.block == "个人记忆块"
    assert result.metadata["personal_memory_count"] == 1


async def test_personal_semantic_recall_timeout_falls_back_to_keyword_ranking(
    tmp_path,
) -> None:
    record = _record("focus", "重要任务尽量安排在上午", MemoryKind.PREFERENCE)

    class _SlowEmbedder:
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            await asyncio.sleep(1)
            return [[1.0, 0.0] for _ in texts]

        async def embed(self, text: str) -> list[float]:
            await asyncio.sleep(1)
            return [1.0, 0.0]

    class _Source:
        def personal_memory_records(self):
            return [record]

        def search_personal_memory(
            self,
            query: str,
            *,
            limit: int = 6,
            context_tags=None,
            now=None,
            semantic_scores=None,
        ) -> PersonalMemoryQueryResult:
            hits = GovernedPersonalMemoryRetriever().retrieve(
                [record],
                query=query,
                semantic_scores=semantic_scores,
                limit=limit,
                now=now or NOW,
            )
            return PersonalMemoryQueryResult(hits=tuple(hits))

    service = PersonalSemanticRecallService(
        store=PersonalMemoryVectorStore(tmp_path / "personal.db"),
        embedder=_SlowEmbedder(),
        model="slow-embedding",
        foreground_timeout_s=0.02,
    )
    try:
        started = time.perf_counter()
        result = await service.search(_Source(), "上午重要任务", now=NOW)
    finally:
        service.close()

    assert time.perf_counter() - started < 0.2
    assert result.hits[0].record.id == "focus"
    assert result.hits[0].signals.semantic == 0.0


async def test_personal_semantic_sidecar_persists_vectors_and_feeds_hybrid_rank(
    tmp_path,
) -> None:
    records = [
        _record("focus", "重要任务尽量安排在上午", MemoryKind.PREFERENCE),
        _record("food", "用户喜欢清淡饮食", MemoryKind.PREFERENCE),
    ]

    class _Embedder:
        batch_calls = 0

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            self.batch_calls += 1
            return [self._vector(text) for text in texts]

        async def embed(self, text: str) -> list[float]:
            return self._vector(text)

        @staticmethod
        def _vector(text: str) -> list[float]:
            return [1.0, 0.0] if "上午" in text or "专注" in text else [0.0, 1.0]

    class _Source:
        def personal_memory_records(self) -> list[PersonalRecord]:
            return records

        def search_personal_memory(
            self,
            query: str,
            *,
            limit: int = 6,
            context_tags=None,
            now=None,
            semantic_scores=None,
        ) -> PersonalMemoryQueryResult:
            hits = GovernedPersonalMemoryRetriever().retrieve(
                records,
                query=query,
                semantic_scores=semantic_scores,
                limit=limit,
                now=now or NOW,
            )
            return PersonalMemoryQueryResult(hits=tuple(hits))

    embedder = _Embedder()
    service = PersonalSemanticRecallService(
        store=PersonalMemoryVectorStore(tmp_path / "personal.db"),
        embedder=embedder,
        model="test-embedding",
    )
    try:
        first = await service.search(_Source(), "适合专注的时间", now=NOW)
        second = await service.search(_Source(), "专注安排", now=NOW)
    finally:
        service.close()

    assert first.hits[0].record.id == "focus"
    assert first.hits[0].signals.semantic > 0.9
    assert second.hits[0].record.id == "focus"
    assert embedder.batch_calls == 1
