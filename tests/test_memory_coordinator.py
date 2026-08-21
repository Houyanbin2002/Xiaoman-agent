from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core.memory.coordinator import CompositeMemoryEngine
from core.memory.engine import (
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryMutation,
    MemoryMutationResult,
    MemoryQuery,
    MemoryQueryResult,
    MemoryRecord,
    MemoryToolProfile,
)


pytestmark = pytest.mark.asyncio


class _Engine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.queries: list[MemoryQuery] = []
        self.mutations: list[MemoryMutation] = []
        self.ingests: list[MemoryIngestRequest] = []

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        self.queries.append(request)
        return MemoryQueryResult(
            text_block=f"{self.name}:{request.intent}",
            records=[
                MemoryRecord(
                    id=f"{self.name}-1",
                    kind=request.intent,
                    summary=self.name,
                    score=1.0,
                    engine_kind=self.name,
                )
            ],
            trace={"engine": self.name},
        )

    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        self.mutations.append(request)
        return MemoryMutationResult(
            accepted=True,
            item_id=f"{self.name}-item",
            actual_kind=request.memory_kind,
            status="stored",
            affected_ids=list(request.ids),
        )

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        self.ingests.append(request)
        return MemoryIngestResult(accepted=True, summary=self.name)

    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile()

    def reinforce_items_batch(self, ids: list[str]) -> None:
        _ = ids


class _Personal:
    def __init__(self) -> None:
        self.remembered: list[str] = []

    def remember_explicit(self, summary: str, **_: object) -> tuple[str, object]:
        self.remembered.append(summary)
        return "created", SimpleNamespace(
            id="personal-1",
            data={"kind": "requested"},
            to_dict=lambda: {"id": "personal-1"},
        )


async def test_context_combines_episodic_and_execution_without_general_duplicate() -> (
    None
):
    structured = _Engine("execution")
    episodic = _Engine("episodic")
    engine = CompositeMemoryEngine(
        structured=cast(Any, structured),
        episodic=cast(Any, episodic),
    )

    result = await engine.query(MemoryQuery(text="修复项目", intent="context"))

    assert [request.intent for request in episodic.queries] == ["context"]
    assert [request.intent for request in structured.queries] == ["execution"]
    assert result.text_block == "episodic:context\n\nexecution:execution"


async def test_explicit_personal_memory_uses_canonical_personal_source() -> None:
    structured = _Engine("execution")
    engine = CompositeMemoryEngine(
        structured=cast(Any, structured),
        episodic=cast(Any, _Engine("episodic")),
    )
    personal = _Personal()
    engine.bind_personal_memory(cast(Any, personal))

    result = await engine.mutate(
        MemoryMutation(
            kind="remember",
            summary="重要任务安排在上午",
            user_confirmed=True,
        )
    )

    assert result.item_id == "personal-1"
    assert personal.remembered == ["重要任务安排在上午"]
    assert structured.mutations == []


async def test_procedure_memory_routes_only_to_execution_store() -> None:
    structured = _Engine("execution")
    engine = CompositeMemoryEngine(
        structured=cast(Any, structured),
        episodic=cast(Any, _Engine("episodic")),
    )

    result = await engine.mutate(
        MemoryMutation(
            kind="remember",
            summary="先读取原文再回答",
            memory_kind="procedure",
            user_confirmed=True,
        )
    )

    assert result.item_id == "execution-item"
    assert len(structured.mutations) == 1


@pytest.mark.parametrize("memory_kind", ["", "procedure"])
async def test_unconfirmed_immediate_memory_write_is_rejected(
    memory_kind: str,
) -> None:
    structured = _Engine("execution")
    engine = CompositeMemoryEngine(
        structured=cast(Any, structured),
        episodic=cast(Any, _Engine("episodic")),
    )
    personal = _Personal()
    engine.bind_personal_memory(cast(Any, personal))

    result = await engine.mutate(
        MemoryMutation(
            kind="remember",
            summary="用户喜欢海底捞",
            memory_kind=memory_kind,
        )
    )

    assert result.accepted is False
    assert result.status == "explicit_confirmation_required"
    assert structured.mutations == []
    assert personal.remembered == []


async def test_conversation_ingest_only_updates_episodic_index() -> None:
    structured = _Engine("execution")
    episodic = _Engine("episodic")
    engine = CompositeMemoryEngine(
        structured=cast(Any, structured),
        episodic=cast(Any, episodic),
    )
    request = MemoryIngestRequest(content={}, source_kind="conversation_turn")

    result = await engine.ingest(request)

    assert result.summary == "episodic"
    assert episodic.ingests == [request]
    assert structured.ingests == []
