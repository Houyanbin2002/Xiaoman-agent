from __future__ import annotations

import asyncio
from typing import Any, cast
from types import SimpleNamespace

import pytest

from memory2.embedder import Embedder, query_embedding_scope
from agent.looping.ports import MemoryServices
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import RetrievalRequest
from core.memory.engine import MemoryQueryResult
from core.memory.personal_retrieval import PersonalMemoryQueryResult

pytestmark = pytest.mark.asyncio


class _CountingEmbedder(Embedder):
    calls = 0

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        type(self).calls += 1
        await asyncio.sleep(0.02)
        return [[float(len(text)), 1.0] for text in texts]


def _embedder() -> _CountingEmbedder:
    return _CountingEmbedder(
        base_url="https://example.invalid/v1",
        api_key="shared-key",
        model="shared-model",
        output_dimensionality=2,
        requester=cast(Any, object()),
    )


async def test_query_embedding_scope_singleflights_across_embedder_instances() -> None:
    _CountingEmbedder.calls = 0
    first = _embedder()
    second = _embedder()

    async with query_embedding_scope():
        vectors = await asyncio.gather(
            first.embed("同一个查询"),
            second.embed("同一个查询"),
            first.embed("同一个查询"),
        )

    assert _CountingEmbedder.calls == 1
    assert vectors == [[5.0, 1.0], [5.0, 1.0], [5.0, 1.0]]
    assert vectors[0] is not vectors[1]


async def test_scope_does_not_cache_across_requests_and_nested_scope_reuses() -> None:
    _CountingEmbedder.calls = 0
    client = _embedder()
    async with query_embedding_scope():
        first = await client.embed("查询")
        first[0] = -99
        async with query_embedding_scope():
            assert await client.embed("查询") == [2.0, 1.0]
        assert await client.embed("查询") == [2.0, 1.0]
    async with query_embedding_scope():
        await client.embed("查询")
    await client.embed("查询")
    assert _CountingEmbedder.calls == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("_model", "other"),
        ("_key", "other"),
        ("_url", "https://other.invalid/embeddings"),
        ("_output_dimensionality", 3),
    ],
)
async def test_incompatible_embedding_requests_never_share(field, value) -> None:
    _CountingEmbedder.calls = 0
    first, second = _embedder(), _embedder()
    setattr(second, field, value)
    async with query_embedding_scope():
        await asyncio.gather(first.embed("查询"), second.embed("查询"))
    assert _CountingEmbedder.calls == 2


async def test_failed_request_can_retry_in_same_scope() -> None:
    class Flaky(_CountingEmbedder):
        attempts = 0

        async def embed_batch(self, texts):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary network failure")
            return [[1.0]]

    client = Flaky("https://example.invalid", "key", requester=cast(Any, object()))
    async with query_embedding_scope():
        with pytest.raises(RuntimeError):
            await client.embed("查询")
        assert await client.embed("查询") == [1.0]
    assert client.attempts == 2


async def test_scope_exit_cancels_orphan_network_request() -> None:
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class Slow(_CountingEmbedder):
        async def embed_batch(self, texts):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return [[0.0]]

    client = Slow("https://example.invalid", "key", requester=cast(Any, object()))
    async with query_embedding_scope():
        waiter = asyncio.create_task(client.embed("查询"))
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not cancelled.is_set()
    assert cancelled.is_set()


async def test_concurrent_scopes_are_isolated() -> None:
    _CountingEmbedder.calls = 0

    async def request():
        async with query_embedding_scope():
            await asyncio.gather(_embedder().embed("查询"), _embedder().embed("查询"))

    await asyncio.gather(request(), request())
    assert _CountingEmbedder.calls == 2


async def test_singleflight_survives_one_waiter_timeout() -> None:
    _CountingEmbedder.calls = 0
    first = _embedder()
    second = _embedder()

    async with query_embedding_scope():
        timed = asyncio.create_task(first.embed("共享查询"))
        await asyncio.sleep(0)
        survivor = asyncio.create_task(second.embed("共享查询"))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(timed, timeout=0.001)
        assert await survivor == [4.0, 1.0]

    assert _CountingEmbedder.calls == 1


async def test_production_pipeline_shares_all_lanes_without_manual_scope() -> None:
    _CountingEmbedder.calls = 0

    async def engine_query(request):
        await asyncio.gather(
            _embedder().embed(request.text), _embedder().embed(request.text)
        )
        return MemoryQueryResult()

    async def personal_query(text, **kwargs):
        await _embedder().embed(text)
        return PersonalMemoryQueryResult()

    services = MemoryServices(
        engine=cast(Any, SimpleNamespace(query=engine_query)),
        runtime=cast(
            Any, SimpleNamespace(retrieve_personal_memory_async=personal_query)
        ),
    )
    pipeline = DefaultMemoryRetrievalPipeline(services)
    request = RetrievalRequest(
        message="共享查询",
        session_key="test:1",
        channel="test",
        chat_id="1",
        history=[],
        session_metadata={},
    )
    await pipeline.retrieve(request)
    assert _CountingEmbedder.calls == 1
    await pipeline.retrieve(request)
    assert _CountingEmbedder.calls == 2
