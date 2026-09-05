from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from memory2.embedder import Embedder, query_embedding_scope


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
