"""Request-local embedding coordination, independent of any provider adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

EmbeddingCacheKey = tuple[type, str, str, str, int | None, str]


@dataclass
class QueryEmbeddings:
    tasks: dict[EmbeddingCacheKey, asyncio.Task[list[float]]] = field(
        default_factory=dict
    )
    closed: bool = False


query_embeddings: ContextVar[QueryEmbeddings | None] = ContextVar(
    "query_embeddings", default=None
)


@asynccontextmanager
async def query_embedding_scope() -> AsyncGenerator[None]:
    """Nested consumers share the owner; independent requests never share vectors."""
    current = query_embeddings.get()
    if current is not None and not current.closed:
        yield
        return
    scope = QueryEmbeddings()
    token = query_embeddings.set(scope)
    try:
        yield
    finally:
        scope.closed = True
        query_embeddings.reset(token)
        tasks = list(scope.tasks.values())
        scope.tasks.clear()
        for task in tasks:
            if not task.done():
                _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)
