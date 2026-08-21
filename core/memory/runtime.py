from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

from core.memory.profile import CanonicalLongTermMemory, UnifiedMemoryProfile
from core.memory.personal_retrieval import (
    PersonalMemoryQueryResult,
    PersonalMemoryRetrievalApi,
)

if TYPE_CHECKING:
    from core.memory.engine import (
        MemoryEngine,
        MemoryMutation,
        MemoryMutationResult,
        MemoryQuery,
        MemoryQueryResult,
    )
    from core.memory.markdown import MarkdownMemoryRuntime

logger = logging.getLogger(__name__)


class _AsyncCloseable(Protocol):
    def aclose(self) -> object: ...


class _Closeable(Protocol):
    def close(self) -> object: ...


@dataclass
class MemoryRuntime:
    markdown: "MarkdownMemoryRuntime"
    engine: "MemoryEngine"
    closeables: list[object] = field(default_factory=list[object])
    profile: UnifiedMemoryProfile = field(init=False)
    long_term: CanonicalLongTermMemory | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.profile = UnifiedMemoryProfile(self.markdown.store)

    def bind_canonical_long_term_memory(
        self,
        source: CanonicalLongTermMemory,
    ) -> None:
        self.long_term = source
        self.profile.bind_canonical(source)
        binder = getattr(self.engine, "bind_personal_memory", None)
        if callable(binder):
            binder(source)

    def read_long_term(self) -> str:
        return self.profile.read_long_term()

    def read_self(self) -> str:
        return self.profile.read_self()

    def read_recent_context(self) -> str:
        return self.profile.read_recent_context()

    def get_memory_context(self) -> str:
        return self.profile.get_memory_context()

    def retrieve_personal_memory(
        self,
        query: str,
        *,
        limit: int = 6,
    ) -> PersonalMemoryQueryResult:
        source = self.long_term
        if not isinstance(source, PersonalMemoryRetrievalApi):
            return PersonalMemoryQueryResult()
        return source.retrieve_personal_memory(query, limit=limit)

    async def retrieve_personal_memory_async(
        self,
        query: str,
        *,
        limit: int = 6,
    ) -> PersonalMemoryQueryResult:
        query_personal = getattr(self.engine, "query_personal_memory", None)
        if callable(query_personal):
            result = query_personal(query, limit=limit)
            if inspect.isawaitable(result):
                return await result
        return self.retrieve_personal_memory(query, limit=limit)

    async def query(
        self,
        request: "MemoryQuery",
    ) -> "MemoryQueryResult":
        return await self.engine.query(request)

    async def mutate(
        self,
        request: "MemoryMutation",
    ) -> "MemoryMutationResult":
        return await self.engine.mutate(request)

    async def aclose(self) -> None:
        first_error: Exception | None = None
        for closeable in reversed(self.closeables):
            try:
                if hasattr(closeable, "aclose"):
                    result = cast(_AsyncCloseable, closeable).aclose()
                    if inspect.isawaitable(result):
                        await result
                elif hasattr(closeable, "close"):
                    _ = cast(_Closeable, closeable).close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.warning(
                    "memory runtime close failed for %s: %s",
                    type(closeable).__name__,
                    exc,
                )
        if first_error is not None:
            raise first_error
