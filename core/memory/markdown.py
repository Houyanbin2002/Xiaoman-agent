from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent.memory import MemoryStore

@dataclass(frozen=True)
class ConsolidateRequest:
    session: object
    archive_all: bool = False
    force: bool = False


@dataclass
class ConsolidateResult:
    consolidated_count: int = 0
    trace: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class MemoryProfileApi(Protocol):
    def read_long_term(self) -> str: ...

    def read_self(self) -> str: ...

    def read_recent_context(self) -> str: ...

    def get_memory_context(self) -> str: ...


class MarkdownMemoryStore(MemoryStore):
    pass


class MarkdownMemoryMaintenance:
    """Manual consolidation adapter for the shared semantic batcher.

    Conversation analysis no longer belongs to the markdown store. Manual
    consolidation requests are forwarded to the durable semantic batcher;
    markdown only exposes the support files used by prompt and dashboard code.
    """

    def __init__(self) -> None:
        self._semantic_flush: Callable[..., Awaitable[None]] | None = None

    def bind_semantic_flush(
        self,
        flush: Callable[..., Awaitable[None]],
    ) -> None:
        self._semantic_flush = flush

    async def consolidate(self, request: ConsolidateRequest) -> ConsolidateResult:
        session_key = str(getattr(request.session, "key", "") or "")
        if self._semantic_flush is None or not session_key:
            return ConsolidateResult(trace={"mode": "skipped"})
        reason = "archive" if request.archive_all else "manual"
        await self._semantic_flush(session_key, reason=reason)
        return ConsolidateResult(trace={"mode": "semantic_batch"})


@dataclass
class MarkdownMemoryRuntime:
    store: MarkdownMemoryStore
    maintenance: MarkdownMemoryMaintenance


def build_markdown_memory_runtime(
    *,
    workspace: Path,
) -> MarkdownMemoryRuntime:
    store = MarkdownMemoryStore(workspace)
    maintenance = MarkdownMemoryMaintenance()
    return MarkdownMemoryRuntime(store=store, maintenance=maintenance)
