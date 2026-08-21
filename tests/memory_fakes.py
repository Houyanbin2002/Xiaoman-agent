from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from agent.memory import MemoryStore
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
    MemoryToolProfile,
)
from core.memory.markdown import ConsolidateRequest, ConsolidateResult


class FakeMemoryEngine:
    def __init__(self, workspace: Path | None = None) -> None:
        self._store = MemoryStore(workspace) if workspace is not None else None
        self.consolidate_calls: list[ConsolidateRequest] = []
        self.retrieve_result = MemoryQueryResult(text_block="")

    def describe(self) -> MemoryEngineDescriptor:
        return MemoryEngineDescriptor(
            name="fake",
            profile=EngineProfile.CLASSIC_MEMORY_SERVICE,
            capabilities=frozenset({MemoryCapability.RETRIEVE_CONTEXT_BLOCK}),
        )

    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile()

    async def query(
        self,
        request: MemoryQuery,
    ) -> MemoryQueryResult:
        return self.retrieve_result

    async def mutate(
        self,
        request: MemoryMutation,
    ) -> MemoryMutationResult:
        if request.kind == "forget":
            return MemoryMutationResult(accepted=False, missing_ids=list(request.ids))
        return MemoryMutationResult(
            accepted=True,
            item_id="mem-1",
            actual_kind=request.memory_kind,
            status="new",
        )

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        return MemoryIngestResult(accepted=True)

    async def consolidate(self, request: ConsolidateRequest) -> ConsolidateResult:
        self.consolidate_calls.append(request)
        return ConsolidateResult(trace={"mode": "markdown"})

    def read_self(self) -> str:
        return self._store.read_self() if self._store is not None else ""

    def read_history(self, max_chars: int = 0) -> str:
        if self._store is None:
            return ""
        return self._store.read_history(max_chars=max_chars)

    def read_recent_context(self) -> str:
        return self._store.read_recent_context() if self._store is not None else ""

    def write_recent_context(self, content: str) -> None:
        if self._store is not None:
            self._store.write_recent_context(content)

    def append_history(self, entry: str) -> None:
        if self._store is not None:
            self._store.append_history(entry)

    def append_history_once(
        self,
        entry: str,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool:
        if self._store is None:
            return False
        return self._store.append_history_once(
            entry,
            source_ref=source_ref,
            kind=kind,
        )

    def append_journal(
        self,
        date_str: str,
        entry: str,
        *,
        source_ref: str = "",
        kind: str = "journal",
    ) -> bool:
        if self._store is None:
            return False
        return self._store.append_journal(
            date_str,
            entry,
            source_ref=source_ref,
            kind=kind,
        )

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        return []

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        return []

    def list_items_for_dashboard(self, **kwargs: Any) -> tuple[list[dict[str, object]], int]:
        return [], 0

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        return None

    def update_item_for_dashboard(self, item_id: str, **kwargs: Any) -> dict[str, object] | None:
        return None

    def delete_item(self, item_id: str) -> bool:
        return False

    def delete_items_batch(self, ids: list[str]) -> int:
        return 0

    def find_similar_items_for_dashboard(self, item_id: str, **kwargs: Any) -> list[dict[str, object]]:
        return []
