from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class SessionUpdatePayload(BaseModel):
    metadata: dict[str, Any] | None = None
    last_consolidated: int | None = None
    last_user_at: str | None = None
    last_proactive_at: str | None = None


class SessionBatchDeletePayload(BaseModel):
    keys: list[str]
    cascade: bool = True


class SessionConsolidatePayload(BaseModel):
    archive_all: bool = False
    force: bool = True


class MessageUpdatePayload(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_chain: Any | None = None
    extra: dict[str, Any] | None = None
    ts: str | None = None


class MessageBatchDeletePayload(BaseModel):
    ids: list[str]


class MemoryUpdatePayload(BaseModel):
    status: str | None = None
    extra_json: dict[str, Any] | None = None
    source_ref: str | None = None
    happened_at: str | None = None
    emotional_weight: int | None = None


class MemoryBatchDeletePayload(BaseModel):
    ids: list[str]


class ManualConsolidator(Protocol):
    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
        force: bool = False,
    ) -> bool: ...


class ManualMemoryOptimizer(Protocol):
    @property
    def is_running(self) -> bool: ...

    async def optimize(self) -> None: ...
