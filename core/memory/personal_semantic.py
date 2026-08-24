from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import logging
from typing import Protocol

from core.memory.personal_retrieval import PersonalMemoryQueryResult
from core.personal.models import PersonalRecord

logger = logging.getLogger(__name__)

_DEFAULT_FOREGROUND_TIMEOUT_S = 1.5


class PersonalEmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class PersonalMemoryVectorStorePort(Protocol):
    def stale_ids(
        self,
        hashes: Mapping[str, str],
        *,
        model: str,
    ) -> list[str]: ...

    def upsert_many(
        self,
        rows: Sequence[tuple[str, str, str, Sequence[float]]],
    ) -> None: ...

    def delete_except(self, record_ids: set[str]) -> None: ...

    def semantic_scores(
        self,
        query: Sequence[float],
        *,
        record_ids: set[str],
        model: str,
    ) -> dict[str, float]: ...

    def close(self) -> None: ...


class PersonalSemanticSource(Protocol):
    def personal_memory_records(self) -> list[PersonalRecord]: ...

    def search_personal_memory(
        self,
        query: str,
        *,
        limit: int = 6,
        context_tags: set[str] | None = None,
        now: datetime | None = None,
        semantic_scores: Mapping[str, float] | None = None,
    ) -> PersonalMemoryQueryResult: ...


class PersonalSemanticRecallService:
    """Keep personal vectors current and feed semantic scores into governance ranking."""

    def __init__(
        self,
        *,
        store: PersonalMemoryVectorStorePort,
        embedder: PersonalEmbeddingProvider,
        model: str,
        foreground_timeout_s: float = _DEFAULT_FOREGROUND_TIMEOUT_S,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._model = model.strip() or "default"
        self._foreground_timeout_s = max(0.05, float(foreground_timeout_s))

    async def search(
        self,
        source: PersonalSemanticSource,
        query: str,
        *,
        limit: int = 6,
        now: datetime | None = None,
    ) -> PersonalMemoryQueryResult:
        records = source.personal_memory_records()
        if not records or not query.strip():
            return PersonalMemoryQueryResult()
        semantic_scores: dict[str, float] = {}
        try:
            semantic_scores = await asyncio.wait_for(
                self._semantic_scores(records, query),
                timeout=self._foreground_timeout_s,
            )
        except Exception as exc:
            # Embedding outages must not disable personal memory; deterministic
            # keyword, governance and type-aware ranking remain available.
            logger.warning(
                "personal semantic recall degraded to keyword ranking: %s", exc
            )
        return source.search_personal_memory(
            query,
            limit=limit,
            now=now,
            semantic_scores=semantic_scores,
        )

    async def _semantic_scores(
        self,
        records: list[PersonalRecord],
        query: str,
    ) -> dict[str, float]:
        hashes = {record.id: _content_hash(record) for record in records}
        by_id = {record.id: record for record in records}
        stale_ids = self._store.stale_ids(hashes, model=self._model)
        if stale_ids:
            texts = [_record_text(by_id[item_id]) for item_id in stale_ids]
            embeddings = await self._embedder.embed_batch(texts)
            self._store.upsert_many(
                [
                    (item_id, hashes[item_id], self._model, embedding)
                    for item_id, embedding in zip(
                        stale_ids,
                        embeddings,
                        strict=False,
                    )
                ]
            )
        record_ids = set(by_id)
        self._store.delete_except(record_ids)
        query_embedding = await self._embedder.embed(query.strip())
        return self._store.semantic_scores(
            query_embedding,
            record_ids=record_ids,
            model=self._model,
        )

    def reconfigure(
        self,
        *,
        embedder: PersonalEmbeddingProvider,
        model: str,
    ) -> None:
        self._embedder = embedder
        self._model = model.strip() or "default"

    def close(self) -> None:
        self._store.close()


def _record_text(record: PersonalRecord) -> str:
    content = str(record.data.get("content") or record.summary or record.title).strip()
    tags = " ".join(
        str(item) for item in record.data.get("tags", []) if str(item).strip()
    )
    return "\n".join(part for part in (record.title, content, tags) if part.strip())


def _content_hash(record: PersonalRecord) -> str:
    return hashlib.sha256(_record_text(record).encode("utf-8")).hexdigest()
