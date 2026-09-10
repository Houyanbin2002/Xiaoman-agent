"""
Embedding 客户端，对接配置指定的 OpenAI 兼容嵌入接口。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

from core.net.http import HttpRequester, RequestBudget, get_default_http_requester
from core.memory.query_embeddings import query_embeddings
from core.memory.query_embeddings import query_embedding_scope as query_embedding_scope

logger = logging.getLogger(__name__)


class Embedder:
    MAX_BATCH = 10  # DashScope 每批上限
    MAX_TEXT_LEN = 2000

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-v3",
        output_dimensionality: int | None = None,
        requester: HttpRequester | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._key = api_key
        self._model = model
        self._output_dimensionality = output_dimensionality
        self._requester = requester or get_default_http_requester("external_default")

    async def embed(self, text: str) -> list[float]:
        """Singleflight for identical effective requests; callers own their copy."""
        text = text[: self.MAX_TEXT_LEN]
        scope = query_embeddings.get()
        if scope is None or scope.closed:
            return await self._embed_one(text)
        key = (
            type(self),
            self._url,
            hashlib.sha256(self._key.encode()).hexdigest(),
            self._model,
            self._output_dimensionality,
            text,
        )
        task = scope.tasks.get(key)
        if task is None:
            # Bounded to the request. Unusual oversized workloads still work,
            # but do not keep an unbounded number of vectors alive.
            if len(scope.tasks) >= 128:
                return await self._embed_one(text)
            task = asyncio.create_task(self._embed_one(text))
            scope.tasks[key] = task

            def discard_failure(done: asyncio.Task[list[float]]) -> None:
                failed = done.cancelled() or done.exception() is not None
                if failed and scope.tasks.get(key) is done:
                    del scope.tasks[key]

            task.add_done_callback(discard_failure)
        # A timeout on one retrieval lane must not cancel the shared request.
        return list(await asyncio.shield(task))

    async def _embed_one(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """分批 embed，每批 ≤ MAX_BATCH，批间 sleep 0.3s"""
        results: list[list[float]] = []
        truncated = [t[: self.MAX_TEXT_LEN] for t in texts]

        for i in range(0, len(truncated), self.MAX_BATCH):
            batch = truncated[i : i + self.MAX_BATCH]
            payload: dict[str, object] = {"model": self._model, "input": batch}
            if self._output_dimensionality is not None:
                payload["dimensions"] = self._output_dimensionality
            resp = await self._requester.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout_s=30.0,
                budget=RequestBudget(total_timeout_s=40.0),
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            results.extend(d["embedding"] for d in data)

            if i + self.MAX_BATCH < len(truncated):
                await asyncio.sleep(0.3)

        return results

    async def aclose(self) -> None:
        return None
