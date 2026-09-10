"""Opt-in configured embedding smoke test; no database writes or channel sends.

Run: python -m eval.verify_embedding
Only reports model, endpoint hostname and aggregate validation, never credentials.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

from agent.config import Config
from core.net.http import HttpRequester, SharedHttpResources
from memory2.embedder import Embedder, query_embedding_scope


async def main() -> None:
    config = Config.load("config.toml")
    embedding = config.memory.embedding
    base_url = embedding.base_url or config.light_base_url or config.base_url or ""
    key = embedding.api_key or config.light_api_key or config.api_key
    resources = SharedHttpResources()
    requests = 0

    class CountingRequester:
        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal requests
            requests += 1
            return await resources.external_default.post(*args, **kwargs)

    def client() -> Embedder:
        return Embedder(
            base_url=base_url,
            api_key=key,
            model=embedding.model,
            output_dimensionality=embedding.output_dimensionality,
            requester=cast(HttpRequester, CountingRequester()),
        )

    first, second = client(), client()
    started = time.perf_counter()
    report: dict[str, object] = {
        "model": embedding.model,
        "host": urlsplit(base_url).hostname,
    }
    try:
        async with query_embedding_scope():
            vectors = await asyncio.gather(
                first.embed("验证小满的记忆向量接口"),
                second.embed("验证小满的记忆向量接口"),
                first.embed("验证小满的记忆向量接口"),
            )
        vector = vectors[0]
        valid = (
            bool(vector)
            and all(math.isfinite(value) for value in vector)
            and any(value != 0 for value in vector)
        )
        dimensions_match = (
            embedding.output_dimensionality is None
            or len(vector) == embedding.output_dimensionality
        )
        shared = vectors[0] == vectors[1] == vectors[2] and vectors[0] is not vectors[1]
        report.update(
            ok=valid and dimensions_match and shared and requests == 1,
            dimensions=len(vector),
            configured_dimensions=embedding.output_dimensionality,
            embedding_requests=requests,
            consumers=3,
            shared_result=shared,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    except Exception as exc:
        report.update(ok=False, error_type=type(exc).__name__)
        if isinstance(exc, httpx.HTTPStatusError):
            report["http_status"] = exc.response.status_code
    finally:
        await resources.aclose()
    print(json.dumps(report, ensure_ascii=False))
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
