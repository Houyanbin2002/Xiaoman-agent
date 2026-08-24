"""记忆内容哈希去重测试。"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from memory2.memorizer import Memorizer
from memory2.store import MemoryStore2


class _FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return list(self._mapping.get(text, [0.0, 0.0, 0.0]))


def test_exact_hash_prevents_double_write(tmp_path):
    """完全相同的 summary 写两次时只保留一条并强化。"""
    store = MemoryStore2(tmp_path / "m.db")
    embedder = _FakeEmbedder({"查 Steam 必须用 steam MCP": [1.0, 0.0]})
    memorizer = Memorizer(store, cast(Any, embedder))

    async def _run():
        await memorizer.save_item(
            summary="查 Steam 必须用 steam MCP",
            memory_type="procedure",
            extra={},
            source_ref="turn1",
        )
        await memorizer.save_item(
            summary="查 Steam 必须用 steam MCP",
            memory_type="procedure",
            extra={},
            source_ref="turn2",
        )

    asyncio.run(_run())

    items = store.list_by_type("procedure")
    assert len(items) == 1, "完全相同内容不应重复写入"
    assert items[0]["reinforcement"] == 2, "重复写入应增加 reinforcement"
