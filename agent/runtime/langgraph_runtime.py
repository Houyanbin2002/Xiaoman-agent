"""Shared LangGraph persistence and store lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


class LangGraphRuntime:
    """Own the process-wide checkpointer and cross-thread LangGraph Store.

    A filesystem path enables durable SQLite checkpoints. Tests and explicitly
    ephemeral agents may omit it and receive an in-memory checkpointer.
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.store: BaseStore = InMemoryStore()
        self._checkpointer: BaseCheckpointSaver[Any] | None = None
        self._sqlite_context: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
        self._lock = asyncio.Lock()

    async def checkpointer(self) -> BaseCheckpointSaver[Any]:
        if self._checkpointer is not None:
            return self._checkpointer
        async with self._lock:
            if self._checkpointer is not None:
                return self._checkpointer
            if self.checkpoint_path is None:
                self._checkpointer = InMemorySaver()
                return self._checkpointer
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            context = AsyncSqliteSaver.from_conn_string(str(self.checkpoint_path))
            saver = await context.__aenter__()
            await saver.setup()
            self._sqlite_context = context
            self._checkpointer = saver
            return saver

    async def aclose(self) -> None:
        async with self._lock:
            context = self._sqlite_context
            self._sqlite_context = None
            self._checkpointer = None
            if context is not None:
                await context.__aexit__(None, None, None)
