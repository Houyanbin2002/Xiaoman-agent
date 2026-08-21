"""Periodic lifecycle maintenance for governed personal memory."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable

from core.memory.governed import GovernedLongTermMemory

logger = logging.getLogger(__name__)


class MemoryOptimizerBusy(RuntimeError):
    pass


class MemoryOptimizer:
    """Run deterministic expiry and governance maintenance on personal.db."""

    def __init__(self, canonical: GovernedLongTermMemory) -> None:
        self._canonical = canonical
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._lock.locked()

    async def optimize(self) -> None:
        if self._lock.locked():
            raise MemoryOptimizerBusy("memory optimizer 正在运行")
        async with self._lock:
            result = await asyncio.to_thread(self._canonical.optimize)
            logger.info(
                "[memory_optimizer] governed lifecycle complete: %s",
                result.to_dict(),
            )


_DEFAULT_INTERVAL_SECONDS = 64800  # 默认每 18 小时整点


class MemoryOptimizerLoop:
    def __init__(
        self,
        optimizer: MemoryOptimizer | None,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        _now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._interval = max(60, interval_seconds)
        self._now_fn = _now_fn or datetime.now
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info(
            "[memory_optimizer] 优化循环已启动，间隔=%ds (%.1fh)，对齐整点",
            self._interval,
            self._interval / 3600,
        )
        while self._running:
            secs = self._seconds_until_next_tick()
            logger.info(
                "[memory_optimizer] 距下次优化 %.0f 秒 (%.1f 小时)",
                secs,
                secs / 3600,
            )
            await asyncio.sleep(secs)
            if not self._running:
                break
            try:
                if self._optimizer:
                    await self._optimizer.optimize()
            except Exception:
                logger.exception("[memory_optimizer] 优化异常")

    def stop(self) -> None:
        self._running = False

    def _seconds_until_next_tick(self) -> float:
        """计算距下一个对齐整点的秒数。"""
        now = self._now_fn()
        now_ts = now.replace(second=0, microsecond=0).timestamp()
        next_ts = (now_ts // self._interval + 1) * self._interval
        return max(1.0, next_ts - now.timestamp())
