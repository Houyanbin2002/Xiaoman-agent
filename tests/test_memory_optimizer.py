"""Tests for governed personal-memory lifecycle scheduling."""

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from proactive_v2.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizerBusy,
    MemoryOptimizerLoop,
)


class _CanonicalMemory:
    def __init__(self) -> None:
        self.calls = 0

    def optimize(self):
        self.calls += 1
        return SimpleNamespace(to_dict=lambda: {"expired": 0})


def test_optimize_runs_governed_lifecycle() -> None:
    canonical = _CanonicalMemory()
    optimizer = MemoryOptimizer(canonical)  # type: ignore[arg-type]

    asyncio.run(optimizer.optimize())

    assert canonical.calls == 1


def test_optimize_reports_busy_instead_of_waiting() -> None:
    async def run_case() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        guarded = MemoryOptimizer(_CanonicalMemory())  # type: ignore[arg-type]

        async def blocked_run() -> None:
            async with guarded._lock:
                started.set()
                await release.wait()

        running = asyncio.create_task(blocked_run())
        await started.wait()
        assert guarded.is_running
        with pytest.raises(MemoryOptimizerBusy):
            await guarded.optimize()
        release.set()
        await running

    asyncio.run(run_case())


def test_seconds_until_next_tick_aligns_to_interval_boundary() -> None:
    now = datetime(2026, 2, 23, 12, 34, 56)
    loop = MemoryOptimizerLoop(None, interval_seconds=3600, _now_fn=lambda: now)

    secs = loop._seconds_until_next_tick()

    assert abs(secs - (25 * 60 + 4)) < 0.001


def test_seconds_until_next_tick_always_positive() -> None:
    for hour in range(24):
        now = datetime(2026, 2, 23, hour, 59, 59)
        loop = MemoryOptimizerLoop(None, interval_seconds=300, _now_fn=lambda n=now: n)
        assert loop._seconds_until_next_tick() > 0
