from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from proactive_v2.loop import ProactiveLoop


async def test_request_tick_interrupts_proactive_fallback_wait() -> None:
    loop = ProactiveLoop.__new__(ProactiveLoop)
    loop._wake_event = asyncio.Event()

    waiting = asyncio.create_task(loop._wait_for_tick(3600))
    await asyncio.sleep(0)
    loop.request_tick()

    await asyncio.wait_for(waiting, timeout=1)
    assert loop._wake_event.is_set() is False


async def test_event_tick_uses_shared_tick_lock() -> None:
    loop = ProactiveLoop.__new__(ProactiveLoop)
    loop._tick_lock = asyncio.Lock()
    loop._tick = AsyncMock(return_value=0.6)

    assert await loop.run_event_tick() == 0.6
    loop._tick.assert_awaited_once()
