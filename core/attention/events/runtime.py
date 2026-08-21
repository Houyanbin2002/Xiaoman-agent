from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

from core.attention._shared import parse_datetime
from core.attention.events.service import EventDrivenAttentionService
from core.attention.ports import AttentionRepository

logger = logging.getLogger(__name__)

TickCallback = Callable[[], Awaitable[float | None] | float | None]


class AttentionWakeRuntime:
    """Wake the existing proactive decision loop only when durable work is due."""

    def __init__(
        self,
        *,
        repository: AttentionRepository,
        events: EventDrivenAttentionService,
        tick: TickCallback | None = None,
        now_fn: Callable[[], datetime] | None = None,
        fallback_seconds: int = 3600,
    ) -> None:
        self._repository = repository
        self._events = events
        self._tick = tick
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._fallback_seconds = max(60, int(fallback_seconds))
        self._changed = asyncio.Event()
        self._running = False
        self._events.bind_wake_notifier(self.notify_changed)

    def bind_tick(self, tick: TickCallback | None) -> None:
        self._tick = tick
        self.notify_changed()

    def notify_changed(self) -> None:
        self._changed.set()

    async def run(self) -> None:
        self._repository.recover_processing_wakes()
        self._running = True
        while self._running:
            now = self._now().astimezone(timezone.utc)
            if await self.run_due_once(now=now):
                continue
            timeout = self._wait_seconds(now)
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=timeout)
            except TimeoutError:
                pass
            self._changed.clear()

    def stop(self) -> None:
        self._running = False
        self.notify_changed()

    async def run_due_once(self, *, now: datetime | None = None) -> int:
        current = (now or self._now()).astimezone(timezone.utc)
        wakes = self._repository.claim_due_wakes(now=current, limit=20)
        for wake in wakes:
            signal = self._events.activate_wake(wake, now=current)
            if signal is None:
                self._repository.complete_wake(wake.id, decision="inactive")
                continue
            if self._tick is None:
                self._defer(wake.id, current, wake.attempt, "runtime_unbound")
                continue
            try:
                result = self._tick()
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                logger.exception("event-driven attention tick failed: %s", exc)
                self._defer(wake.id, current, wake.attempt, "tick_error")
                continue
            if result is None:
                self._defer(wake.id, current, wake.attempt, "no_action")
            else:
                self._repository.complete_wake(wake.id, decision="evaluated")
        return len(wakes)

    def _defer(
        self,
        wake_id: str,
        now: datetime,
        attempt: int,
        decision: str,
    ) -> None:
        delay_minutes = min(120, 15 * (2 ** max(0, attempt - 1)))
        self._repository.defer_wake(
            wake_id,
            wake_at=now + timedelta(minutes=delay_minutes),
            decision=decision,
        )

    def _wait_seconds(self, now: datetime) -> float:
        next_at = parse_datetime(self._repository.next_wake_at())
        if next_at is None:
            return float(self._fallback_seconds)
        return max(
            0.05,
            min(float(self._fallback_seconds), (next_at - now).total_seconds()),
        )


__all__ = ["AttentionWakeRuntime"]
