from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable

from core.attention._shared import parse_datetime
from core.attention.signals.models import AttentionSignal

logger = logging.getLogger(__name__)

SignalCollector = Callable[
    [datetime],
    list[AttentionSignal] | Awaitable[list[AttentionSignal]],
]


@dataclass(frozen=True)
class SignalProviderManifest:
    id: str
    version: int
    domains: tuple[str, ...]
    enabled: bool = True
    effective_from: str | None = None
    expires_at: str | None = None
    refresh_minutes: int = 30
    source_type: str = "builtin"

    def is_active_at(self, now: datetime) -> bool:
        if not self.enabled:
            return False
        starts = parse_datetime(self.effective_from)
        ends = parse_datetime(self.expires_at)
        current = parse_datetime(now.isoformat())
        if current is None or (starts is not None and current < starts):
            return False
        return ends is None or current <= ends


@dataclass(frozen=True)
class RegisteredSignalProvider:
    manifest: SignalProviderManifest
    collect: SignalCollector


@dataclass(frozen=True)
class SignalProviderFailure:
    provider_id: str
    error_type: str
    message: str


class SignalProviderRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, RegisteredSignalProvider] = {}
        self._last_failures: tuple[SignalProviderFailure, ...] = ()

    def register(
        self,
        manifest: SignalProviderManifest,
        collector: SignalCollector,
        *,
        replace: bool = False,
    ) -> None:
        with self._lock:
            if manifest.id in self._items and not replace:
                raise ValueError(f"signal provider already registered: {manifest.id}")
            self._items[manifest.id] = RegisteredSignalProvider(manifest, collector)

    def unregister(self, provider_id: str) -> bool:
        with self._lock:
            return self._items.pop(provider_id, None) is not None

    def list(self) -> list[RegisteredSignalProvider]:
        with self._lock:
            return sorted(self._items.values(), key=lambda item: item.manifest.id)

    @property
    def last_failures(self) -> tuple[SignalProviderFailure, ...]:
        """Failures from the latest collection pass, for diagnostics only."""
        with self._lock:
            return self._last_failures

    async def collect(self, *, now: datetime) -> list[AttentionSignal]:
        signals: list[AttentionSignal] = []
        failures: list[SignalProviderFailure] = []
        for provider in self.list():
            if not provider.manifest.is_active_at(now):
                continue
            try:
                result = provider.collect(now)
                if inspect.isawaitable(result):
                    result = await result
                signals.extend(result)
            except Exception as exc:
                failure = SignalProviderFailure(
                    provider_id=provider.manifest.id,
                    error_type=type(exc).__name__,
                    message=str(exc)[:500],
                )
                failures.append(failure)
                logger.warning(
                    "attention signal provider %s failed: %s",
                    provider.manifest.id,
                    exc,
                )
        with self._lock:
            self._last_failures = tuple(failures)
        return signals


__all__ = [
    "RegisteredSignalProvider",
    "SignalCollector",
    "SignalProviderManifest",
    "SignalProviderFailure",
    "SignalProviderRegistry",
]
