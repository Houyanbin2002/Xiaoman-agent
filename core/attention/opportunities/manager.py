from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Iterable

from core.attention._shared import parse_datetime, positive_int, utc_iso
from core.attention.opportunities.models import (
    OpportunityKind,
    OpportunityStatus,
    OpportunityWindow,
)
from core.attention.patterns import BehaviorPattern
from core.attention.signals import AttentionSignal


class OpportunityManager:
    """Materialize current opportunities from patterns and live signals."""

    def is_signal_actionable_now(
        self,
        signal: AttentionSignal,
        now: datetime,
    ) -> bool:
        return self._signal_interval(
            signal,
            now.astimezone(timezone.utc),
        ) is not None

    def materialize(
        self,
        *,
        patterns: Iterable[BehaviorPattern],
        signals: Iterable[AttentionSignal],
        now: datetime,
        scene: str,
    ) -> list[OpportunityWindow]:
        current = now.astimezone(timezone.utc)
        signal_items = list(signals)
        eligible_signal_ids = tuple(
            signal.id
            for signal in signal_items
            if self._signal_interval(signal, current) is not None
        )
        windows: list[OpportunityWindow] = []
        for pattern in patterns:
            interval = pattern.interval_containing(current)
            if interval is None:
                continue
            starts, ends = interval
            windows.append(
                OpportunityWindow(
                    id=self._stable_id("pattern", pattern.id, utc_iso(starts)),
                    kind=OpportunityKind.RECURRING,
                    scene=pattern.scene,
                    available_from=utc_iso(starts),
                    available_until=utc_iso(ends),
                    available_minutes=pattern.available_minutes,
                    confidence=pattern.confidence,
                    status=OpportunityStatus.ACTIVE,
                    source_pattern_id=pattern.id,
                    signal_ids=eligible_signal_ids,
                    metadata={"pattern_kind": pattern.kind, **pattern.metadata},
                )
            )
        for signal in signal_items:
            if not signal.is_active_at(current) or signal.actionability <= 0:
                continue
            interval = self._signal_interval(signal, current)
            if interval is None:
                continue
            starts, ends, config = interval
            windows.append(
                OpportunityWindow(
                    id=self._stable_id("signal", signal.id, utc_iso(starts)),
                    kind=OpportunityKind.EVENT,
                    scene=str(config.get("scene") or scene or "neutral"),
                    available_from=utc_iso(starts),
                    available_until=utc_iso(ends),
                    available_minutes=positive_int(
                        config.get("available_minutes"),
                        signal.estimated_attention_minutes,
                    ),
                    confidence=signal.confidence,
                    status=OpportunityStatus.ACTIVE,
                    signal_ids=(signal.id,),
                    validation={"signal_active": True},
                    metadata={"signal_kind": signal.kind},
                )
            )
        return self._dedupe(windows)

    @staticmethod
    def _signal_interval(
        signal: AttentionSignal,
        current: datetime,
    ) -> tuple[datetime, datetime, dict[str, object]] | None:
        """Return the declared actionable interval only when it is open now.

        A signal may exist before it is appropriate to contact the user.  In
        particular, conversation appraisal deliberately creates delayed
        follow-up signals.  Recurring patterns must not make those signals
        actionable early.
        """
        if not signal.is_active_at(current) or signal.actionability <= 0:
            return None
        raw = signal.metadata.get("opportunity")
        config: dict[str, object] = dict(raw) if isinstance(raw, dict) else {}
        starts = parse_datetime(config.get("starts_at")) or parse_datetime(
            signal.occurred_at
        )
        if starts is None:
            return None
        duration = positive_int(
            config.get("duration_minutes"),
            max(30, signal.estimated_attention_minutes * 3),
            maximum=7 * 24 * 60,
        )
        ends = parse_datetime(config.get("ends_at")) or (
            starts + timedelta(minutes=duration)
        )
        signal_expires = parse_datetime(signal.expires_at)
        if signal_expires is not None:
            ends = min(ends, signal_expires)
        if not starts <= current <= ends:
            return None
        return starts, ends, config

    @staticmethod
    def _stable_id(*parts: str) -> str:
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
        return f"win_{digest}"

    @staticmethod
    def _dedupe(windows: list[OpportunityWindow]) -> list[OpportunityWindow]:
        return list({window.id: window for window in windows}.values())


__all__ = ["OpportunityManager"]
