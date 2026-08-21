from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from core.attention.actions import ActionPlanStatus
from core.attention.engine import AttentionEngine
from core.attention.feedback.service import FeedbackService
from core.attention.policies import DecisionContext
from core.attention.providers import McpAlertSignalAdapter
from core.personal.models import PersonalEntityType, PersonalRecord
from core.personal.rhythm import PersonalRhythmService
from core.personal.service import PersonalDataService

_SNAPSHOT_LIMITS: tuple[tuple[PersonalEntityType, str, int], ...] = (
    (PersonalEntityType.COMMITMENT, "active_commitments", 8),
    (PersonalEntityType.CALENDAR_EVENT, "upcoming_events", 5),
    (PersonalEntityType.DAILY_PLAN, "daily_plans", 3),
    (PersonalEntityType.CHECK_IN, "recent_check_ins", 5),
    (PersonalEntityType.HEALTH_OBSERVATION, "recent_health", 5),
    (PersonalEntityType.GOAL, "active_goals", 5),
)
logger = logging.getLogger(__name__)


class PersonalAttentionSource:
    """Bridge the generic attention engine to the proactive delivery gateway."""

    ack_server = "attention"

    def __init__(
        self,
        *,
        personal_data: PersonalDataService,
        rhythm: PersonalRhythmService,
        engine: AttentionEngine,
        feedback: FeedbackService | None = None,
    ) -> None:
        self.personal_data = personal_data
        self.rhythm = rhythm
        self.engine = engine
        self.feedback = feedback

    async def alert_fn(
        self,
        *,
        now: datetime | None = None,
        external_alerts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        await self.engine.refresh(now=current)
        for event in external_alerts or []:
            self.engine.ingest_signal(McpAlertSignalAdapter.convert(event, now=current))
        snapshot = self.rhythm.snapshot(now=current)
        decision_attributes = (
            self.feedback.decision_attributes(now=current)
            if self.feedback is not None
            else {}
        )
        evaluation = self.engine.evaluate(
            context=DecisionContext(
                now=current,
                scene=snapshot.scene.value,
                focus_active=snapshot.focus_active,
                do_not_disturb=snapshot.do_not_disturb,
                allow_high_priority=snapshot.allow_high_priority,
                channel="proactive",
                permission_mode="delegated",
                attributes=decision_attributes,
            )
        )
        plan = evaluation.plan
        if (
            plan is None
            or plan.status not in {ActionPlanStatus.PROPOSED, ActionPlanStatus.APPROVED}
            or plan.capability_id != "message.notify"
        ):
            return []
        signals = [
            signal
            for signal_id in plan.signal_ids
            if (signal := self.engine.repository.get_signal(signal_id)) is not None
        ]
        if not signals:
            return []
        return [self._render_alert(plan, signals[0])]

    def has_priority_signal(
        self,
        now: datetime,
        *,
        threshold: float = 0.8,
    ) -> bool:
        """Cheap pre-gate probe so urgent evidence is never randomized away."""
        current = now.astimezone(timezone.utc)
        handled = {
            signal_id
            for plan in self.engine.repository.list_plans(limit=1000)
            if plan.status
            in {
                ActionPlanStatus.SUCCEEDED,
                ActionPlanStatus.SKIPPED,
                ActionPlanStatus.EXPIRED,
                ActionPlanStatus.FAILED,
            }
            for signal_id in plan.signal_ids
        }
        return any(
            signal.id not in handled
            and signal.severity >= threshold
            and signal.confidence >= 0.7
            and self.engine.opportunity_manager.is_signal_actionable_now(
                signal,
                current,
            )
            for signal in self.engine.repository.list_active_signals(now=current)
        )

    @staticmethod
    def _render_alert(plan: Any, signal: Any) -> dict[str, Any]:
        content = str(signal.metadata.get("content") or "").strip()
        reason = str(signal.metadata.get("reason") or "").strip()
        suggested_action = str(signal.metadata.get("suggested_action") or "").strip()
        details = [item for item in (content, reason, suggested_action) if item]
        severity = (
            "high"
            if signal.severity >= 0.8
            else "warning" if signal.severity >= 0.5 else "info"
        )
        return {
            "ack_server": "attention",
            "event_id": plan.id,
            "title": signal.summary,
            "content": "\n".join(dict.fromkeys(details)) or signal.summary,
            "severity": severity,
            "detected_at": signal.occurred_at,
            "data_category": signal.domain,
            "signal_kind": signal.kind,
            "evidence": list(signal.evidence),
            "delivery_allowed": True,
            "delivery_block_reason": "",
            "action_plan_id": plan.id,
            "canonical_event_id": str(signal.metadata.get("event_id") or ""),
            "capability_id": plan.capability_id,
            "decision_score": plan.score,
            "signal_ids": list(plan.signal_ids),
            "metrics": {
                "delivery_allowed": True,
                "delivery_block_reason": "",
                "attention_score": plan.score,
            },
        }

    def external_alert_ack_targets(self, plan_id: str) -> list[tuple[str, str]]:
        plan = self.engine.repository.get_plan(plan_id)
        if plan is None:
            return []
        targets: list[tuple[str, str]] = []
        for signal_id in plan.signal_ids:
            signal = self.engine.repository.get_signal(signal_id)
            if signal is None or signal.source.type != "mcp":
                continue
            server = str(signal.metadata.get("source_ack_server") or "").strip()
            event_id = str(signal.metadata.get("source_event_id") or "").strip()
            if server and event_id:
                targets.append((server, event_id))
        return list(dict.fromkeys(targets))

    def complete_action_plan(
        self,
        plan_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        plan = self.engine.repository.get_plan(plan_id)
        if plan is None:
            return
        if plan.status in {ActionPlanStatus.PROPOSED, ActionPlanStatus.APPROVED}:
            plan = self.engine.repository.transition_plan(
                plan.id,
                ActionPlanStatus.EXECUTING,
            )
            updated_records = self._apply_delivery_updates(
                plan.signal_ids,
                now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc),
            )
            self.engine.repository.transition_plan(
                plan.id,
                ActionPlanStatus.SUCCEEDED,
                result={
                    "delivered": True,
                    "updated_record_ids": updated_records,
                },
            )

    def _apply_delivery_updates(
        self,
        signal_ids: tuple[str, ...],
        *,
        now: datetime,
    ) -> list[str]:
        """Apply the small, declared post-delivery protocol carried by signals."""
        updated: list[str] = []
        for signal_id in signal_ids:
            signal = self.engine.repository.get_signal(signal_id)
            if signal is None or signal.source.type != "personal_record":
                continue
            instruction = signal.metadata.get("on_delivery")
            if not isinstance(instruction, dict):
                continue
            if str(instruction.get("operation") or "") != "advance_time":
                continue
            field = str(instruction.get("field") or "").strip()
            if field != "next_trigger_at":
                continue
            try:
                interval = max(5, int(instruction.get("interval_minutes") or 0))
            except (TypeError, ValueError):
                continue
            record_id = str(signal.metadata.get("source_record_id") or "").strip()
            record = self.personal_data.get(record_id)
            if record is None:
                continue
            base = self._parse_datetime(record.data.get(field)) or now
            next_at = base
            while next_at <= now:
                next_at = next_at + timedelta(minutes=interval)
            try:
                self.personal_data.update(
                    record.id,
                    {"data": {field: next_at.isoformat()}},
                    actor="attention-engine",
                    reason="advance declared recurring record after delivery",
                    automatic=True,
                )
            except (ValueError, PermissionError) as exc:
                logger.warning(
                    "unable to apply attention delivery update for %s: %s",
                    record.id,
                    exc,
                )
                continue
            updated.append(record.id)
        return updated

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        else:
            text = str(value or "").strip()
            if not text:
                return None
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)

    async def alert_ack_fn(self, compound_key: str) -> None:
        prefix, separator, plan_id = str(compound_key).partition(":")
        if separator and prefix == self.ack_server and plan_id:
            self.complete_action_plan(plan_id)

    async def context_fn(
        self,
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        snapshot: dict[str, Any] = {
            "_source": "personal_state",
            "observed_at": current.isoformat(),
            "rhythm": self.rhythm.snapshot(now=current).to_dict(),
        }
        for entity_type, key, limit in _SNAPSHOT_LIMITS:
            records = self.personal_data.list(entity_type=entity_type, limit=limit)
            snapshot[key] = [self._compact_record(item) for item in records[:limit]]
        return [snapshot]

    @staticmethod
    def _compact_record(record: PersonalRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "entity_type": record.entity_type.value,
            "title": record.title[:160],
            "summary": record.summary[:300],
            "data": PersonalAttentionSource._compact_payload(record.data),
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _compact_payload(value: Any, *, depth: int = 0) -> Any:
        if depth >= 3:
            return str(value)[:300]
        if isinstance(value, str):
            return value[:500]
        if isinstance(value, dict):
            return {
                str(key)[:80]: PersonalAttentionSource._compact_payload(
                    item,
                    depth=depth + 1,
                )
                for key, item in list(value.items())[:20]
            }
        if isinstance(value, (list, tuple)):
            return [
                PersonalAttentionSource._compact_payload(item, depth=depth + 1)
                for item in list(value)[:12]
            ]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)[:300]


__all__ = ["PersonalAttentionSource"]
