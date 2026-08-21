from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from core.attention._shared import clamp01, parse_datetime, positive_int
from core.attention.signals import AttentionSignal, SignalSource, SignalValence
from core.personal.models import PersonalRecord, RecordStatus
from core.personal.service import PersonalDataService

_TERMINAL_STATES = {"done", "completed", "cancelled", "closed", "inactive"}
_TEMPORAL_FIELDS = ("due_at", "next_trigger_at", "depart_at", "date")


class PersonalRecordSignalProvider:
    """Translate personal records through one open, data-driven protocol.

    Records can contribute an explicit ``attention_signal`` mapping. Records
    with a standard temporal field receive a conservative generic due signal.
    No entity-specific life scenario or notification rule lives here.
    """

    def __init__(self, personal_data: PersonalDataService) -> None:
        self.personal_data = personal_data

    def collect(self, now: datetime) -> list[AttentionSignal]:
        current = now.astimezone(timezone.utc)
        signals: list[AttentionSignal] = []
        records = self.personal_data.list(
            statuses=[RecordStatus.ACTIVE],
            limit=1000,
        )
        for record in records:
            explicit = record.data.get("attention_signal")
            if isinstance(explicit, dict):
                signal = self._explicit(record, explicit, current)
            else:
                signal = self._temporal(record, current)
            if signal is not None:
                signals.append(signal)
        return signals

    @classmethod
    def _explicit(
        cls,
        record: PersonalRecord,
        config: dict[str, Any],
        now: datetime,
    ) -> AttentionSignal | None:
        if config.get("enabled", True) is False:
            return None
        occurred = (
            parse_datetime(config.get("occurred_at"))
            or parse_datetime(record.updated_at)
            or now
        )
        expires = (
            parse_datetime(config.get("expires_at"))
            or parse_datetime(record.expires_at)
            or occurred
            + timedelta(minutes=positive_int(config.get("valid_for_minutes"), 360))
        )
        if expires < now:
            return None
        kind = cls._kind(config.get("kind") or f"personal.{record.entity_type.value}")
        opportunity = config.get("opportunity")
        if not isinstance(opportunity, dict):
            opportunity = {
                "scene": str(config.get("scene") or "neutral"),
                "starts_at": occurred.isoformat(),
                "ends_at": expires.isoformat(),
                "available_minutes": positive_int(
                    config.get("estimated_attention_minutes"),
                    5,
                ),
            }
        capabilities = config.get("suggested_capabilities")
        if isinstance(capabilities, str):
            suggested = (capabilities,)
        elif isinstance(capabilities, (list, tuple)):
            suggested = tuple(str(item) for item in capabilities if str(item).strip())
        else:
            suggested = ("message.notify",)
        digest = cls._digest(record.id, kind, occurred.isoformat())
        delivery_update = config.get("on_delivery")
        return AttentionSignal.create(
            signal_id=f"sig_personal_{digest}",
            kind=kind,
            domain=cls._token(
                config.get("domain")
                or record.data_category.value
                or record.entity_type.value
            ),
            summary=str(config.get("summary") or record.title)[:300],
            source=SignalSource(
                type="personal_record",
                name=record.entity_type.value,
                reference=record.id,
            ),
            occurred_at=occurred,
            expires_at=expires,
            valence=cls._valence(config.get("valence")),
            severity=clamp01(config.get("severity"), 0.5),
            urgency=clamp01(config.get("urgency"), 0.5),
            actionability=clamp01(config.get("actionability"), 0.75),
            confidence=clamp01(config.get("confidence"), record.confidence),
            freshness=clamp01(config.get("freshness"), 1.0),
            estimated_attention_minutes=positive_int(
                config.get("estimated_attention_minutes"),
                5,
            ),
            risk_domain=str(config.get("risk_domain") or record.data_category.value),
            evidence=cls._evidence(config.get("evidence"), record),
            suggested_capabilities=suggested,
            metadata={
                "source_record_id": record.id,
                "content": str(config.get("content") or record.summary)[:2000],
                "reason": str(config.get("reason") or "")[:500],
                "suggested_action": str(config.get("suggested_action") or "")[:500],
                "goal_alignment": clamp01(config.get("goal_alignment"), 0.5),
                "opportunity": dict(opportunity),
                **(
                    {"on_delivery": dict(delivery_update)}
                    if isinstance(delivery_update, dict)
                    else {}
                ),
                **(
                    {"pattern_observation": dict(config["pattern_observation"])}
                    if isinstance(config.get("pattern_observation"), dict)
                    else {}
                ),
                **(
                    {"policy_observation": dict(config["policy_observation"])}
                    if isinstance(config.get("policy_observation"), dict)
                    else {}
                ),
            },
        )

    @classmethod
    def _temporal(
        cls,
        record: PersonalRecord,
        now: datetime,
    ) -> AttentionSignal | None:
        data = record.data
        state = str(data.get("state") or data.get("status") or "").lower()
        if state in _TERMINAL_STATES or data.get("enabled") is False:
            return None
        due = None
        due_field = ""
        for field in _TEMPORAL_FIELDS:
            due = parse_datetime(data.get(field))
            if due is not None:
                due_field = field
                break
        if due is None:
            return None
        horizon_minutes = positive_int(
            data.get("attention_horizon_minutes"),
            48 * 60,
            maximum=30 * 24 * 60,
        )
        starts = due - timedelta(minutes=horizon_minutes)
        # An open commitment does not stop mattering six hours after its due
        # time. Keep overdue records eligible until their canonical state is
        # completed/cancelled; plan history, cooldown and policy still prevent
        # repeated delivery.
        ends = max(due + timedelta(hours=6), now + timedelta(hours=6))
        if not starts <= now <= ends:
            return None
        remaining = max(0.0, (due - now).total_seconds() / 60)
        urgency = 1.0 if due <= now else 1.0 - min(remaining / horizon_minutes, 1.0)
        priority = str(data.get("priority") or "normal").lower()
        severity = 0.85 if priority in {"high", "urgent", "critical"} else 0.6
        progress = clamp01(data.get("progress"), 0.0)
        actionability = max(0.2, 0.9 - progress * 0.5)
        delivery_update = data.get("attention_on_delivery")
        if not isinstance(delivery_update, dict):
            interval_minutes = positive_int(data.get("interval_minutes"), 0)
            delivery_update = (
                {
                    "operation": "advance_time",
                    "field": due_field,
                    "interval_minutes": interval_minutes,
                }
                if due_field == "next_trigger_at" and interval_minutes > 0
                else None
            )
        digest = cls._digest(record.id, due_field, due.isoformat())
        return AttentionSignal.create(
            signal_id=f"sig_personal_due_{digest}",
            kind="personal.temporal_due",
            domain=cls._token(
                record.data_category.value
                if record.data_category.value != "general"
                else record.entity_type.value
            ),
            summary=record.title[:300],
            source=SignalSource(
                type="personal_record",
                name=record.entity_type.value,
                reference=record.id,
            ),
            occurred_at=starts,
            expires_at=ends,
            valence=SignalValence.NEUTRAL,
            severity=severity,
            urgency=urgency,
            actionability=actionability,
            confidence=record.confidence,
            estimated_attention_minutes=positive_int(
                data.get("estimated_attention_minutes")
                or data.get("estimated_minutes"),
                5,
            ),
            risk_domain=record.data_category.value,
            evidence=[
                {
                    "record_id": record.id,
                    "field": due_field,
                    "value": due.isoformat(),
                    "progress": progress,
                }
            ],
            suggested_capabilities=("message.notify",),
            metadata={
                "source_record_id": record.id,
                "content": record.summary[:2000],
                "reason": "记录进入其声明的关注时间范围",
                "suggested_action": str(data.get("next_action") or "")[:500],
                "goal_alignment": 0.65,
                "opportunity": {
                    "scene": str(data.get("scene") or "neutral"),
                    "starts_at": starts.isoformat(),
                    "ends_at": ends.isoformat(),
                    "available_minutes": positive_int(
                        data.get("estimated_attention_minutes"),
                        5,
                    ),
                },
                **(
                    {"on_delivery": dict(delivery_update)}
                    if isinstance(delivery_update, dict)
                    else {}
                ),
            },
        )

    @staticmethod
    def _kind(value: Any) -> str:
        token = PersonalRecordSignalProvider._token(value)
        return token if "." in token else f"personal.{token}"

    @staticmethod
    def _token(value: Any) -> str:
        token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip().lower())
        return token.strip("_.-") or "general"

    @staticmethod
    def _valence(value: Any) -> SignalValence:
        try:
            return SignalValence(str(value or "neutral").lower())
        except ValueError:
            return SignalValence.NEUTRAL

    @staticmethod
    def _evidence(value: Any, record: PersonalRecord) -> list[dict[str, Any]]:
        if isinstance(value, (list, tuple)):
            return [
                dict(item) if isinstance(item, dict) else {"value": str(item)[:500]}
                for item in value[:20]
            ]
        return [{"record_id": record.id, "revision": record.revision}]

    @staticmethod
    def _digest(*parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:28]


__all__ = ["PersonalRecordSignalProvider"]
