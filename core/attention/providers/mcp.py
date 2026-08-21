from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from core.attention._shared import clamp01, parse_datetime, positive_int
from core.attention.signals import AttentionSignal, SignalSource, SignalValence

_SEVERITY = {
    "info": 0.35,
    "low": 0.35,
    "warning": 0.65,
    "medium": 0.65,
    "high": 0.9,
    "critical": 0.98,
}


class McpAlertSignalAdapter:
    """Normalize any proactive MCP alert into the open attention contract.

    The adapter deliberately understands only protocol fields. Domain-specific
    meanings stay in the MCP provider's manifest and payload rather than being
    hard-coded in the attention engine.
    """

    @classmethod
    def convert(
        cls,
        event: dict[str, Any],
        *,
        now: datetime,
    ) -> AttentionSignal:
        current = now.astimezone(timezone.utc)
        ack_server = str(event.get("ack_server") or event.get("source") or "mcp")
        event_id = str(event.get("event_id") or event.get("id") or "").strip()
        raw_kind = str(
            event.get("signal_kind")
            or event.get("kind")
            or event.get("event_type")
            or "event"
        )
        kind = cls._namespaced(raw_kind)
        domain = cls._token(
            event.get("domain")
            or event.get("data_category")
            or event.get("category")
            or "general"
        )
        severity = cls._feature(event.get("severity"), fallback=0.55)
        urgency = cls._feature(
            event.get("urgency"),
            fallback=max(0.35, severity - 0.05),
        )
        occurred = (
            parse_datetime(event.get("occurred_at"))
            or parse_datetime(event.get("detected_at"))
            or parse_datetime(event.get("published_at"))
            or current
        )
        duration = positive_int(
            event.get("valid_for_minutes"),
            120 if severity >= 0.8 else 360 if severity >= 0.5 else 720,
            maximum=7 * 24 * 60,
        )
        expires = parse_datetime(event.get("expires_at")) or (
            occurred + timedelta(minutes=duration)
        )
        capabilities = event.get("suggested_capabilities")
        if isinstance(capabilities, str):
            suggested = (capabilities,)
        elif isinstance(capabilities, (list, tuple)):
            suggested = tuple(str(item) for item in capabilities if str(item).strip())
        else:
            suggested = (str(event.get("capability_id") or "message.notify"),)
        evidence = cls._evidence(event.get("evidence"))
        opportunity = event.get("opportunity")
        if not isinstance(opportunity, dict):
            opportunity = {
                "scene": str(event.get("scene") or "neutral"),
                "duration_minutes": duration,
                "available_minutes": positive_int(
                    event.get("estimated_attention_minutes"),
                    5,
                ),
            }
        provider_metadata = event.get("metadata")
        provider_metadata = (
            dict(provider_metadata) if isinstance(provider_metadata, dict) else {}
        )
        pattern_observation = event.get("pattern_observation")
        if not isinstance(pattern_observation, dict):
            pattern_observation = provider_metadata.get("pattern_observation")
        policy_observation = event.get("policy_observation")
        if not isinstance(policy_observation, dict):
            policy_observation = provider_metadata.get("policy_observation")
        digest = hashlib.sha256(
            f"{ack_server}|{event_id}|{kind}".encode("utf-8")
        ).hexdigest()[:28]
        return AttentionSignal.create(
            signal_id=f"sig_mcp_{digest}",
            kind=kind,
            domain=domain,
            summary=str(event.get("title") or event.get("summary") or raw_kind)[:300],
            source=SignalSource(
                type="mcp",
                name=ack_server,
                reference=event_id,
            ),
            occurred_at=occurred,
            expires_at=expires,
            valence=cls._valence(event.get("valence")),
            severity=severity,
            urgency=urgency,
            actionability=cls._feature(event.get("actionability"), fallback=0.85),
            confidence=cls._feature(event.get("confidence"), fallback=0.8),
            freshness=cls._feature(event.get("freshness"), fallback=1.0),
            estimated_attention_minutes=positive_int(
                event.get("estimated_attention_minutes"),
                5,
            ),
            risk_domain=str(event.get("risk_domain") or domain),
            evidence=evidence,
            suggested_capabilities=suggested,
            metadata={
                "source_event_id": event_id,
                "source_ack_server": ack_server,
                "content": str(event.get("content") or event.get("summary") or "")[
                    :2000
                ],
                "reason": str(event.get("reason") or "")[:500],
                "suggested_action": str(event.get("suggested_action") or "")[:500],
                "opportunity": dict(opportunity),
                "provider_metadata": cls._compact(provider_metadata),
                **(
                    {"pattern_observation": dict(pattern_observation)}
                    if isinstance(pattern_observation, dict)
                    else {}
                ),
                **(
                    {"policy_observation": dict(policy_observation)}
                    if isinstance(policy_observation, dict)
                    else {}
                ),
            },
        )

    @staticmethod
    def _feature(value: Any, *, fallback: float) -> float:
        if isinstance(value, str):
            mapped = _SEVERITY.get(value.strip().lower())
            if mapped is not None:
                return mapped
        return clamp01(value, fallback)

    @staticmethod
    def _namespaced(value: str) -> str:
        token = McpAlertSignalAdapter._token(value)
        return token if "." in token else f"mcp.{token or 'event'}"

    @staticmethod
    def _token(value: Any) -> str:
        token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip().lower())
        return token.strip("_.-") or "general"

    @staticmethod
    def _valence(value: Any) -> SignalValence:
        try:
            return SignalValence(str(value or "neutral").strip().lower())
        except ValueError:
            return SignalValence.NEUTRAL

    @staticmethod
    def _evidence(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, (list, tuple)):
            return []
        rows: list[dict[str, Any]] = []
        for item in value[:20]:
            if isinstance(item, dict):
                rows.append(dict(item))
            else:
                rows.append({"value": str(item)[:500]})
        return rows

    @staticmethod
    def _compact(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:30]:
            if isinstance(item, (str, int, float, bool)) or item is None:
                result[str(key)[:80]] = (
                    item if not isinstance(item, str) else item[:500]
                )
        return result


__all__ = ["McpAlertSignalAdapter"]
