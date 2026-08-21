from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException

from core.attention.events.models import DeliverySemantics
from core.attention.feedback import FeedbackKind
from core.attention.policies import DecisionContext
from core.attention.patterns import (
    BehaviorPattern,
    PatternSource,
    PatternStatus,
    RecurrenceSpec,
)
from core.attention.policies import PolicyEffect, PolicyRule, PolicyStatus

from ..contracts import DashboardRuntimeServices
from ..schemas import (
    AttentionFeedbackPayload,
    AttentionPatternPayload,
    AttentionPatternStatusPayload,
    AttentionPolicyPayload,
    AttentionPolicyStatusPayload,
    AttentionRuntimePayload,
)
from ..support import load_config, require_attention_runtime, save_config


def _proactive_targets(config: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    telegram = getattr(getattr(config, "channels", None), "telegram", None)
    if telegram is not None:
        for chat_id in getattr(telegram, "allow_from", []) or []:
            value = str(chat_id).strip()
            if value:
                rows.append({"channel": "telegram", "chat_id": value})
    plugins = getattr(config, "plugins", {}) or {}
    if isinstance(plugins, dict):
        for channel in ("qqbot", "weixin", "wecom"):
            item = plugins.get(channel) or {}
            if not isinstance(item, dict) or not item.get("enabled"):
                continue
            for chat_id in item.get("allow_from") or []:
                value = str(chat_id).strip()
                if value:
                    rows.append({"channel": channel, "chat_id": value})
    proactive = getattr(config, "proactive", None)
    current_channel = str(getattr(proactive, "default_channel", "") or "").strip()
    current_chat_id = str(getattr(proactive, "default_chat_id", "") or "").strip()
    if current_channel and current_chat_id:
        rows.append({"channel": current_channel, "chat_id": current_chat_id})
    return list({(row["channel"], row["chat_id"]): row for row in rows}.values())


def _datetime(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("时间必须包含时区")
    return parsed


def _visible_active_events(runtime: Any) -> list[Any]:
    return [
        event
        for event in runtime.store.list_active_events()
        if event.delivery_semantics is not DeliverySemantics.SILENT
    ]


def register_attention_routes(
    app: FastAPI,
    services: DashboardRuntimeServices,
) -> None:
    @app.get("/api/dashboard/control/attention/overview")
    def attention_overview() -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        now = datetime.now().astimezone()
        patterns = runtime.store.list_patterns()
        policies = runtime.store.list_policies()
        plans = runtime.store.list_plans(limit=1000)
        events = _visible_active_events(runtime)
        wakes = runtime.store.list_pending_wakes()
        entities = runtime.store.list_entities(limit=10000)
        proactive = getattr(services.config, "proactive", None)
        provider_failures = [
            {
                "provider_id": item.provider_id,
                "error_type": item.error_type,
                "message": item.message,
            }
            for item in runtime.providers.last_failures
        ]
        return {
            "active_signals": len(runtime.store.list_active_signals(now=now)),
            "active_events": len(events),
            "pending_wakes": len(wakes),
            "next_wake_at": runtime.store.next_wake_at(),
            "source_sync_pending": sum(
                entity is not None
                and entity.local_override.get("source_sync") == "pending"
                for entity in entities
            ),
            "patterns": len(patterns),
            "active_patterns": sum(
                item.status == PatternStatus.ACTIVE for item in patterns
            ),
            "policies": len(policies),
            "enabled_policies": sum(item.is_active_at(now) for item in policies),
            "plans": len(plans),
            "pending_approval": sum(
                item.status.value == "pending_approval" for item in plans
            ),
            "capabilities": len(runtime.capabilities.list()),
            "runtime_enabled": bool(getattr(proactive, "enabled", False)),
            "target_configured": bool(
                str(getattr(proactive, "default_chat_id", "") or "").strip()
            ),
            "target_channel": str(
                getattr(proactive, "default_channel", "") or ""
            ),
            "target_chat_id": str(
                getattr(proactive, "default_chat_id", "") or ""
            ),
            "available_targets": _proactive_targets(services.config),
            "provider_failures": provider_failures,
        }

    @app.get("/api/dashboard/control/attention/events")
    def list_attention_events(limit: int = 100) -> list[dict[str, Any]]:
        runtime = require_attention_runtime(services)
        rows: list[dict[str, Any]] = []
        for event in _visible_active_events(runtime)[: max(1, min(limit, 500))]:
            entity = runtime.store.get_entity(event.entity_id)
            row = event.to_dict()
            row["entity"] = entity.to_dict() if entity is not None else None
            rows.append(row)
        return rows

    @app.get("/api/dashboard/control/attention/wakes")
    def list_attention_wakes(limit: int = 100) -> list[dict[str, Any]]:
        runtime = require_attention_runtime(services)
        rows: list[dict[str, Any]] = []
        for wake in runtime.store.list_pending_wakes()[: max(1, min(limit, 500))]:
            event = runtime.store.get_event(wake.event_id)
            entity = (
                runtime.store.get_entity(event.entity_id)
                if event is not None
                else None
            )
            row = wake.to_dict()
            row["event"] = event.to_dict() if event is not None else None
            row["entity"] = entity.to_dict() if entity is not None else None
            rows.append(row)
        return rows

    @app.patch("/api/dashboard/control/attention/runtime")
    def update_attention_runtime(
        payload: AttentionRuntimePayload,
    ) -> dict[str, Any]:
        if payload.enabled and not payload.chat_id.strip():
            raise HTTPException(status_code=422, detail="开启主动联系前请选择接收账号")
        config_data = load_config(services.config_path)
        proactive_data = config_data.setdefault("proactive", {})
        if not isinstance(proactive_data, dict):
            raise HTTPException(status_code=400, detail="proactive 配置格式不正确")
        proactive_data["enabled"] = payload.enabled
        target = proactive_data.setdefault("target", {})
        if not isinstance(target, dict):
            target = {}
            proactive_data["target"] = target
        target["channel"] = payload.channel
        target["chat_id"] = payload.chat_id.strip()
        # Remove legacy aliases so one source of truth remains.
        proactive_data.pop("default_channel", None)
        proactive_data.pop("default_chat_id", None)
        save_config(services.config_path, config_data)
        runtime_config = getattr(services.config, "proactive", None)
        if runtime_config is not None:
            runtime_config.enabled = payload.enabled
            runtime_config.default_channel = payload.channel
            runtime_config.default_chat_id = payload.chat_id.strip()
        return {
            "saved": True,
            "restart_required": True,
            "enabled": payload.enabled,
            "channel": payload.channel,
            "target_configured": bool(payload.chat_id.strip()),
        }

    @app.get("/api/dashboard/control/attention/patterns")
    def list_attention_patterns() -> list[dict[str, Any]]:
        runtime = require_attention_runtime(services)
        return [item.to_dict() for item in runtime.store.list_patterns()]

    @app.get("/api/dashboard/control/attention/signals")
    def list_attention_signals(limit: int = 100) -> list[dict[str, Any]]:
        runtime = require_attention_runtime(services)
        now = datetime.now(timezone.utc)
        return [
            item.to_dict()
            for item in runtime.store.list_active_signals(now=now)[
                : max(1, min(limit, 500))
            ]
        ]

    @app.post("/api/dashboard/control/attention/evaluate")
    async def evaluate_attention() -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        now = datetime.now(timezone.utc)
        await runtime.engine.refresh(now=now)
        rhythm = services.personal_rhythm
        snapshot = rhythm.snapshot(now=now) if rhythm is not None else None
        result = runtime.engine.evaluate(
            context=DecisionContext(
                now=now,
                scene=(snapshot.scene.value if snapshot is not None else "neutral"),
                focus_active=(snapshot.focus_active if snapshot is not None else False),
                do_not_disturb=(
                    snapshot.do_not_disturb if snapshot is not None else False
                ),
                allow_high_priority=(
                    snapshot.allow_high_priority if snapshot is not None else True
                ),
                channel="dashboard",
                permission_mode="delegated",
            )
        )
        return {
            "reason": result.reason,
            "candidate_count": result.candidate_count,
            "denied_count": result.denied_count,
            "below_threshold_count": result.below_threshold_count,
            "plan": result.plan.to_dict() if result.plan is not None else None,
            "windows": [item.to_dict() for item in result.windows],
        }

    @app.post("/api/dashboard/control/attention/patterns")
    def create_attention_pattern(
        payload: AttentionPatternPayload,
    ) -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        try:
            pattern = BehaviorPattern.create(
                pattern_id=payload.id,
                kind=payload.kind,
                scene=payload.scene,
                recurrence=RecurrenceSpec(
                    timezone=payload.timezone,
                    days=tuple(payload.days),
                    start=payload.start,
                    end=payload.end,
                ),
                available_minutes=payload.available_minutes,
                confidence=1.0,
                source=PatternSource.USER,
                status=PatternStatus.ACTIVE,
                expires_at=_datetime(payload.expires_at),
                user_locked=payload.user_locked,
                metadata=payload.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return runtime.store.upsert_pattern(pattern).to_dict()

    @app.patch("/api/dashboard/control/attention/patterns/{pattern_id}/status")
    def set_attention_pattern_status(
        pattern_id: str,
        payload: AttentionPatternStatusPayload,
    ) -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        pattern = runtime.store.get_pattern(pattern_id)
        if pattern is None:
            raise HTTPException(status_code=404, detail="机会窗口不存在")
        try:
            updated = replace(pattern, status=PatternStatus(payload.status))
            if updated.status == PatternStatus.ACTIVE:
                updated = replace(
                    updated,
                    source=PatternSource.USER,
                    user_locked=True,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return runtime.store.upsert_pattern(updated).to_dict()

    @app.get("/api/dashboard/control/attention/policies")
    def list_attention_policies() -> list[dict[str, Any]]:
        runtime = require_attention_runtime(services)
        return [item.to_dict() for item in runtime.store.list_policies()]

    @app.get("/api/dashboard/control/attention/observations")
    def list_attention_observations(limit: int = 100) -> list[dict[str, Any]]:
        runtime = require_attention_runtime(services)
        return [
            item.to_dict()
            for item in runtime.store.list_observations(limit=max(1, min(limit, 500)))
        ]

    @app.post("/api/dashboard/control/attention/policies")
    def create_attention_policy(
        payload: AttentionPolicyPayload,
    ) -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        try:
            policy = PolicyRule.create(
                policy_id=payload.id,
                scope=payload.scope,
                conditions=payload.conditions,
                effect=PolicyEffect(payload.effect),
                priority=payload.priority,
                score_adjustment=payload.score_adjustment,
                enabled=payload.enabled,
                effective_from=_datetime(payload.effective_from),
                expires_at=_datetime(payload.expires_at),
                source="user",
                user_locked=payload.user_locked,
                metadata=payload.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return runtime.store.upsert_policy(policy).to_dict()

    @app.patch("/api/dashboard/control/attention/policies/{policy_id}/status")
    def set_attention_policy_status(
        policy_id: str,
        payload: AttentionPolicyStatusPayload,
    ) -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        policy = runtime.store.get_policy(policy_id)
        if policy is None:
            raise HTTPException(status_code=404, detail="主动行为策略不存在")
        try:
            status = PolicyStatus(payload.status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        updated = replace(
            policy,
            status=status,
            enabled=status not in {PolicyStatus.REJECTED, PolicyStatus.EXPIRED},
            source="user" if status == PolicyStatus.ACTIVE else policy.source,
            user_locked=True if status == PolicyStatus.ACTIVE else policy.user_locked,
        )
        return runtime.store.upsert_policy(updated).to_dict()

    @app.get("/api/dashboard/control/attention/plans")
    def list_attention_plans(limit: int = 100) -> list[dict[str, Any]]:
        runtime = require_attention_runtime(services)
        return [
            item.to_dict()
            for item in runtime.store.list_plans(limit=max(1, min(limit, 500)))
        ]

    @app.get("/api/dashboard/control/attention/capabilities")
    def list_attention_capabilities() -> list[dict[str, Any]]:
        runtime = require_attention_runtime(services)
        return [item.to_dict() for item in runtime.capabilities.list()]

    @app.post("/api/dashboard/control/attention/plans/{plan_id}/feedback")
    def record_attention_feedback(
        plan_id: str,
        payload: AttentionFeedbackPayload,
    ) -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        try:
            feedback = runtime.feedback.record(
                plan_id=plan_id,
                kind=FeedbackKind(payload.kind),
                note=payload.note,
                metadata=payload.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return feedback.to_dict()

    @app.post("/api/dashboard/control/attention/plans/{plan_id}/approve")
    def approve_attention_plan(plan_id: str) -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        try:
            return runtime.execution.approve(plan_id).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dashboard/control/attention/plans/{plan_id}/skip")
    def skip_attention_plan(plan_id: str) -> dict[str, Any]:
        runtime = require_attention_runtime(services)
        try:
            return runtime.execution.skip(plan_id).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["register_attention_routes"]
