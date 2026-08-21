from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from agent.tools.base import Tool
from agent.tools.personal._shared import json_text
from core.attention.feedback import FeedbackKind
from core.attention.feedback.service import FeedbackService
from core.attention.events.acknowledgement import EventAcknowledgementService
from core.attention.patterns import (
    BehaviorPattern,
    PatternSource,
    PatternStatus,
    RecurrenceSpec,
)
from core.attention.policies import PolicyEffect, PolicyRule
from core.attention.policies import PolicyStatus
from core.attention.ports import AttentionRepository


class AttentionControlTool(Tool):
    name = "attention_control"
    description = (
        "管理通用注意力机会窗口、动态策略和主动行动反馈。"
        "仅把用户明确描述的生活规律保存为 active；模型推断必须保存为 proposed。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "save_opportunity",
                    "list_opportunities",
                    "set_opportunity_status",
                    "save_policy",
                    "list_policies",
                    "set_policy_status",
                    "feedback",
                    "list_events",
                    "complete_event",
                    "cancel_event",
                ],
            },
            "pattern_id": {"type": "string"},
            "kind": {"type": "string"},
            "scene": {"type": "string"},
            "timezone": {"type": "string"},
            "days": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                },
            },
            "start_time": {"type": "string", "description": "HH:MM"},
            "end_time": {"type": "string", "description": "HH:MM"},
            "available_minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "user_confirmed": {"type": "boolean"},
            "user_locked": {"type": "boolean"},
            "status": {
                "type": "string",
                "enum": [item.value for item in PatternStatus],
            },
            "expires_at": {"type": "string"},
            "metadata": {"type": "object"},
            "policy_id": {"type": "string"},
            "scope": {"type": "object"},
            "conditions": {"type": "object"},
            "effect": {
                "type": "string",
                "enum": [item.value for item in PolicyEffect],
            },
            "priority": {"type": "integer", "minimum": 0, "maximum": 1000},
            "score_adjustment": {"type": "number"},
            "effective_from": {"type": "string"},
            "enabled": {"type": "boolean"},
            "plan_id": {"type": "string"},
            "event_id": {"type": "string"},
            "feedback_kind": {
                "type": "string",
                "enum": [item.value for item in FeedbackKind],
            },
            "note": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(
        self,
        repository: AttentionRepository,
        feedback: FeedbackService,
        acknowledgements: EventAcknowledgementService | None = None,
    ) -> None:
        self._repository = repository
        self._feedback = feedback
        self._acknowledgements = acknowledgements

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action") or "")
        try:
            if action == "save_opportunity":
                result: Any = self._save_opportunity(kwargs).to_dict()
            elif action == "list_opportunities":
                result = [item.to_dict() for item in self._repository.list_patterns()]
            elif action == "set_opportunity_status":
                result = self._set_opportunity_status(kwargs).to_dict()
            elif action == "save_policy":
                result = self._save_policy(kwargs).to_dict()
            elif action == "list_policies":
                result = [item.to_dict() for item in self._repository.list_policies()]
            elif action == "set_policy_status":
                result = self._set_policy_status(kwargs).to_dict()
            elif action == "feedback":
                result = self._feedback.record(
                    plan_id=self._required(kwargs, "plan_id"),
                    kind=FeedbackKind(self._required(kwargs, "feedback_kind")),
                    note=str(kwargs.get("note") or ""),
                    metadata=dict(kwargs.get("metadata") or {}),
                ).to_dict()
            elif action == "list_events":
                result = [item.to_dict() for item in self._repository.list_active_events()]
            elif action == "complete_event":
                result = self._require_acknowledgements().complete(
                    self._required(kwargs, "event_id"),
                    actor="user",
                )
            elif action == "cancel_event":
                result = self._require_acknowledgements().cancel(
                    self._required(kwargs, "event_id"),
                    actor="user",
                )
            else:
                return "错误：未知 action"
        except (KeyError, TypeError, ValueError) as exc:
            return f"错误：{exc}"
        return json_text(result)

    def _require_acknowledgements(self) -> EventAcknowledgementService:
        if self._acknowledgements is None:
            raise ValueError("主动事件确认服务不可用")
        return self._acknowledgements

    def _save_opportunity(self, kwargs: dict[str, Any]) -> BehaviorPattern:
        confirmed = bool(kwargs.get("user_confirmed", False))
        source = PatternSource.USER if confirmed else PatternSource.LEARNED
        status = PatternStatus.ACTIVE if confirmed else PatternStatus.PROPOSED
        expires = self._optional_datetime(kwargs.get("expires_at"))
        pattern = BehaviorPattern.create(
            pattern_id=str(kwargs.get("pattern_id") or "") or None,
            kind=str(kwargs.get("kind") or "availability_pattern"),
            scene=str(kwargs.get("scene") or "neutral"),
            recurrence=RecurrenceSpec(
                timezone=str(kwargs.get("timezone") or "Asia/Shanghai"),
                days=tuple(str(item) for item in kwargs.get("days") or ()),
                start=self._required(kwargs, "start_time"),
                end=self._required(kwargs, "end_time"),
            ),
            available_minutes=int(kwargs.get("available_minutes") or 15),
            confidence=(1.0 if confirmed else float(kwargs.get("confidence") or 0.5)),
            source=source,
            status=status,
            observation_count=1,
            expires_at=expires,
            user_locked=confirmed and bool(kwargs.get("user_locked", True)),
            metadata=dict(kwargs.get("metadata") or {}),
        )
        return self._repository.upsert_pattern(pattern)

    def _set_opportunity_status(self, kwargs: dict[str, Any]) -> BehaviorPattern:
        pattern_id = self._required(kwargs, "pattern_id")
        pattern = self._repository.get_pattern(pattern_id)
        if pattern is None:
            raise ValueError(f"机会窗口不存在: {pattern_id}")
        status = PatternStatus(self._required(kwargs, "status"))
        if status == PatternStatus.ACTIVE and not bool(
            kwargs.get("user_confirmed", False)
        ):
            raise ValueError("激活推断窗口需要用户确认")
        updated = replace(
            pattern,
            status=status,
            source=(
                PatternSource.USER
                if status == PatternStatus.ACTIVE
                and bool(kwargs.get("user_confirmed", False))
                else pattern.source
            ),
            user_locked=(
                True
                if status == PatternStatus.ACTIVE
                and bool(kwargs.get("user_confirmed", False))
                else pattern.user_locked
            ),
        )
        return self._repository.upsert_pattern(updated)

    def _save_policy(self, kwargs: dict[str, Any]) -> PolicyRule:
        confirmed = bool(kwargs.get("user_confirmed", False))
        if not confirmed:
            raise ValueError("保存主动行为策略需要用户明确确认")
        policy = PolicyRule.create(
            policy_id=str(kwargs.get("policy_id") or "") or None,
            effect=PolicyEffect(self._required(kwargs, "effect")),
            scope=dict(kwargs.get("scope") or {}),
            conditions=dict(kwargs.get("conditions") or {}),
            priority=int(kwargs.get("priority") or 50),
            score_adjustment=float(kwargs.get("score_adjustment") or 0.0),
            enabled=bool(kwargs.get("enabled", True)),
            effective_from=self._optional_datetime(kwargs.get("effective_from")),
            expires_at=self._optional_datetime(kwargs.get("expires_at")),
            source="user",
            user_locked=bool(kwargs.get("user_locked", True)),
            metadata=dict(kwargs.get("metadata") or {}),
        )
        return self._repository.upsert_policy(policy)

    def _set_policy_status(self, kwargs: dict[str, Any]) -> PolicyRule:
        policy_id = self._required(kwargs, "policy_id")
        policy = self._repository.get_policy(policy_id)
        if policy is None:
            raise ValueError(f"主动行为策略不存在: {policy_id}")
        status = PolicyStatus(self._required(kwargs, "status"))
        if status == PolicyStatus.ACTIVE and not bool(
            kwargs.get("user_confirmed", False)
        ):
            raise ValueError("激活主动行为策略需要用户确认")
        updated = replace(
            policy,
            status=status,
            enabled=status not in {PolicyStatus.REJECTED, PolicyStatus.EXPIRED},
            source=(
                "user"
                if status == PolicyStatus.ACTIVE
                and bool(kwargs.get("user_confirmed", False))
                else policy.source
            ),
            user_locked=(
                True
                if status == PolicyStatus.ACTIVE
                and bool(kwargs.get("user_confirmed", False))
                else policy.user_locked
            ),
        )
        return self._repository.upsert_policy(updated)

    @staticmethod
    def _required(kwargs: dict[str, Any], key: str) -> str:
        value = str(kwargs.get(key) or "").strip()
        if not value:
            raise ValueError(f"缺少参数 {key}")
        return value

    @staticmethod
    def _optional_datetime(value: Any) -> datetime | None:
        if value is None or not str(value).strip():
            return None
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("时间必须包含时区")
        return parsed


__all__ = ["AttentionControlTool"]
