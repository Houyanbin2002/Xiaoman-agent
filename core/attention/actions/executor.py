from __future__ import annotations

import inspect
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from core.attention._shared import parse_datetime
from core.attention.actions.models import ActionPlan, ActionPlanStatus
from core.attention.ports import AttentionRepository

ActionHandler = Callable[
    [ActionPlan],
    dict[str, Any] | Awaitable[dict[str, Any]],
]


class ActionHandlerRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, ActionHandler] = {}

    def register(
        self,
        capability_id: str,
        handler: ActionHandler,
        *,
        replace: bool = False,
    ) -> None:
        with self._lock:
            if capability_id in self._items and not replace:
                raise ValueError(f"action handler already registered: {capability_id}")
            self._items[capability_id] = handler

    def unregister(self, capability_id: str) -> bool:
        with self._lock:
            return self._items.pop(capability_id, None) is not None

    def get(self, capability_id: str) -> ActionHandler | None:
        with self._lock:
            return self._items.get(capability_id)


class ActionExecutionService:
    """Execute approved plans with durable, validated state transitions."""

    def __init__(
        self,
        repository: AttentionRepository,
        handlers: ActionHandlerRegistry,
    ) -> None:
        self.repository = repository
        self.handlers = handlers

    def approve(self, plan_id: str) -> ActionPlan:
        plan = self.repository.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"attention action plan not found: {plan_id}")
        expires = parse_datetime(plan.expires_at)
        if expires is not None and expires < datetime.now(timezone.utc):
            if plan.status in {
                ActionPlanStatus.PROPOSED,
                ActionPlanStatus.PENDING_APPROVAL,
                ActionPlanStatus.APPROVED,
                ActionPlanStatus.DEFERRED,
            }:
                self.repository.transition_plan(plan.id, ActionPlanStatus.EXPIRED)
            raise ValueError("action plan has expired")
        if plan.status != ActionPlanStatus.PENDING_APPROVAL:
            raise ValueError(f"action plan is not awaiting approval: {plan.status.value}")
        return self.repository.transition_plan(plan_id, ActionPlanStatus.APPROVED)

    def skip(self, plan_id: str) -> ActionPlan:
        plan = self.repository.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"attention action plan not found: {plan_id}")
        if plan.status not in {
            ActionPlanStatus.PROPOSED,
            ActionPlanStatus.PENDING_APPROVAL,
            ActionPlanStatus.APPROVED,
            ActionPlanStatus.DEFERRED,
        }:
            raise ValueError(f"action plan cannot be skipped: {plan.status.value}")
        return self.repository.transition_plan(plan_id, ActionPlanStatus.SKIPPED)

    async def execute(self, plan_id: str) -> ActionPlan:
        plan = self.repository.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"attention action plan not found: {plan_id}")
        if plan.status == ActionPlanStatus.PENDING_APPROVAL:
            raise PermissionError("action plan requires approval")
        if plan.status not in {ActionPlanStatus.PROPOSED, ActionPlanStatus.APPROVED}:
            raise ValueError(f"action plan is not executable: {plan.status.value}")
        handler = self.handlers.get(plan.capability_id)
        if handler is None:
            raise ValueError(f"no action handler registered: {plan.capability_id}")
        executing = self.repository.transition_plan(
            plan.id,
            ActionPlanStatus.EXECUTING,
        )
        try:
            result = handler(executing)
            if inspect.isawaitable(result):
                result = await result
            return self.repository.transition_plan(
                plan.id,
                ActionPlanStatus.SUCCEEDED,
                result=dict(result),
            )
        except Exception as exc:
            self.repository.transition_plan(
                plan.id,
                ActionPlanStatus.FAILED,
                error=str(exc),
            )
            raise


__all__ = [
    "ActionExecutionService",
    "ActionHandler",
    "ActionHandlerRegistry",
]
