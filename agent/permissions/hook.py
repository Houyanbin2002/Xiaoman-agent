from __future__ import annotations

from collections.abc import Callable

from agent.permissions.classifier import PermissionClassifier
from agent.permissions.models import normalize_permission_mode
from agent.permissions.service import PermissionService
from agent.tool_hooks.base import ToolHook
from agent.tool_hooks.types import HookContext, HookOutcome


class PermissionGuardHook(ToolHook):
    name = "permission_guard"
    event = "pre_tool_use"

    def __init__(
        self,
        *,
        classifier: PermissionClassifier,
        service: PermissionService,
        risk_resolver: Callable[[str], str],
    ) -> None:
        self._classifier = classifier
        self._service = service
        self._risk_resolver = risk_resolver

    def matches(self, ctx: HookContext) -> bool:
        request = ctx.request
        return (
            request.source == "passive"
            and request.channel == "dashboard"
            and request.enforce_permissions
        )

    async def run(self, ctx: HookContext) -> HookOutcome:
        request = ctx.request
        mode = normalize_permission_mode(request.permission_mode)
        classification = self._classifier.classify(
            request.tool_name,
            ctx.current_arguments,
            self._risk_resolver(request.tool_name),
        )
        if not classification.requires_approval(mode):
            return HookOutcome(
                reason=(
                    f"permission:{mode}:allowed:{classification.category}:"
                    f"{classification.risk}"
                )
            )
        approval = self._service.create_request(
            session_key=request.session_key,
            call_id=request.call_id,
            tool_name=request.tool_name,
            mode=mode,
            classification=classification,
        )
        if await self._service.request(approval):
            return HookOutcome(
                reason=f"permission:{mode}:approved:{approval.id}"
            )
        return HookOutcome(
            decision="deny",
            reason=(
                "用户拒绝了这次工具调用。不要执行该操作；可以说明未产生更改，"
                "并在可能时提供更安全的替代方案。"
            ),
        )
