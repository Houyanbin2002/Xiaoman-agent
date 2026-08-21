from __future__ import annotations

from typing import Any

from agent.tools.base import Tool
from agent.tools.personal._shared import json_text
from core.personal.governance import MemoryGovernanceService
from core.personal.models import (
    AccessPolicy,
    DataCategory,
    PersonalEntityType,
    RecordSource,
    RecordStatus,
    SensitivityLevel,
)
from core.personal.service import PersonalDataService


class PersonalContextTool(Tool):
    name = "personal_context"
    description = (
        "读取小满统一个人数据中的资料、承诺、计划、健康观测、签到、通知策略和受治理记忆。"
        "用于制定每日计划或回顾，不会修改数据。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "entity_type": {
                "type": "string",
                "enum": [item.value for item in PersonalEntityType],
                "description": "可选的个人数据类型过滤",
            },
            "include_inactive": {
                "type": "boolean",
                "description": "是否包含已过期、已替代和已遗忘记录，默认 false",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
    }

    def __init__(self, service: PersonalDataService) -> None:
        self._service = service

    async def execute(self, **kwargs: Any) -> str:
        self._service.expire_due()
        entity_value = str(kwargs.get("entity_type") or "").strip()
        entity_type = PersonalEntityType(entity_value) if entity_value else None
        statuses = list(RecordStatus) if kwargs.get("include_inactive", False) else None
        records = self._service.list(
            entity_type=entity_type,
            statuses=statuses,
            limit=int(kwargs.get("limit") or 50),
        )
        visible = [
            record
            for record in records
            if record.access_policy
            not in {AccessPolicy.CONFIRM_READ, AccessPolicy.OWNER_ONLY}
        ]
        return json_text(
            {
                "count": len(visible),
                "restricted_count": len(records) - len(visible),
                "records": [record.to_dict() for record in visible],
            }
        )


class PersonalRecordTool(Tool):
    name = "personal_record"
    description = (
        "创建、更新、确认或遗忘统一个人记录。自动更新会遵守用户锁定和禁止自动更新设置；"
        "create 必须提供 entity_type 和 title。长期记忆统一由后台语义分析提炼，"
        "本工具不创建、更新或确认长期记忆。"
        "待办事项（commitment）应提供 next_action、estimated_minutes，并按已知精度使用 due_at"
        "或 due_date/due_period；"
        "confirm、forget 以及覆盖锁定记录必须基于用户本轮明确确认。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "confirm", "forget"],
                "description": "create 新建；update 修改；confirm 确认；forget 遗忘",
            },
            "record_id": {"type": "string"},
            "entity_type": {
                "type": "string",
                "enum": [item.value for item in PersonalEntityType],
            },
            "record_key": {"type": "string"},
            "title": {
                "type": "string",
                "description": "记录标题；action=create 时必填，应使用简短、可识别的自然语言",
            },
            "summary": {"type": "string"},
            "data": {"type": "object"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "sensitivity": {
                "type": "string",
                "enum": [item.value for item in SensitivityLevel],
            },
            "data_category": {
                "type": "string",
                "enum": [item.value for item in DataCategory],
            },
            "access_policy": {
                "type": "string",
                "enum": [item.value for item in AccessPolicy],
            },
            "expires_at": {"type": "string"},
            "reason": {"type": "string"},
            "user_confirmed": {
                "type": "boolean",
                "description": "仅在用户本轮明确确认该写操作时为 true",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        service: PersonalDataService,
        memory_governance: MemoryGovernanceService | None = None,
    ) -> None:
        self._service = service
        self._memory_governance = memory_governance

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action") or "").strip()
        confirmed = bool(kwargs.get("user_confirmed", False))
        record_id = str(kwargs.get("record_id") or "").strip()
        reason = str(kwargs.get("reason") or "").strip()
        try:
            if action == "create":
                entity_type = PersonalEntityType(str(kwargs.get("entity_type") or ""))
                create_data = dict(kwargs.get("data") or {})
                if entity_type == PersonalEntityType.MEMORY:
                    return self._background_memory_result()
                if entity_type == PersonalEntityType.PROACTIVE_INTENT and not confirmed:
                    create_data.update({"enabled": False, "status": "proposed"})
                record = self._service.create(
                    entity_type=entity_type,
                    record_key=str(kwargs.get("record_key") or ""),
                    title=str(kwargs.get("title") or ""),
                    summary=str(kwargs.get("summary") or ""),
                    data=create_data,
                    source=self.source(kwargs),
                    confidence=float(kwargs.get("confidence", 0.8)),
                    sensitivity=(
                        SensitivityLevel(str(kwargs["sensitivity"]))
                        if kwargs.get("sensitivity")
                        else None
                    ),
                    data_category=(
                        DataCategory(str(kwargs["data_category"]))
                        if kwargs.get("data_category")
                        else None
                    ),
                    access_policy=(
                        AccessPolicy(str(kwargs["access_policy"]))
                        if kwargs.get("access_policy")
                        else None
                    ),
                    expires_at=str(kwargs.get("expires_at") or "") or None,
                    actor="assistant",
                )
            elif action == "update":
                self._require_id(record_id)
                changes = {
                    key: kwargs[key]
                    for key in (
                        "record_key",
                        "title",
                        "summary",
                        "data",
                        "confidence",
                        "sensitivity",
                        "data_category",
                        "access_policy",
                        "expires_at",
                        "user_locked",
                        "allow_auto_update",
                    )
                    if key in kwargs
                }
                existing = self._service.get(record_id)
                if (
                    self._memory_governance is not None
                    and existing is not None
                    and existing.entity_type == PersonalEntityType.MEMORY
                ):
                    return self._background_memory_result()
                else:
                    record = self._service.update(
                        record_id,
                        changes,
                        actor="user" if confirmed else "assistant",
                        reason=reason,
                        automatic=not confirmed,
                    )
            elif action == "confirm":
                self._require_confirmation(confirmed)
                self._require_id(record_id)
                existing = self._service.get(record_id)
                if (
                    self._memory_governance is not None
                    and existing is not None
                    and existing.entity_type == PersonalEntityType.MEMORY
                ):
                    return self._background_memory_result()
                else:
                    record = self._service.confirm(record_id)
            elif action == "forget":
                self._require_confirmation(confirmed)
                self._require_id(record_id)
                existing = self._service.get(record_id)
                if (
                    self._memory_governance is not None
                    and existing is not None
                    and existing.entity_type == PersonalEntityType.MEMORY
                ):
                    record = self._memory_governance.forget(record_id, reason=reason)
                else:
                    record = self._service.forget(record_id, reason=reason)
            else:
                return "错误：未知 action"
        except (PermissionError, TypeError, ValueError) as exc:
            return f"错误：{exc}"
        return json_text({"ok": True, "record": record.to_dict()})

    @staticmethod
    def source(kwargs: dict[str, Any]) -> RecordSource:
        channel = str(kwargs.get("channel") or "assistant")
        chat_id = str(kwargs.get("chat_id") or "")
        return RecordSource(channel, f"{channel}:{chat_id}" if chat_id else channel)

    @staticmethod
    def _require_id(record_id: str) -> None:
        if not record_id:
            raise ValueError("record_id is required")

    @staticmethod
    def _require_confirmation(confirmed: bool) -> None:
        if not confirmed:
            raise PermissionError("该操作需要用户本轮明确确认")

    @staticmethod
    def _background_memory_result() -> str:
        return json_text(
            {
                "ok": False,
                "status": "background_extraction_required",
                "reason": "long_term_memory_is_background_managed",
            }
        )
