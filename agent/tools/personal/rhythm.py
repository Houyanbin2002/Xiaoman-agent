from __future__ import annotations

from typing import Any

from agent.tools.base import Tool
from agent.tools.personal._shared import json_text
from agent.tools.personal.records import PersonalRecordTool
from core.attention.feedback import FeedbackKind
from core.attention.feedback.service import FeedbackService
from core.personal.models import FollowUpTrigger, PersonalEntityType, SceneMode
from core.personal.rhythm import PersonalRhythmService


class PersonalRhythmControlTool(Tool):
    name = "personal_rhythm"
    description = (
        "更新个人助手当前场景或专注状态，沉淀可扩展的主动关注意图，"
        "并记录用户对最近提醒的接受、忽略或稍后处理反馈。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "set_scene",
                    "start_focus",
                    "stop_focus",
                    "create_follow_up",
                    "feedback",
                ],
            },
            "scene": {
                "type": "string",
                "enum": [item.value for item in SceneMode],
            },
            "duration_minutes": {"type": "integer", "minimum": 5, "maximum": 1440},
            "label": {"type": "string"},
            "allow_high_priority": {"type": "boolean"},
            "title": {"type": "string"},
            "message": {"type": "string"},
            "reason": {"type": "string"},
            "trigger_type": {
                "type": "string",
                "enum": [item.value for item in FollowUpTrigger],
            },
            "next_trigger_at": {"type": "string"},
            "interval_minutes": {"type": "integer", "minimum": 5},
            "target_entity_type": {
                "type": "string",
                "enum": [item.value for item in PersonalEntityType],
            },
            "target_record_key": {"type": "string"},
            "inactivity_days": {"type": "integer", "minimum": 1},
            "condition": {"type": "object"},
            "cooldown_minutes": {"type": "integer", "minimum": 5},
            "user_confirmed": {
                "type": "boolean",
                "description": "仅当用户本轮明确要求该场景、专注或定期关注时为 true",
            },
            "plan_id": {"type": "string"},
            "feedback": {
                "type": "string",
                "enum": [
                    "accepted",
                    "ignored",
                    "deferred",
                    "disliked",
                    "wrong_time",
                    "too_frequent",
                    "inaccurate",
                ],
            },
            "note": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(
        self,
        rhythm: PersonalRhythmService,
        feedback: FeedbackService,
    ) -> None:
        self._rhythm = rhythm
        self._feedback = feedback

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action") or "")
        try:
            if action == "set_scene":
                record = self._rhythm.set_scene(
                    SceneMode(str(kwargs.get("scene") or "neutral")),
                    duration_minutes=(
                        int(kwargs["duration_minutes"])
                        if kwargs.get("duration_minutes") is not None
                        else None
                    ),
                    source=PersonalRecordTool.source(kwargs),
                )
                result: Any = {
                    "record": record.to_dict(),
                    "context": self._rhythm.snapshot().to_dict(),
                }
            elif action == "start_focus":
                record = self._rhythm.start_focus(
                    minutes=int(kwargs.get("duration_minutes") or 30),
                    label=str(kwargs.get("label") or "专注"),
                    allow_high_priority=bool(kwargs.get("allow_high_priority", True)),
                    source=PersonalRecordTool.source(kwargs),
                )
                result = {
                    "record": record.to_dict(),
                    "context": self._rhythm.snapshot().to_dict(),
                }
            elif action == "stop_focus":
                result = {
                    "stopped": self._rhythm.stop_focus(),
                    "context": self._rhythm.snapshot().to_dict(),
                }
            elif action == "create_follow_up":
                record = self._rhythm.create_follow_up(
                    title=str(kwargs.get("title") or "主动关注"),
                    message=str(kwargs.get("message") or "进行一次状态确认。"),
                    reason=str(kwargs.get("reason") or "需要持续关注"),
                    trigger_type=FollowUpTrigger(
                        str(kwargs.get("trigger_type") or "interval")
                    ),
                    next_trigger_at=str(kwargs.get("next_trigger_at") or "") or None,
                    interval_minutes=(
                        int(kwargs["interval_minutes"])
                        if kwargs.get("interval_minutes") is not None
                        else None
                    ),
                    target_entity_type=str(kwargs.get("target_entity_type") or ""),
                    target_record_key=str(kwargs.get("target_record_key") or ""),
                    inactivity_days=(
                        int(kwargs["inactivity_days"])
                        if kwargs.get("inactivity_days") is not None
                        else None
                    ),
                    condition=dict(kwargs.get("condition") or {}),
                    cooldown_minutes=int(kwargs.get("cooldown_minutes") or 60),
                    user_confirmed=bool(kwargs.get("user_confirmed", False)),
                    source=PersonalRecordTool.source(kwargs),
                )
                result = {
                    "record": record.to_dict(),
                    "enabled": bool(record.data.get("enabled", False)),
                    "requires_confirmation": not bool(
                        record.data.get("enabled", False)
                    ),
                }
            elif action == "feedback":
                result = self._record_feedback(kwargs)
            else:
                return "错误：未知 action"
        except (PermissionError, TypeError, ValueError) as exc:
            return f"错误：{exc}"
        return json_text(result)

    def _record_feedback(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        plan_id = str(kwargs.get("plan_id") or "").strip()
        if not plan_id:
            plans = self._feedback.repository.list_plans(limit=1)
            if not plans:
                raise ValueError("没有可关联的最近行动计划")
            plan_id = plans[0].id
        kind = FeedbackKind(str(kwargs.get("feedback") or ""))
        return self._feedback.record(
            plan_id=plan_id,
            kind=kind,
            note=str(kwargs.get("note") or ""),
            metadata={"source": "conversation"},
        ).to_dict()
