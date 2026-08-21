from __future__ import annotations

from typing import Any

from agent.tools.base import Tool
from agent.tools.personal._shared import json_text
from agent.workflows.personal import PersonalRoutineService, RoutineKind
from core.personal.rhythm import PersonalRhythmService


class PersonalRoutineTool(Tool):
    name = "personal_routine"
    description = (
        "创建标准个人助手任务：晨间简报、晚间回顾或承诺捕获。"
        "任务进入统一任务中心，可等待用户反馈并跨重启继续。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "routine": {
                "type": "string",
                "enum": [item.value for item in RoutineKind],
            },
            "local_date": {"type": "string", "description": "YYYY-MM-DD"},
            "candidate": {"type": "string", "description": "承诺候选文本"},
            "timezone": {"type": "string"},
        },
        "required": ["routine"],
    }

    def __init__(self, routines: PersonalRoutineService) -> None:
        self._routines = routines

    async def execute(self, **kwargs: Any) -> str:
        try:
            workflow, created = self._routines.create(
                RoutineKind(str(kwargs.get("routine") or "")),
                session_key=self._session_key(kwargs),
                channel=str(kwargs.get("channel") or "dashboard"),
                chat_id=str(kwargs.get("chat_id") or "personal"),
                local_date=str(kwargs.get("local_date") or ""),
                timezone_name=str(kwargs.get("timezone") or "Asia/Shanghai"),
                candidate=str(kwargs.get("candidate") or ""),
            )
        except (TypeError, ValueError) as exc:
            return f"错误：{exc}"
        return json_text(
            {
                "created": created,
                "task_id": workflow.id,
                "name": workflow.name,
                "status": workflow.status.value,
            }
        )

    @staticmethod
    def _session_key(kwargs: dict[str, Any]) -> str:
        channel = str(kwargs.get("channel") or "dashboard")
        chat_id = str(kwargs.get("chat_id") or "personal")
        return f"{channel}:{chat_id}"


class PersonalGuidanceTool(Tool):
    name = "personal_guidance"
    description = (
        "读取当前场景、专注和精力上下文；根据可用时间动态推荐下一项任务；"
        "或生成周报、月报和目标偏差分析。它是统一决策入口，不会执行外部写操作。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["snapshot", "recommend", "report"],
            },
            "available_minutes": {
                "type": "integer",
                "minimum": 5,
                "maximum": 480,
            },
            "period": {"type": "string", "enum": ["week", "month"]},
            "persist": {"type": "boolean"},
        },
        "required": ["action"],
    }

    def __init__(self, rhythm: PersonalRhythmService) -> None:
        self._rhythm = rhythm

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action") or "")
        try:
            if action == "snapshot":
                result: Any = self._rhythm.snapshot().to_dict()
            elif action == "recommend":
                result = self._rhythm.recommend(
                    minutes=int(kwargs.get("available_minutes") or 30)
                )
            elif action == "report":
                result = self._rhythm.generate_report(
                    period=str(kwargs.get("period") or "week"),
                    persist=bool(kwargs.get("persist", True)),
                ).to_dict()
            else:
                return "错误：未知 action"
        except (TypeError, ValueError) as exc:
            return f"错误：{exc}"
        return json_text(result)
