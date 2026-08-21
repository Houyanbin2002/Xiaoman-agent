from __future__ import annotations

import json
from typing import Any

from agent.tools.base import Tool
from agent.workflows.runtime import WorkflowRuntime
from core.workflow.models import StepExecutor, StepKind, StepSpec


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


class TaskCreateTool(Tool):
    name = "task_create"
    description = (
        "创建小满统一管理的持久化任务。它是长任务和后台任务的唯一入口，"
        "负责步骤依赖、状态、失败重试、用户确认和进程重启恢复。"
        "普通问答或 1-3 步能立即完成的操作不要创建任务，直接执行。"
        "独立调研、分析、写报告等步骤使用 executor=subagent；"
        "需要当前会话完整能力或外部操作的步骤使用 executor=agent。"
        "agent 步骤默认只能调用只读工具；需要写入或外部副作用工具时，"
        "必须在 allowed_tools 中逐项声明，并直接依赖一个 approval 步骤。"
        "subagent 的 scripting/general profile 也必须直接依赖 approval；research 可默认执行。"
        "task_create、task_manage、message_push 永远不能由离线步骤调用。"
        "需要用户补充信息用 wait_user。"
        "subagent 只是内部步骤执行器，不是另一套用户任务。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "任务短名称，例如：每日训练复盘",
                "minLength": 1,
                "maxLength": 80,
            },
            "goal": {
                "type": "string",
                "description": "完整任务目标和完成标准",
                "minLength": 1,
            },
            "steps": {
                "type": "array",
                "description": "按依赖关系定义的步骤，最多 32 个",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "稳定步骤 ID，只用字母、数字、下划线或短横线，且以字母开头",
                        },
                        "title": {"type": "string", "description": "步骤短标题"},
                        "description": {
                            "type": "string",
                            "description": "agent 执行指令，或向用户展示的问题/审批内容",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["agent", "wait_user", "approval"],
                            "description": "agent=自动执行，wait_user=等待回答，approval=等待同意",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "必须先完成的步骤 ID",
                        },
                        "max_attempts": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "description": "agent 步骤最大尝试次数，默认 2",
                        },
                        "input": {
                            "type": "object",
                            "description": "传给步骤的结构化输入",
                        },
                        "allowed_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "仅 executor=agent 有效。默认无需填写，只读工具自动可用；"
                                "写入/外部副作用工具必须逐项填写，且步骤必须直接依赖 approval。"
                                "任务管理和主动消息工具不可填写。"
                            ),
                        },
                        "executor": {
                            "type": "string",
                            "enum": ["agent", "subagent"],
                            "description": (
                                "仅 agent 步骤有效。agent=使用小满主执行循环；"
                                "subagent=使用隔离的步骤执行器，适合独立调研/分析/产出。"
                            ),
                        },
                        "profile": {
                            "type": "string",
                            "enum": ["research", "scripting", "general"],
                            "description": (
                                "仅 subagent 有效。research=只读调研；"
                                "scripting=在任务目录执行脚本/写文件；general=两者兼有。"
                                "scripting/general 必须直接依赖 approval 步骤。"
                            ),
                        },
                    },
                    "required": ["id", "title", "description", "kind"],
                },
            },
            "context": {
                "type": "object",
                "description": "整个任务共享的结构化上下文",
            },
            "auto_start": {
                "type": "boolean",
                "description": "创建后立即执行，默认 true",
            },
        },
        "required": ["name", "goal", "steps"],
    }

    def __init__(self, runtime: WorkflowRuntime) -> None:
        self.runtime = runtime

    async def execute(self, **kwargs: Any) -> str:
        raw_steps = kwargs.get("steps")
        if not isinstance(raw_steps, list):
            return "错误：steps 必须是数组"
        try:
            steps = [self._parse_step(item) for item in raw_steps]
            channel = str(kwargs.get("channel") or "").strip()
            chat_id = str(kwargs.get("chat_id") or "").strip()
            session_key = f"{channel}:{chat_id}" if channel and chat_id else ""
            workflow = self.runtime.create_workflow(
                name=str(kwargs.get("name") or ""),
                goal=str(kwargs.get("goal") or ""),
                steps=steps,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                context=dict(kwargs.get("context") or {}),
                auto_start=bool(kwargs.get("auto_start", True)),
            )
        except (TypeError, ValueError) as exc:
            return f"错误：{exc}"
        self.runtime.wake()
        return _json_text(
            {
                "created": True,
                "task_id": workflow.id,
                "short_id": workflow.id[:8],
                "name": workflow.name,
                "status": workflow.status.value,
                "steps": [
                    {
                        "id": step.id,
                        "kind": step.kind.value,
                        "depends_on": list(step.depends_on),
                        "allowed_tools": list(step.allowed_tools),
                    }
                    for step in workflow.steps
                ],
                "message": "任务已进入统一任务中心，将按依赖执行并在需要时联系用户。",
            }
        )

    @staticmethod
    def _parse_step(raw: Any) -> StepSpec:
        if not isinstance(raw, dict):
            raise ValueError("每个 step 必须是对象")
        depends_on = raw.get("depends_on") or []
        if not isinstance(depends_on, list):
            raise ValueError("depends_on 必须是数组")
        input_value = raw.get("input") or {}
        if not isinstance(input_value, dict):
            raise ValueError("step.input 必须是对象")
        allowed_tools = raw.get("allowed_tools") or []
        if not isinstance(allowed_tools, list):
            raise ValueError("allowed_tools 必须是数组")
        return StepSpec(
            id=str(raw.get("id") or "").strip(),
            title=str(raw.get("title") or "").strip(),
            description=str(raw.get("description") or "").strip(),
            kind=StepKind(str(raw.get("kind") or StepKind.AGENT.value)),
            depends_on=tuple(
                str(item).strip() for item in depends_on if str(item).strip()
            ),
            max_attempts=int(raw.get("max_attempts", 2)),
            input=dict(input_value),
            executor=StepExecutor(str(raw.get("executor") or StepExecutor.AGENT.value)),
            profile=str(raw.get("profile") or "research"),
            allowed_tools=tuple(
                dict.fromkeys(
                    name for item in allowed_tools if (name := str(item).strip())
                )
            ),
        )


class TaskManageTool(Tool):
    name = "task_manage"
    description = (
        "查看或控制小满任务中心里的持久化任务。用户回复了任务的问题时，使用 respond；"
        "用户同意/拒绝审批时，使用 approve；还可 list/get/events/start/retry/cancel。"
        "当用户问任务进度、后台任务状态、说“继续”“同意”“不同意”或补充信息时，"
        "应先用 list/get 找到当前会话中的任务，再执行对应动作。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list",
                    "get",
                    "events",
                    "start",
                    "respond",
                    "approve",
                    "retry",
                    "cancel",
                ],
            },
            "task_id": {
                "type": "string",
                "description": "完整任务 ID 或唯一前缀",
            },
            "step_id": {"type": "string", "description": "步骤 ID"},
            "status": {
                "type": "string",
                "enum": [
                    "draft",
                    "running",
                    "waiting",
                    "blocked",
                    "succeeded",
                    "failed",
                    "cancelled",
                ],
                "description": "list 时的可选状态过滤",
            },
            "response": {
                "type": "string",
                "description": "respond 时保存的用户回答",
            },
            "approved": {
                "type": "boolean",
                "description": "approve 时的审批结论",
            },
            "note": {
                "type": "string",
                "description": "审批说明或取消原因",
            },
            "scope": {
                "type": "string",
                "enum": ["current_session", "all"],
                "description": "list 范围，默认当前会话",
            },
        },
        "required": ["action"],
    }

    def __init__(self, runtime: WorkflowRuntime) -> None:
        self.runtime = runtime

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action") or "").strip()
        task_id = str(kwargs.get("task_id") or "").strip()
        step_id = str(kwargs.get("step_id") or "").strip()
        try:
            if action == "list":
                channel = str(kwargs.get("channel") or "").strip()
                chat_id = str(kwargs.get("chat_id") or "").strip()
                session_key = (
                    None
                    if kwargs.get("scope") == "all" or not channel or not chat_id
                    else f"{channel}:{chat_id}"
                )
                workflows = self.runtime.store.list_workflows(
                    status=str(kwargs.get("status") or "") or None,
                    session_key=session_key,
                    limit=30,
                )
                return _json_text(
                    [
                        {
                            "id": item.id,
                            "short_id": item.id[:8],
                            "name": item.name,
                            "status": item.status.value,
                            "updated_at": item.updated_at,
                            "waiting_steps": [
                                step.id
                                for step in item.steps
                                if step.status.value == "waiting"
                            ],
                            "failed_steps": [
                                step.id
                                for step in item.steps
                                if step.status.value == "failed"
                            ],
                        }
                        for item in workflows
                    ]
                )

            if not task_id:
                return "错误：该 action 需要 task_id"
            if action == "get":
                workflow = self.runtime.store.require_workflow(task_id)
                return _json_text(workflow.to_dict())
            if action == "events":
                events = self.runtime.store.list_events(task_id, limit=50)
                return _json_text([event.to_dict() for event in reversed(events)])
            if action == "start":
                workflow = self.runtime.store.start_workflow(task_id)
            elif action == "respond":
                if not step_id or not str(kwargs.get("response") or "").strip():
                    return "错误：respond 需要 step_id 和 response"
                workflow = self.runtime.store.respond_to_step(
                    task_id,
                    step_id,
                    response=str(kwargs.get("response") or "").strip(),
                )
            elif action == "approve":
                if not step_id or "approved" not in kwargs:
                    return "错误：approve 需要 step_id 和 approved"
                workflow = self.runtime.store.approve_step(
                    task_id,
                    step_id,
                    approved=bool(kwargs.get("approved")),
                    note=str(kwargs.get("note") or "").strip(),
                )
            elif action == "retry":
                if not step_id:
                    return "错误：retry 需要 step_id"
                workflow = self.runtime.store.retry_step(task_id, step_id)
            elif action == "cancel":
                workflow = await self.runtime.cancel_workflow(
                    task_id,
                    reason=str(kwargs.get("note") or "").strip(),
                )
            else:
                return f"错误：不支持的 action {action!r}"
        except (TypeError, ValueError) as exc:
            return f"错误：{exc}"
        self.runtime.wake()
        return _json_text(
            {
                "task_id": workflow.id,
                "short_id": workflow.id[:8],
                "status": workflow.status.value,
                "revision": workflow.revision,
                "steps": [
                    {"id": step.id, "status": step.status.value}
                    for step in workflow.steps
                ],
            }
        )
