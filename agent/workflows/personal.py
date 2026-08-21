from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

from agent.workflows.runtime import WorkflowRuntime
from core.workflow.models import StepKind, StepSpec, WorkflowInstance, WorkflowStatus


class RoutineKind(StrEnum):
    MORNING_BRIEF = "morning_brief"
    EVENING_REVIEW = "evening_review"
    CAPTURE_COMMITMENT = "capture_commitment"


class PersonalRoutineService:
    def __init__(self, runtime: WorkflowRuntime) -> None:
        self.runtime = runtime

    def create(
        self,
        routine: RoutineKind,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        local_date: str = "",
        timezone_name: str = "Asia/Shanghai",
        candidate: str = "",
    ) -> tuple[WorkflowInstance, bool]:
        date_value = self._local_date(local_date, timezone_name)
        if routine == RoutineKind.MORNING_BRIEF:
            name, goal, steps, context = self._morning(date_value, timezone_name)
        elif routine == RoutineKind.EVENING_REVIEW:
            name, goal, steps, context = self._evening(date_value, timezone_name)
        elif routine == RoutineKind.CAPTURE_COMMITMENT:
            if not candidate.strip():
                raise ValueError("capture_commitment requires candidate")
            name, goal, steps, context = self._commitment(
                candidate.strip(), date_value, timezone_name
            )
        else:
            raise ValueError(f"unsupported routine: {routine}")
        routine_key = self._routine_key(routine, session_key, date_value, candidate)
        existing = self._find_active(routine_key)
        if existing is not None:
            return existing, False
        context["routine"] = routine.value
        context["routine_key"] = routine_key
        workflow = self.runtime.create_workflow(
            name=name,
            goal=goal,
            steps=steps,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            context=context,
        )
        self.runtime.wake()
        return workflow, True

    def _find_active(self, routine_key: str) -> WorkflowInstance | None:
        terminal = {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }
        for workflow in self.runtime.store.list_workflows(limit=200):
            if (
                workflow.context.get("routine_key") == routine_key
                and workflow.status not in terminal
            ):
                return workflow
        return None

    @staticmethod
    def _morning(
        local_date: str, timezone_name: str
    ) -> tuple[str, str, list[StepSpec], dict[str, str]]:
        return (
            f"{local_date} 晨间简报",
            "形成符合当前承诺、健康状态和打扰边界的今日计划，并经用户确认后保存。",
            [
                StepSpec(
                    id="collect_context",
                    title="汇总今日上下文",
                    description=(
                        "使用 personal_context 读取 active 的 profile、commitment、daily_plan、"
                        "health_observation、check_in 和 notification_policy，整理今天的约束、缺失信息与优先事项。"
                        "同时结合系统上下文中的长期记忆；当需要确认用户以往的作息偏好、安排习惯或类似决定时，"
                        "使用 recall_memory 查找相关原始对话，不要把一次性对话误当成稳定偏好。"
                    ),
                ),
                StepSpec(
                    id="draft_plan",
                    title="生成今日建议",
                    description=(
                        "根据前置数据生成简洁晨间简报和按时间块排列的今日计划；"
                        "计划应尊重已确认的长期偏好、协作边界和历史纠正，"
                        "健康数据不足时明确说明，不作医疗推断。"
                    ),
                    depends_on=("collect_context",),
                ),
                StepSpec(
                    id="user_adjustment",
                    title="等待用户调整",
                    description="请确认今日计划，或告诉我需要调整的优先级、时间和训练安排。",
                    kind=StepKind.WAIT_USER,
                    depends_on=("draft_plan",),
                ),
                StepSpec(
                    id="approve_plan",
                    title="确认保存今日计划",
                    description="请确认是否按草案和你的调整保存今日计划。",
                    kind=StepKind.APPROVAL,
                    depends_on=("draft_plan", "user_adjustment"),
                ),
                StepSpec(
                    id="save_plan",
                    title="保存确认后的计划",
                    description=(
                        "结合草案与用户反馈形成最终计划，使用 personal_record 创建或更新 daily_plan，"
                        "最后返回今日三项核心事项和提醒边界。"
                    ),
                    depends_on=("draft_plan", "user_adjustment", "approve_plan"),
                    allowed_tools=("personal_record",),
                ),
            ],
            {"local_date": local_date, "timezone": timezone_name},
        )

    @staticmethod
    def _evening(
        local_date: str, timezone_name: str
    ) -> tuple[str, str, list[StepSpec], dict[str, str]]:
        return (
            f"{local_date} 晚间回顾",
            "回顾今天的完成情况、精力与训练反馈，并沉淀明天可使用的可靠上下文。",
            [
                StepSpec(
                    id="collect_day",
                    title="汇总今日记录",
                    description=(
                        "使用 personal_context 汇总今天的 daily_plan、commitment、health_observation 和 check_in，"
                        "只陈述已有证据并列出缺口；如需判断反复出现的习惯或用户曾经的明确取舍，"
                        "使用 recall_memory 核对相关原始对话。"
                    ),
                ),
                StepSpec(
                    id="ask_reflection",
                    title="获取主观反馈",
                    description="今天完成得怎么样？请补充精力、情绪、训练感受和未完成事项的原因。",
                    kind=StepKind.WAIT_USER,
                    depends_on=("collect_day",),
                ),
                StepSpec(
                    id="approve_store",
                    title="确认保存晚间回顾",
                    description="请确认是否将本次主观反馈整理并保存到个人记录。",
                    kind=StepKind.APPROVAL,
                    depends_on=("collect_day", "ask_reflection"),
                ),
                StepSpec(
                    id="review_and_store",
                    title="生成回顾并沉淀",
                    description=(
                        "结合记录和用户反馈生成晚间回顾；使用 personal_record 保存 check_in，"
                        "只有稳定且明确的信息才保存为 memory，临时状态要设置合理有效期；"
                        "不得用单日表现覆盖已有长期偏好。"
                    ),
                    depends_on=("collect_day", "ask_reflection", "approve_store"),
                    allowed_tools=("personal_record",),
                ),
            ],
            {"local_date": local_date, "timezone": timezone_name},
        )

    @staticmethod
    def _commitment(
        candidate: str, local_date: str, timezone_name: str,
    ) -> tuple[str, str, list[StepSpec], dict[str, str]]:
        return (
            "记录一项待办",
            "把用户想做的事整理成明确、可执行且经过确认的待办事项。",
            [
                StepSpec(
                    id="normalize",
                    title="整理待办",
                    description=(
                        "把候选文本整理为标题、下一步行动、预计用时、优先级和截止范围。"
                        "精确时间用带时区的 due_at；只有日期时用 due_date；上午、下午或晚上"
                        "分别用 due_period=morning、afternoon、evening。不要猜测用户没有提供的"
                        "具体时刻。"
                    ),
                    input={"candidate": candidate},
                ),
                StepSpec(
                    id="approve",
                    title="确认待办",
                    description="请确认是否将这项待办加入我的一天；也可以补充截止时间或修改内容。",
                    kind=StepKind.APPROVAL,
                    depends_on=("normalize",),
                ),
                StepSpec(
                    id="store",
                    title="保存待办",
                    description=(
                        "根据整理结果和用户确认内容，使用 personal_record 创建 commitment；"
                        "data 使用统一字段：state=open、next_action、priority、estimated_minutes、"
                        "energy、contexts、progress，以及适用的 due_at 或 due_date/due_period。"
                        "priority 只能是 urgent/high/normal/low，energy 只能是 low/medium/high，"
                        "contexts 只能从 any/neutral/home/leaving/bedtime/travel 中选择，progress 是"
                        " 0 到 1 的数字。"
                        "保留原始对话来源并避免重复创建。"
                    ),
                    depends_on=("normalize", "approve"),
                    allowed_tools=("personal_record",),
                ),
            ],
            {
                "candidate": candidate,
                "local_date": local_date,
                "timezone": timezone_name,
            },
        )

    @staticmethod
    def _local_date(local_date: str, timezone_name: str) -> str:
        if local_date:
            try:
                return datetime.strptime(local_date, "%Y-%m-%d").date().isoformat()
            except ValueError as exc:
                raise ValueError("local_date must be YYYY-MM-DD") from exc
        try:
            zone = ZoneInfo(timezone_name)
        except Exception as exc:
            raise ValueError(f"invalid timezone: {timezone_name}") from exc
        return datetime.now(zone).date().isoformat()

    @staticmethod
    def _routine_key(
        routine: RoutineKind,
        session_key: str,
        local_date: str,
        candidate: str,
    ) -> str:
        suffix = local_date
        if routine == RoutineKind.CAPTURE_COMMITMENT:
            suffix = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
        return f"{routine.value}:{session_key}:{suffix}"
