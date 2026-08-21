from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class WorkflowStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StepKind(StrEnum):
    AGENT = "agent"
    WAIT_USER = "wait_user"
    APPROVAL = "approval"


class StepExecutor(StrEnum):
    AGENT = "agent"
    SUBAGENT = "subagent"


TERMINAL_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    }
)
SUCCESS_STEP_STATUSES = frozenset({StepStatus.SUCCEEDED, StepStatus.SKIPPED})
TERMINAL_STEP_STATUSES = frozenset(
    {
        StepStatus.SUCCEEDED,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
        StepStatus.CANCELLED,
    }
)


@dataclass(frozen=True)
class StepSpec:
    id: str
    title: str
    description: str
    kind: StepKind = StepKind.AGENT
    depends_on: tuple[str, ...] = ()
    max_attempts: int = 2
    input: dict[str, Any] = field(default_factory=dict)
    executor: StepExecutor = StepExecutor.AGENT
    profile: str = "research"
    allowed_tools: tuple[str, ...] = ()


@dataclass
class WorkflowStep:
    workflow_id: str
    id: str
    position: int
    title: str
    description: str
    kind: StepKind
    status: StepStatus
    depends_on: tuple[str, ...]
    input: dict[str, Any]
    executor: StepExecutor
    profile: str
    allowed_tools: tuple[str, ...]
    output: Any
    error: str
    attempt_count: int
    max_attempts: int
    next_run_at: str | None
    notified_at: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position,
            "title": self.title,
            "description": self.description,
            "kind": self.kind.value,
            "status": self.status.value,
            "depends_on": list(self.depends_on),
            "input": self.input,
            "executor": self.executor.value,
            "profile": self.profile,
            "allowed_tools": list(self.allowed_tools),
            "output": self.output,
            "error": self.error,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "next_run_at": self.next_run_at,
            "notified_at": self.notified_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class WorkflowInstance:
    id: str
    name: str
    goal: str
    status: WorkflowStatus
    session_key: str
    channel: str
    chat_id: str
    context: dict[str, Any]
    revision: int
    error: str
    notified_status: str | None
    created_at: str
    updated_at: str
    trace_id: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)

    def to_dict(self, *, include_steps: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "status": self.status.value,
            "session_key": self.session_key,
            "channel": self.channel,
            "chat_id": self.chat_id,
            "trace_id": self.trace_id,
            "context": self.context,
            "revision": self.revision,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_steps:
            result["steps"] = [step.to_dict() for step in self.steps]
        return result


@dataclass(frozen=True)
class WorkflowEvent:
    id: int
    workflow_id: str
    revision: int
    event_type: str
    payload: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "revision": self.revision,
            "event_type": self.event_type,
            "payload": self.payload,
            "created_at": self.created_at,
        }
