from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from core.workflow.models import (
    StepSpec,
    WorkflowEvent,
    WorkflowInstance,
    WorkflowStatus,
    WorkflowStep,
)


class WorkflowStorePort(Protocol):
    """Persistence boundary consumed by the workflow application runtime."""

    def close(self) -> None: ...

    def create_workflow(
        self,
        *,
        name: str,
        goal: str,
        steps: Sequence[StepSpec],
        session_key: str,
        channel: str,
        chat_id: str,
        trace_id: str = "",
        context: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> WorkflowInstance: ...

    def get_workflow(self, workflow_id: str) -> WorkflowInstance | None: ...

    def require_workflow(self, workflow_id: str) -> WorkflowInstance: ...

    def list_workflows(
        self,
        *,
        status: str | None = None,
        session_key: str | None = None,
        limit: int = 20,
    ) -> list[WorkflowInstance]: ...

    def list_events(
        self,
        workflow_id: str,
        *,
        limit: int = 50,
    ) -> list[WorkflowEvent]: ...

    def start_workflow(self, workflow_id: str) -> WorkflowInstance: ...

    def cancel_workflow(
        self,
        workflow_id: str,
        *,
        reason: str = "",
    ) -> WorkflowInstance: ...

    def prepare_human_steps(self) -> list[tuple[WorkflowInstance, WorkflowStep]]: ...

    def claim_runnable_steps(
        self,
        *,
        limit: int = 3,
    ) -> list[tuple[WorkflowInstance, WorkflowStep]]: ...

    def claim_workflow_steps(
        self,
        workflow_id: str,
        *,
        limit: int = 3,
    ) -> list[tuple[WorkflowInstance, WorkflowStep]]: ...

    def complete_step(
        self,
        workflow_id: str,
        step_id: str,
        *,
        output: Any,
    ) -> WorkflowInstance: ...

    def fail_step(
        self,
        workflow_id: str,
        step_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
    ) -> WorkflowInstance: ...

    def respond_to_step(
        self,
        workflow_id: str,
        step_id: str,
        *,
        response: str,
    ) -> WorkflowInstance: ...

    def approve_step(
        self,
        workflow_id: str,
        step_id: str,
        *,
        approved: bool,
        note: str = "",
    ) -> WorkflowInstance: ...

    def retry_step(self, workflow_id: str, step_id: str) -> WorkflowInstance: ...

    def recover_interrupted(self) -> int: ...

    def list_unnotified_waiting(
        self,
        *,
        limit: int = 20,
    ) -> list[tuple[WorkflowInstance, WorkflowStep]]: ...

    def mark_step_notified(self, workflow_id: str, step_id: str) -> None: ...

    def list_unnotified_terminal(
        self,
        *,
        limit: int = 20,
    ) -> list[WorkflowInstance]: ...

    def mark_workflow_notified(
        self,
        workflow_id: str,
        status: WorkflowStatus,
    ) -> None: ...
