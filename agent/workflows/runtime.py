from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.tools.registry import ToolRegistry
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.runtime.execution_guard import ExecutionGuardConfig
from agent.runtime.execution_policy import IncompleteExecutionError
from agent.runtime.delegation import (
    CONTEXT_KEY,
    DelegationScope,
    DelegationBudgetExceeded,
    current_scope,
    new_scope,
)
from core.workflow.models import (
    SUCCESS_STEP_STATUSES,
    StepExecutor,
    StepKind,
    StepSpec,
    StepStatus,
    WorkflowInstance,
    WorkflowStatus,
    WorkflowStep,
)
from core.workflow.ports import WorkflowStorePort
from core.tracing import current_trace_id, record_trace_event, trace_root
from core.tracing.ports import TraceRecorder
from agent.runtime.reasoning_policy import (
    REASONING_EFFORTS,
    WORKFLOW_REASONING_CONTEXT_KEY,
)

logger = logging.getLogger(__name__)

_READ_ONLY_RISK = "read-only"
_APPROVAL_REQUIRED_SUBAGENT_PROFILES = frozenset({"scripting", "general"})
_RESERVED_WORKFLOW_TOOLS = frozenset({"task_create", "task_manage", "message_push"})


class WorkflowGraphState(TypedDict):
    workflow_id: str
    route: str
    cycle: int


class AgentLoopPort(Protocol):
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        busy_session_key: str | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        omit_user_turn: bool = False,
        skip_post_memory: bool = False,
        skip_memory_retrieval: bool = False,
        stream_events: bool = False,
        disabled_tools: list[str] | None = None,
        trace_id: str = "",
        trace_flow: str = "workflow",
        trace_title: str = "",
        reasoning_effort: str = "",
        require_completed: bool = False,
    ) -> str: ...


class SubagentExecutorPort(Protocol):
    async def execute(
        self,
        *,
        task: str,
        label: str | None,
        profile: str = "research",
        execution_id: str | None = None,
        reasoning_effort: str = "",
    ) -> str: ...


class WorkflowNotificationPort(Protocol):
    async def execute(self, **kwargs: Any) -> str: ...


class WorkflowRuntime:
    """Durable workflow worker that executes runnable agent steps."""

    def __init__(
        self,
        *,
        store: WorkflowStorePort,
        agent_loop_provider: Callable[[], AgentLoopPort | None],
        push_tool: WorkflowNotificationPort,
        tool_registry: ToolRegistry | None = None,
        subagent_executor: SubagentExecutorPort | None = None,
        trace_recorder: TraceRecorder | None = None,
        graph_runtime: LangGraphRuntime | None = None,
        poll_interval_seconds: float = 1.0,
        max_concurrency: int = 2,
        step_timeout_seconds: float = 180.0,
        max_subagent_steps: int = 4,
        delegation_guard: ExecutionGuardConfig | None = None,
    ) -> None:
        self.store = store
        self._agent_loop_provider = agent_loop_provider
        self._push_tool = push_tool
        self._tool_registry = tool_registry or ToolRegistry()
        self.subagent_executor = subagent_executor
        self.trace_recorder = trace_recorder
        self._graph_runtime = graph_runtime or LangGraphRuntime()
        self._graph: Any | None = None
        self._graph_lock = asyncio.Lock()
        self._poll_interval_seconds = max(0.1, poll_interval_seconds)
        self._max_concurrency = max(1, max_concurrency)
        self._step_timeout_seconds = max(10.0, float(step_timeout_seconds))
        self._max_subagent_steps = max(1, int(max_subagent_steps))
        self._delegation_guard = (
            delegation_guard or ExecutionGuardConfig()
        ).normalized()
        self._execution_slots = asyncio.Semaphore(self._max_concurrency)
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._active: dict[asyncio.Task[None], str] = {}
        self._running = False
        self._closed = False
        self._stopped = asyncio.Event()
        self._stopped.set()

    def create_workflow(
        self,
        *,
        name: str,
        goal: str,
        steps: Sequence[StepSpec],
        session_key: str,
        channel: str,
        chat_id: str,
        context: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> WorkflowInstance:
        """Validate execution permissions before persisting a workflow."""

        self._validate_step_permissions(steps)
        context = dict(context or {})
        policy = context.get(CONTEXT_KEY)
        if not isinstance(policy, dict):
            policy = new_scope(
                self._graph_runtime.delegation_ledger,
                self._delegation_guard,
                allowed=self._delegation_guard.autonomous_delegation,
            ).metadata()
            context[CONTEXT_KEY] = policy
        self._validate_delegation(steps, policy)
        return self.store.create_workflow(
            name=name,
            goal=goal,
            steps=steps,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            trace_id=current_trace_id(),
            context=context,
            auto_start=auto_start,
        )

    @staticmethod
    def _validate_delegation(steps: Sequence[StepSpec], policy: dict[str, Any]) -> None:
        if policy.get("allowed") is True:
            return
        if any(step.executor == StepExecutor.SUBAGENT for step in steps):
            raise ValueError(
                "自主多 Agent 已关闭，不能创建 SubAgent。请留在主 AgentLoop 执行，或请用户在聊天框开启多 Agent 后重新创建任务。"
            )
        # Keep durable sequential plans available, but don't bypass the switch
        # by naming concurrent children 'agent' rather than 'subagent'.
        ancestors: dict[str, set[str]] = {}
        by_id = {step.id: step for step in steps}

        def dependencies(key: str, visiting: set[str]) -> set[str]:
            if key in visiting or key not in by_id:
                return set()
            found = set(by_id[key].depends_on)
            for dep in by_id[key].depends_on:
                found |= dependencies(dep, visiting | {key})
            return found

        automatic = [step for step in steps if step.kind == StepKind.AGENT]
        for step in automatic:
            ancestors[step.id] = dependencies(step.id, set())
        for index, step in enumerate(automatic):
            for other in automatic[index + 1 :]:
                if (
                    step.id not in ancestors[other.id]
                    and other.id not in ancestors[step.id]
                ):
                    raise ValueError(
                        "自主多 Agent 已关闭，不允许并行自动步骤。请在主循环完成独立工作，或开启多 Agent。"
                    )

    def _validate_step_permissions(self, steps: Sequence[StepSpec]) -> None:
        subagent_steps = sum(
            1
            for step in steps
            if step.kind == StepKind.AGENT and step.executor == StepExecutor.SUBAGENT
        )
        if subagent_steps > self._max_subagent_steps:
            raise ValueError(
                "单个 Workflow 最多允许 "
                f"{self._max_subagent_steps} 个 SubAgent 步骤，当前为 {subagent_steps}"
            )
        tool_risks = self._tool_risks()
        by_id = {step.id: step for step in steps}
        for step in steps:
            allowed = set(step.allowed_tools)
            reserved = sorted(allowed & _RESERVED_WORKFLOW_TOOLS)
            if reserved:
                raise ValueError(f"步骤 {step.id} 不可放开任务保留工具: {reserved}")
            unknown = sorted(allowed - tool_risks.keys())
            if unknown:
                raise ValueError(f"步骤 {step.id} 引用了不存在的工具: {unknown}")
            if allowed and (
                step.kind != StepKind.AGENT or step.executor != StepExecutor.AGENT
            ):
                raise ValueError(
                    f"步骤 {step.id} 只有 executor=agent 的自动步骤可设置 allowed_tools"
                )
            side_effect_tools = sorted(
                name for name in allowed if tool_risks[name] != _READ_ONLY_RISK
            )
            has_direct_approval = any(
                (dependency := by_id.get(dependency_id)) is not None
                and dependency.kind == StepKind.APPROVAL
                for dependency_id in step.depends_on
            )
            if side_effect_tools and not has_direct_approval:
                raise ValueError(
                    f"步骤 {step.id} 放开非只读工具 {side_effect_tools} 时，"
                    "必须直接依赖 approval 步骤"
                )
            profile = str(step.profile or "research").strip().lower()
            if (
                step.kind == StepKind.AGENT
                and step.executor == StepExecutor.SUBAGENT
                and profile in _APPROVAL_REQUIRED_SUBAGENT_PROFILES
                and not has_direct_approval
            ):
                raise ValueError(
                    f"步骤 {step.id} 使用 subagent profile={profile} 时，"
                    "必须直接依赖 approval 步骤"
                )

    def replan_workflow(
        self,
        workflow_id: str,
        *,
        remaining_steps: Sequence[StepSpec],
        expected_revision: int,
        reason: str,
    ) -> WorkflowInstance:
        """Validate and replace the unresolved tail of a persisted plan."""

        current = self.store.require_workflow(workflow_id)
        self._validate_delegation(
            remaining_steps, dict(current.context.get(CONTEXT_KEY) or {})
        )
        preserved = [
            step for step in current.steps if step.status in SUCCESS_STEP_STATUSES
        ]
        preserved_specs = [
            StepSpec(
                id=step.id,
                title=step.title,
                description=step.description,
                kind=step.kind,
                depends_on=step.depends_on,
                max_attempts=step.max_attempts,
                input=step.input,
                executor=step.executor,
                profile=step.profile,
                allowed_tools=step.allowed_tools,
            )
            for step in preserved
        ]
        combined = [*preserved_specs, *remaining_steps]
        self._validate_step_permissions(combined)

        # A completed approval only covered the old plan. Any newly introduced
        # side effect or privileged child must carry a new approval step.
        new_by_id = {step.id: step for step in remaining_steps}
        tool_risks = self._tool_risks()
        for step in remaining_steps:
            requires_fresh_approval = any(
                tool_risks.get(name, _READ_ONLY_RISK) != _READ_ONLY_RISK
                for name in step.allowed_tools
            ) or (
                step.kind == StepKind.AGENT
                and step.executor == StepExecutor.SUBAGENT
                and step.profile in _APPROVAL_REQUIRED_SUBAGENT_PROFILES
            )
            if requires_fresh_approval and not any(
                (dependency := new_by_id.get(dependency_id)) is not None
                and dependency.kind == StepKind.APPROVAL
                for dependency_id in step.depends_on
            ):
                raise ValueError(
                    f"重规划步骤 {step.id} 扩大了权限，必须依赖本次计划中新建的 approval 步骤"
                )
        return self.store.replan_workflow(
            current.id,
            remaining_steps=remaining_steps,
            expected_revision=expected_revision,
            reason=reason,
        )

    def _tool_risks(self) -> dict[str, str]:
        return {
            document.name: document.risk
            for document in self._tool_registry.get_documents()
        }

    def wake(self) -> None:
        self._wake.set()

    async def run(self) -> None:
        if self._running:
            await self._stop.wait()
            return
        self._running = True
        self._stopped.clear()
        recovered = self.store.recover_interrupted()
        if recovered:
            logger.info("workflow runtime recovered %d interrupted step(s)", recovered)
        try:
            while not self._stop.is_set():
                for workflow in self.store.list_workflows(
                    status=WorkflowStatus.RUNNING.value,
                    limit=100,
                ):
                    self._ensure_graph_task(workflow.id)
                await self._deliver_terminal_notifications()
                self._wake.clear()
                await self._wake.wait()
        finally:
            self._running = False
            active = list(self._active)
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            self._active.clear()
            self._stopped.set()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.set()
        active = list(self._active)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        if self._running:
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=5.0)
            except TimeoutError:
                logger.warning("workflow runtime did not stop before store close")
        close_subagent = getattr(self.subagent_executor, "aclose", None)
        if callable(close_subagent):
            close_result = close_subagent()
            if inspect.isawaitable(close_result):
                await close_result
        await self._graph_runtime.aclose()
        self.store.close()

    async def _compiled_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        async with self._graph_lock:
            if self._graph is not None:
                return self._graph
            builder = StateGraph(WorkflowGraphState)
            builder.add_node("advance", self._advance_node)
            builder.add_node("wait_human", self._wait_human_node)
            builder.add_node("delay", self._delay_node)
            builder.add_edge(START, "advance")
            builder.add_conditional_edges(
                "advance",
                lambda state: state["route"],
                {"wait": "wait_human", "delay": "delay", "done": END},
            )
            builder.add_edge("wait_human", "advance")
            builder.add_edge("delay", "advance")
            self._graph = builder.compile(
                checkpointer=await self._graph_runtime.checkpointer(),
                store=self._graph_runtime.store,
            )
            return self._graph

    def _ensure_graph_task(self, workflow_id: str) -> None:
        if any(
            active_workflow_id == workflow_id and not task.done()
            for task, active_workflow_id in self._active.items()
        ):
            return
        task = asyncio.create_task(
            self._drive_workflow(workflow_id),
            name=f"workflow-graph:{workflow_id[:8]}",
        )
        self._active[task] = workflow_id
        task.add_done_callback(self._on_step_task_done)

    async def _drive_workflow(self, workflow_id: str) -> None:
        graph = await self._compiled_graph()
        config = {
            "configurable": {"thread_id": f"workflow:{workflow_id}"},
            "recursion_limit": 10_000,
        }
        snapshot = await graph.aget_state(config)
        workflow = self.store.require_workflow(workflow_id)
        if snapshot.next:
            if workflow.status != WorkflowStatus.RUNNING:
                return
            graph_input: WorkflowGraphState | Command = Command(
                resume={"workflow_id": workflow_id, "action": "continue"}
            )
        else:
            graph_input = WorkflowGraphState(
                workflow_id=workflow_id,
                route="delay",
                cycle=0,
            )
        await graph.ainvoke(graph_input, config, durability="sync")

    async def _advance_node(self, state: WorkflowGraphState) -> dict[str, Any]:
        workflow_id = state["workflow_id"]
        # Projection updates are atomic; graph checkpoints own execution
        # position while this table remains query-friendly for tools and UI.
        self.store.prepare_human_steps()
        workflow = self.store.require_workflow(workflow_id)
        if workflow.status in {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
        }:
            return {"route": "done", "cycle": state["cycle"] + 1}
        claimed = self.store.claim_workflow_steps(
            workflow_id,
            limit=self._max_concurrency,
        )
        if claimed:
            await asyncio.gather(
                *(self._execute_claimed(workflow, step) for workflow, step in claimed)
            )
        latest = self.store.require_workflow(workflow_id)
        if latest.status == WorkflowStatus.WAITING:
            route = "wait"
        elif latest.status in {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
        }:
            route = "done"
        else:
            route = "delay"
        return {"route": route, "cycle": state["cycle"] + 1}

    async def _execute_claimed(
        self,
        workflow: WorkflowInstance,
        step: WorkflowStep,
    ) -> None:
        async with self._execution_slots:
            await self._execute_step(workflow, step)

    async def _wait_human_node(self, state: WorkflowGraphState) -> dict[str, Any]:
        await self._deliver_waiting_prompts()
        workflow = self.store.require_workflow(state["workflow_id"])
        if workflow.status == WorkflowStatus.WAITING:
            _ = interrupt(
                {
                    "workflow_id": workflow.id,
                    "waiting_steps": [
                        step.id
                        for step in workflow.steps
                        if step.status == StepStatus.WAITING
                    ],
                }
            )
        return {"route": "delay"}

    async def _delay_node(self, state: WorkflowGraphState) -> dict[str, Any]:
        del state
        await asyncio.sleep(self._poll_interval_seconds)
        return {"route": "delay"}

    async def cancel_workflow(
        self,
        workflow_id: str,
        *,
        reason: str = "",
    ) -> WorkflowInstance:
        """Persist cancellation and stop every active step owned by the workflow."""

        workflow = self.store.cancel_workflow(workflow_id, reason=reason)
        active = [
            task
            for task, active_workflow_id in self._active.items()
            if active_workflow_id == workflow.id
        ]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        self.wake()
        cancelled = self.store.require_workflow(workflow.id)
        _finish_trace(
            self.trace_recorder,
            cancelled.trace_id or f"tr_workflow_{cancelled.id}",
            status="cancelled",
            metadata={"workflow_status": cancelled.status.value},
        )
        return cancelled

    def _on_step_task_done(self, task: asyncio.Task[None]) -> None:
        self._active.pop(task, None)
        if not task.cancelled():
            try:
                error = task.exception()
            except Exception:
                error = None
            if error is not None:
                logger.error(
                    "workflow step task stopped unexpectedly",
                    exc_info=(type(error), error, error.__traceback__),
                )
        self.wake()

    async def _execute_step(
        self, workflow: WorkflowInstance, step: WorkflowStep
    ) -> None:
        latest = self.store.require_workflow(workflow.id)
        # Workflows created before trace persistence still need one stable ID
        # across all of their later steps.
        trace_id = latest.trace_id or f"tr_workflow_{latest.id}"
        started_wall = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        with trace_root(
            self.trace_recorder,
            trace_id=trace_id,
            flow="workflow",
            session_key=latest.session_key,
            title=latest.name,
            metadata={"workflow_id": latest.id, "goal": latest.goal[:500]},
            finish=False,
        ):
            try:
                await self._execute_step_bound(workflow, step)
            except asyncio.CancelledError:
                _finish_trace(self.trace_recorder, trace_id, status="interrupted")
                raise
            resolved = self.store.require_workflow(workflow.id)
            current = next(item for item in resolved.steps if item.id == step.id)
            record_trace_event(
                category="workflow",
                name=current.id,
                summary=f"任务步骤「{current.title}」{_step_status_label(current.status)}",
                status=(
                    "completed"
                    if current.status in {StepStatus.SUCCEEDED, StepStatus.SKIPPED}
                    else current.status.value
                ),
                started_at=started_wall,
                duration_ms=int((time.perf_counter() - started) * 1000),
                payload={
                    "workflow_id": resolved.id,
                    "step_id": current.id,
                    "executor": current.executor.value,
                    "attempt": current.attempt_count,
                    "depends_on": list(current.depends_on),
                    "error": current.error[:500],
                },
            )
        if resolved.status in {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
        }:
            _finish_trace(
                self.trace_recorder,
                trace_id,
                status=(
                    "completed"
                    if resolved.status == WorkflowStatus.SUCCEEDED
                    else resolved.status.value
                ),
                metadata={"workflow_status": resolved.status.value},
            )

    async def _execute_step_bound(
        self, workflow: WorkflowInstance, step: WorkflowStep
    ) -> None:
        try:
            latest = self.store.require_workflow(workflow.id)
            current = next((item for item in latest.steps if item.id == step.id), None)
            if current is None or current.status != StepStatus.RUNNING:
                return
            prompt = self._build_step_prompt(latest, current)
            result = await asyncio.wait_for(
                self._run_step_executor(latest, current, prompt),
                timeout=self._step_timeout_seconds,
            )
            output = result.strip()
            if not output:
                raise IncompleteExecutionError("empty_result")
            self.store.complete_step(latest.id, current.id, output=output)
        except asyncio.CancelledError:
            raise
        except (DelegationBudgetExceeded, IncompleteExecutionError) as exc:
            self.store.fail_step(
                workflow.id,
                step.id,
                error=str(exc),
                retry_delay_seconds=0,
                retryable=False,
            )
        except TimeoutError:
            logger.warning(
                "workflow step timed out workflow=%s step=%s timeout=%ss",
                workflow.id,
                step.id,
                self._step_timeout_seconds,
            )
            delay = min(60.0, 2.0 ** max(1, step.attempt_count))
            self.store.fail_step(
                workflow.id,
                step.id,
                error=(
                    "步骤执行超时（"
                    f"{self._step_timeout_seconds:g}s），已停止本次尝试"
                ),
                retry_delay_seconds=delay,
            )
        except Exception as exc:
            logger.exception(
                "workflow step failed workflow=%s step=%s", workflow.id, step.id
            )
            delay = min(60.0, 2.0 ** max(1, step.attempt_count))
            self.store.fail_step(
                workflow.id,
                step.id,
                error=str(exc),
                retry_delay_seconds=delay,
            )

    async def _run_step_executor(
        self,
        workflow: WorkflowInstance,
        step: WorkflowStep,
        prompt: str,
    ) -> str:
        policy = workflow.context.get(CONTEXT_KEY) or {}
        if step.executor == StepExecutor.SUBAGENT and policy.get("allowed") is not True:
            raise RuntimeError(
                "任务没有多 Agent 授权，未启动 SubAgent。请开启后重新创建。"
            )
        key = str(policy.get("budget_id") or "")
        # Legacy workflows receive a stable task budget; new workflows use
        # their origin turn's ledger. Never reset an existing budget on retry.
        if not key:
            scope = new_scope(
                self._graph_runtime.delegation_ledger,
                self._delegation_guard,
                key=f"workflow:{workflow.id}",
            )
        else:
            scope = DelegationScope(self._graph_runtime.delegation_ledger, key)
        scope.ledger.claim_child(scope.key, f"{workflow.id}:{step.id}")
        token = current_scope.set(DelegationScope(scope.ledger, scope.key, False, True))
        try:
            return await self._run_step_executor_inner(workflow, step, prompt)
        finally:
            current_scope.reset(token)

    async def _run_step_executor_inner(
        self,
        workflow: WorkflowInstance,
        step: WorkflowStep,
        prompt: str,
    ) -> str:
        if step.executor == StepExecutor.SUBAGENT:
            if self.subagent_executor is None:
                raise RuntimeError("Subagent 步骤执行器未启用")
            if (
                step.profile in _APPROVAL_REQUIRED_SUBAGENT_PROFILES
                and not self._has_approved_dependency(workflow, step)
            ):
                raise RuntimeError(
                    f"步骤 {step.id} 使用 subagent profile={step.profile} "
                    "时缺少已批准的直接 approval 依赖"
                )
            reasoning_effort = self._workflow_reasoning_effort(workflow)
            execute_kwargs: dict[str, Any] = {
                "task": prompt,
                "label": f"{workflow.name}-{step.title}"[:30],
                "profile": step.profile,
                "execution_id": f"task-{workflow.id[:12]}-{step.id}",
            }
            if reasoning_effort:
                execute_kwargs["reasoning_effort"] = reasoning_effort
            return await self.subagent_executor.execute(
                **execute_kwargs,
            )

        loop = self._agent_loop_provider()
        if loop is None:
            raise RuntimeError("AgentLoop 尚未就绪")
        reasoning_effort = self._workflow_reasoning_effort(workflow)
        process_kwargs: dict[str, Any] = {
            "session_key": f"workflow:{workflow.id}:step:{step.id}",
            "busy_session_key": workflow.session_key or None,
            "channel": workflow.channel or "workflow",
            "chat_id": workflow.chat_id or workflow.id,
            "omit_user_turn": True,
            "skip_post_memory": True,
            "stream_events": False,
            "disabled_tools": self._disabled_tools_for_step(workflow, step),
        }
        # Third-party/test loop adapters may implement the older port. Use the
        # strict completion contract when available, while preserving the port
        # compatibility required by existing integrations.
        try:
            if "require_completed" in inspect.signature(loop.process_direct).parameters:
                process_kwargs["require_completed"] = True
        except (TypeError, ValueError):
            pass
        if reasoning_effort:
            process_kwargs["reasoning_effort"] = reasoning_effort
        return await loop.process_direct(
            prompt,
            **process_kwargs,
        )

    @staticmethod
    def _workflow_reasoning_effort(workflow: WorkflowInstance) -> str:
        """Read only the trusted, persisted parent choice for this workflow."""

        value = (
            str((workflow.context or {}).get(WORKFLOW_REASONING_CONTEXT_KEY) or "")
            .strip()
            .lower()
        )
        return value if value in REASONING_EFFORTS else ""

    def _disabled_tools_for_step(
        self,
        workflow: WorkflowInstance,
        step: WorkflowStep,
    ) -> list[str]:
        tool_risks = self._tool_risks()
        allowed = set(step.allowed_tools)
        reserved = sorted(allowed & _RESERVED_WORKFLOW_TOOLS)
        if reserved:
            raise RuntimeError(f"步骤 {step.id} 试图放开任务保留工具: {reserved}")
        unknown = sorted(allowed - tool_risks.keys())
        if unknown:
            raise RuntimeError(f"步骤 {step.id} 引用了不存在的工具: {unknown}")

        allowed_side_effects = {
            name for name in allowed if tool_risks[name] != _READ_ONLY_RISK
        }
        if allowed_side_effects and not self._has_approved_dependency(workflow, step):
            raise RuntimeError(
                f"步骤 {step.id} 的非只读工具授权缺少已批准的直接 approval 依赖"
            )

        disabled = {
            name
            for name, risk in tool_risks.items()
            if risk != _READ_ONLY_RISK and name not in allowed_side_effects
        }
        disabled.update(_RESERVED_WORKFLOW_TOOLS)
        ordered = [
            name
            for name in self._tool_registry.get_registered_order()
            if name in disabled
        ]
        ordered.extend(sorted(disabled - set(ordered)))
        return ordered

    @staticmethod
    def _has_approved_dependency(
        workflow: WorkflowInstance,
        step: WorkflowStep,
    ) -> bool:
        by_id = {item.id: item for item in workflow.steps}
        return any(
            (dependency := by_id.get(dependency_id)) is not None
            and dependency.kind == StepKind.APPROVAL
            and dependency.status == StepStatus.SUCCEEDED
            and isinstance(dependency.output, dict)
            and dependency.output.get("approved") is True
            for dependency_id in step.depends_on
        )

    def _build_step_prompt(self, workflow: WorkflowInstance, step: WorkflowStep) -> str:
        dependency_outputs: list[str] = []
        by_id = {item.id: item for item in workflow.steps}
        for dependency_id in step.depends_on:
            dependency = by_id.get(dependency_id)
            if dependency is None:
                continue
            dependency_outputs.append(
                f"- {dependency.title} ({dependency.id}): {self._compact_value(dependency.output, 1800)}"
            )
        # The parent reasoning choice is an execution concern, not task data;
        # keep it durable for retries without leaking it into the model prompt.
        prompt_context = dict(workflow.context or {})
        prompt_context.pop(WORKFLOW_REASONING_CONTEXT_KEY, None)
        prompt_context.pop(CONTEXT_KEY, None)
        context = self._compact_value(prompt_context, 1800)
        inputs = self._compact_value(step.input, 1800)
        previous = "\n".join(dependency_outputs) or "（无前置步骤）"
        return (
            "你正在执行小满任务中心的一个持久化步骤。只完成当前步骤，不要重做整个计划。\n"
            f"任务: {workflow.name}\n"
            f"总目标: {workflow.goal}\n"
            f"当前步骤: {step.title} ({step.id})\n"
            f"步骤要求: {step.description}\n"
            f"步骤输入: {inputs}\n"
            f"任务上下文: {context}\n"
            f"前置步骤结果:\n{previous}\n\n"
            "可以调用当前可用工具完成任务。不要创建新任务或后台子任务，不要主动发消息给用户。"
            "若步骤包含外部写操作，只执行步骤描述明确授权的内容。"
            "最后返回清晰、可供后续步骤使用的结果摘要；不要询问用户。"
        )

    async def _deliver_waiting_prompts(self) -> None:
        for workflow, step in self.store.list_unnotified_waiting(limit=20):
            if not workflow.channel or not workflow.chat_id:
                self.store.mark_step_notified(workflow.id, step.id)
                continue
            if step.kind == StepKind.APPROVAL:
                action = "请回复是否同意，并可补充说明。"
                permission_notice = self._approval_permission_notice(workflow, step)
            else:
                action = "请回复所需信息，我会从这里继续执行。"
                permission_notice = ""
            evidence = self._human_step_context(workflow, step)
            message = (
                f"小满正在执行「{workflow.name}」，现在需要你的确认。\n\n"
                f"{evidence}{step.description}{permission_notice}\n\n{action}\n"
                f"任务编号：{workflow.id[:8]}，步骤：{step.id}"
            )
            if await self._push(workflow, message):
                self.store.mark_step_notified(workflow.id, step.id)

    def _approval_permission_notice(
        self,
        workflow: WorkflowInstance,
        approval: WorkflowStep,
    ) -> str:
        tool_risks = self._tool_risks()
        downstream: list[tuple[str, list[str]]] = []
        privileged_subagents: list[tuple[str, str]] = []
        for step in workflow.steps:
            if approval.id not in step.depends_on:
                continue
            side_effect_tools = sorted(
                name
                for name in step.allowed_tools
                if tool_risks.get(name) != _READ_ONLY_RISK
            )
            if side_effect_tools:
                downstream.append((step.title, side_effect_tools))
            if (
                step.executor == StepExecutor.SUBAGENT
                and step.profile in _APPROVAL_REQUIRED_SUBAGENT_PROFILES
            ):
                privileged_subagents.append((step.title, step.profile))
        if not downstream and not privileged_subagents:
            return "\n\n本次审批不会开放任何非只读工具。"
        detail_lines = [
            f"- {title}：{', '.join(tool_names)}" for title, tool_names in downstream
        ]
        detail_lines.extend(
            f"- {title}：subagent profile={profile}（具执行或网络能力）"
            for title, profile in privileged_subagents
        )
        details = "\n".join(detail_lines)
        return (
            "\n\n批准后仅为以下直接后续步骤开放权限：\n"
            f"{details}\n"
            "未列出的写入或外部副作用工具仍保持禁用。"
        )

    def _human_step_context(
        self, workflow: WorkflowInstance, step: WorkflowStep
    ) -> str:
        by_id = {item.id: item for item in workflow.steps}
        parts: list[str] = []
        for dependency_id in step.depends_on:
            dependency = by_id.get(dependency_id)
            if dependency is None or dependency.output in (None, "", {}):
                continue
            parts.append(
                f"{dependency.title}：\n{self._compact_value(dependency.output, 2400)}"
            )
        if not parts:
            return ""
        return "\n\n".join(parts) + "\n\n"

    async def _deliver_terminal_notifications(self) -> None:
        for workflow in self.store.list_unnotified_terminal(limit=20):
            if workflow.status == WorkflowStatus.SUCCEEDED:
                message = self._success_message(workflow)
            else:
                failed = next(
                    (
                        step
                        for step in workflow.steps
                        if step.status == StepStatus.FAILED
                    ),
                    None,
                )
                detail = failed.error if failed is not None else workflow.error
                step_text = (
                    f"，停在步骤「{failed.title}」" if failed is not None else ""
                )
                message = (
                    f"「{workflow.name}」暂时无法继续{step_text}。\n"
                    f"原因：{detail or '未知错误'}\n"
                    f"任务编号：{workflow.id[:8]}。修复条件后可以让我重试。"
                )
            if not workflow.channel or not workflow.chat_id:
                self.store.mark_workflow_notified(workflow.id, workflow.status)
                continue
            if await self._push(workflow, message):
                self.store.mark_workflow_notified(workflow.id, workflow.status)

    async def _push(self, workflow: WorkflowInstance, message: str) -> bool:
        try:
            result = await self._push_tool.execute(
                channel=workflow.channel,
                chat_id=workflow.chat_id,
                message=message,
            )
        except Exception:
            logger.exception("workflow notification failed workflow=%s", workflow.id)
            return False
        failed_markers = ("发送失败", "未注册", "没有可用")
        return not any(marker in result for marker in failed_markers)

    def _success_message(self, workflow: WorkflowInstance) -> str:
        outputs = [
            f"- {step.title}：{self._compact_value(step.output, 500)}"
            for step in workflow.steps
            if step.kind == StepKind.AGENT and step.output is not None
        ]
        summary = "\n".join(outputs[-4:]) or "所有步骤均已完成。"
        return (
            f"「{workflow.name}」已经完成。\n\n{summary}\n\n"
            f"任务编号：{workflow.id[:8]}"
        )

    @staticmethod
    def _compact_value(value: Any, limit: int) -> str:
        text = str(value) if value is not None else "（无）"
        if len(text) <= limit:
            return text
        return text[:limit] + "…"


def _step_status_label(status: StepStatus) -> str:
    return {
        StepStatus.SUCCEEDED: "已完成",
        StepStatus.SKIPPED: "已跳过",
        StepStatus.FAILED: "失败",
        StepStatus.CANCELLED: "已取消",
        StepStatus.PENDING: "等待重试",
        StepStatus.RUNNING: "仍在运行",
        StepStatus.WAITING: "等待确认",
    }.get(status, status.value)


def _finish_trace(
    recorder: TraceRecorder | None,
    trace_id: str,
    *,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if recorder is None or not trace_id:
        return
    try:
        recorder.finish_trace(trace_id, status=status, metadata=metadata)
    except Exception:
        logger.warning(
            "workflow trace finish failed trace_id=%s", trace_id, exc_info=True
        )
