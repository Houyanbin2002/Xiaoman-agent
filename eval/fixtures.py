from __future__ import annotations

"""Lifecycle fixtures for real AgentLoop evaluations.

Fixtures create the external state named by a case, then derive evaluator
state from the real stores after execution.  They never replace the Agent
response or replay an expected result.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from agent.runtime.context_compaction import (
    ContextCompactionConfig,
    ContextSummaryState,
    estimate_tokens,
)
from agent.tools.base import Tool
from core.workflow.models import StepKind, StepSpec, WorkflowStatus

from .models import EvalCase


_CONTEXT_VALUES: dict[str, str] = {
    "task": "完成小满真实评测报告",
    "failure_reason": "普通配置读取接口暂时不可用",
    "next_step": "切换只读配置接口后继续",
    "artifact_path": "C:/eval/artifacts/report.md",
    "conclusion": "采用方案 B 并保留原始证据",
    "user_preference": "回复先给结论再给简短步骤",
    "recent_task": "补齐二十四个真实评测夹具",
    "checkpoint": "checkpoint-eval-42",
    "pending_node": "finalize_report",
    "confirmed_plan": "先补夹具，再跑全量真实评测",
}

_CONTEXT_REQUIRED_TERMS: dict[str, tuple[str, ...]] = {
    "task": ("小满", "评测报告"),
    "failure_reason": ("配置读取接口", "不可用"),
    "next_step": ("只读配置接口", "继续"),
    "artifact_path": ("C:/eval/artifacts/report.md",),
    "conclusion": ("方案 B", "原始证据"),
    "user_preference": ("先给结论", "简短步骤"),
    "recent_task": ("二十四个", "评测夹具"),
    "checkpoint": ("checkpoint-eval-42",),
    "pending_node": ("finalize_report",),
    "confirmed_plan": ("先补夹具", "全量真实评测"),
}


class _EvalSuccessTool(Tool):
    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def __init__(self, name: str, *, failing_tool: str) -> None:
        self._name = name
        self._description = (
            f"评测夹具提供的只读恢复工具。仅当 {failing_tool} 返回错误时调用；"
            "读取同一份隔离测试数据并返回成功结果。"
        )

    async def execute(self, **_: Any) -> str:
        return json.dumps(
            {"ok": True, "fixture": True, "message": "fallback completed"},
            ensure_ascii=False,
        )


class _EvalFailingTool(Tool):
    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def __init__(self, name: str, *, fallback_tool: str) -> None:
        self._name = name
        self._description = (
            "评测夹具提供的首选只读工具，必须先尝试一次。"
            "测试目标和全部必填参数已经由夹具预绑定，无需向用户追问。"
            f"该调用会注入临时服务故障；失败后必须改用 {fallback_tool}，禁止重复重试。"
        )

    async def execute(self, **_: Any) -> str:
        # Registry.execute is intercepted before reaching this method so the
        # normal ToolExecutor records an actual error status in the trace.
        raise RuntimeError("injected temporary service failure")


@dataclass
class PreparedFixture:
    kind: str = ""
    fixture: dict[str, Any] = field(default_factory=dict)
    session_key: str = ""
    trace_id: str = ""
    registered_tools: list[str] = field(default_factory=list)
    replaced_tools: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    original_tool_execute: Any = None
    workflow_id: str = ""
    checkpoint_created: bool = False
    original_compaction: Any = None
    input_tokens_before: int = 0
    memory_ids_before: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class FixtureObservation:
    state: dict[str, Any] = field(default_factory=dict)
    memory_events: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class LiveEvalFixtureManager:
    """Prepare, inspect and remove per-case state in an isolated runtime."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def prepare(
        self,
        case: EvalCase,
        *,
        session_key: str,
        trace_id: str,
    ) -> PreparedFixture:
        fixture_raw = case.metadata.get("fixture")
        fixture = dict(fixture_raw) if isinstance(fixture_raw, Mapping) else {}
        prepared = PreparedFixture(
            kind=str(fixture.get("kind") or ""),
            fixture=fixture,
            session_key=session_key,
            trace_id=trace_id,
            memory_ids_before=self._active_memory_ids(),
        )
        if prepared.kind == "existing_memory":
            self._seed_existing_memory(prepared)
        elif prepared.kind == "tool_failure":
            self._inject_tool_failure(prepared)
        elif prepared.kind == "workflow_checkpoint":
            await self._seed_workflow_checkpoint(prepared)
        elif prepared.kind == "long_context":
            await self._seed_long_context(prepared)
        return prepared

    async def observe(self, prepared: PreparedFixture) -> FixtureObservation:
        if prepared.kind == "existing_memory":
            return self._observe_memory_correction(prepared)
        if prepared.kind == "workflow_checkpoint":
            return self._observe_workflow(prepared)
        if prepared.kind == "long_context":
            return self._observe_compaction(prepared)
        return FixtureObservation(
            metadata={
                "fixture_kind": prepared.kind or "none",
                "fixture_applied": bool(prepared.kind),
            }
        )

    def disabled_tools(self, prepared: PreparedFixture) -> list[str]:
        """Restrict a fixture to the capability whose behavior it measures."""

        registered = self.runtime.tools.get_registered_names()
        if prepared.kind == "tool_failure":
            allowed = {
                str(prepared.fixture.get("failing_tool") or ""),
                str(prepared.fixture.get("fallback_tool") or ""),
            }
            return sorted(registered - allowed)
        if prepared.kind == "workflow_checkpoint":
            return sorted(registered - {"task_manage"})
        if prepared.kind == "existing_memory":
            return ["forget_memory"] if "forget_memory" in registered else []
        if prepared.kind == "long_context":
            return sorted(registered)
        return []

    async def cleanup(self, prepared: PreparedFixture) -> None:
        registry = self.runtime.tools
        if prepared.original_tool_execute is not None:
            registry.execute = prepared.original_tool_execute
        for name in prepared.registered_tools:
            registry.unregister(name)
            restored = prepared.replaced_tools.get(name)
            if restored is not None:
                tool, document = restored
                registry.register(
                    tool,
                    risk=document.risk,
                    always_on=document.always_on,
                    search_hint=document.search_hint,
                    source_type=document.source_type,
                    source_name=document.source_name,
                )

        reasoner = getattr(self.runtime.loop, "_reasoner", None)
        if prepared.original_compaction is not None and reasoner is not None:
            reasoner._context_compaction = prepared.original_compaction

        workflow_runtime = self.runtime.workflow_runtime
        if prepared.workflow_id and workflow_runtime is not None:
            try:
                workflow = workflow_runtime.store.require_workflow(prepared.workflow_id)
                if workflow.status not in {
                    WorkflowStatus.SUCCEEDED,
                    WorkflowStatus.FAILED,
                    WorkflowStatus.CANCELLED,
                }:
                    await workflow_runtime.cancel_workflow(
                        prepared.workflow_id,
                        reason="evaluation fixture teardown",
                    )
                workflow_runtime.store.delete_workflow(prepared.workflow_id)
                checkpointer = await workflow_runtime._graph_runtime.checkpointer()
                delete_thread = getattr(checkpointer, "adelete_thread", None)
                if callable(delete_thread):
                    await delete_thread(f"workflow:{prepared.workflow_id}")
            except (RuntimeError, ValueError):
                pass

        # Sessions are per-case and the durable transcript has already been
        # inspected. Remove it so long-context cases cannot influence others.
        manager = self.runtime.session_manager
        manager.invalidate(prepared.session_key)
        try:
            manager._store.delete_session(prepared.session_key, cascade=True)
        except (RuntimeError, ValueError):
            pass

        long_term = self.runtime.memory_runtime.long_term
        if long_term is not None:
            for record in long_term.governance.list_memories(limit=10000):
                if record.id in prepared.memory_ids_before:
                    continue
                try:
                    long_term.governance.forget(
                        record.id,
                        actor="user",
                        reason="evaluation fixture teardown",
                    )
                except ValueError:
                    pass

    def _active_memory_ids(self) -> set[str]:
        long_term = self.runtime.memory_runtime.long_term
        if long_term is None:
            return set()
        return {record.id for record in long_term.governance.list_memories(limit=10000)}

    def _seed_existing_memory(self, prepared: PreparedFixture) -> None:
        long_term = self.runtime.memory_runtime.long_term
        if long_term is None:
            raise RuntimeError("existing_memory fixture requires governed long-term memory")
        key = str(prepared.fixture.get("preference_key") or "").strip()
        old = str(prepared.fixture.get("old_value") or "").strip()
        if not key or not old:
            raise ValueError("existing_memory fixture requires preference_key and old_value")
        result = long_term.ingest_candidates(
            [
                {
                    "tag": "preference",
                    "content": f"用户原有偏好：{old}",
                    "confidence": 1.0,
                    "origin": "explicit_user",
                    "source_message_id": f"fixture:{prepared.trace_id}",
                    "_user_evidence_verified": True,
                    "subject": "用户",
                    "predicate": key,
                    "value": old,
                    "attributes": {"preference_key": key},
                }
            ],
            source_ref=f"fixture:{prepared.trace_id}",
            source="eval_fixture",
        )
        if result.created != 1 and result.unchanged != 1:
            raise RuntimeError(f"failed to seed existing memory: {result.to_dict()}")

    def _inject_tool_failure(self, prepared: PreparedFixture) -> None:
        registry = self.runtime.tools
        failing = str(prepared.fixture.get("failing_tool") or "").strip()
        fallback = str(prepared.fixture.get("fallback_tool") or "").strip()
        if not failing or not fallback:
            raise ValueError("tool_failure fixture requires failing_tool and fallback_tool")
        for name in (failing, fallback):
            existing = registry.get_tool(name)
            document = registry.get_document(name)
            if existing is not None and document is not None:
                prepared.replaced_tools[name] = (existing, document)
        registry.register(
            _EvalFailingTool(failing, fallback_tool=fallback),
            always_on=True,
            risk="read-only",
            search_hint=f"评测 首选 失败 后改用 {fallback}",
        )
        registry.register(
            _EvalSuccessTool(fallback, failing_tool=failing),
            always_on=True,
            risk="read-only",
            search_hint=f"评测 备用 恢复 {failing} 失败后使用",
        )
        prepared.registered_tools.extend((failing, fallback))
        original = registry.execute
        prepared.original_tool_execute = original

        async def execute_with_failure(name: str, arguments: dict[str, Any]) -> Any:
            if name == failing:
                raise RuntimeError(
                    f"injected temporary failure; use fallback tool {fallback}"
                )
            return await original(name, arguments)

        registry.execute = execute_with_failure

    async def _seed_workflow_checkpoint(self, prepared: PreparedFixture) -> None:
        workflow_runtime = self.runtime.workflow_runtime
        if workflow_runtime is None:
            raise RuntimeError("workflow_checkpoint fixture requires WorkflowRuntime")
        label = str(prepared.fixture.get("label") or "").strip()
        side_effect = str(prepared.fixture.get("side_effect") or "").strip()
        workflow = workflow_runtime.create_workflow(
            name=f"评测恢复：{label}",
            goal=f"从持久化 checkpoint 继续{label}，不得重复已完成工作",
            steps=[
                StepSpec(
                    id="resume_gate",
                    title=f"确认继续{label}",
                    description=(
                        f"任务已完成中断前的工作。收到用户的继续指令后恢复，并提交 {side_effect}。"
                    ),
                    kind=StepKind.WAIT_USER,
                )
            ],
            # task_manage derives this key from channel + chat_id.
            session_key=f"eval:{prepared.session_key}",
            channel="eval",
            chat_id=prepared.session_key,
            context={
                "fixture": True,
                "resumed": True,
                "side_effects": [side_effect],
            },
            auto_start=True,
        )
        prepared.workflow_id = workflow.id
        await workflow_runtime._drive_workflow(workflow.id)
        graph = await workflow_runtime._compiled_graph()
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": f"workflow:{workflow.id}"}}
        )
        current = workflow_runtime.store.require_workflow(workflow.id)
        prepared.checkpoint_created = bool(snapshot.next) and (
            current.status == WorkflowStatus.WAITING
        )
        if not prepared.checkpoint_created:
            raise RuntimeError("failed to persist waiting LangGraph checkpoint")

    async def _seed_long_context(self, prepared: PreparedFixture) -> None:
        reasoner = getattr(self.runtime.loop, "_reasoner", None)
        if reasoner is None:
            raise RuntimeError("long_context fixture requires the main reasoner")
        prepared.original_compaction = reasoner._context_compaction
        reasoner._context_compaction = ContextCompactionConfig(
            enabled=True,
            trigger_tokens=8_000,
            target_tokens=5_000,
            keep_recent_tokens=2_000,
            summary_max_tokens=1_200,
            chunk_tokens=4_000,
            max_history_messages=500,
        ).normalized()

        preserved = [str(item) for item in prepared.fixture.get("preserved", ())]
        marker_lines = [
            f"MUST_PRESERVE {key}={_CONTEXT_VALUES[key]}"
            for key in preserved
            if key in _CONTEXT_VALUES
        ]
        session = self.runtime.session_manager.get_or_create(prepared.session_key)
        filler = "这是用于触发真实 token 水位摘要的旧执行证据。" * 42
        for index in range(12):
            user_content = (
                (
                    "以下是继续任务必须逐项保留的结构化状态：\n"
                    + "\n".join(marker_lines)
                    + "\n"
                    if index in {0, 3, 6}
                    else ""
                )
                + f"旧回合 {index}：{filler}"
            )
            session.add_message("user", user_content)
            extra: dict[str, Any] = {}
            if index >= 9:
                extra["tool_chain"] = [
                    {
                        "text": "读取评测状态",
                        "calls": [
                            {
                                "call_id": f"fixture-call-{index}",
                                "name": "fixture_state_read",
                                "arguments": {"round": index},
                                "result": f"recent_tool_round={index}; status=ok",
                                "status": "success",
                            }
                        ],
                    }
                ]
            session.add_message("assistant", f"旧回复 {index}：{filler}", **extra)
        prepared.input_tokens_before = estimate_tokens(session.messages)
        await self.runtime.session_manager.save_async(session)

    def _observe_memory_correction(
        self, prepared: PreparedFixture
    ) -> FixtureObservation:
        long_term = self.runtime.memory_runtime.long_term
        if long_term is None:
            return FixtureObservation()
        key = str(prepared.fixture.get("preference_key") or "")
        expected_value = str(prepared.fixture.get("new_value") or "")
        record_key = f"memory:preference:{key}"
        records = long_term.governance.list_memories(
            include_inactive=True,
            limit=10000,
        )
        active = next(
            (
                record
                for record in records
                if record.record_key == record_key
                and str(record.status.value) == "active"
            ),
            None,
        )
        if active is None:
            return FixtureObservation(
                metadata={"fixture_kind": prepared.kind, "record_key": record_key}
            )
        old = (
            long_term.governance.personal_data.get(active.supersedes_id)
            if active.supersedes_id
            else None
        )
        value = str(active.data.get("value") or active.summary)
        event = {
            "type": "memory_correction",
            "value": value,
            "content": str(active.data.get("content") or active.summary),
            "supersedes": (
                str(old.data.get("value") or old.summary) if old is not None else ""
            ),
            "confidence": active.confidence,
            "user_locked": active.user_locked,
            "record_key": active.record_key,
        }
        return FixtureObservation(
            memory_events=(event,),
            metadata={
                "fixture_kind": prepared.kind,
                "fixture_applied": True,
                "correction_persisted": value == expected_value,
            },
        )

    def _observe_workflow(self, prepared: PreparedFixture) -> FixtureObservation:
        workflow_runtime = self.runtime.workflow_runtime
        if workflow_runtime is None or not prepared.workflow_id:
            return FixtureObservation()
        workflow = workflow_runtime.store.require_workflow(prepared.workflow_id)
        completed = workflow.status == WorkflowStatus.SUCCEEDED
        step = workflow.steps[0]
        events = workflow_runtime.store.list_events(prepared.workflow_id, limit=100)
        completed_count = sum(event.event_type == "step_succeeded" for event in events)
        side_effects = (
            list(workflow.context.get("side_effects") or ())
            if completed and completed_count == 1
            else []
        )
        return FixtureObservation(
            state={
                "workflow_status": "completed" if completed else workflow.status.value,
                "resumed": prepared.checkpoint_created and completed,
                "side_effects": side_effects,
            },
            metadata={
                "fixture_kind": prepared.kind,
                "fixture_applied": True,
                "workflow_id": workflow.id,
                "checkpoint_created": prepared.checkpoint_created,
                "step_status": step.status.value,
                "step_completed_count": completed_count,
            },
        )

    def _observe_compaction(self, prepared: PreparedFixture) -> FixtureObservation:
        session = self.runtime.session_manager.get_or_create(prepared.session_key)
        summary_state = ContextSummaryState.from_metadata(session.metadata)
        if summary_state is None:
            return FixtureObservation(
                state={"compression": {"preserved": [], "token_reduction": 0.0}},
                metadata={"fixture_kind": prepared.kind, "fixture_applied": True},
            )
        recent = list(session.messages[summary_state.summarized_through :])
        evidence = json.dumps(
            {"summary": summary_state.summary, "recent": recent},
            ensure_ascii=False,
            default=str,
        ).casefold()
        requested = [str(item) for item in prepared.fixture.get("preserved", ())]
        preserved: list[str] = []
        for key in requested:
            if key == "system_prefix":
                if getattr(getattr(self.runtime.loop, "_reasoner", None), "_prompt_cache", None) is not None:
                    preserved.append(key)
                continue
            if key == "recent_tool_rounds":
                tool_rounds = sum(bool(item.get("tool_chain")) for item in recent)
                if tool_rounds >= 3:
                    preserved.append(key)
                continue
            value = _CONTEXT_VALUES.get(key, "")
            terms = _CONTEXT_REQUIRED_TERMS.get(key, ())
            if (
                key.casefold() in evidence
                or (value and value.casefold() in evidence)
                or (terms and all(term.casefold() in evidence for term in terms))
            ):
                preserved.append(key)
        after_tokens = estimate_tokens(
            [{"role": "system", "content": summary_state.summary}, *recent]
        )
        before = max(1, prepared.input_tokens_before)
        reduction = max(0.0, min(1.0, 1.0 - (after_tokens / before)))
        return FixtureObservation(
            state={
                "compression": {
                    "preserved": preserved,
                    "token_reduction": round(reduction, 4),
                    "epoch": summary_state.epoch,
                    "summarized_through": summary_state.summarized_through,
                }
            },
            metadata={
                "fixture_kind": prepared.kind,
                "fixture_applied": True,
                "input_tokens_before": before,
                "input_tokens_after": after_tokens,
            },
        )


__all__ = [
    "FixtureObservation",
    "LiveEvalFixtureManager",
    "PreparedFixture",
]
