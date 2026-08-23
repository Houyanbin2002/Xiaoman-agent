"""LangGraph execution graph shared by the main agent and sub-agents."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, TypedDict, cast

from langchain_core.messages import AIMessage, convert_to_messages
from langgraph.graph import END, START, StateGraph

import agent.core.passive_support as support
from agent.core.types import LLMToolCall, ReasonerResult
from agent.lifecycle.types import (
    AfterStepCtx,
    AfterToolResultCtx,
    BeforeStepInput,
    BeforeToolCallCtx,
)
from agent.runtime.langchain_adapters import (
    ProviderChatModel,
    registry_tools_as_langchain,
)
from agent.runtime.langgraph_runtime import LangGraphRuntime
from agent.runtime.execution_guard import bound_tool_result
from agent.tool_hooks import ToolExecutionRequest
from agent.tool_runtime import (
    append_assistant_tool_calls,
    append_tool_result,
    tool_call_batch_snapshot,
)
from agent.tools.base import normalize_tool_result
from core.llm import ToolArgumentsDecodeError

logger = logging.getLogger(__name__)

_MAX_MODEL_PROTOCOL_RETRIES = 2
_MAX_INCOMPLETE_REPLY_RETRIES = 2
_NORMAL_FINISH_REASONS = frozenset(
    {"", "stop", "tool_calls", "function_call", "end_turn"}
)
_ACTION_WITHOUT_TOOL_RE = re.compile(
    r"^\s*(?:我(?:先|来|会|将|准备|正在|看看|检查|查询|创建|执行|处理|读取|搜索|确认)|"
    r"让我|接下来我)"
)
_INCOMPLETE_TAIL_RE = re.compile(
    r"(?:[，、：:；;(（]|再把|然后|以及|并且|因为|所以|在于|如下|包括)\s*$"
)


def _tool_arguments_feedback(error: ToolArgumentsDecodeError) -> str:
    preview = error.raw_arguments.replace("\x00", "")[:800]
    return (
        "【运行时协议反馈：工具参数无效】\n"
        f"工具：{error.tool_name or '<unknown>'}\n"
        f"固定原因：{error.reason}\n"
        f"原参数预览：{preview}\n"
        "该工具尚未执行。请严格按照工具 schema 重新生成一个完整且唯一的工具调用；"
        "arguments 顶层必须是一个 JSON 对象，不要在对象后追加说明、代码块或第二个 JSON。"
    )


def _incomplete_reply_reason(content: str, metadata: dict[str, Any]) -> str:
    """Return a deterministic reason when a no-tool response is likely incomplete."""

    text = content.strip()
    if not text:
        return ""
    finish_reason = str(metadata.get("finish_reason") or "").strip().lower()
    if finish_reason not in _NORMAL_FINISH_REASONS:
        return f"Provider 结束原因为 {finish_reason!r}，不是正常完成"
    if text.count("```") % 2 == 1:
        return "回复包含未闭合的代码块"
    has_terminal_punctuation = text.endswith(("。", "！", "？", ".", "!", "?", "…"))
    if _ACTION_WITHOUT_TOOL_RE.search(text) and not has_terminal_punctuation:
        return "回复表示将继续检查或执行操作，但没有产生工具调用且句子未结束"
    if len(text) <= 120 and _INCOMPLETE_TAIL_RE.search(text):
        return "短回复停在连接词或未闭合标点处"
    return ""


class AgentGraphState(TypedDict):
    """Serializable execution state persisted at every graph super-step."""

    messages: list[dict[str, Any]]
    tools_used: list[str]
    tools_unlocked: list[str]
    tool_chain: list[dict[str, Any]]
    visible_names: list[str] | None
    visible_order: list[str] | None
    disabled_tools: list[str]
    guard_state: dict[str, Any]
    iteration: int
    react_input_samples: list[int]
    cache_prompt_tokens: int
    cache_hit_tokens: int
    cache_seen: bool
    cache_plan: dict[str, Any]
    streamed: bool
    run_id: str
    session_key: str
    channel: str
    chat_id: str
    request_text: str
    permission_mode: str
    pending_tool_calls: list[dict[str, Any]]
    pending_tool_index: int
    pending_tool_results: list[dict[str, Any]]
    pending_content: str
    pending_thinking: str | None
    pending_provider_fields: dict[str, Any]
    reply: str
    thinking: str | None
    exit_reason: str
    summary_reason: str
    early_stop_reply: str
    summary_used_fallback: bool
    request_time: NotRequired[str | None]
    model_retry_pending: NotRequired[bool]
    protocol_retry_count: NotRequired[int]
    incomplete_reply_retry_count: NotRequired[int]


def _tool_call_from_dict(raw: dict[str, Any]) -> LLMToolCall:
    arguments = raw.get("arguments")
    return LLMToolCall(
        id=str(raw.get("id") or ""),
        name=str(raw.get("name") or ""),
        arguments=dict(arguments) if isinstance(arguments, dict) else {},
    )


def _tool_call_to_dict(call: LLMToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}


def _trace_to_dict(item: object) -> dict[str, Any]:
    return {
        "hook_name": getattr(item, "hook_name", ""),
        "event": getattr(item, "event", ""),
        "matched": bool(getattr(item, "matched", False)),
        "decision": getattr(item, "decision", ""),
        "reason": getattr(item, "reason", ""),
        "extra_message": getattr(item, "extra_message", ""),
    }


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(cast(str, block["text"]))
        return "".join(parts)
    return str(content or "")


class LangGraphAgentExecutor:
    """Compile and run the common Agent graph against an execution-kernel host."""

    def __init__(self, host: Any, runtime: LangGraphRuntime) -> None:
        self._host = host
        self._runtime = runtime
        self._graph: Any | None = None
        self._compile_lock = asyncio.Lock()
        self._stream_callbacks: dict[
            str, Callable[[dict[str, str]], Awaitable[None]]
        ] = {}

    async def _compiled_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        async with self._compile_lock:
            if self._graph is not None:
                return self._graph
            builder = StateGraph(AgentGraphState)
            builder.add_node("model", self._model_node)
            builder.add_node("tool", self._tool_node)
            builder.add_node("summarize", self._summarize_node)
            builder.add_edge(START, "model")
            builder.add_conditional_edges(
                "model",
                self._route_after_model,
                {
                    "model": "model",
                    "tool": "tool",
                    "summarize": "summarize",
                    "done": END,
                },
            )
            builder.add_conditional_edges(
                "tool",
                self._route_after_tool,
                {"tool": "tool", "model": "model", "summarize": "summarize"},
            )
            builder.add_edge("summarize", END)
            self._graph = builder.compile(
                checkpointer=await self._runtime.checkpointer(),
                store=self._runtime.store,
            )
            return self._graph

    def _initial_state(
        self,
        initial_messages: list[dict[str, Any]],
        *,
        run_id: str,
        request_time: object,
        preloaded_tools: set[str] | None,
        preloaded_tool_order: list[str] | None,
        session_key: str,
        channel: str,
        chat_id: str,
        request_text: str,
        permission_mode: str,
        disabled_tools: set[str] | None,
    ) -> AgentGraphState:
        disabled = set(disabled_tools or ())
        visible_names: list[str] | None = None
        visible_order: list[str] | None = None
        if self._host._tool_search_enabled:
            always_on = self._host._tools.get_always_on_names()
            visible = (always_on | (preloaded_tools or set())) - disabled
            visible_names = self._host._tools.get_registered_order(visible)
            visible_order = self._host._tools.get_registered_order(always_on - disabled)
            assert visible_order is not None
            seen = set(visible_order)
            for name in preloaded_tool_order or sorted(preloaded_tools or set()):
                if name in visible and name not in seen:
                    visible_order.append(name)
                    seen.add(name)
            logger.info(
                "[tool_search] visible=%d 个工具 always_on=%d preloaded=%d need_search=%s",
                len(visible),
                len(always_on),
                len(preloaded_tools or set()),
                "yes" if len(visible) == len(always_on) else "maybe",
            )
        return AgentGraphState(
            messages=list(initial_messages),
            tools_used=[],
            tools_unlocked=[],
            tool_chain=[],
            visible_names=visible_names,
            visible_order=visible_order,
            disabled_tools=sorted(disabled),
            guard_state=self._host._execution_guard.initial_state(),
            iteration=0,
            react_input_samples=[],
            cache_prompt_tokens=0,
            cache_hit_tokens=0,
            cache_seen=False,
            cache_plan={},
            streamed=False,
            run_id=run_id,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            request_text=request_text,
            permission_mode=permission_mode,
            pending_tool_calls=[],
            pending_tool_index=0,
            pending_tool_results=[],
            pending_content="",
            pending_thinking=None,
            pending_provider_fields={},
            reply="",
            thinking=None,
            exit_reason="",
            summary_reason="",
            early_stop_reply="",
            summary_used_fallback=False,
            request_time=(
                request_time.isoformat() if hasattr(request_time, "isoformat") else None
            ),
            model_retry_pending=False,
            protocol_retry_count=0,
            incomplete_reply_retry_count=0,
        )

    async def run(
        self,
        initial_messages: list[dict[str, Any]],
        *,
        request_time: object = None,
        preloaded_tools: set[str] | None = None,
        preloaded_tool_order: list[str] | None = None,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
        request_text: str = "",
        permission_mode: str = "full_access",
        disabled_tools: set[str] | None = None,
        resume_from_checkpoint: bool = False,
    ) -> ReasonerResult:
        graph = await self._compiled_graph()
        run_id = uuid.uuid4().hex
        callback_keys = {run_id}
        if on_content_delta is not None:
            self._stream_callbacks[run_id] = on_content_delta
        thread_key = tool_event_session_key or run_id
        config = {
            "configurable": {"thread_id": f"agent:{thread_key}"},
            "recursion_limit": max(100, self._host._llm_config.max_iterations * 4 + 20),
        }
        try:
            graph_input: AgentGraphState | None
            snapshot = await graph.aget_state(config)
            # A pending checkpoint also survives a process restart, where the
            # old in-memory interrupt marker no longer exists. Resume it by
            # default; an explicit new-thread action should use a new session id.
            if snapshot.next:
                graph_input = None
                await graph.aupdate_state(
                    config,
                    {
                        "guard_state": self._host._execution_guard.resume_state(
                            snapshot.values.get("guard_state")
                        )
                    },
                )
                persisted_run_id = str(snapshot.values.get("run_id") or "")
                if on_content_delta is not None and persisted_run_id:
                    self._stream_callbacks[persisted_run_id] = on_content_delta
                    callback_keys.add(persisted_run_id)
                logger.info(
                    "[LangGraph恢复] thread=%s next=%s requested=%s",
                    thread_key,
                    snapshot.next,
                    resume_from_checkpoint,
                )
            else:
                graph_input = self._initial_state(
                    initial_messages,
                    run_id=run_id,
                    request_time=request_time,
                    preloaded_tools=preloaded_tools,
                    preloaded_tool_order=preloaded_tool_order,
                    session_key=tool_event_session_key,
                    channel=tool_event_channel,
                    chat_id=tool_event_chat_id,
                    request_text=request_text,
                    permission_mode=permission_mode,
                    disabled_tools=disabled_tools,
                )
            state = cast(
                AgentGraphState,
                await graph.ainvoke(graph_input, config, durability="sync"),
            )
            visible_raw = state.get("visible_names")
            visible_names = set(visible_raw) if visible_raw is not None else None
            return self._host._build_result(
                reply=state.get("reply") or "（无响应）",
                tools_used=list(state.get("tools_used") or []),
                tool_chain=list(state.get("tool_chain") or []),
                visible_names=visible_names,
                thinking=state.get("thinking"),
                streamed=bool(state.get("streamed")),
                react_input_samples=list(state.get("react_input_samples") or []),
                cache_prompt_tokens=int(state.get("cache_prompt_tokens") or 0),
                cache_hit_tokens=int(state.get("cache_hit_tokens") or 0),
                cache_seen=bool(state.get("cache_seen")),
                cache_plan=dict(state.get("cache_plan") or {}),
                tools_unlocked=list(state.get("tools_unlocked") or []),
                exit_reason=state.get("exit_reason") or "completed",
                execution_guard_state=dict(state.get("guard_state") or {}),
            )
        finally:
            for key in callback_keys:
                self._stream_callbacks.pop(key, None)

    async def _model_node(self, state: AgentGraphState) -> dict[str, Any]:
        iteration = state["iteration"]
        max_iterations = self._host._llm_config.max_iterations
        if max_iterations > 0 and iteration >= max_iterations:
            logger.warning(
                "[迭代上限] 达到最大轮次%d，触发收尾总结，已调用工具: %s",
                iteration,
                state["tools_used"] or "无",
            )
            return {"summary_reason": "max_iterations"}

        messages = list(state["messages"])
        visible_raw = state.get("visible_names")
        visible_names = set(visible_raw) if visible_raw is not None else None
        cache_view = self._host._prompt_cache.prepare_model_messages(
            messages,
            keep_recent_tool_rounds=self._host._execution_policy.recent_tool_rounds,
        )
        model_messages = cache_view.messages
        step_ctx = await self._host._before_step.run(
            BeforeStepInput(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=iteration,
                messages=model_messages,
                visible_names=visible_names,
            )
        )
        if step_ctx.early_stop:
            return {
                "messages": messages,
                "summary_reason": "early_stop",
                "early_stop_reply": step_ctx.early_stop_reply or "",
            }

        guard = self._host._execution_guard.before_model(
            state.get("guard_state"),
            context_tokens=step_ctx.input_tokens_estimate,
        )
        if guard.stop_reason:
            return {
                "messages": messages,
                "guard_state": guard.state,
                "summary_reason": guard.stop_reason,
            }
        if guard.hint:
            messages.append(
                support.build_context_hint_message("execution_guard", guard.hint)
            )
            cache_view = self._host._prompt_cache.prepare_model_messages(
                messages,
                keep_recent_tool_rounds=self._host._execution_policy.recent_tool_rounds,
            )
            model_messages = cache_view.messages

        samples = [*state["react_input_samples"], step_ctx.input_tokens_estimate]
        visible_order_raw = state.get("visible_order")
        schema_names: list[str] | set[str] | None = (
            list(visible_order_raw) if visible_order_raw is not None else None
        )
        disabled = set(state["disabled_tools"])
        if schema_names is None and disabled:
            schema_names = self._host._tools.get_registered_names() - disabled
        elif schema_names is not None:
            schema_names = [name for name in schema_names if name not in disabled]
        tools = registry_tools_as_langchain(self._host._tools, schema_names)
        callback = self._stream_callbacks.get(state["run_id"])
        model = ProviderChatModel(
            provider=self._host._llm.provider,
            model_name=self._host._llm_config.model,
            max_output_tokens=self._host._llm_config.max_tokens,
            source=self._host._execution_policy.source,
            iteration=iteration + 1,
            on_content_delta=callback,
            cache_metadata=cache_view.plan.to_metadata(),
        ).bind_tools(tools)
        logger.info(
            "[LLM调用] 第%d轮，可见工具=%s input_tokens~=%d",
            iteration + 1,
            f"{len(visible_names)}个" if visible_names is not None else "全部",
            step_ctx.input_tokens_estimate,
        )
        if cache_view.plan.chars_saved or cache_view.plan.capped_recent_tool_messages:
            logger.info(
                "[CacheBreakpoint] index=%d protected_rounds=%d saved_chars=%d "
                "cold_tools=%d recent_capped=%d prefix=%s",
                cache_view.plan.breakpoint_index,
                cache_view.plan.protected_tool_rounds,
                cache_view.plan.chars_saved,
                cache_view.plan.compacted_tool_messages,
                cache_view.plan.capped_recent_tool_messages,
                cache_view.plan.stable_prefix_hash,
            )
        try:
            response = cast(
                AIMessage,
                await asyncio.wait_for(
                    model.ainvoke(convert_to_messages(model_messages)),
                    timeout=self._host._execution_guard.config.model_call_timeout_seconds,
                ),
            )
        except ToolArgumentsDecodeError as exc:
            retry_count = int(state.get("protocol_retry_count") or 0) + 1
            logger.warning(
                "[ReAct协议修复] 工具参数解析失败 tool=%s retry=%d reason=%s",
                exc.tool_name,
                retry_count,
                exc.reason,
            )
            if retry_count > _MAX_MODEL_PROTOCOL_RETRIES:
                return {
                    "messages": messages,
                    "iteration": iteration + 1,
                    "react_input_samples": samples,
                    "guard_state": guard.state,
                    "reply": "模型连续生成了无法解析的工具参数，本轮已安全停止，工具没有执行。",
                    "thinking": None,
                    "exit_reason": "tool_arguments_invalid",
                    "model_retry_pending": False,
                    "protocol_retry_count": retry_count,
                }
            messages.append({"role": "user", "content": _tool_arguments_feedback(exc)})
            return {
                "messages": messages,
                "iteration": iteration + 1,
                "react_input_samples": samples,
                "guard_state": guard.state,
                "cache_plan": cache_view.plan.to_metadata(),
                "pending_tool_calls": [],
                "pending_tool_index": 0,
                "pending_tool_results": [],
                "model_retry_pending": True,
                "protocol_retry_count": retry_count,
            }
        except TimeoutError:
            logger.warning("[执行保护] 模型调用超时，交由现有调用方错误边界处理")
            raise
        content = _message_text(response)
        thinking_raw = response.response_metadata.get("thinking")
        thinking = str(thinking_raw) if thinking_raw is not None else None
        prompt_tokens = response.response_metadata.get("cache_prompt_tokens")
        hit_tokens = response.response_metadata.get("cache_hit_tokens")
        update: dict[str, Any] = {
            "messages": messages,
            "iteration": iteration + 1,
            "react_input_samples": samples,
            "streamed": bool(state["streamed"] or (callback is not None and content)),
            "cache_prompt_tokens": state["cache_prompt_tokens"]
            + int(prompt_tokens or 0),
            "cache_hit_tokens": state["cache_hit_tokens"] + int(hit_tokens or 0),
            "cache_seen": bool(state["cache_seen"] or prompt_tokens is not None),
            "cache_plan": cache_view.plan.to_metadata(),
            "guard_state": guard.state,
            "model_retry_pending": False,
        }
        if response.tool_calls:
            tool_calls = [
                LLMToolCall(
                    id=str(call.get("id") or ""),
                    name=str(call.get("name") or ""),
                    arguments=dict(call.get("args") or {}),
                )
                for call in response.tool_calls
            ]
            append_assistant_tool_calls(
                messages,
                content=content,
                tool_calls=tool_calls,
                provider_fields=dict(response.additional_kwargs),
            )
            logger.info(
                "[LLM决策→工具] 第%d轮，调用: %s",
                iteration + 1,
                [call.name for call in tool_calls],
            )
            update.update(
                messages=messages,
                pending_tool_calls=[_tool_call_to_dict(call) for call in tool_calls],
                pending_tool_index=0,
                pending_tool_results=[],
                pending_content=content,
                pending_thinking=thinking,
                pending_provider_fields=dict(response.additional_kwargs),
            )
            return update

        incomplete_reason = _incomplete_reply_reason(
            content,
            dict(response.response_metadata),
        )
        incomplete_retries = int(state.get("incomplete_reply_retry_count") or 0)
        if incomplete_reason and incomplete_retries < _MAX_INCOMPLETE_REPLY_RETRIES:
            output_tokens = response.response_metadata.get("output_tokens")
            logger.warning(
                "[ReAct完成性修复] retry=%d output_tokens=%s reason=%s",
                incomplete_retries + 1,
                output_tokens,
                incomplete_reason,
            )
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "【运行时协议反馈：上一回复疑似未完成】\n"
                        f"固定原因：{incomplete_reason}。\n"
                        "请回到当前用户目标继续执行：需要操作就立即调用合适工具；"
                        "不需要工具就给出语义完整的最终答复。不要只描述接下来准备做什么。"
                    ),
                }
            )
            update.update(
                messages=messages,
                reply="",
                thinking=thinking,
                model_retry_pending=True,
                incomplete_reply_retry_count=incomplete_retries + 1,
            )
            return update

        if not content and thinking:
            messages.append({"role": "assistant", "content": ""})
            messages.append(
                {
                    "role": "user",
                    "content": "你刚才只输出了思考过程，没有给出正式回复。请直接回复用户，不要重复思考。",
                }
            )
            retry_cache_view = self._host._prompt_cache.prepare_model_messages(
                messages,
                keep_recent_tool_rounds=self._host._execution_policy.recent_tool_rounds,
            )
            retry_model = ProviderChatModel(
                provider=self._host._llm.provider,
                model_name=self._host._llm_config.model,
                max_output_tokens=self._host._llm_config.max_tokens,
                source=self._host._execution_policy.source,
                iteration=iteration + 2,
                purpose="empty_reply_retry",
                on_content_delta=callback,
                cache_metadata=retry_cache_view.plan.to_metadata(),
            )
            try:
                retry = cast(
                    AIMessage,
                    await asyncio.wait_for(
                        retry_model.ainvoke(
                            convert_to_messages(retry_cache_view.messages)
                        ),
                        timeout=self._host._execution_guard.config.model_call_timeout_seconds,
                    ),
                )
            except TimeoutError:
                logger.warning("[执行保护] 空回复重试超时")
                raise
            retry_content = _message_text(retry)
            retry_prompt_tokens = retry.response_metadata.get("cache_prompt_tokens")
            update["cache_prompt_tokens"] += int(retry_prompt_tokens or 0)
            update["cache_hit_tokens"] += int(
                retry.response_metadata.get("cache_hit_tokens") or 0
            )
            update["cache_seen"] = bool(
                update["cache_seen"] or retry_prompt_tokens is not None
            )
            update["cache_plan"] = retry_cache_view.plan.to_metadata()
            if retry_content:
                content = retry_content
                retry_thinking = retry.response_metadata.get("thinking")
                thinking = (
                    str(retry_thinking) if retry_thinking is not None else thinking
                )
                update["streamed"] = bool(update["streamed"] or callback is not None)

        messages.append({"role": "assistant", "content": content})
        await self._host._after_step.run(
            AfterStepCtx(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=iteration,
                context_tokens_estimate=support.estimate_messages_tokens(messages),
                tools_called=(),
                partial_reply=content,
                tools_used_so_far=tuple(state["tools_used"]),
                tool_chain_partial=tuple(state["tool_chain"]),
                partial_thinking=thinking,
                has_more=False,
            )
        )
        logger.info(
            "[LLM决策→回复] 第%d轮，共调用工具%d次: %s",
            iteration + 1,
            len(state["tools_used"]),
            state["tools_used"] or "无",
        )
        update.update(
            messages=messages,
            reply=content or "（无响应）",
            thinking=thinking,
            exit_reason="completed",
            model_retry_pending=False,
        )
        return update

    async def _tool_node(self, state: AgentGraphState) -> dict[str, Any]:
        calls = [_tool_call_from_dict(raw) for raw in state["pending_tool_calls"]]
        index = state["pending_tool_index"]
        if index >= len(calls):
            return await self._finish_tool_round(state)
        call = calls[index]
        disabled = set(state["disabled_tools"])
        visible_raw = state.get("visible_names")
        visible_names = set(visible_raw) if visible_raw is not None else None
        messages = list(state["messages"])
        iter_calls = list(state["pending_tool_results"])
        tools_used = list(state["tools_used"])
        tools_unlocked = list(state["tools_unlocked"])
        visible_order_raw = state.get("visible_order")
        visible_order = (
            list(visible_order_raw) if visible_order_raw is not None else None
        )
        batch = tool_call_batch_snapshot(calls)
        iteration = state["iteration"]

        if index == 0:
            guard = self._host._execution_guard.before_tool_batch(
                state.get("guard_state"),
                calls,
                risk_resolver=self._host._tools.get_risk,
            )
            if guard.stop_reason:
                blocked_calls: list[dict[str, Any]] = []
                for blocked in calls:
                    result = (
                        "工具调用已被执行保护器阻止："
                        f"{guard.stop_reason}。请基于已有结果收尾。"
                    )
                    append_tool_result(
                        messages,
                        tool_call_id=blocked.id,
                        content=result,
                        tool_name=blocked.name,
                    )
                    blocked_calls.append(
                        {
                            "call_id": blocked.id,
                            "name": blocked.name,
                            "status": "blocked",
                            "arguments": blocked.arguments,
                            "result": result,
                        }
                    )
                return {
                    "messages": messages,
                    "guard_state": guard.state,
                    "pending_tool_results": blocked_calls,
                    "pending_tool_index": len(calls),
                    "tool_chain": [
                        *state["tool_chain"],
                        {"text": state["pending_content"], "calls": blocked_calls},
                    ],
                    "summary_reason": guard.stop_reason,
                }

        unknown = not self._host._tools.has_tool(call.name)
        if unknown or call.name in disabled:
            await self._host._observe_tool_call_started(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=iteration,
                call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
            )
            result = (
                f"未知工具: {call.name}。"
                f'请先调用 tool_search(query="select:{call.name}") 精确查找；'
                "若仍不存在，再按目标能力描述进行搜索。"
                if unknown
                else f"工具 '{call.name}' 在当前后台任务中不可用。请直接返回要发送的最终内容，不要主动推送。"
            )
            status = "error" if unknown else "blocked"
            append_tool_result(
                messages, tool_call_id=call.id, content=result, tool_name=call.name
            )
            await self._host._observe_tool_call_completed(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=iteration,
                call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
                final_arguments=call.arguments,
                status=status,
                result_preview=support.log_preview(result),
            )
            iter_calls.append(
                {
                    "call_id": call.id,
                    "name": call.name,
                    "status": status,
                    "arguments": call.arguments,
                    "result": result,
                }
            )
            return {
                "messages": messages,
                "pending_tool_results": iter_calls,
                "pending_tool_index": index + 1,
            }

        if visible_names is not None and call.name not in visible_names:
            await self._host._observe_tool_call_started(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=iteration,
                call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
            )
            result = (
                f"工具 '{call.name}' 当前未加载（schema 不可见）。"
                f'请先调用 tool_search(query="select:{call.name}") 加载，'
                "然后再调用该工具。不要放弃当前任务。"
            )
            status = "blocked"
            final_arguments = call.arguments
            append_tool_result(
                messages, tool_call_id=call.id, content=result, tool_name=call.name
            )
            await self._host._observe_tool_call_completed(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=iteration,
                call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
                final_arguments=final_arguments,
                status=status,
                result_preview=support.log_preview(result),
            )
            iter_calls.append(
                {
                    "call_id": call.id,
                    "name": call.name,
                    "status": status,
                    "arguments": call.arguments,
                    "final_arguments": final_arguments,
                    "result": result,
                }
            )
            return {
                "messages": messages,
                "pending_tool_results": iter_calls,
                "pending_tool_index": index + 1,
            }

        async def _execute_tool(name: str, arguments: dict[str, Any]) -> Any:
            if name == "tool_search" and visible_names is not None:
                arguments = {**arguments, "excluded_names": visible_names | disabled}
            if name == "message_push":
                arguments = {**arguments, "_commit_role": "passive"}
            return await self._host._tools.execute(name, arguments)

        logger.info(
            "[工具执行→] %s args=%s",
            call.name,
            support.log_preview(call.arguments, 120),
        )
        await self._host._observe_tool_call_started(
            session_key=state["session_key"],
            channel=state["channel"],
            chat_id=state["chat_id"],
            iteration=iteration,
            call_id=call.id,
            tool_name=call.name,
            arguments=call.arguments,
        )
        await self._host._bus.fanout(
            BeforeToolCallCtx(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                tool_name=call.name,
                arguments=dict(call.arguments),
            )
        )
        exec_result = await self._host._tool_executor.execute(
            ToolExecutionRequest(
                call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
                source=self._host._execution_policy.source,
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                request_text=state["request_text"],
                permission_mode=state["permission_mode"],
                tool_batch=batch,
                tool_batch_index=index,
                timeout_seconds=self._host._execution_guard.tool_timeout(
                    self._host._tools.get_risk(call.name),
                    tool_name=call.name,
                    arguments=call.arguments,
                ),
            ),
            _execute_tool,
        )
        if exec_result.status == "success":
            tools_used.append(call.name)
        await self._host._bus.fanout(
            AfterToolResultCtx(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                tool_name=call.name,
                arguments=dict(exec_result.final_arguments),
                result=str(exec_result.output),
                status=exec_result.status,
            )
        )
        normalized = self._host._execution_policy.limit_tool_result(
            normalize_tool_result(exec_result.output)
        )
        round_chars = sum(len(str(item.get("result") or "")) for item in iter_calls)
        normalized = bound_tool_result(
            normalized,
            self._host._execution_guard.result_limit(
                state.get("guard_state"),
                round_chars=round_chars,
            ),
        )
        await self._host._observe_tool_call_completed(
            session_key=state["session_key"],
            channel=state["channel"],
            chat_id=state["chat_id"],
            iteration=iteration,
            call_id=call.id,
            tool_name=call.name,
            arguments=call.arguments,
            final_arguments=exec_result.final_arguments,
            status=exec_result.status,
            result_preview=normalized.preview(),
        )
        append_tool_result(
            messages,
            tool_call_id=call.id,
            content=normalized,
            tool_name=call.name,
        )
        if (
            exec_result.status == "success"
            and call.name == "tool_search"
            and visible_names is not None
        ):
            newly_unlocked = [
                name
                for name in self._host._discovery.unlock_names_from_result(
                    normalized.text
                )
                if name not in visible_names and name not in disabled
            ]
            if newly_unlocked:
                visible_names.update(newly_unlocked)
                tools_unlocked.extend(newly_unlocked)
                if visible_order is not None:
                    seen = set(visible_order)
                    visible_order.extend(
                        name for name in newly_unlocked if name not in seen
                    )
        iter_calls.append(
            {
                "call_id": call.id,
                "name": call.name,
                "status": exec_result.status,
                "arguments": call.arguments,
                "final_arguments": exec_result.final_arguments,
                "pre_hook_trace": [
                    _trace_to_dict(item) for item in exec_result.pre_hook_trace
                ],
                "post_hook_trace": [
                    _trace_to_dict(item) for item in exec_result.post_hook_trace
                ],
                "result": normalized.text or normalized.preview(),
            }
        )
        update: dict[str, Any] = {
            "messages": messages,
            "tools_used": tools_used,
            "tools_unlocked": tools_unlocked,
            "visible_names": (
                self._host._tools.get_registered_order(visible_names)
                if visible_names is not None
                else None
            ),
            "visible_order": visible_order,
            "pending_tool_results": iter_calls,
            "pending_tool_index": index + 1,
        }
        return update

    async def _finish_tool_round(self, state: AgentGraphState) -> dict[str, Any]:
        chain_group: dict[str, Any] = {
            "text": state["pending_content"],
            "calls": list(state["pending_tool_results"]),
        }
        if state.get("pending_thinking") is not None:
            chain_group["reasoning_content"] = state["pending_thinking"]
        tool_chain = [*state["tool_chain"], chain_group]
        messages = list(state["messages"])
        guard = self._host._execution_guard.after_tool_round(
            state.get("guard_state"),
            list(state["pending_tool_results"]),
        )
        if guard.hint:
            messages.append(
                support.build_context_hint_message("execution_guard", guard.hint)
            )
        self._host._execution_policy.append_after_tool_round(
            messages,
            completed_iterations=state["iteration"],
            max_iterations=self._host._llm_config.max_iterations,
        )
        pressure_view = self._host._prompt_cache.prepare_model_messages(
            messages,
            keep_recent_tool_rounds=self._host._execution_policy.recent_tool_rounds,
        )
        pressure = support.estimate_messages_tokens(pressure_view.messages)
        calls = [_tool_call_from_dict(raw) for raw in state["pending_tool_calls"]]
        after_step = await self._host._after_step.run(
            AfterStepCtx(
                session_key=state["session_key"],
                channel=state["channel"],
                chat_id=state["chat_id"],
                iteration=state["iteration"] - 1,
                context_tokens_estimate=pressure,
                tools_called=tuple(call.name for call in calls),
                partial_reply=state["pending_content"],
                tools_used_so_far=tuple(state["tools_used"]),
                tool_chain_partial=tuple(tool_chain),
                partial_thinking=state.get("pending_thinking"),
                has_more=True,
            )
        )
        update: dict[str, Any] = {
            "messages": messages,
            "tool_chain": tool_chain,
            "pending_tool_calls": [],
            "pending_tool_index": 0,
            "pending_tool_results": [],
            "pending_content": "",
            "pending_thinking": None,
            "pending_provider_fields": {},
            "guard_state": guard.state,
            "disabled_tools": sorted(
                set(state["disabled_tools"]) | set(guard.disabled_tools)
            ),
        }
        if guard.stop_reason:
            update["summary_reason"] = guard.stop_reason
        elif after_step.early_stop:
            update["summary_reason"] = after_step.early_stop_reason or "after_step"
        return update

    async def _summarize_node(self, state: AgentGraphState) -> dict[str, Any]:
        reason = state.get("summary_reason") or "incomplete"
        summary = await self._host._summarize_incomplete_progress(
            list(state["messages"]),
            reason=reason,
            iteration=state["iteration"],
            tools_used=list(state["tools_used"]),
        )
        reply = state.get("early_stop_reply") or summary.text
        exit_reason = reason
        if reason == "tool_call_loop":
            exit_reason = "tool_loop"
        elif reason == "max_iterations" and summary.used_fallback:
            exit_reason = "max_iterations_fallback"
        return {
            "reply": reply,
            "thinking": None,
            "streamed": False,
            "exit_reason": exit_reason,
            "summary_used_fallback": summary.used_fallback,
        }

    @staticmethod
    def _route_after_model(state: AgentGraphState) -> str:
        if state.get("summary_reason"):
            return "summarize"
        if state.get("exit_reason"):
            return "done"
        if state.get("model_retry_pending"):
            return "model"
        return "tool"

    @staticmethod
    def _route_after_tool(state: AgentGraphState) -> str:
        if state.get("summary_reason"):
            return "summarize"
        if state["pending_tool_index"] < len(state["pending_tool_calls"]):
            return "tool"
        if state["pending_tool_calls"]:
            return "tool"
        return "model"
