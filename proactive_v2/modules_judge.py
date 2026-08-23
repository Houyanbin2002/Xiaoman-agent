from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from agent.tool_hooks import ToolExecutionRequest
from agent.runtime.execution_guard import (
    ExecutionGuard,
    ExecutionGuardConfig,
    bound_tool_result,
)
from agent.tools.base import ToolResult
from core.common.diagnostic_log import diagnostic_line
from proactive_v2.config import ProactiveConfig
from proactive_v2.context import AgentTickContext
from proactive_v2.gateway import GatewayResult
from proactive_v2.tools import TOOL_SCHEMAS, ToolDeps, dispatch

logger = logging.getLogger(__name__)


class ProactiveJudge:
    def __init__(
        self,
        *,
        cfg: ProactiveConfig,
        session_key: str,
        llm_fn: Any | None,
        tool_deps: ToolDeps,
        tool_executor: Any,
        record_step_fn: Callable[..., None],
    ) -> None:
        self._cfg = cfg
        self._session_key = session_key
        self._llm_fn = llm_fn
        self._tool_deps = tool_deps
        self._tool_executor = tool_executor
        self._record_step_fn = record_step_fn
        self._guard = ExecutionGuard(
            ExecutionGuardConfig(
                max_tool_calls=cfg.agent_tick_max_steps,
                soft_timeout_seconds=45,
                hard_timeout_seconds=90,
                max_tool_result_chars=cfg.agent_tick_web_fetch_max_chars,
                max_tool_round_chars=cfg.agent_tick_web_fetch_max_chars,
                max_turn_tool_result_chars=max(
                    cfg.agent_tick_web_fetch_max_chars * 4,
                    24_000,
                ),
            )
        )
        self._guard_state = self._guard.initial_state()
        self._guard_stopped = False

    async def evaluate(
        self,
        ctx: AgentTickContext,
        messages: list[dict],
        gw_result: GatewayResult | None,
    ) -> None:
        if self._llm_fn is None:
            return
        self._guard_state = self._guard.initial_state()
        self._guard_stopped = False

        while ctx.steps_taken < self._cfg.agent_tick_max_steps:
            ok = await self._run_tool_step(
                messages, ctx, loop_tag="loop", tool_choice="auto"
            )
            if not ok:
                break
            if ctx.terminal_action is not None:
                break

        if self._guard_stopped:
            return

        if (
            ctx.terminal_action == "skip"
            and gw_result is not None
            and gw_result.content_meta
        ):
            all_content_ids = {m["id"] for m in gw_result.content_meta}
            classified_ids = ctx.interesting_item_ids | ctx.discarded_item_ids
            unclassified_ids = all_content_ids - classified_ids
            if unclassified_ids:
                ctx.terminal_action = None
                ctx.skip_reason = ""
                ctx.skip_note = ""
                titles_hint = "; ".join(
                    f"{m['id']}（{m['title'][:40]}）"
                    for m in gw_result.content_meta
                    if m["id"] in unclassified_ids
                )
                completeness_msg = (
                    f"【系统提示】以下 {len(unclassified_ids)} 个条目尚未完成分类：\n"
                    f"{titles_hint}\n"
                    "请对每条调用 mark_interesting 或 mark_not_interesting，"
                    "全部分类完毕后再调用 message_push + finish_turn(decision=reply)，或 finish_turn(decision=skip, reason=...)。"
                )
                logger.info(
                    "[proactive_v2] judge completeness: %d unclassified, resetting → %s",
                    len(unclassified_ids),
                    sorted(unclassified_ids),
                )
                messages.append({"role": "user", "content": completeness_msg})
                for _ in range(5):
                    if (
                        ctx.terminal_action is not None
                        or ctx.steps_taken >= self._cfg.agent_tick_max_steps
                    ):
                        break
                    ok = await self._run_tool_step(messages, ctx, loop_tag="complete")
                    if not ok:
                        break

        if self._guard_stopped:
            return

        if (
            ctx.terminal_action is None
            and ctx.interesting_item_ids
            and ctx.steps_taken < self._cfg.agent_tick_max_steps
        ):
            ids_str = ", ".join(sorted(ctx.interesting_item_ids))
            reflection = (
                f"【系统提示】你已将以下条目标记为 interesting：{ids_str}。\n"
                "所有条目均已分类完毕。你必须现在调用 message_push 撰写推送，然后调用 finish_turn(decision=reply)；"
                "或直接调用 finish_turn(decision=skip, reason=...)。不允许直接结束。"
            )
            logger.info(
                "[proactive_v2] judge reflection: interesting=%d, injecting prompt",
                len(ctx.interesting_item_ids),
            )
            messages.append({"role": "user", "content": reflection})
            for _ in range(3):
                if (
                    ctx.terminal_action is not None
                    or ctx.steps_taken >= self._cfg.agent_tick_max_steps
                ):
                    break
                ok = await self._run_tool_step(
                    messages, ctx, loop_tag="reflect", tool_choice="auto"
                )
                if not ok:
                    break

    async def _run_tool_step(
        self,
        messages: list[dict],
        ctx: AgentTickContext,
        *,
        loop_tag: str,
        tool_choice: str | dict = "auto",
        schemas: list[dict] | None = None,
    ) -> bool:
        active_schemas = schemas or TOOL_SCHEMAS
        llm_fn = self._llm_fn
        if llm_fn is None:
            return False
        try:
            tool_call = await asyncio.wait_for(
                llm_fn(messages, active_schemas, tool_choice),
                timeout=self._guard.config.model_call_timeout_seconds,
            )
        except TimeoutError:
            logger.warning("[proactive_v2] %s: model call timed out", loop_tag)
            self._guard_stopped = True
            return False
        if tool_call is None:
            logger.warning(
                "[proactive_v2] %s: llm_fn returned None at step %d, stopping",
                loop_tag,
                ctx.steps_taken,
            )
            return False
        ctx.record_llm_cache(
            cache_prompt_tokens=tool_call.get("_cache_prompt_tokens"),
            cache_hit_tokens=tool_call.get("_cache_hit_tokens"),
        )
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("input", {})
        arg_summary = json.dumps(tool_args, ensure_ascii=False)[:200]
        logger.info(
            diagnostic_line(
                "ProactiveJudge._run_tool_step",
                event="tool_call",
                flow="proactive",
                phase="agent_loop",
                session=self._session_key,
                tick=ctx.tick_id,
                action=str(tool_name or "-"),
                counts=f"step:{ctx.steps_taken}",
            )
        )
        logger.info(
            "[proactive_v2] %s step %d: %s  args=%s",
            loop_tag,
            ctx.steps_taken,
            tool_name,
            arg_summary,
        )
        ctx.steps_taken += 1
        call_id = str(tool_call.get("id") or f"call_{ctx.steps_taken}")
        call_record = {
            "call_id": call_id,
            "name": tool_name,
            "arguments": tool_args,
        }
        before = self._guard.before_tool_batch(
            self._guard_state,
            [call_record],
            risk_resolver=lambda name: (
                "external-side-effect" if name == "message_push" else "read-only"
            ),
        )
        self._guard_state = before.state
        if before.stop_reason:
            result = f"主动执行已收敛：{before.stop_reason}"
            self._append_tool_messages(
                messages,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=call_id,
                result=result,
            )
            self._guard_stopped = True
            return False
        exec_result = await self._tool_executor.execute(
            ToolExecutionRequest(
                call_id=str(tool_call.get("id") or f"call_{ctx.steps_taken}"),
                tool_name=tool_name,
                arguments=tool_args,
                source="proactive",
                session_key=self._session_key,
                timeout_seconds=self._guard.tool_timeout(
                    (
                        "external-side-effect"
                        if tool_name == "message_push"
                        else "read-only"
                    ),
                    tool_name=tool_name,
                    arguments=tool_args,
                ),
            ),
            lambda name, args: dispatch(name, args, ctx, self._tool_deps),
        )
        if exec_result.status == "error":
            logger.warning(
                diagnostic_line(
                    "ProactiveJudge._run_tool_step",
                    event="tool_result",
                    flow="proactive",
                    phase="agent_loop",
                    session=self._session_key,
                    tick=ctx.tick_id,
                    action=str(tool_name or "-"),
                    reason="tool_error",
                    counts=f"step:{ctx.steps_taken}",
                    note=str(exec_result.output)[:160],
                )
            )
            logger.warning(
                "[proactive_v2] %s: tool error: %s", loop_tag, exec_result.output
            )
            result = bound_tool_result(
                ToolResult(text=str(exec_result.output)),
                self._guard.config.max_tool_result_chars,
            ).text
            self._record_step_fn(
                ctx,
                phase=f"{loop_tag}:error",
                tool_name=tool_name,
                tool_call_id=call_id,
                tool_args=tool_args,
                tool_result_text=result,
            )
            self._append_tool_messages(
                messages,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=call_id,
                result=result,
            )
            after = self._guard.after_tool_round(
                self._guard_state,
                [{**call_record, "status": exec_result.status, "result": result}],
            )
            self._guard_state = after.state
            self._guard_stopped = bool(after.stop_reason)
            return False
        result = bound_tool_result(
            ToolResult(text=str(exec_result.output)),
            self._guard.config.max_tool_result_chars,
        ).text
        logger.info(
            diagnostic_line(
                "ProactiveJudge._run_tool_step",
                event="tool_result",
                flow="proactive",
                phase="agent_loop",
                session=self._session_key,
                tick=ctx.tick_id,
                action=str(tool_name or "-"),
                reason="-",
                counts=f"step:{ctx.steps_taken}",
            )
        )
        self._record_step_fn(
            ctx,
            phase=loop_tag,
            tool_name=tool_name,
            tool_call_id=call_id,
            tool_args=tool_args,
            tool_result_text=result,
        )
        self._append_tool_messages(
            messages,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=call_id,
            result=result,
        )
        after = self._guard.after_tool_round(
            self._guard_state,
            [{**call_record, "status": exec_result.status, "result": result}],
        )
        self._guard_state = after.state
        if after.hint:
            messages.append({"role": "system", "content": f"【执行保护】{after.hint}"})
        if after.stop_reason:
            self._guard_stopped = True
            return False
        return True

    @staticmethod
    def _append_tool_messages(
        messages: list[dict],
        *,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        result: str,
    ) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": f"调用工具 {tool_name}",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }
        )
