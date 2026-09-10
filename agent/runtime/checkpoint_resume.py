"""Reconcile interrupted execution state before accepting a new model decision."""

from __future__ import annotations

from typing import Any, Mapping

from agent.tool_runtime import append_tool_result


def reconcile_checkpoint(
    saved: Mapping[str, Any],
    initial_messages: list[dict[str, Any]],
    *,
    request_text: str,
    channel: str,
    chat_id: str,
    permission_mode: str,
    delegation_allowed: bool,
) -> dict[str, Any]:
    """Keep committed results, close dangling calls, then re-plan with current intent.

    A checkpoint is not a remote side-effect receipt. An interrupted call is
    explicitly unknown, not silently replayed or falsely labelled unexecuted.
    """
    messages = [dict(message) for message in saved.get("messages", [])]
    current_system = [dict(m) for m in initial_messages if m.get("role") == "system"]
    if current_system:
        messages = current_system + [m for m in messages if m.get("role") != "system"]
    calls = saved.get("pending_tool_calls") or []
    index = int(saved.get("pending_tool_index") or 0)
    results = list(saved.get("pending_tool_results") or [])
    for call in calls[index:]:
        if not isinstance(call, Mapping):
            continue
        call_id = str(call.get("id") or call.get("call_id") or "unknown")
        call_name = str(call.get("name") or "unknown")
        feedback = (
            "执行在本次工具结果落盘前中断，结果未确认，可能尚未执行或已经产生副作用。"
            "恢复时已取消旧的待执行计划。先检查文件、进程状态或发送回执，"
            "不得直接重复发送、支付、删除等副作用；无法核验时向用户说明并请求确认。"
        )
        append_tool_result(messages, tool_call_id=call_id, content=feedback, tool_name=call_name)
        results.append({
            "call_id": call_id, "name": call_name,
            "arguments": call.get("arguments", {}),
            "status": "interrupted_unknown", "result": feedback,
        })
    chain = list(saved.get("tool_chain") or [])
    if calls:
        chain.append({"text": saved.get("pending_content") or "", "calls": results})
    messages.append({
        "role": "user",
        "content": (
            "[运行时恢复上下文] 已保留此前已提交的工具结果；先根据本次用户要求重新判断下一步，"
            "不要盲目继续旧计划，也不要重复已完成操作。若用户已改换任务，以新要求为准。\n"
            f"当前渠道={channel}，会话={chat_id}，权限={permission_mode}，"
            f"自主多 Agent={'开启' if delegation_allowed else '关闭'}。\n"
            "旧后台任务的进程状态不能仅凭历史 task_id 判断；先核验是否仍存在。"
        ),
    })
    latest_user = next((m for m in reversed(initial_messages) if m.get("role") == "user"), None)
    if latest_user is not None:
        messages.append(dict(latest_user))
    elif request_text:
        messages.append({"role": "user", "content": request_text})
    return {
        "messages": messages, "tool_chain": chain,
        "pending_tool_calls": [], "pending_tool_index": 0,
        "pending_tool_results": [], "pending_content": "",
        "pending_thinking": None, "pending_provider_fields": {},
        "reply": "", "thinking": None, "streamed": False,
        "exit_reason": "", "summary_reason": "", "early_stop_reply": "",
    }
