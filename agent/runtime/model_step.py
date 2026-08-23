from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from core.llm import LLMProvider, LLMResponse, StreamDelta
from core.tracing import new_span_id, record_trace_event


async def run_model_step(
    provider: LLMProvider,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    tool_choice: str | dict[str, Any] = "auto",
    extra_body: dict[str, Any] | None = None,
    disable_thinking: bool = False,
    on_content_delta: Callable[[StreamDelta], Awaitable[None]] | None = None,
    source: str,
    iteration: int,
    purpose: str = "reasoning",
    cache_metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Invoke one model step and attach mode-neutral trace metadata."""

    span_id = new_span_id()
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        response = await provider.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            extra_body=extra_body,
            disable_thinking=disable_thinking,
            on_content_delta=on_content_delta,
        )
    except BaseException as exc:
        record_trace_event(
            category="model",
            name=purpose,
            summary=f"{source} 第 {iteration} 轮模型调用失败",
            status=(
                "interrupted" if type(exc).__name__ == "CancelledError" else "failed"
            ),
            started_at=started_wall,
            duration_ms=int((time.perf_counter() - started) * 1000),
            span_id=span_id,
            payload={
                "source": source,
                "iteration": iteration,
                "model": model,
                "input": _trace_snapshot(messages, max_chars=24000),
                "message_count": len(messages),
                "tool_schema_count": len(tools),
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
                "prompt_cache": dict(cache_metadata or {}),
            },
        )
        raise

    tool_calls = list(getattr(response, "tool_calls", ()) or ())
    tool_names = [str(getattr(call, "name", "") or "") for call in tool_calls]
    tool_names = [name for name in tool_names if name]
    record_trace_event(
        category="model",
        name=purpose,
        summary=(
            f"模型选择工具：{'、'.join(tool_names)}" if tool_names else "模型生成回复"
        ),
        started_at=started_wall,
        duration_ms=int((time.perf_counter() - started) * 1000),
        span_id=span_id,
        payload={
            "source": source,
            "iteration": iteration,
            "model": model,
            "input": _trace_snapshot(messages, max_chars=24000),
            "output": _trace_snapshot(
                {
                    "content": getattr(response, "content", None),
                    "thinking": getattr(response, "thinking", None),
                    "tool_calls": [
                        {
                            "id": str(getattr(call, "id", "") or ""),
                            "name": str(getattr(call, "name", "") or ""),
                            "arguments": dict(getattr(call, "arguments", {}) or {}),
                        }
                        for call in tool_calls
                    ],
                },
                max_chars=12000,
            ),
            "message_count": len(messages),
            "tool_schema_count": len(tools),
            "tool_names": tool_names,
            "content_chars": len(getattr(response, "content", "") or ""),
            "thinking_chars": len(getattr(response, "thinking", "") or ""),
            "input_tokens": getattr(response, "input_tokens", None),
            "output_tokens": getattr(response, "output_tokens", None),
            "total_tokens": getattr(response, "total_tokens", None),
            "cache_prompt_tokens": getattr(response, "cache_prompt_tokens", None),
            "cache_hit_tokens": getattr(response, "cache_hit_tokens", None),
            "finish_reason": getattr(response, "finish_reason", None),
            "prompt_cache": dict(cache_metadata or {}),
        },
    )
    return response


def _trace_snapshot(value: Any, *, max_chars: int) -> Any:
    """Detach mutable provider payloads and cap local trace growth."""

    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= max_chars:
        return json.loads(encoded)
    return {
        "truncated": True,
        "original_chars": len(encoded),
        "preview": encoded[:max_chars],
    }
