"""Cache-aware model views for long-running Agent loops.

The execution state always keeps the complete tool trace.  This module only
builds a deterministic, bounded view for a model call.  Keeping the transform
pure is important: LangGraph checkpoints remain lossless and an interrupted
run produces the same cache prefix after it is resumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromptCacheConfig:
    enabled: bool = True
    keep_recent_tool_rounds: int = 3
    cold_tool_result_chars: int = 1800
    recent_tool_result_chars: int = 24000

    def normalized(self) -> "PromptCacheConfig":
        return PromptCacheConfig(
            enabled=bool(self.enabled),
            keep_recent_tool_rounds=max(1, int(self.keep_recent_tool_rounds)),
            cold_tool_result_chars=max(400, int(self.cold_tool_result_chars)),
            recent_tool_result_chars=max(
                max(800, int(self.cold_tool_result_chars)),
                int(self.recent_tool_result_chars),
            ),
        )


@dataclass(frozen=True)
class PromptCachePlan:
    enabled: bool
    breakpoint_index: int
    stable_prefix_messages: int
    protected_tool_rounds: int
    raw_chars: int
    model_view_chars: int
    chars_saved: int
    compacted_tool_messages: int
    compacted_artifact_messages: int
    capped_recent_tool_messages: int
    stable_prefix_hash: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptCacheView:
    messages: list[dict[str, Any]]
    plan: PromptCachePlan


class PromptCacheOptimizer:
    """Build a cache-stable view without mutating the durable graph state."""

    def __init__(self, config: PromptCacheConfig | None = None) -> None:
        self.config = (config or PromptCacheConfig()).normalized()

    def prepare_model_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        keep_recent_tool_rounds: int | None = None,
    ) -> PromptCacheView:
        config = self.config
        rounds = max(
            1,
            int(
                config.keep_recent_tool_rounds
                if keep_recent_tool_rounds is None
                else keep_recent_tool_rounds
            ),
        )
        tool_round_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        breakpoint_index = (
            tool_round_indices[max(0, len(tool_round_indices) - rounds)]
            if tool_round_indices
            else len(messages)
        )
        protected_rounds = min(rounds, len(tool_round_indices))

        if not config.enabled:
            return PromptCacheView(
                messages=messages,
                plan=_build_plan(
                    enabled=False,
                    messages=messages,
                    view=messages,
                    breakpoint_index=breakpoint_index,
                    protected_rounds=protected_rounds,
                ),
            )

        tool_names = _tool_names_by_call_id(messages)
        view: list[dict[str, Any]] = []
        compacted_tools = 0
        compacted_artifacts = 0
        capped_recent = 0
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            content_text = _content_text(content)
            limit: int | None = None
            marker = ""

            if role == "tool":
                is_cold = index < breakpoint_index
                limit = (
                    config.cold_tool_result_chars
                    if is_cold
                    else config.recent_tool_result_chars
                )
                if len(content_text) > limit:
                    call_id = str(message.get("tool_call_id") or "")
                    marker = _tool_marker(
                        content_text,
                        tool_name=tool_names.get(call_id, "unknown"),
                        cold=is_cold,
                    )
                    if is_cold:
                        compacted_tools += 1
                    else:
                        capped_recent += 1
            elif (
                index < breakpoint_index
                and role == "user"
                and isinstance(content, list)
                and index > 0
                and messages[index - 1].get("role") == "tool"
            ):
                # Tool content blocks (file/image payloads) are represented as a
                # following user message.  Once the producing tool round becomes
                # cold, keep a deterministic textual preview instead of replaying
                # the whole artifact on every model call.
                limit = config.cold_tool_result_chars
                if len(content_text) > limit:
                    marker = _artifact_marker(content_text)
                    compacted_artifacts += 1

            if limit is None or not marker:
                view.append(message)
                continue
            view.append(
                {
                    **message,
                    "content": _bounded_preview(content_text, limit, marker),
                }
            )

        return PromptCacheView(
            messages=view,
            plan=_build_plan(
                enabled=True,
                messages=messages,
                view=view,
                breakpoint_index=breakpoint_index,
                protected_rounds=protected_rounds,
                compacted_tools=compacted_tools,
                compacted_artifacts=compacted_artifacts,
                capped_recent=capped_recent,
            ),
        )


def tool_schema_fingerprint(schemas: list[dict[str, Any]]) -> str:
    """Return a stable signature so schema drift is visible in model traces."""
    canonical = json.dumps(
        schemas,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _build_plan(
    *,
    enabled: bool,
    messages: list[dict[str, Any]],
    view: list[dict[str, Any]],
    breakpoint_index: int,
    protected_rounds: int,
    compacted_tools: int = 0,
    compacted_artifacts: int = 0,
    capped_recent: int = 0,
) -> PromptCachePlan:
    raw_chars = _messages_chars(messages)
    view_chars = _messages_chars(view)
    stable_prefix = view[:breakpoint_index]
    canonical = json.dumps(
        stable_prefix,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return PromptCachePlan(
        enabled=enabled,
        breakpoint_index=breakpoint_index,
        stable_prefix_messages=len(stable_prefix),
        protected_tool_rounds=protected_rounds,
        raw_chars=raw_chars,
        model_view_chars=view_chars,
        chars_saved=max(0, raw_chars - view_chars),
        compacted_tool_messages=compacted_tools,
        compacted_artifact_messages=compacted_artifacts,
        capped_recent_tool_messages=capped_recent,
        stable_prefix_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    )


def _messages_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(_content_text(message.get("content"))) for message in messages)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(
                        json.dumps(
                            block,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        )
                    )
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return str(content)


def _tool_names_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            function = call.get("function")
            name = (
                str(function.get("name") or "")
                if isinstance(function, dict)
                else str(call.get("name") or "")
            )
            if call_id:
                names[call_id] = name or "unknown"
    return names


def _tool_marker(text: str, *, tool_name: str, cold: bool) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    status = _result_status(text)
    tier = "cold" if cold else "recent-capped"
    return (
        f"[cache-aware tool_result tier={tier} tool={tool_name} "
        f"status={status} original_chars={len(text)} sha256={digest}]"
    )


def _artifact_marker(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return (
        "[cache-aware historical_tool_artifact "
        f"original_chars={len(text)} sha256={digest}]"
    )


def _result_status(text: str) -> str:
    lowered = text[:2000].lower()
    if any(token in lowered for token in ("traceback", "exception", "error", "失败")):
        return "error"
    return "completed"


def _bounded_preview(text: str, limit: int, marker: str) -> str:
    if len(text) <= limit:
        return text
    available = max(0, limit - len(marker) - 12)
    head = (available * 2) // 3
    tail = available - head
    suffix = text[-tail:] if tail else ""
    return f"{marker}\n{text[:head]}\n…\n{suffix}"
