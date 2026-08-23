"""Serializable execution guard used by the existing LangGraph agent loop.

The guard does not execute tools or create another loop.  It only evaluates
the model/tool rounds already owned by ``LangGraphAgentExecutor`` and returns
one of three outcomes: continue, continue with a convergence hint, or route
the current graph to its existing summarize node.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from agent.tools.base import ToolResult

_POLLING_TOOLS = frozenset({"process_output", "process_stop"})
_VOLATILE_ARGUMENTS = frozenset({"description", "_commit_role"})


@dataclass(frozen=True)
class ExecutionGuardConfig:
    """Budgets and convergence thresholds for one Agent execution."""

    enabled: bool = True
    window_rounds: int = 6
    same_signature_warn: int = 2
    same_signature_stop: int = 3
    no_progress_rounds: int = 4
    max_tool_calls: int = 12
    soft_timeout_seconds: float = 600.0
    hard_timeout_seconds: float = 3900.0
    model_call_timeout_seconds: float = 180.0
    tool_timeout_seconds: float = 300.0
    side_effect_tool_timeout_seconds: float = 300.0
    blocking_tool_timeout_seconds: float = 3600.0
    context_soft_tokens: int = 120_000
    context_hard_tokens: int = 160_000
    max_tool_result_chars: int = 12_000
    max_tool_round_chars: int = 24_000
    max_turn_tool_result_chars: int = 60_000
    subagent_max_iterations: int = 10
    subagent_timeout_seconds: float = 3900.0
    subagent_result_chars: int = 12_000
    workflow_max_concurrency: int = 2
    workflow_step_timeout_seconds: float = 4200.0
    workflow_max_subagent_steps: int = 4

    def normalized(self) -> "ExecutionGuardConfig":
        warn = max(2, int(self.same_signature_warn))
        stop = max(warn + 1, int(self.same_signature_stop))
        soft_timeout = max(5.0, float(self.soft_timeout_seconds))
        hard_timeout = max(soft_timeout + 5.0, float(self.hard_timeout_seconds))
        context_soft = max(8_000, int(self.context_soft_tokens))
        context_hard = max(context_soft + 4_000, int(self.context_hard_tokens))
        result_chars = max(1_000, int(self.max_tool_result_chars))
        round_chars = max(result_chars, int(self.max_tool_round_chars))
        turn_chars = max(round_chars, int(self.max_turn_tool_result_chars))
        return replace(
            self,
            enabled=bool(self.enabled),
            window_rounds=max(4, int(self.window_rounds)),
            same_signature_warn=warn,
            same_signature_stop=stop,
            no_progress_rounds=max(2, int(self.no_progress_rounds)),
            max_tool_calls=max(1, int(self.max_tool_calls)),
            soft_timeout_seconds=soft_timeout,
            hard_timeout_seconds=hard_timeout,
            model_call_timeout_seconds=max(5.0, float(self.model_call_timeout_seconds)),
            tool_timeout_seconds=max(1.0, float(self.tool_timeout_seconds)),
            side_effect_tool_timeout_seconds=max(
                1.0, float(self.side_effect_tool_timeout_seconds)
            ),
            blocking_tool_timeout_seconds=max(
                float(self.tool_timeout_seconds),
                float(self.side_effect_tool_timeout_seconds),
                float(self.blocking_tool_timeout_seconds),
            ),
            context_soft_tokens=context_soft,
            context_hard_tokens=context_hard,
            max_tool_result_chars=result_chars,
            max_tool_round_chars=round_chars,
            max_turn_tool_result_chars=turn_chars,
            subagent_max_iterations=max(1, int(self.subagent_max_iterations)),
            subagent_timeout_seconds=max(5.0, float(self.subagent_timeout_seconds)),
            subagent_result_chars=max(1_000, int(self.subagent_result_chars)),
            workflow_max_concurrency=max(1, int(self.workflow_max_concurrency)),
            workflow_step_timeout_seconds=max(
                10.0, float(self.workflow_step_timeout_seconds)
            ),
            workflow_max_subagent_steps=max(1, int(self.workflow_max_subagent_steps)),
        )

    def for_subagent(self) -> "ExecutionGuardConfig":
        """Return the same guard with a child-sized result/call budget."""

        normalized = self.normalized()
        return replace(
            normalized,
            max_tool_calls=min(
                normalized.max_tool_calls,
                normalized.subagent_max_iterations,
            ),
            soft_timeout_seconds=min(
                normalized.soft_timeout_seconds,
                normalized.subagent_timeout_seconds * 0.75,
            ),
            hard_timeout_seconds=normalized.subagent_timeout_seconds,
            max_tool_result_chars=min(
                normalized.max_tool_result_chars,
                normalized.subagent_result_chars,
            ),
            max_tool_round_chars=min(
                normalized.max_tool_round_chars,
                normalized.subagent_result_chars * 2,
            ),
            max_turn_tool_result_chars=min(
                normalized.max_turn_tool_result_chars,
                normalized.subagent_result_chars * 4,
            ),
        ).normalized()


@dataclass(frozen=True)
class GuardDecision:
    state: dict[str, Any]
    stop_reason: str = ""
    hint: str = ""
    disabled_tools: tuple[str, ...] = ()


class ExecutionGuard:
    """Evaluate bounded execution signals without owning control flow."""

    def __init__(self, config: ExecutionGuardConfig | None = None) -> None:
        self.config = (config or ExecutionGuardConfig()).normalized()

    def initial_state(self) -> dict[str, Any]:
        return {
            "started_at": time.time(),
            "recent_rounds": [],
            "tool_calls_total": 0,
            "tool_result_chars": 0,
            "no_progress_streak": 0,
            "failure_counts": {},
            "soft_timeout_warned": False,
            "context_warned": False,
            "convergence_stage": 0,
        }

    def resume_state(self, value: object) -> dict[str, Any]:
        state = dict(value) if isinstance(value, Mapping) else self.initial_state()
        state["started_at"] = time.time()
        return state

    def before_model(
        self,
        value: object,
        *,
        context_tokens: int,
    ) -> GuardDecision:
        state = self._state(value)
        if not self.config.enabled:
            return GuardDecision(state)
        elapsed = self._elapsed(state)
        if elapsed >= self.config.hard_timeout_seconds:
            return GuardDecision(state, stop_reason="turn_timeout")
        if context_tokens >= self.config.context_hard_tokens:
            return GuardDecision(state, stop_reason="context_budget")

        hints: list[str] = []
        if elapsed >= self.config.soft_timeout_seconds and not state.get(
            "soft_timeout_warned"
        ):
            state["soft_timeout_warned"] = True
            state["convergence_stage"] = max(
                1, int(state.get("convergence_stage") or 0)
            )
            hints.append(
                "本轮已接近时间预算。停止扩展非必要步骤；已有证据足够时立即给出结论，"
                "复杂剩余工作转为可恢复 Workflow。"
            )
        if context_tokens >= self.config.context_soft_tokens and not state.get(
            "context_warned"
        ):
            state["context_warned"] = True
            state["convergence_stage"] = max(
                1, int(state.get("convergence_stage") or 0)
            )
            hints.append(
                "本轮上下文已接近预算。禁止继续获取大段原文，只保留关键证据并尽快收敛。"
            )
        return GuardDecision(state, hint="\n".join(hints))

    def before_tool_batch(
        self,
        value: object,
        calls: Sequence[object],
        *,
        risk_resolver: Callable[[str], str],
    ) -> GuardDecision:
        state = self._state(value)
        if not self.config.enabled:
            return GuardDecision(state)
        if self._elapsed(state) >= self.config.hard_timeout_seconds:
            return GuardDecision(state, stop_reason="turn_timeout")
        if (
            int(state.get("tool_calls_total") or 0) + len(calls)
            > self.config.max_tool_calls
        ):
            return GuardDecision(state, stop_reason="tool_budget")

        signature = tool_batch_signature(calls)
        if not signature:
            return GuardDecision(state)
        recent = self._recent_rounds(state)
        same_seen = any(item.get("signature") == signature for item in recent)
        has_side_effect = any(
            risk_resolver(_call_name(call)) != "read-only" for call in calls
        )
        if has_side_effect and same_seen:
            return GuardDecision(state, stop_reason="duplicate_side_effect")

        consecutive = 1
        for item in reversed(recent):
            if item.get("signature") != signature:
                break
            consecutive += 1
        if consecutive >= self.config.same_signature_stop:
            return GuardDecision(state, stop_reason="tool_call_loop")
        return GuardDecision(state)

    def after_tool_round(
        self,
        value: object,
        calls: Sequence[Mapping[str, Any]],
    ) -> GuardDecision:
        state = self._state(value)
        if not self.config.enabled or not calls:
            return GuardDecision(state)

        signature = tool_batch_signature(calls)
        result_digest = _result_digest(calls)
        result_chars = sum(len(str(item.get("result") or "")) for item in calls)
        all_failed = all(
            str(item.get("status") or "") not in {"success", "completed"}
            for item in calls
        )
        recent = self._recent_rounds(state)
        previous = recent[-1] if recent else {}
        repeated_result = bool(
            result_digest and result_digest == previous.get("result_digest")
        )
        no_progress = all_failed or repeated_result
        no_progress_streak = (
            int(state.get("no_progress_streak") or 0) + 1 if no_progress else 0
        )

        record = {
            "signature": signature,
            "result_digest": result_digest,
            "all_failed": all_failed,
            "tools": [_call_name(call) for call in calls],
        }
        recent.append(record)
        recent = recent[-self.config.window_rounds :]
        state["recent_rounds"] = recent
        state["tool_calls_total"] = int(state.get("tool_calls_total") or 0) + len(calls)
        state["tool_result_chars"] = (
            int(state.get("tool_result_chars") or 0) + result_chars
        )
        state["no_progress_streak"] = no_progress_streak

        failure_counts = {
            str(key): int(count)
            for key, count in dict(state.get("failure_counts") or {}).items()
        }
        disabled: list[str] = []
        for call in calls:
            name = _call_name(call)
            status = str(call.get("status") or "")
            if status in {"success", "completed"}:
                failure_counts[name] = 0
                continue
            failure_counts[name] = failure_counts.get(name, 0) + 1
            if failure_counts[name] >= 3 and name not in _POLLING_TOOLS:
                disabled.append(name)
        state["failure_counts"] = failure_counts

        hints: list[str] = []
        if signature:
            consecutive = 0
            for item in reversed(recent):
                if item.get("signature") != signature:
                    break
                consecutive += 1
            if consecutive >= self.config.same_signature_warn:
                hints.append(
                    "相同工具和参数已经重复且没有证明获得新进展。下一步必须更换方法，"
                    "或直接基于已有证据结束；禁止再次原样调用。"
                )
        if disabled:
            hints.append(
                "以下工具本轮连续失败，已临时禁用："
                + "、".join(sorted(set(disabled)))
                + "。只允许选择替代能力或结束任务。"
            )
        if hints:
            state["convergence_stage"] = max(
                1, int(state.get("convergence_stage") or 0)
            )

        stop_reason = ""
        if _is_oscillation(recent):
            stop_reason = "tool_oscillation"
        elif no_progress_streak >= self.config.no_progress_rounds:
            stop_reason = "no_progress"
        elif int(state["tool_result_chars"]) >= self.config.max_turn_tool_result_chars:
            stop_reason = "tool_result_budget"
        return GuardDecision(
            state,
            stop_reason=stop_reason,
            hint="\n".join(hints),
            disabled_tools=tuple(sorted(set(disabled))),
        )

    def result_limit(
        self,
        value: object,
        *,
        round_chars: int,
    ) -> int:
        state = self._state(value)
        turn_remaining = max(
            512,
            self.config.max_turn_tool_result_chars
            - int(state.get("tool_result_chars") or 0),
        )
        round_remaining = max(
            512,
            self.config.max_tool_round_chars - max(0, int(round_chars)),
        )
        return max(
            512,
            min(
                self.config.max_tool_result_chars,
                turn_remaining,
                round_remaining,
            ),
        )

    def tool_timeout(
        self,
        risk: str,
        *,
        tool_name: str = "",
        arguments: Mapping[str, Any] | None = None,
    ) -> float:
        base_timeout = (
            self.config.tool_timeout_seconds
            if risk == "read-only"
            else self.config.side_effect_tool_timeout_seconds
        )
        if tool_name != "shell" or not isinstance(arguments, Mapping):
            return base_timeout
        if arguments.get("auto_promote") is not False:
            return base_timeout

        # shell(auto_promote=false) is deliberately synchronous (for example
        # codex-delegate).  Its own timeout must get a chance to return a
        # structured result instead of being cancelled by the generic
        # side-effect budget first.
        raw_timeout = arguments.get("timeout")
        if raw_timeout is None:
            return self.config.blocking_tool_timeout_seconds
        try:
            requested = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            return self.config.blocking_tool_timeout_seconds
        return min(
            self.config.blocking_tool_timeout_seconds,
            max(base_timeout, requested + 5.0),
        )

    def _state(self, value: object) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return self.initial_state()
        state = dict(value)
        state.setdefault("started_at", time.time())
        state.setdefault("recent_rounds", [])
        state.setdefault("tool_calls_total", 0)
        state.setdefault("tool_result_chars", 0)
        state.setdefault("no_progress_streak", 0)
        state.setdefault("failure_counts", {})
        return state

    @staticmethod
    def _recent_rounds(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in (state.get("recent_rounds") or [])
            if isinstance(item, Mapping)
        ]

    @staticmethod
    def _elapsed(state: Mapping[str, Any]) -> float:
        try:
            return max(0.0, time.time() - float(state.get("started_at") or 0.0))
        except (TypeError, ValueError):
            return 0.0


def tool_batch_signature(calls: Sequence[object]) -> str:
    payload: list[dict[str, Any]] = []
    for call in calls:
        name = _call_name(call)
        if not name or name in _POLLING_TOOLS:
            continue
        arguments = _call_arguments(call)
        canonical = {
            str(key): value
            for key, value in arguments.items()
            if str(key) not in _VOLATILE_ARGUMENTS
        }
        payload.append({"name": name, "arguments": canonical})
    if not payload:
        return ""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def bound_tool_result(result: ToolResult, limit: int) -> ToolResult:
    """Deterministically retain the head/tail instead of filling graph state."""

    text = result.text
    if len(text) <= limit:
        return result
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    marker = (
        f"[tool_result bounded original_chars={len(text)} sha256={digest} "
        f"limit={limit}]"
    )
    available = max(0, limit - len(marker) - 8)
    head = (available * 2) // 3
    tail = available - head
    suffix = text[-tail:] if tail else ""
    return ToolResult(
        text=f"{marker}\n{text[:head]}\n…\n{suffix}",
        content_blocks=list(result.content_blocks),
    )


def _call_name(call: object) -> str:
    if isinstance(call, Mapping):
        return str(call.get("name") or "")
    return str(getattr(call, "name", "") or "")


def _call_arguments(call: object) -> dict[str, Any]:
    raw = (
        call.get("arguments")
        if isinstance(call, Mapping)
        else getattr(call, "arguments", {})
    )
    return dict(raw) if isinstance(raw, Mapping) else {}


def _result_digest(calls: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "name": _call_name(call),
            "status": str(call.get("status") or ""),
            "result": str(call.get("result") or "")[:8_000],
        }
        for call in calls
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _is_oscillation(recent: Sequence[Mapping[str, Any]]) -> bool:
    if len(recent) < 4:
        return False
    a, b, c, d = recent[-4:]
    signature_a = str(a.get("signature") or "")
    signature_b = str(b.get("signature") or "")
    if not signature_a or signature_a == signature_b:
        return False
    same_pattern = signature_a == c.get("signature") and signature_b == d.get(
        "signature"
    )
    if not same_pattern:
        return False
    same_results = a.get("result_digest") == c.get("result_digest") and b.get(
        "result_digest"
    ) == d.get("result_digest")
    all_failed = all(bool(item.get("all_failed")) for item in (a, b, c, d))
    return bool(same_results or all_failed)


__all__ = [
    "ExecutionGuard",
    "ExecutionGuardConfig",
    "GuardDecision",
    "bound_tool_result",
    "tool_batch_signature",
]
