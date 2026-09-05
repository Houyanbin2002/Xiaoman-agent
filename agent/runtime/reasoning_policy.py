"""Explicit, request-scoped reasoning controls for Agent executions.

This module intentionally does not classify task text. The selected effort is
resolved from an explicit turn/session override or the configured default, then
translated to the closest setting supported by the active model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
_VALID_EFFORTS = frozenset(REASONING_EFFORTS)
_QWEN_BUDGETS = {
    "minimal": 1_024,
    "low": 4_096,
    "medium": 8_192,
    "high": 16_384,
    "xhigh": 32_768,
    "max": 65_536,
}


def normalize_reasoning_effort(
    value: object,
    *,
    default: str = "",
) -> str:
    """Return one supported generic effort or ``default`` for an empty value."""

    effort = str(value or "").strip().lower()
    if not effort:
        return default
    if effort not in _VALID_EFFORTS:
        allowed = "/".join(REASONING_EFFORTS)
        raise ValueError(f"思考等级必须是 {allowed}")
    return effort


@dataclass(frozen=True)
class ReasoningPolicyConfig:
    """Global default plus optional executor-specific overrides.

    Empty executor overrides mean "inherit the originating turn", matching
    Hermes' delegation behavior.
    """

    default_effort: str = "medium"
    subagent_effort: str = ""
    workflow_effort: str = ""

    def normalized(self) -> ReasoningPolicyConfig:
        default_effort = normalize_reasoning_effort(
            self.default_effort,
            default="medium",
        )
        subagent_effort = normalize_reasoning_effort(self.subagent_effort)
        workflow_effort = normalize_reasoning_effort(self.workflow_effort)
        return replace(
            self,
            default_effort=default_effort,
            subagent_effort=subagent_effort,
            workflow_effort=workflow_effort,
        )


@dataclass(frozen=True)
class ReasoningDecision:
    """One immutable reasoning selection for an Agent execution/checkpoint."""

    requested_effort: str
    effective_effort: str
    enabled: bool
    reason: str

    @property
    def mode(self) -> str:
        return self.requested_effort

    @property
    def effort(self) -> str:
        """Compatibility alias used by existing runtime metadata."""

        return self.effective_effort if self.enabled else ""

    def request_extra_body(self, model: str) -> dict[str, object]:
        """Translate the generic selection to Bailian/OpenAI-compatible fields."""

        if not self.enabled:
            return {}

        model_name = model.strip().lower()
        payload: dict[str, object] = {"enable_thinking": True}
        if _is_deepseek_v4(model_name):
            payload["reasoning_effort"] = self.effective_effort
        elif "qwen3.8-max" in model_name:
            payload["reasoning_effort"] = self.effective_effort
        elif _is_qwen_hybrid(model_name):
            payload["thinking_budget"] = _QWEN_BUDGETS[self.effective_effort]
        elif "gpt-oss" in model_name or model_name.startswith("step-"):
            payload["reasoning_effort"] = self.effective_effort
        return payload

    def to_metadata(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "requested_effort": self.requested_effort,
            "effective_effort": self.effective_effort if self.enabled else None,
            "reason": self.reason,
        }


class ReasoningPolicy:
    """Resolve explicit choices without inspecting or classifying user text."""

    def __init__(self, config: ReasoningPolicyConfig | None = None) -> None:
        self.config = (config or ReasoningPolicyConfig()).normalized()

    def decide(
        self,
        requested_effort: object = "",
        *,
        source: str,
        session_key: str = "",
        model: str = "",
    ) -> ReasoningDecision:
        explicit = normalize_reasoning_effort(requested_effort)
        if source == "subagent" and self.config.subagent_effort:
            selected = self.config.subagent_effort
            reason = "subagent_override"
        elif session_key.startswith("workflow:") and self.config.workflow_effort:
            selected = self.config.workflow_effort
            reason = "workflow_override"
        elif explicit:
            selected = explicit
            reason = "turn_or_session_override"
        else:
            selected = self.config.default_effort
            reason = "global_default"

        if selected == "none":
            return ReasoningDecision(
                requested_effort="none",
                effective_effort="none",
                enabled=False,
                reason=reason,
            )
        return ReasoningDecision(
            requested_effort=selected,
            effective_effort=_effective_effort(model, selected),
            enabled=True,
            reason=reason,
        )


def _effective_effort(model: str, effort: str) -> str:
    model_name = model.strip().lower()
    if _is_deepseek_v4(model_name):
        return "max" if effort in {"xhigh", "max"} else "high"
    if "qwen3.8-max" in model_name:
        if effort in {"minimal", "low"}:
            return "low"
        if effort == "medium":
            return "medium"
        return "xhigh"
    if _is_qwen_hybrid(model_name):
        return effort
    if "gpt-oss" in model_name or model_name.startswith("step-"):
        if effort in {"minimal", "low"}:
            return "low"
        if effort in {"xhigh", "max"}:
            return "high"
        return effort
    return effort


def _is_deepseek_v4(model: str) -> bool:
    return "deepseek-v4" in model


def _is_qwen_hybrid(model: str) -> bool:
    return any(
        marker in model
        for marker in (
            "qwen3.7",
            "qwen3.6",
            "qwen3.5",
            "qwen3-vl",
        )
    )


__all__ = [
    "REASONING_EFFORTS",
    "ReasoningDecision",
    "ReasoningPolicy",
    "ReasoningPolicyConfig",
    "normalize_reasoning_effort",
]
