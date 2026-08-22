from __future__ import annotations

"""Serializable contracts shared by runners, evaluators and reports."""

from dataclasses import dataclass, field
from typing import Any, Mapping


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool invocation extracted from an Agent trace."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    output: Any = None
    error: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolCall":
        args = value.get("arguments", value.get("args", {}))
        return cls(
            name=str(value.get("name", value.get("tool", ""))),
            arguments=_as_dict(args if isinstance(args, Mapping) else {}),
            status=str(value.get("status", value.get("outcome", "completed"))),
            output=value.get("output", value.get("result")),
            error=str(value.get("error", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "status": self.status,
            "output": self.output,
            "error": self.error,
        }


@dataclass(frozen=True)
class AgentRun:
    """The small, evaluator-oriented view of one Agent execution."""

    response: str = ""
    tools: tuple[ToolCall, ...] = ()
    state: dict[str, Any] = field(default_factory=dict)
    memory_events: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    latency_ms: float | None = None
    trace_id: str = ""

    @classmethod
    def from_value(cls, value: "AgentRun | Mapping[str, Any] | str") -> "AgentRun":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(response=value)
        tools = value.get("tools", value.get("tool_calls", ()))
        normalized_tools = tuple(
            item if isinstance(item, ToolCall) else ToolCall.from_dict(item)
            for item in (tools or ())
            if isinstance(item, (ToolCall, Mapping))
        )
        memory_events = value.get("memory_events", value.get("memory", ()))
        return cls(
            response=str(value.get("response", value.get("output", ""))),
            tools=normalized_tools,
            state=_as_dict(value.get("state")),
            memory_events=tuple(
                dict(item) for item in (memory_events or ()) if isinstance(item, Mapping)
            ),
            metadata=_as_dict(value.get("metadata")),
            status=str(value.get("status", "completed")),
            latency_ms=(
                float(value["latency_ms"])
                if value.get("latency_ms") is not None
                else None
            ),
            trace_id=str(value.get("trace_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "tools": [tool.to_dict() for tool in self.tools],
            "state": dict(self.state),
            "memory_events": [dict(item) for item in self.memory_events],
            "metadata": dict(self.metadata),
            "status": self.status,
            "latency_ms": self.latency_ms,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class RubricCriterion:
    """One criterion in a case-level rubric.

    ``evaluator`` selects the implementation behind the criterion.  The Rubric
    remains the only public evaluation contract while a criterion can be backed
    by an exact deterministic check or an LLM judge.
    """

    criterion_id: str
    description: str
    weight: float = 1.0
    threshold: float = 0.5
    hard: bool = False
    check: str = ""
    evaluator: str = "deterministic"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RubricCriterion":
        check = str(value.get("check", ""))
        evaluator = str(value.get("evaluator", "deterministic" if check else "llm"))
        if evaluator not in {"deterministic", "llm"}:
            raise ValueError(f"unsupported rubric evaluator: {evaluator}")
        return cls(
            criterion_id=str(value.get("id", value.get("criterion_id", ""))),
            description=str(value.get("description", "")),
            weight=float(value.get("weight", 1.0)),
            threshold=float(value.get("threshold", 0.5)),
            hard=bool(value.get("hard", False)),
            check=check,
            evaluator=evaluator,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.criterion_id,
            "description": self.description,
            "weight": self.weight,
            "threshold": self.threshold,
            "hard": self.hard,
            "check": self.check,
            "evaluator": self.evaluator,
        }


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    title: str
    input: str
    expected: dict[str, Any] = field(default_factory=dict)
    rubric: tuple[RubricCriterion, ...] = ()
    initial_state: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)
    replay: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvalCase":
        rubric = value.get("rubric", ())
        return cls(
            case_id=str(value.get("case_id", "")),
            title=str(value.get("title", value.get("case_id", ""))),
            input=str(value.get("input", value.get("request", ""))),
            expected=_as_dict(value.get("expected")),
            rubric=tuple(
                item if isinstance(item, RubricCriterion) else RubricCriterion.from_dict(item)
                for item in (rubric or ())
                if isinstance(item, (RubricCriterion, Mapping))
            ),
            initial_state=_as_dict(value.get("initial_state")),
            tags=tuple(str(item) for item in (value.get("tags", ()) or ())),
            version=str(value.get("version", "v1")),
            metadata=_as_dict(value.get("metadata")),
            replay=_as_dict(value.get("replay")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "input": self.input,
            "expected": dict(self.expected),
            "rubric": [item.to_dict() for item in self.rubric],
            "initial_state": dict(self.initial_state),
            "tags": list(self.tags),
            "version": self.version,
            "metadata": dict(self.metadata),
            "replay": dict(self.replay),
        }


@dataclass(frozen=True)
class Score:
    name: str
    value: float
    passed: bool
    weight: float = 1.0
    hard: bool = False
    reason: str = ""
    source: str = "deterministic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", max(0.0, min(1.0, float(self.value))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "weight": self.weight,
            "hard": self.hard,
            "reason": self.reason,
            "source": self.source,
        }


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    title: str
    passed: bool
    reward: float
    scores: tuple[Score, ...]
    run: AgentRun
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "passed": self.passed,
            "reward": self.reward,
            "scores": [score.to_dict() for score in self.scores],
            "run": self.run.to_dict(),
            "error": self.error,
        }


@dataclass(frozen=True)
class EvalSummary:
    dataset: str
    version: str
    total: int
    passed: int
    pass_rate: float
    mean_reward: float
    results: tuple[CaseResult, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    slices: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "mean_reward": self.mean_reward,
            "metrics": dict(self.metrics),
            "slices": {name: dict(values) for name, values in self.slices.items()},
            "results": [result.to_dict() for result in self.results],
        }
