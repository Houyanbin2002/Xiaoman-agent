"""Local evaluation harness for Xiaoman Agent.

The package deliberately has no dependency on a hosted evaluator.  It can run
against deterministic replay fixtures in CI and against a real ``AgentLoop``
through :class:`eval.runner.ProcessDirectExecutor`.
"""

from .models import (
    AgentRun,
    CaseResult,
    EvalCase,
    EvalSummary,
    RubricCriterion,
    Score,
    ToolCall,
)
from .runner import EvalHarness, ProcessDirectExecutor, ReplayExecutor
from .store import EvalResultStore
from .publishers import LangfuseScorePublisher, publish_best_effort
from .compare import EvalComparison, compare
from .scorers import canonical_rubric
from .analysis import failure_hotspots
from .judge import OpenAICompatibleRubricJudge, RubricJudgeError, build_judge_from_config

__all__ = [
    "AgentRun",
    "CaseResult",
    "EvalCase",
    "EvalHarness",
    "EvalSummary",
    "EvalResultStore",
    "EvalComparison",
    "LangfuseScorePublisher",
    "ProcessDirectExecutor",
    "ReplayExecutor",
    "RubricCriterion",
    "Score",
    "ToolCall",
    "publish_best_effort",
    "compare",
    "canonical_rubric",
    "failure_hotspots",
    "OpenAICompatibleRubricJudge",
    "RubricJudgeError",
    "build_judge_from_config",
]
