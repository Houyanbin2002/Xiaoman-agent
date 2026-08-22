from __future__ import annotations

"""Optional score exporters; Langfuse is not a runtime dependency of eval."""

from collections.abc import Iterable
from typing import Any, Protocol

from .models import EvalSummary, Score


class ScorePublisher(Protocol):
    def publish(self, summary: EvalSummary) -> None: ...


class LangfuseScorePublisher:
    """Publish per-case scores to an already configured Langfuse client."""

    def __init__(self, client: Any, *, environment: str = "development") -> None:
        self.client = client
        self.environment = environment

    def publish(self, summary: EvalSummary) -> None:
        create_score = getattr(self.client, "create_score", None)
        if not callable(create_score):
            raise TypeError("Langfuse client does not expose create_score")
        for result in summary.results:
            if not result.run.trace_id:
                continue
            trace_id = _resolve_trace_id(self.client, result.run.trace_id)
            for score in result.scores:
                _publish_score(create_score, trace_id, score, summary)
            _publish_score(
                create_score,
                trace_id,
                Score("eval_reward", result.reward, result.passed, reason="aggregated reward"),
                summary,
            )


def _publish_score(create_score: Any, trace_id: str, score: Score, summary: EvalSummary) -> None:
    create_score(
        name=score.name,
        value=score.value,
        trace_id=trace_id,
        data_type="NUMERIC",
        comment=score.reason,
        metadata={
            "dataset": summary.dataset,
            "version": summary.version,
            "passed": score.passed,
            "hard": score.hard,
            "source": score.source,
        },
    )


def _resolve_trace_id(client: Any, trace_id: str) -> str:
    """Map Xiaoman's deterministic local id to the Langfuse trace id.

    ``LangfuseTraceRecorder`` creates remote traces with
    ``create_trace_id(seed=<local id>)``. Repeating that deterministic mapping
    lets the evaluator attach scores to the remote trace instead of sending a
    score for a non-existent local-only id.
    """

    create_trace_id = getattr(client, "create_trace_id", None)
    if not callable(create_trace_id):
        return trace_id
    try:
        return str(create_trace_id(seed=trace_id))
    except Exception:
        return trace_id


def publish_best_effort(summary: EvalSummary, publishers: Iterable[ScorePublisher]) -> list[str]:
    """Publish to optional sinks and return non-fatal error messages."""
    errors: list[str] = []
    for publisher in publishers:
        try:
            publisher.publish(summary)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return errors
