from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from core.personal.models import PersonalRecord, SceneMode


@dataclass(frozen=True)
class PersonalContextSnapshot:
    observed_at: str
    timezone: str
    scene: SceneMode
    scene_ends_at: str | None
    focus_active: bool
    focus_label: str
    focus_ends_at: str | None
    do_not_disturb: bool
    allow_high_priority: bool
    energy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at,
            "timezone": self.timezone,
            "scene": self.scene.value,
            "scene_ends_at": self.scene_ends_at,
            "focus_active": self.focus_active,
            "focus_label": self.focus_label,
            "focus_ends_at": self.focus_ends_at,
            "do_not_disturb": self.do_not_disturb,
            "allow_high_priority": self.allow_high_priority,
            "energy": self.energy,
        }


@dataclass(frozen=True)
class TaskRecommendation:
    candidate_id: str
    source_type: str
    title: str
    next_action: str
    estimated_minutes: int
    score: float
    reason: str
    due_at: str | None = None
    due_text: str = ""
    context: str = ""
    energy: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_type": self.source_type,
            "title": self.title,
            "next_action": self.next_action,
            "estimated_minutes": self.estimated_minutes,
            "score": round(self.score, 3),
            "reason": self.reason,
            "due_at": self.due_at,
            "due_text": self.due_text,
            "context": self.context,
            "energy": self.energy,
        }


@dataclass(frozen=True)
class PeriodicReport:
    period: str
    period_start: str
    period_end: str
    metrics: dict[str, Any]
    deviations: list[dict[str, Any]]
    recommendations: list[str]
    record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "metrics": self.metrics,
            "deviations": self.deviations,
            "recommendations": self.recommendations,
            "record_id": self.record_id,
        }


@dataclass(frozen=True)
class DeliveryPolicy:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ReportContribution:
    metrics: dict[str, Any] = field(default_factory=dict)
    deviations: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


RecommendationProvider = Callable[
    [int, PersonalContextSnapshot, datetime],
    list[TaskRecommendation],
]
ReportContributor = Callable[
    [list[PersonalRecord], datetime, datetime],
    ReportContribution,
]

__all__ = [
    "DeliveryPolicy",
    "PeriodicReport",
    "PersonalContextSnapshot",
    "RecommendationProvider",
    "ReportContribution",
    "ReportContributor",
    "TaskRecommendation",
]
