from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, StrEnum
from typing import Any, Mapping


class PersonalEntityType(StrEnum):
    PROFILE = "profile"
    COMMITMENT = "commitment"
    CALENDAR_EVENT = "calendar_event"
    HEALTH_OBSERVATION = "health_observation"
    DAILY_PLAN = "daily_plan"
    CHECK_IN = "check_in"
    NOTIFICATION_POLICY = "notification_policy"
    MEMORY = "memory"
    MONITOR_OBSERVATION = "monitor_observation"
    CONTEXT_STATE = "context_state"
    RELATIONSHIP = "relationship"
    IMPORTANT_DATE = "important_date"
    FINANCIAL_OBLIGATION = "financial_obligation"
    TRIP = "trip"
    GOAL = "goal"
    PERIODIC_REPORT = "periodic_report"
    PROACTIVE_INTENT = "proactive_intent"


class RecordStatus(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


class SensitivityLevel(StrEnum):
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class DataCategory(StrEnum):
    GENERAL = "general"
    HEALTH = "health"
    EMOTIONAL = "emotional"
    ACCOUNT = "account"
    RELATIONSHIP = "relationship"
    LOCATION = "location"
    FINANCIAL = "financial"


class AccessPolicy(StrEnum):
    STANDARD = "standard"
    CONFIRM_WRITE = "confirm_write"
    CONFIRM_READ = "confirm_read"
    OWNER_ONLY = "owner_only"


class MemoryKind(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    REQUESTED = "requested"
    TEMPORARY_STATE = "temporary_state"
    HISTORICAL_EVENT = "historical_event"
    EPISODE = "episode"
    RELATIONSHIP = "relationship"
    PROCEDURE = "procedure"


class SceneMode(StrEnum):
    NEUTRAL = "neutral"
    LEAVING = "leaving"
    HOME = "home"
    BEDTIME = "bedtime"
    TRAVEL = "travel"


class FollowUpTrigger(StrEnum):
    INTERVAL = "interval"
    AT_TIME = "at_time"
    INACTIVITY = "inactivity"
    CONDITION = "condition"


@dataclass(frozen=True)
class RecordSource:
    source: str
    source_ref: str = ""


@dataclass(frozen=True)
class PersonalProfileData:
    display_name: str = ""
    timezone: str = "Asia/Shanghai"
    locale: str = "zh-CN"
    preferences: dict[str, Any] = field(default_factory=dict)
    boundaries: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryData:
    kind: MemoryKind
    content: str
    tags: list[str] = field(default_factory=list)
    category: DataCategory = DataCategory.GENERAL
    subject: str = ""
    predicate: str = ""
    value: str = ""
    scope: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    identity_quality: str = "weak"


@dataclass(frozen=True)
class ContextStateData:
    context_type: str
    mode: str
    started_at: str
    ends_at: str | None = None
    label: str = ""
    do_not_disturb: bool = False
    allow_high_priority: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProactiveIntentData:
    trigger_type: FollowUpTrigger
    message: str
    reason: str
    next_trigger_at: str | None = None
    interval_minutes: int | None = None
    target_entity_type: str = ""
    target_record_key: str = ""
    inactivity_days: int | None = None
    condition: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    status: str = "active"
    cooldown_minutes: int = 60
    last_triggered_at: str | None = None


@dataclass(frozen=True)
class PeriodicReportData:
    period: str
    period_start: str
    period_end: str
    metrics: dict[str, Any] = field(default_factory=dict)
    deviations: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def normalize_payload(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_payload(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): normalize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_payload(item) for item in value]
    return value


@dataclass(frozen=True)
class PersonalRecord:
    id: str
    entity_type: PersonalEntityType
    record_key: str
    title: str
    summary: str
    data: dict[str, Any]
    source: RecordSource
    confidence: float
    sensitivity: SensitivityLevel
    data_category: DataCategory
    access_policy: AccessPolicy
    status: RecordStatus
    valid_from: str | None
    expires_at: str | None
    last_confirmed_at: str | None
    user_locked: bool
    allow_auto_update: bool
    supersedes_id: str | None
    revision: int
    created_at: str
    updated_at: str
    forgotten_at: str | None = None
    valid_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type.value,
            "record_key": self.record_key,
            "title": self.title,
            "summary": self.summary,
            "data": self.data,
            "source": self.source.source,
            "source_ref": self.source.source_ref,
            "confidence": self.confidence,
            "sensitivity": self.sensitivity.value,
            "data_category": self.data_category.value,
            "access_policy": self.access_policy.value,
            "status": self.status.value,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "expires_at": self.expires_at,
            "last_confirmed_at": self.last_confirmed_at,
            "user_locked": self.user_locked,
            "allow_auto_update": self.allow_auto_update,
            "supersedes_id": self.supersedes_id,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "forgotten_at": self.forgotten_at,
        }


@dataclass(frozen=True)
class RecordRevision:
    id: int
    record_id: str
    revision: int
    action: str
    actor: str
    reason: str
    snapshot: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class MemoryEvidence:
    id: str
    record_id: str
    source: RecordSource
    statement: str
    confidence: float
    observed_at: str
