from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.marketplace.models import MarketplaceKind

from core.personal.models import DataCategory, PersonalEntityType


class ModelUpdatePayload(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    provider: str | None = Field(default=None, max_length=80)
    base_url: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=1000)
    output_dimensionality: int | None = Field(default=None, ge=64, le=4096)


class ModelCatalogPayload(BaseModel):
    base_url: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=1000)


class ChannelUpdatePayload(BaseModel):
    enabled: bool = True
    token: str | None = Field(default=None, max_length=1000)
    app_id: str | None = Field(default=None, max_length=128)
    client_secret: str | None = Field(default=None, max_length=1000)
    bot_id: str | None = Field(default=None, max_length=256)
    secret: str | None = Field(default=None, max_length=1000)
    allow_from: list[str] = Field(default_factory=list)


class ConversationStyleUpdatePayload(BaseModel):
    style_id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_-]*$")


class McpCreatePayload(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    transport: str = Field(default="stdio", pattern=r"^(stdio|streamable_http|sse)$")
    command: list[str] = Field(default_factory=list, max_length=30)
    url: str = Field(default="", max_length=2000)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = Field(default=None, max_length=1000)
    auth_type: str = Field(default="none", pattern=r"^(none|oauth|bearer|headers)$")
    scopes: str = Field(default="", max_length=1000)
    bearer_token: str = Field(default="", max_length=8000)
    headers: dict[str, str] = Field(default_factory=dict)
    oauth_client_id: str = Field(default="", max_length=2000)
    oauth_client_secret: str = Field(default="", max_length=8000)


class ScheduleCreatePayload(BaseModel):
    tier: str
    trigger: str
    when: str
    channel: str
    chat_id: str
    name: str | None = None
    message: str | None = None
    prompt: str | None = None
    timezone: str = "Asia/Shanghai"
    request_time: str | None = None


class WorkflowActionPayload(BaseModel):
    step_id: str | None = None
    note: str = ""


class WorkflowApprovalPayload(BaseModel):
    step_id: str = Field(min_length=1, max_length=200)
    approved: bool
    note: str = Field(default="", max_length=2000)


class WorkflowResponsePayload(BaseModel):
    step_id: str = Field(min_length=1, max_length=200)
    response: str = Field(min_length=1, max_length=10000)


class SkillInstallPayload(BaseModel):
    source: str = Field(min_length=1, max_length=1000)
    ref: str = Field(default="", max_length=200)
    subdir: str = Field(default="", max_length=500)


class MarketplaceInstallPayload(BaseModel):
    kind: MarketplaceKind
    item_id: str = Field(min_length=1, max_length=500)
    configuration: dict[str, Any] = Field(default_factory=dict, max_length=30)


class PersonalRecordCreatePayload(BaseModel):
    entity_type: str
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=2000)
    data: dict[str, Any] = Field(default_factory=dict)
    record_key: str = Field(default="", max_length=200)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    sensitivity: str | None = None
    data_category: str | None = None
    access_policy: str | None = None
    expires_at: str | None = None


class PersonalRecordUpdatePayload(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    summary: str | None = Field(default=None, max_length=2000)
    data: dict[str, Any] | None = None
    record_key: str | None = Field(default=None, max_length=200)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sensitivity: str | None = None
    data_category: str | None = None
    access_policy: str | None = None
    expires_at: str | None = None
    user_locked: bool | None = None
    allow_auto_update: bool | None = None
    reason: str = Field(default="", max_length=500)


class PersonalActionPayload(BaseModel):
    reason: str = Field(default="", max_length=500)


class PersonalRoutinePayload(BaseModel):
    routine: str
    local_date: str = ""
    timezone: str = "Asia/Shanghai"
    candidate: str = Field(default="", max_length=4000)
    chat_id: str = "xiaoman-console"


class ExternalSourceCreatePayload(BaseModel):
    provider: str = Field(default="notion", pattern=r"^[a-z0-9_-]+$")
    server_name: str = Field(default="notion", min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=200)
    resource_url: str = Field(min_length=1, max_length=2000)
    entity_type: str = PersonalEntityType.COMMITMENT.value
    mapping: dict[str, Any] = Field(default_factory=dict)
    poll_interval_minutes: int = Field(default=15, ge=1, le=1440)
    enabled: bool = True


class ExternalSourceUpdatePayload(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    server_name: str | None = Field(default=None, min_length=1, max_length=80)
    resource_url: str | None = Field(default=None, min_length=1, max_length=2000)
    mapping: dict[str, Any] | None = None
    poll_interval_minutes: int | None = Field(default=None, ge=1, le=1440)
    enabled: bool | None = None


class GovernedMemoryCreatePayload(BaseModel):
    kind: str
    content: str = Field(min_length=1, max_length=10000)
    summary: str = Field(min_length=1, max_length=2000)
    subject: str = Field(default="", max_length=500)
    predicate: str = Field(default="", max_length=256)
    value: str = Field(default="", max_length=1000)
    scope: str = Field(default="", max_length=500)
    attributes: dict[str, Any] = Field(default_factory=dict)
    record_key: str = Field(default="", max_length=300)
    tags: list[str] = Field(default_factory=list, max_length=50)
    data_category: str = DataCategory.GENERAL.value
    sensitivity: str | None = None
    access_policy: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    expires_at: str | None = None


class MemoryConflictResolvePayload(BaseModel):
    action: str
    note: str = Field(default="", max_length=1000)
    merged: dict[str, Any] | None = None


class AttentionPatternPayload(BaseModel):
    id: str | None = Field(default=None, max_length=200)
    kind: str = Field(default="availability_pattern", max_length=120)
    scene: str = Field(default="neutral", max_length=120)
    timezone: str = Field(default="Asia/Shanghai", max_length=120)
    days: list[str] = Field(min_length=1, max_length=7)
    start: str = Field(pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    end: str = Field(pattern=r"^\d{2}:\d{2}(:\d{2})?$")
    available_minutes: int = Field(default=15, ge=1, le=1440)
    expires_at: str | None = None
    user_locked: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttentionPatternStatusPayload(BaseModel):
    status: str


class AttentionPolicyPayload(BaseModel):
    id: str | None = Field(default=None, max_length=200)
    scope: dict[str, Any] = Field(default_factory=dict)
    conditions: dict[str, Any] = Field(default_factory=dict)
    effect: str
    priority: int = Field(default=50, ge=0, le=1000)
    score_adjustment: float = 0.0
    enabled: bool = True
    effective_from: str | None = None
    expires_at: str | None = None
    user_locked: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttentionPolicyStatusPayload(BaseModel):
    status: str


class AttentionFeedbackPayload(BaseModel):
    kind: str
    note: str = Field(default="", max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttentionRuntimePayload(BaseModel):
    enabled: bool = True
    channel: str = Field(pattern=r"^(telegram|qqbot|weixin|wecom)$")
    chat_id: str = Field(default="", max_length=500)


class RhythmScenePayload(BaseModel):
    scene: str
    duration_minutes: int | None = Field(default=None, ge=15, le=1440)


class RhythmFocusPayload(BaseModel):
    minutes: int = Field(default=30, ge=5, le=720)
    label: str = Field(default="专注", max_length=120)
    allow_high_priority: bool = True


class RhythmRecordPayload(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=2000)
    record_key: str = Field(default="", max_length=300)
    data: dict[str, Any] = Field(default_factory=dict)


class RhythmFollowUpPayload(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=1000)
    trigger_type: str
    next_trigger_at: str | None = None
    interval_minutes: int | None = Field(default=None, ge=5, le=525600)
    target_entity_type: str = ""
    target_record_key: str = ""
    inactivity_days: int | None = Field(default=None, ge=1, le=3650)
    condition: dict[str, Any] = Field(default_factory=dict)
    cooldown_minutes: int = Field(default=60, ge=5, le=525600)
    enabled: bool = True


class RhythmReportPayload(BaseModel):
    period: str
    persist: bool = True
