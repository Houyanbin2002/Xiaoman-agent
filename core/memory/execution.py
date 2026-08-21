from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
import math
import re
from types import MappingProxyType


class ExecutionMemoryKind(StrEnum):
    ENVIRONMENT = "environment"
    PROJECT_CONVENTION = "project_convention"
    TOOL_LESSON = "tool_lesson"
    PROCEDURE = "procedure"
    DECISION = "decision"
    CAPABILITY = "capability"


class ExecutionScopeKind(StrEnum):
    GLOBAL = "global"
    WORKSPACE = "workspace"
    PROJECT = "project"
    TOOL = "tool"
    PLUGIN = "plugin"


class ExecutionVerificationStatus(StrEnum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    STALE = "stale"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"


class ExecutionAuthority(StrEnum):
    LEARNED = "learned"
    USER = "user"
    SYSTEM = "system"


class ExecutionLifecycleStatus(StrEnum):
    CANDIDATE = "candidate"
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    SUPERSEDED = "superseded"


_EXECUTION_USE_MARKER_RE = re.compile(
    r'<used-execution-memory\s+id=["\']([^"\']+)["\']\s*/>'
)


def used_execution_memory_ids(
    thinking: str | None,
    retrieved_ids: object,
) -> list[str]:
    """Return only explicit use markers for memories retrieved in this turn."""

    retrieved = (
        {str(item) for item in retrieved_ids if str(item).strip()}
        if isinstance(retrieved_ids, list)
        else set()
    )
    return list(
        dict.fromkeys(
            match
            for match in _EXECUTION_USE_MARKER_RE.findall(thinking or "")
            if match in retrieved
        )
    )


@dataclass(frozen=True)
class ExecutionContext:
    """Runtime facts used to decide whether an execution memory is applicable."""

    workspace_id: str = ""
    project_id: str = ""
    tools: tuple[str, ...] = ()
    plugins: tuple[str, ...] = ()
    platform: str = ""
    environment_fingerprint: str = ""
    versions: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_id", _normal(self.workspace_id))
        object.__setattr__(self, "project_id", _normal(self.project_id))
        object.__setattr__(self, "tools", _normal_tuple(self.tools))
        object.__setattr__(self, "plugins", _normal_tuple(self.plugins))
        object.__setattr__(self, "platform", _normal(self.platform))
        object.__setattr__(
            self,
            "environment_fingerprint",
            str(self.environment_fingerprint or "").strip(),
        )
        object.__setattr__(
            self,
            "versions",
            MappingProxyType(
                {
                    _normal(key): str(value).strip()
                    for key, value in self.versions.items()
                    if _normal(key) and str(value).strip()
                }
            ),
        )


@dataclass(frozen=True)
class ExecutionScope:
    kind: ExecutionScopeKind = ExecutionScopeKind.GLOBAL
    workspace_id: str = ""
    project_id: str = ""
    tool_name: str = ""
    plugin_name: str = ""
    platform: str = ""
    environment_fingerprint: str = ""
    version_key: str = ""
    version_value: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_id", _normal(self.workspace_id))
        object.__setattr__(self, "project_id", _normal(self.project_id))
        object.__setattr__(self, "tool_name", _normal(self.tool_name))
        object.__setattr__(self, "plugin_name", _normal(self.plugin_name))
        object.__setattr__(self, "platform", _normal(self.platform))
        object.__setattr__(
            self,
            "environment_fingerprint",
            str(self.environment_fingerprint or "").strip(),
        )
        object.__setattr__(self, "version_key", _normal(self.version_key))
        object.__setattr__(self, "version_value", str(self.version_value or "").strip())

    def matches(self, context: ExecutionContext) -> bool:
        if self.kind is ExecutionScopeKind.WORKSPACE and (
            not self.workspace_id or self.workspace_id != context.workspace_id
        ):
            return False
        if self.kind is ExecutionScopeKind.PROJECT and (
            not self.project_id or self.project_id != context.project_id
        ):
            return False
        if self.kind is ExecutionScopeKind.TOOL and (
            not self.tool_name or self.tool_name not in context.tools
        ):
            return False
        if self.kind is ExecutionScopeKind.PLUGIN and (
            not self.plugin_name or self.plugin_name not in context.plugins
        ):
            return False
        if self.workspace_id and self.workspace_id != context.workspace_id:
            return False
        if self.project_id and self.project_id != context.project_id:
            return False
        if self.platform and self.platform != context.platform:
            return False
        if (
            self.environment_fingerprint
            and self.environment_fingerprint != context.environment_fingerprint
        ):
            return False
        if self.version_key and (
            context.versions.get(self.version_key) != self.version_value
        ):
            return False
        return True

    def specificity(self, context: ExecutionContext) -> float:
        if not self.matches(context):
            return 0.0
        base = {
            ExecutionScopeKind.GLOBAL: 0.25,
            ExecutionScopeKind.WORKSPACE: 0.55,
            ExecutionScopeKind.PROJECT: 0.75,
            ExecutionScopeKind.TOOL: 0.85,
            ExecutionScopeKind.PLUGIN: 0.85,
        }[self.kind]
        constraints = sum(
            bool(value)
            for value in (
                self.workspace_id,
                self.project_id,
                self.platform,
                self.environment_fingerprint,
                self.version_key,
            )
        )
        return min(1.0, base + constraints * 0.04)


@dataclass(frozen=True)
class ExecutionMemoryState:
    item_id: str
    kind: ExecutionMemoryKind = ExecutionMemoryKind.PROCEDURE
    scope: ExecutionScope = field(default_factory=ExecutionScope)
    verification_status: ExecutionVerificationStatus = (
        ExecutionVerificationStatus.CANDIDATE
    )
    authority: ExecutionAuthority = ExecutionAuthority.LEARNED
    lifecycle_status: ExecutionLifecycleStatus = ExecutionLifecycleStatus.PROPOSED
    user_locked: bool = False
    extraction_confidence: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_verified_at: datetime | None = None
    expires_at: datetime | None = None
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", str(self.item_id or "").strip())
        object.__setattr__(self, "success_count", max(0, int(self.success_count)))
        object.__setattr__(self, "failure_count", max(0, int(self.failure_count)))
        object.__setattr__(
            self,
            "extraction_confidence",
            _clamp(self.extraction_confidence),
        )
        object.__setattr__(self, "evidence_refs", _clean_refs(self.evidence_refs))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def is_available(
        self, context: ExecutionContext, now: datetime | None = None
    ) -> bool:
        current = _aware(now or datetime.now(timezone.utc))
        if self.verification_status in {
            ExecutionVerificationStatus.QUARANTINED,
            ExecutionVerificationStatus.SUPERSEDED,
        }:
            return False
        if self.lifecycle_status not in {
            ExecutionLifecycleStatus.PROPOSED,
            ExecutionLifecycleStatus.ACTIVE,
        }:
            return False
        if self.expires_at is not None and _aware(self.expires_at) <= current:
            return False
        return self.scope.matches(context)

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "kind": self.kind.value,
            "scope": {
                "kind": self.scope.kind.value,
                "workspace_id": self.scope.workspace_id,
                "project_id": self.scope.project_id,
                "tool_name": self.scope.tool_name,
                "plugin_name": self.scope.plugin_name,
                "platform": self.scope.platform,
                "environment_fingerprint": self.scope.environment_fingerprint,
                "version_key": self.scope.version_key,
                "version_value": self.scope.version_value,
            },
            "verification_status": self.verification_status.value,
            "authority": self.authority.value,
            "lifecycle_status": self.lifecycle_status.value,
            "user_locked": self.user_locked,
            "extraction_confidence": self.extraction_confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_verified_at": _datetime_iso(self.last_verified_at),
            "expires_at": _datetime_iso(self.expires_at),
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }


_HALF_LIFE_DAYS: dict[ExecutionMemoryKind, float | None] = {
    ExecutionMemoryKind.ENVIRONMENT: 30.0,
    ExecutionMemoryKind.PROJECT_CONVENTION: None,
    ExecutionMemoryKind.TOOL_LESSON: 60.0,
    ExecutionMemoryKind.PROCEDURE: 180.0,
    ExecutionMemoryKind.DECISION: None,
    ExecutionMemoryKind.CAPABILITY: 30.0,
}


def execution_reliability_score(
    state: ExecutionMemoryState,
    *,
    now: datetime | None = None,
) -> float:
    """Reliability reflects verified outcomes, not how often an item was recalled."""

    status_weight = {
        ExecutionVerificationStatus.CANDIDATE: 0.45,
        ExecutionVerificationStatus.VERIFIED: 1.0,
        ExecutionVerificationStatus.STALE: 0.3,
        ExecutionVerificationStatus.QUARANTINED: 0.0,
        ExecutionVerificationStatus.SUPERSEDED: 0.0,
    }[state.verification_status]
    lifecycle_weight = {
        ExecutionLifecycleStatus.CANDIDATE: 0.0,
        ExecutionLifecycleStatus.PROPOSED: 0.65,
        ExecutionLifecycleStatus.ACTIVE: 1.0,
        ExecutionLifecycleStatus.SUSPENDED: 0.0,
        ExecutionLifecycleStatus.SUPERSEDED: 0.0,
    }[state.lifecycle_status]
    status_weight *= lifecycle_weight
    if status_weight == 0.0:
        return 0.0

    evidence_score = (state.success_count + 1.0) / (
        state.success_count + state.failure_count + 2.0
    )
    freshness = 1.0
    half_life = _HALF_LIFE_DAYS[state.kind]
    current = _aware(now or datetime.now(timezone.utc))
    if half_life is not None:
        if state.last_verified_at is None:
            freshness = 0.55 if state.success_count == 0 else 0.7
        else:
            age_days = max(
                0.0,
                (current - _aware(state.last_verified_at)).total_seconds() / 86400.0,
            )
            freshness = math.exp(-math.log(2.0) * age_days / half_life)
    return _clamp(0.55 * status_weight + 0.3 * evidence_score + 0.15 * freshness)


def execution_rank_score(
    *,
    semantic_score: float,
    state: ExecutionMemoryState,
    context: ExecutionContext,
    now: datetime | None = None,
) -> float:
    if not state.is_available(context, now=now):
        return 0.0
    return _clamp(
        0.4 * _clamp(semantic_score)
        + 0.3 * state.scope.specificity(context)
        + 0.3 * execution_reliability_score(state, now=now)
    )


def is_skill_promotion_candidate(state: ExecutionMemoryState) -> bool:
    if state.kind not in {
        ExecutionMemoryKind.PROCEDURE,
        ExecutionMemoryKind.TOOL_LESSON,
    }:
        return False
    if state.verification_status is not ExecutionVerificationStatus.VERIFIED:
        return False
    attempts = state.success_count + state.failure_count
    return (
        state.success_count >= 3
        and attempts > 0
        and state.failure_count / attempts <= 0.2
    )


def apply_execution_outcome(
    state: ExecutionMemoryState,
    *,
    success: bool,
    evidence_ref: str = "",
    verified_at: datetime | None = None,
) -> ExecutionMemoryState:
    timestamp = _aware(verified_at or datetime.now(timezone.utc))
    refs = _clean_refs((*state.evidence_refs, evidence_ref))
    if success:
        success_count = state.success_count + 1
        return replace(
            state,
            verification_status=ExecutionVerificationStatus.VERIFIED,
            lifecycle_status=(
                ExecutionLifecycleStatus.ACTIVE
                if state.user_locked or success_count >= 2
                else ExecutionLifecycleStatus.PROPOSED
            ),
            success_count=success_count,
            last_verified_at=timestamp,
            evidence_refs=refs,
        )

    failure_count = state.failure_count + 1
    status = ExecutionVerificationStatus.STALE
    if failure_count >= max(2, state.success_count + 1):
        status = ExecutionVerificationStatus.QUARANTINED
    return replace(
        state,
        verification_status=status,
        lifecycle_status=(
            state.lifecycle_status
            if state.user_locked
            else (
                ExecutionLifecycleStatus.SUSPENDED
                if status is ExecutionVerificationStatus.QUARANTINED
                else state.lifecycle_status
            )
        ),
        failure_count=failure_count,
        evidence_refs=refs,
    )


def build_execution_state(
    *,
    item_id: str,
    metadata: Mapping[str, object] | None = None,
    source_ref: str = "",
    verified: bool = False,
) -> ExecutionMemoryState:
    payload = dict(metadata or {})
    raw_authority = str(payload.get("authority") or "learned").strip().lower()
    try:
        authority = ExecutionAuthority(raw_authority)
    except ValueError:
        authority = ExecutionAuthority.LEARNED
    user_locked = (
        bool(payload.get("user_locked")) or authority is ExecutionAuthority.USER
    )
    raw_lifecycle = str(payload.get("lifecycle_status") or "").strip().lower()
    if not raw_lifecycle:
        raw_lifecycle = (
            ExecutionLifecycleStatus.ACTIVE.value
            if user_locked or verified
            else ExecutionLifecycleStatus.PROPOSED.value
        )
    raw_scope = payload.get("execution_scope")
    scope_data = dict(raw_scope) if isinstance(raw_scope, Mapping) else {}
    tool_name = str(
        scope_data.get("tool_name") or payload.get("tool_requirement") or ""
    ).strip()
    plugin_name = str(scope_data.get("plugin_name") or "").strip()
    raw_scope_kind = str(scope_data.get("kind") or "").strip()
    if not raw_scope_kind:
        raw_scope_kind = (
            ExecutionScopeKind.TOOL.value
            if tool_name
            else (
                ExecutionScopeKind.PLUGIN.value
                if plugin_name
                else ExecutionScopeKind.GLOBAL.value
            )
        )
    raw_kind = str(
        payload.get("execution_kind") or ExecutionMemoryKind.PROCEDURE.value
    ).strip()
    raw_evidence_refs = payload.get("evidence_refs")
    evidence_refs = (
        tuple(str(item) for item in raw_evidence_refs)
        if isinstance(raw_evidence_refs, list)
        else ()
    )
    try:
        extraction_confidence = float(payload.get("extraction_confidence") or 0.0)
    except (TypeError, ValueError):
        extraction_confidence = 0.0
    try:
        kind = ExecutionMemoryKind(raw_kind)
    except ValueError:
        kind = ExecutionMemoryKind.PROCEDURE
    try:
        scope_kind = ExecutionScopeKind(raw_scope_kind)
    except ValueError:
        scope_kind = ExecutionScopeKind.GLOBAL
    try:
        lifecycle = ExecutionLifecycleStatus(raw_lifecycle)
    except ValueError:
        lifecycle = ExecutionLifecycleStatus.PROPOSED
    return ExecutionMemoryState(
        item_id=item_id,
        kind=kind,
        scope=ExecutionScope(
            kind=scope_kind,
            workspace_id=str(scope_data.get("workspace_id") or ""),
            project_id=str(scope_data.get("project_id") or ""),
            tool_name=tool_name,
            plugin_name=plugin_name,
            platform=str(scope_data.get("platform") or ""),
            environment_fingerprint=str(
                scope_data.get("environment_fingerprint") or ""
            ),
            version_key=str(scope_data.get("version_key") or ""),
            version_value=str(scope_data.get("version_value") or ""),
        ),
        verification_status=(
            ExecutionVerificationStatus.VERIFIED
            if verified
            else ExecutionVerificationStatus.CANDIDATE
        ),
        authority=authority,
        lifecycle_status=lifecycle,
        user_locked=user_locked,
        extraction_confidence=_clamp(extraction_confidence),
        last_verified_at=datetime.now(timezone.utc) if verified else None,
        evidence_refs=_clean_refs((*evidence_refs, source_ref)),
        metadata={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "execution_scope",
                "tool_requirement",
                "steps",
                "authority",
                "lifecycle_status",
                "user_locked",
                "extraction_confidence",
                "evidence_refs",
            }
        },
    )


def execution_context_from_mapping(
    raw: Mapping[str, object] | None,
) -> ExecutionContext:
    payload = dict(raw or {})
    raw_versions = payload.get("versions")
    versions = dict(raw_versions) if isinstance(raw_versions, Mapping) else {}
    return ExecutionContext(
        workspace_id=str(payload.get("workspace_id") or ""),
        project_id=str(payload.get("project_id") or ""),
        tools=_string_tuple(payload.get("tools")),
        plugins=_string_tuple(payload.get("plugins")),
        platform=str(payload.get("platform") or ""),
        environment_fingerprint=str(payload.get("environment_fingerprint") or ""),
        versions={str(key): str(value) for key, value in versions.items()},
    )


def _normal(value: object) -> str:
    return str(value or "").strip().lower()


def _normal_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_normal(value) for value in values if _normal(value)))


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return ()


def _clean_refs(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value or "").strip() for value in values if str(value or "").strip()
        )
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _datetime_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _aware(value).astimezone(timezone.utc).isoformat()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
