from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

PermissionMode = Literal["request_approval", "auto_approve", "full_access"]
PermissionRisk = Literal["low", "medium", "high"]

DEFAULT_DASHBOARD_PERMISSION_MODE: PermissionMode = "request_approval"
DEFAULT_BACKGROUND_PERMISSION_MODE: PermissionMode = "full_access"
PERMISSION_MODES: frozenset[str] = frozenset(
    {"request_approval", "auto_approve", "full_access"}
)


def normalize_permission_mode(
    value: object,
    *,
    fallback: PermissionMode = DEFAULT_BACKGROUND_PERMISSION_MODE,
) -> PermissionMode:
    normalized = str(value or "").strip().lower()
    if normalized in PERMISSION_MODES:
        return normalized  # type: ignore[return-value]
    return fallback


@dataclass(frozen=True)
class PermissionClassification:
    category: str
    risk: PermissionRisk
    title: str
    description: str
    preview: str = ""

    def requires_approval(self, mode: PermissionMode) -> bool:
        if mode == "full_access":
            return False
        if mode == "auto_approve":
            return self.risk == "high"
        return self.risk != "low"


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    session_key: str
    call_id: str
    tool_name: str
    mode: PermissionMode
    classification: PermissionClassification
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        approval_id: str,
        session_key: str,
        call_id: str,
        tool_name: str,
        mode: PermissionMode,
        classification: PermissionClassification,
    ) -> "ApprovalRequest":
        return cls(
            id=approval_id,
            session_key=session_key,
            call_id=call_id,
            tool_name=tool_name,
            mode=mode,
            classification=classification,
            created_at=datetime.now().astimezone().isoformat(),
        )

    def as_event(self) -> dict[str, str]:
        return {
            "type": "approval_request",
            "approval_id": self.id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "permission_mode": self.mode,
            "category": self.classification.category,
            "risk": self.classification.risk,
            "title": self.classification.title,
            "description": self.classification.description,
            "preview": self.classification.preview,
            "created_at": self.created_at,
        }
