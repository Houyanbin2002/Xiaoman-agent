from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ScheduledJob:
    trigger: str
    tier: str
    fire_at: datetime
    channel: str
    chat_id: str
    interval_seconds: int | None = None
    cron_expr: str | None = None
    message: str | None = None
    prompt: str | None = None
    name: str | None = None
    timezone: str = "UTC"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_count: int = 0
    enabled: bool = True
    last_attempt_at: datetime | None = None
    last_status: str = "pending"
    last_error: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class ScheduledJobChanged:
    """Immutable scheduler lifecycle event for projections and integrations."""

    action: str
    job_id: str
    job: dict[str, Any]
