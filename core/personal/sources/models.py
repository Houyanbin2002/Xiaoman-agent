from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from core.personal.models import PersonalEntityType


def _stable_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExternalSourceSubscription:
    id: str
    provider: str
    server_name: str
    name: str
    resource_url: str
    entity_type: PersonalEntityType
    mapping: dict[str, Any]
    poll_interval_minutes: int
    enabled: bool
    last_synced_at: str | None
    last_error: str
    last_item_count: int
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "server_name": self.server_name,
            "name": self.name,
            "resource_url": self.resource_url,
            "entity_type": self.entity_type.value,
            "mapping": self.mapping,
            "poll_interval_minutes": self.poll_interval_minutes,
            "enabled": self.enabled,
            "last_synced_at": self.last_synced_at,
            "last_error": self.last_error,
            "last_item_count": self.last_item_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ExternalSourceItem:
    external_id: str
    title: str
    summary: str
    data: dict[str, Any]
    source_ref: str
    content_hash: str

    @classmethod
    def build(
        cls,
        *,
        external_id: str,
        title: str,
        summary: str,
        data: dict[str, Any],
        source_ref: str,
    ) -> "ExternalSourceItem":
        normalized = {
            "external_id": external_id.strip(),
            "title": title.strip(),
            "summary": summary.strip(),
            "data": data,
            "source_ref": source_ref.strip(),
        }
        return cls(content_hash=_stable_hash(normalized), **normalized)


@dataclass(frozen=True)
class ExternalSourceSyncResult:
    subscription_id: str
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    error: str = ""
    record_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "created": self.created,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "error": self.error,
            "record_ids": list(self.record_ids),
        }
