from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.personal.models import PersonalEntityType
from core.personal.sources.models import ExternalSourceItem, ExternalSourceSubscription


class ExternalSourceAdapter(Protocol):
    async def fetch(
        self,
        subscription: ExternalSourceSubscription,
    ) -> list[ExternalSourceItem]: ...


class ExternalSourceSubscriptionStore(Protocol):
    def create_subscription(
        self,
        *,
        provider: str,
        server_name: str,
        name: str,
        resource_url: str,
        entity_type: PersonalEntityType,
        mapping: dict,
        poll_interval_minutes: int,
        enabled: bool = True,
    ) -> ExternalSourceSubscription: ...

    def get_subscription(self, subscription_id: str) -> ExternalSourceSubscription | None: ...

    def list_subscriptions(self) -> list[ExternalSourceSubscription]: ...

    def list_due(self, now: datetime) -> list[ExternalSourceSubscription]: ...

    def update_subscription(
        self,
        subscription_id: str,
        *,
        changes: dict,
    ) -> ExternalSourceSubscription: ...

    def delete_subscription(self, subscription_id: str) -> bool: ...

    def get_item_hash(self, subscription_id: str, external_id: str) -> str: ...

    def save_item(
        self,
        *,
        subscription_id: str,
        external_id: str,
        content_hash: str,
        record_id: str,
    ) -> None: ...

    def mark_synced(
        self,
        subscription_id: str,
        *,
        item_count: int,
        error: str = "",
    ) -> ExternalSourceSubscription: ...

    def close(self) -> None: ...
