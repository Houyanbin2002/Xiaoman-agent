from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from core.personal.models import RecordSource
from core.personal.service import PersonalDataService
from core.personal.sources.models import ExternalSourceSyncResult
from core.personal.sources.ports import (
    ExternalSourceAdapter,
    ExternalSourceSubscriptionStore,
)

logger = logging.getLogger(__name__)


class ExternalSourceSyncService:
    """Mirror subscribed external facts into the canonical personal store."""

    def __init__(
        self,
        *,
        store: ExternalSourceSubscriptionStore,
        personal_data: PersonalDataService,
        adapters: dict[str, ExternalSourceAdapter],
    ) -> None:
        self.store = store
        self.personal_data = personal_data
        self.adapters = dict(adapters)
        self._stop_event = asyncio.Event()

    async def sync_one(self, subscription_id: str) -> ExternalSourceSyncResult:
        subscription = self.store.get_subscription(subscription_id)
        if subscription is None:
            raise ValueError(
                f"external source subscription not found: {subscription_id}"
            )
        adapter = self.adapters.get(subscription.provider)
        if adapter is None:
            message = f"unsupported external source provider: {subscription.provider}"
            self.store.mark_synced(subscription.id, item_count=0, error=message)
            return ExternalSourceSyncResult(
                subscription_id=subscription.id, error=message
            )
        try:
            items = await adapter.fetch(subscription)
            created = updated = unchanged = skipped = 0
            record_ids: list[str] = []
            for item in items:
                if not item.external_id or not item.title:
                    skipped += 1
                    continue
                previous_hash = self.store.get_item_hash(
                    subscription.id,
                    item.external_id,
                )
                record_key = (
                    f"external:{subscription.provider}:{subscription.id}:"
                    f"{item.external_id}"
                )
                existing = self.personal_data.find_active_by_key(
                    subscription.entity_type,
                    record_key,
                )
                if existing is not None and previous_hash == item.content_hash:
                    unchanged += 1
                    record_ids.append(existing.id)
                    continue
                try:
                    record, was_created = self.personal_data.upsert_external(
                        entity_type=subscription.entity_type,
                        record_key=record_key,
                        title=item.title,
                        summary=item.summary,
                        data={
                            **item.data,
                            "external_id": item.external_id,
                            "external_subscription_id": subscription.id,
                            "external_provider": subscription.provider,
                        },
                        source=RecordSource(
                            subscription.server_name
                            if subscription.provider == "mcp"
                            else subscription.provider,
                            item.source_ref,
                        ),
                    )
                except PermissionError:
                    skipped += 1
                    continue
                if was_created:
                    created += 1
                else:
                    updated += 1
                record_ids.append(record.id)
                self.store.save_item(
                    subscription_id=subscription.id,
                    external_id=item.external_id,
                    content_hash=item.content_hash,
                    record_id=record.id,
                )
            self.store.mark_synced(subscription.id, item_count=len(items))
            return ExternalSourceSyncResult(
                subscription_id=subscription.id,
                created=created,
                updated=updated,
                unchanged=unchanged,
                skipped=skipped,
                record_ids=tuple(record_ids),
            )
        except Exception as exc:
            message = str(exc)[:1000]
            self.store.mark_synced(subscription.id, item_count=0, error=message)
            logger.warning("external source sync failed (%s): %s", subscription.id, exc)
            return ExternalSourceSyncResult(
                subscription_id=subscription.id, error=message
            )

    async def sync_due(
        self, *, now: datetime | None = None
    ) -> list[ExternalSourceSyncResult]:
        current = now or datetime.now(timezone.utc)
        return [await self.sync_one(item.id) for item in self.store.list_due(current)]

    async def run(self, *, interval_seconds: int = 60) -> None:
        self._stop_event.clear()
        while not self._stop_event.is_set():
            await self.sync_due()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=max(30, interval_seconds),
                )
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def close(self) -> None:
        self.stop()
        self.store.close()
