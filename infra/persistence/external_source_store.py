from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from core.personal.models import PersonalEntityType
from core.personal.sources.models import ExternalSourceSubscription


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class ExternalSourceStore:
    """Subscription, cursor and item identity state stored beside personal facts."""

    _UPDATABLE = {
        "name",
        "mapping",
        "poll_interval_minutes",
        "enabled",
        "resource_url",
        "server_name",
    }

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("external source store is closed")
            try:
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def _init_schema(self) -> None:
        with self._transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS external_source_subscriptions (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    server_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    resource_url TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    mapping_json TEXT NOT NULL DEFAULT '{}',
                    poll_interval_minutes INTEGER NOT NULL DEFAULT 15,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_synced_at TEXT,
                    last_error TEXT NOT NULL DEFAULT '',
                    last_item_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(provider, server_name, resource_url, entity_type)
                );
                CREATE INDEX IF NOT EXISTS idx_external_sources_due
                    ON external_source_subscriptions(enabled, last_synced_at);

                CREATE TABLE IF NOT EXISTS external_source_items (
                    subscription_id TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    PRIMARY KEY(subscription_id, external_id),
                    FOREIGN KEY(subscription_id)
                        REFERENCES external_source_subscriptions(id) ON DELETE CASCADE
                );
                """
            )

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
    ) -> ExternalSourceSubscription:
        provider = provider.strip().lower()
        server_name = server_name.strip()
        resource_url = resource_url.strip()
        if not provider or not server_name or not resource_url:
            raise ValueError("provider、server_name 和 resource_url 不能为空")
        interval = max(1, min(int(poll_interval_minutes), 24 * 60))
        with self._transaction() as db:
            existing = db.execute(
                """
                SELECT * FROM external_source_subscriptions
                WHERE provider = ? AND server_name = ? AND resource_url = ?
                    AND entity_type = ?
                """,
                (provider, server_name, resource_url, entity_type.value),
            ).fetchone()
            if existing is not None:
                return self._row(existing)
            subscription_id = f"src_{uuid.uuid4().hex}"
            now = _now()
            db.execute(
                """
                INSERT INTO external_source_subscriptions(
                    id, provider, server_name, name, resource_url, entity_type,
                    mapping_json, poll_interval_minutes, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subscription_id,
                    provider,
                    server_name,
                    name.strip() or resource_url,
                    resource_url,
                    entity_type.value,
                    _json_dump(mapping),
                    interval,
                    int(enabled),
                    now,
                    now,
                ),
            )
            return self._require(db, subscription_id)

    def get_subscription(self, subscription_id: str) -> ExternalSourceSubscription | None:
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM external_source_subscriptions WHERE id = ?",
                (subscription_id,),
            ).fetchone()
            return self._row(row) if row is not None else None

    def list_subscriptions(self) -> list[ExternalSourceSubscription]:
        with self._transaction() as db:
            rows = db.execute(
                "SELECT * FROM external_source_subscriptions ORDER BY created_at"
            ).fetchall()
            return [self._row(row) for row in rows]

    def list_due(self, now: datetime) -> list[ExternalSourceSubscription]:
        current = now.astimezone(timezone.utc)
        result: list[ExternalSourceSubscription] = []
        for item in self.list_subscriptions():
            if not item.enabled:
                continue
            if item.last_synced_at is None:
                result.append(item)
                continue
            try:
                last = datetime.fromisoformat(item.last_synced_at.replace("Z", "+00:00"))
            except ValueError:
                result.append(item)
                continue
            if last + timedelta(minutes=item.poll_interval_minutes) <= current:
                result.append(item)
        return result

    def update_subscription(
        self,
        subscription_id: str,
        *,
        changes: dict,
    ) -> ExternalSourceSubscription:
        unknown = set(changes) - self._UPDATABLE
        if unknown:
            raise ValueError(f"不支持的订阅字段: {sorted(unknown)}")
        if not changes:
            item = self.get_subscription(subscription_id)
            if item is None:
                raise ValueError("外部数据订阅不存在")
            return item
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            column = "mapping_json" if key == "mapping" else key
            if key == "mapping":
                value = _json_dump(value)
            elif key == "enabled":
                value = int(bool(value))
            elif key == "poll_interval_minutes":
                value = max(1, min(int(value), 24 * 60))
            assignments.append(f"{column} = ?")
            values.append(value)
        assignments.append("updated_at = ?")
        values.extend([_now(), subscription_id])
        with self._transaction() as db:
            self._require(db, subscription_id)
            db.execute(
                f"UPDATE external_source_subscriptions SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            return self._require(db, subscription_id)

    def delete_subscription(self, subscription_id: str) -> bool:
        with self._transaction() as db:
            cursor = db.execute(
                "DELETE FROM external_source_subscriptions WHERE id = ?",
                (subscription_id,),
            )
            return cursor.rowcount > 0

    def get_item_hash(self, subscription_id: str, external_id: str) -> str:
        with self._transaction() as db:
            row = db.execute(
                """
                SELECT content_hash FROM external_source_items
                WHERE subscription_id = ? AND external_id = ?
                """,
                (subscription_id, external_id),
            ).fetchone()
            return str(row["content_hash"]) if row is not None else ""

    def save_item(
        self,
        *,
        subscription_id: str,
        external_id: str,
        content_hash: str,
        record_id: str,
    ) -> None:
        with self._transaction() as db:
            db.execute(
                """
                INSERT INTO external_source_items(
                    subscription_id, external_id, content_hash, record_id, last_seen_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(subscription_id, external_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    record_id = excluded.record_id,
                    last_seen_at = excluded.last_seen_at
                """,
                (subscription_id, external_id, content_hash, record_id, _now()),
            )

    def mark_synced(
        self,
        subscription_id: str,
        *,
        item_count: int,
        error: str = "",
    ) -> ExternalSourceSubscription:
        now = _now()
        with self._transaction() as db:
            self._require(db, subscription_id)
            db.execute(
                """
                UPDATE external_source_subscriptions
                SET last_synced_at = ?, last_error = ?, last_item_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, error[:1000], max(0, int(item_count)), now, subscription_id),
            )
            return self._require(db, subscription_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()

    def _require(self, db: sqlite3.Connection, subscription_id: str) -> ExternalSourceSubscription:
        row = db.execute(
            "SELECT * FROM external_source_subscriptions WHERE id = ?",
            (subscription_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"external source subscription not found: {subscription_id}")
        return self._row(row)

    @staticmethod
    def _row(row: sqlite3.Row) -> ExternalSourceSubscription:
        return ExternalSourceSubscription(
            id=str(row["id"]),
            provider=str(row["provider"]),
            server_name=str(row["server_name"]),
            name=str(row["name"]),
            resource_url=str(row["resource_url"]),
            entity_type=PersonalEntityType(row["entity_type"]),
            mapping=_json_load(str(row["mapping_json"])),
            poll_interval_minutes=int(row["poll_interval_minutes"]),
            enabled=bool(row["enabled"]),
            last_synced_at=row["last_synced_at"],
            last_error=str(row["last_error"]),
            last_item_count=int(row["last_item_count"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
