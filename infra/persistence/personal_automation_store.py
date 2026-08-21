from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from core.personal.events import (
    EventStatus,
    OperationAuditEntry,
    OperationReceipt,
    OperationStatus,
    PersonalEvent,
)


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class PersonalAutomationStore:
    """Durable personal events, idempotent operations, and approval audit."""

    SCHEMA_VERSION = 2

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._transaction() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS personal_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS personal_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    dedupe_key TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 5,
                    available_at TEXT NOT NULL,
                    lease_owner TEXT NOT NULL DEFAULT '',
                    lease_until TEXT,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_personal_event_dedupe
                    ON personal_events(source, dedupe_key)
                    WHERE dedupe_key <> '';
                CREATE INDEX IF NOT EXISTS idx_personal_event_claim
                    ON personal_events(status, available_at, lease_until);

                CREATE TABLE IF NOT EXISTS personal_operations (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    request_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    requires_approval INTEGER NOT NULL DEFAULT 0,
                    approval_actor TEXT NOT NULL DEFAULT '',
                    approval_note TEXT NOT NULL DEFAULT '',
                    approved_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_personal_operation_status
                    ON personal_operations(status, updated_at);

                CREATE TABLE IF NOT EXISTS personal_operation_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (operation_id) REFERENCES personal_operations(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_personal_operation_audit
                    ON personal_operation_audit(operation_id, id);
                """
            )
            db.execute(
                "INSERT OR IGNORE INTO personal_schema_migrations(version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, _iso(_now_dt())),
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("personal automation store is closed")
            try:
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()

    def enqueue_event(
        self,
        *,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        source_ref: str = "",
        dedupe_key: str = "",
        available_at: datetime | None = None,
        max_attempts: int = 5,
    ) -> tuple[PersonalEvent, bool]:
        if not event_type.strip() or not source.strip():
            raise ValueError("event_type and source are required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        with self._transaction() as db:
            if dedupe_key:
                existing = db.execute(
                    "SELECT * FROM personal_events WHERE source = ? AND dedupe_key = ?",
                    (source, dedupe_key),
                ).fetchone()
                if existing is not None:
                    return self._row_to_event(existing), False
            now = _now_dt()
            event_id = f"evt_{uuid.uuid4().hex}"
            db.execute(
                """
                INSERT INTO personal_events(
                    id, event_type, source, source_ref, payload_json, dedupe_key,
                    status, attempts, max_attempts, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type.strip(),
                    source.strip(),
                    source_ref.strip(),
                    _json_dump(payload),
                    dedupe_key.strip(),
                    max_attempts,
                    _iso(available_at or now),
                    _iso(now),
                    _iso(now),
                ),
            )
            return self._require_event(db, event_id), True

    def claim_events(
        self,
        *,
        worker_id: str,
        limit: int = 10,
        lease_seconds: int = 60,
    ) -> list[PersonalEvent]:
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        now = _now_dt()
        now_text = _iso(now)
        lease_until = _iso(now + timedelta(seconds=max(1, lease_seconds)))
        actual_limit = max(1, min(int(limit), 100))
        with self._transaction() as db:
            db.execute(
                """
                UPDATE personal_events
                SET status = 'pending', lease_owner = '', lease_until = NULL,
                    updated_at = ?
                WHERE status = 'processing' AND lease_until <= ?
                """,
                (now_text, now_text),
            )
            rows = db.execute(
                """
                SELECT id FROM personal_events
                WHERE status = 'pending' AND available_at <= ?
                    AND attempts < max_attempts
                ORDER BY created_at LIMIT ?
                """,
                (now_text, actual_limit),
            ).fetchall()
            claimed: list[PersonalEvent] = []
            for row in rows:
                event_id = str(row["id"])
                db.execute(
                    """
                    UPDATE personal_events
                    SET status = 'processing', attempts = attempts + 1,
                        lease_owner = ?, lease_until = ?, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (worker_id, lease_until, now_text, event_id),
                )
                claimed.append(self._require_event(db, event_id))
            return claimed

    def complete_event(self, event_id: str, *, worker_id: str) -> PersonalEvent:
        with self._transaction() as db:
            current = self._require_event(db, event_id)
            self._assert_event_owner(current, worker_id)
            now = _iso(_now_dt())
            db.execute(
                """
                UPDATE personal_events
                SET status = 'succeeded', lease_owner = '', lease_until = NULL,
                    last_error = '', updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (now, now, event_id),
            )
            return self._require_event(db, event_id)

    def fail_event(
        self,
        event_id: str,
        *,
        worker_id: str,
        error: str,
        retry_delay_seconds: int = 30,
    ) -> PersonalEvent:
        with self._transaction() as db:
            current = self._require_event(db, event_id)
            self._assert_event_owner(current, worker_id)
            dead = current.attempts >= current.max_attempts
            now = _now_dt()
            db.execute(
                """
                UPDATE personal_events
                SET status = ?, available_at = ?, lease_owner = '', lease_until = NULL,
                    last_error = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    EventStatus.DEAD.value if dead else EventStatus.PENDING.value,
                    _iso(now + timedelta(seconds=max(0, retry_delay_seconds))),
                    error,
                    _iso(now),
                    _iso(now) if dead else None,
                    event_id,
                ),
            )
            return self._require_event(db, event_id)

    def create_operation(
        self,
        *,
        idempotency_key: str,
        action: str,
        target: str,
        request: dict[str, Any],
        requires_approval: bool,
        actor: str = "assistant",
    ) -> tuple[OperationReceipt, bool]:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key is required")
        with self._transaction() as db:
            existing = db.execute(
                "SELECT * FROM personal_operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return self._row_to_operation(existing), False
            operation_id = f"op_{uuid.uuid4().hex}"
            now = _iso(_now_dt())
            status = (
                OperationStatus.AWAITING_APPROVAL
                if requires_approval
                else OperationStatus.READY
            )
            db.execute(
                """
                INSERT INTO personal_operations(
                    id, idempotency_key, action, target, request_json, status,
                    requires_approval, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    idempotency_key.strip(),
                    action.strip(),
                    target.strip(),
                    _json_dump(request),
                    status.value,
                    int(requires_approval),
                    now,
                    now,
                ),
            )
            receipt = self._require_operation(db, operation_id)
            self._audit(db, receipt.id, "created", actor, {"status": status.value})
            return receipt, True

    def approve_operation(
        self,
        operation_id: str,
        *,
        actor: str,
        note: str = "",
    ) -> OperationReceipt:
        return self._approval_transition(
            operation_id,
            status=OperationStatus.READY,
            actor=actor,
            note=note,
        )

    def reject_operation(
        self,
        operation_id: str,
        *,
        actor: str,
        note: str = "",
    ) -> OperationReceipt:
        return self._approval_transition(
            operation_id,
            status=OperationStatus.REJECTED,
            actor=actor,
            note=note,
        )

    def _approval_transition(
        self,
        operation_id: str,
        *,
        status: OperationStatus,
        actor: str,
        note: str,
    ) -> OperationReceipt:
        with self._transaction() as db:
            current = self._require_operation(db, operation_id)
            if current.status != OperationStatus.AWAITING_APPROVAL:
                raise ValueError("operation is not awaiting approval")
            now = _iso(_now_dt())
            db.execute(
                """
                UPDATE personal_operations
                SET status = ?, approval_actor = ?, approval_note = ?,
                    approved_at = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    actor,
                    note,
                    now if status == OperationStatus.READY else None,
                    now,
                    now if status == OperationStatus.REJECTED else None,
                    operation_id,
                ),
            )
            self._audit(db, operation_id, status.value, actor, {"note": note})
            return self._require_operation(db, operation_id)

    def start_operation(
        self, operation_id: str, *, actor: str = "system"
    ) -> OperationReceipt:
        with self._transaction() as db:
            current = self._require_operation(db, operation_id)
            if current.status not in {OperationStatus.READY, OperationStatus.FAILED}:
                raise ValueError(f"operation cannot start from {current.status.value}")
            now = _iso(_now_dt())
            db.execute(
                """
                UPDATE personal_operations
                SET status = 'running', attempt_count = attempt_count + 1,
                    error = '', updated_at = ? WHERE id = ?
                """,
                (now, operation_id),
            )
            self._audit(db, operation_id, "started", actor, {})
            return self._require_operation(db, operation_id)

    def complete_operation(
        self,
        operation_id: str,
        *,
        result: dict[str, Any],
        actor: str = "system",
    ) -> OperationReceipt:
        return self._finish_operation(
            operation_id,
            status=OperationStatus.SUCCEEDED,
            result=result,
            error="",
            actor=actor,
        )

    def fail_operation(
        self,
        operation_id: str,
        *,
        error: str,
        actor: str = "system",
    ) -> OperationReceipt:
        return self._finish_operation(
            operation_id,
            status=OperationStatus.FAILED,
            result={},
            error=error,
            actor=actor,
        )

    def _finish_operation(
        self,
        operation_id: str,
        *,
        status: OperationStatus,
        result: dict[str, Any],
        error: str,
        actor: str,
    ) -> OperationReceipt:
        with self._transaction() as db:
            current = self._require_operation(db, operation_id)
            if current.status != OperationStatus.RUNNING:
                raise ValueError("operation is not running")
            now = _iso(_now_dt())
            db.execute(
                """
                UPDATE personal_operations
                SET status = ?, result_json = ?, error = ?, updated_at = ?,
                    completed_at = ? WHERE id = ?
                """,
                (
                    status.value,
                    _json_dump(result),
                    error,
                    now,
                    now if status == OperationStatus.SUCCEEDED else None,
                    operation_id,
                ),
            )
            self._audit(
                db,
                operation_id,
                status.value,
                actor,
                {"error": error} if error else {"result": result},
            )
            return self._require_operation(db, operation_id)

    def get_operation_by_key(self, idempotency_key: str) -> OperationReceipt | None:
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM personal_operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            return self._row_to_operation(row) if row is not None else None

    def list_operation_audit(self, operation_id: str) -> list[OperationAuditEntry]:
        with self._transaction() as db:
            self._require_operation(db, operation_id)
            rows = db.execute(
                """
                SELECT * FROM personal_operation_audit
                WHERE operation_id = ? ORDER BY id
                """,
                (operation_id,),
            ).fetchall()
            return [
                OperationAuditEntry(
                    id=int(row["id"]),
                    operation_id=str(row["operation_id"]),
                    action=str(row["action"]),
                    actor=str(row["actor"]),
                    details=_json_load(row["details_json"]),
                    created_at=str(row["created_at"]),
                )
                for row in rows
            ]

    @staticmethod
    def _assert_event_owner(event: PersonalEvent, worker_id: str) -> None:
        if event.status != EventStatus.PROCESSING or event.lease_owner != worker_id:
            raise PermissionError("event is not leased by this worker")

    def _require_event(self, db: sqlite3.Connection, event_id: str) -> PersonalEvent:
        row = db.execute(
            "SELECT * FROM personal_events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"personal event not found: {event_id}")
        return self._row_to_event(row)

    def _require_operation(
        self, db: sqlite3.Connection, operation_id: str
    ) -> OperationReceipt:
        row = db.execute(
            "SELECT * FROM personal_operations WHERE id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"personal operation not found: {operation_id}")
        return self._row_to_operation(row)

    def _audit(
        self,
        db: sqlite3.Connection,
        operation_id: str,
        action: str,
        actor: str,
        details: dict[str, Any],
    ) -> None:
        db.execute(
            """
            INSERT INTO personal_operation_audit(
                operation_id, action, actor, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (operation_id, action, actor, _json_dump(details), _iso(_now_dt())),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> PersonalEvent:
        return PersonalEvent(
            id=str(row["id"]),
            event_type=str(row["event_type"]),
            source=str(row["source"]),
            source_ref=str(row["source_ref"]),
            payload=_json_load(row["payload_json"]),
            dedupe_key=str(row["dedupe_key"]),
            status=EventStatus(row["status"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            available_at=str(row["available_at"]),
            lease_owner=str(row["lease_owner"]),
            lease_until=row["lease_until"],
            last_error=str(row["last_error"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            completed_at=row["completed_at"],
        )

    @staticmethod
    def _row_to_operation(row: sqlite3.Row) -> OperationReceipt:
        return OperationReceipt(
            id=str(row["id"]),
            idempotency_key=str(row["idempotency_key"]),
            action=str(row["action"]),
            target=str(row["target"]),
            request=_json_load(row["request_json"]),
            result=_json_load(row["result_json"]),
            status=OperationStatus(row["status"]),
            requires_approval=bool(row["requires_approval"]),
            approval_actor=str(row["approval_actor"]),
            approval_note=str(row["approval_note"]),
            approved_at=row["approved_at"],
            attempt_count=int(row["attempt_count"]),
            error=str(row["error"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            completed_at=row["completed_at"],
        )
