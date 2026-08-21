from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from core.personal.governance import MemoryConflict, MemoryConflictStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class MemoryGovernanceStore:
    SCHEMA_VERSION = 4

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
                CREATE TABLE IF NOT EXISTS memory_conflicts (
                    id TEXT PRIMARY KEY,
                    record_key TEXT NOT NULL,
                    existing_record_id TEXT,
                    candidate_json TEXT NOT NULL,
                    candidate_hash TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    resolved_record_id TEXT,
                    resolution_note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    FOREIGN KEY (existing_record_id) REFERENCES personal_records(id)
                        ON DELETE SET NULL,
                    FOREIGN KEY (resolved_record_id) REFERENCES personal_records(id)
                        ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_conflicts_status
                    ON memory_conflicts(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_conflicts_key
                    ON memory_conflicts(record_key, status);
                """
            )
            db.execute(
                "INSERT OR IGNORE INTO personal_schema_migrations(version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, _now()),
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("memory governance store is closed")
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

    def create_conflict(
        self,
        *,
        record_key: str,
        existing_record_id: str | None,
        candidate: dict[str, Any],
        reason: str,
    ) -> tuple[MemoryConflict, bool]:
        payload = _dump(candidate)
        candidate_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self._transaction() as db:
            row = db.execute(
                """
                SELECT * FROM memory_conflicts
                WHERE record_key = ? AND candidate_hash = ? AND status = 'pending'
                ORDER BY created_at DESC LIMIT 1
                """,
                (record_key, candidate_hash),
            ).fetchone()
            if row is not None:
                return self._row(row), False
            conflict_id = f"mc_{uuid.uuid4().hex}"
            db.execute(
                """
                INSERT INTO memory_conflicts(
                    id, record_key, existing_record_id, candidate_json,
                    candidate_hash, reason, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    conflict_id,
                    record_key,
                    existing_record_id,
                    payload,
                    candidate_hash,
                    reason,
                    _now(),
                ),
            )
            return self._require(db, conflict_id), True

    def get_conflict(self, conflict_id: str) -> MemoryConflict | None:
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM memory_conflicts WHERE id = ?", (conflict_id,)
            ).fetchone()
            return self._row(row) if row is not None else None

    def list_conflicts(
        self,
        *,
        statuses: Sequence[MemoryConflictStatus] | None = None,
        limit: int = 200,
    ) -> list[MemoryConflict]:
        selected = list(statuses or [])
        values: list[Any] = []
        where = ""
        if selected:
            where = "WHERE status IN ({})".format(",".join("?" for _ in selected))
            values.extend(item.value for item in selected)
        values.append(max(1, min(int(limit), 10000)))
        with self._transaction() as db:
            rows = db.execute(
                f"SELECT * FROM memory_conflicts {where} ORDER BY created_at DESC LIMIT ?",
                values,
            ).fetchall()
            return [self._row(row) for row in rows]

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        status: MemoryConflictStatus,
        resolved_record_id: str | None,
        note: str,
    ) -> MemoryConflict:
        if status == MemoryConflictStatus.PENDING:
            raise ValueError("resolved conflict cannot remain pending")
        with self._transaction() as db:
            current = self._require(db, conflict_id)
            if current.status != MemoryConflictStatus.PENDING:
                raise ValueError("memory conflict is already resolved")
            db.execute(
                """
                UPDATE memory_conflicts
                SET status = ?, resolved_record_id = ?, resolution_note = ?,
                    candidate_json = '{}', resolved_at = ?
                WHERE id = ?
                """,
                (status.value, resolved_record_id, note, _now(), conflict_id),
            )
            return self._require(db, conflict_id)

    def purge_record_references(self, record_id: str) -> None:
        with self._transaction() as db:
            db.execute(
                """
                DELETE FROM memory_conflicts
                WHERE existing_record_id = ? OR resolved_record_id = ?
                """,
                (record_id, record_id),
            )

    def cancel_pending_for_record_key(self, record_key: str, *, note: str) -> int:
        normalized_key = record_key.strip()
        if not normalized_key:
            return 0
        with self._transaction() as db:
            cursor = db.execute(
                """
                UPDATE memory_conflicts
                SET status = ?, candidate_json = '{}', resolution_note = ?,
                    resolved_record_id = NULL, resolved_at = ?
                WHERE record_key = ? AND status = 'pending'
                """,
                (
                    MemoryConflictStatus.CANCELLED.value,
                    note,
                    _now(),
                    normalized_key,
                ),
            )
            return max(0, int(cursor.rowcount))

    def _require(self, db: sqlite3.Connection, conflict_id: str) -> MemoryConflict:
        row = db.execute(
            "SELECT * FROM memory_conflicts WHERE id = ?", (conflict_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"memory conflict not found: {conflict_id}")
        return self._row(row)

    @staticmethod
    def _row(row: sqlite3.Row) -> MemoryConflict:
        return MemoryConflict(
            id=str(row["id"]),
            record_key=str(row["record_key"]),
            existing_record_id=row["existing_record_id"],
            candidate=_load(row["candidate_json"]),
            candidate_hash=str(row["candidate_hash"]),
            reason=str(row["reason"]),
            status=MemoryConflictStatus(row["status"]),
            resolved_record_id=row["resolved_record_id"],
            resolution_note=str(row["resolution_note"]),
            created_at=str(row["created_at"]),
            resolved_at=row["resolved_at"],
        )
