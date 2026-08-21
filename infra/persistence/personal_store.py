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

from core.personal.models import (
    AccessPolicy,
    DataCategory,
    MemoryEvidence,
    PersonalEntityType,
    PersonalRecord,
    RecordRevision,
    RecordSource,
    RecordStatus,
    SensitivityLevel,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value}")
    return parsed.astimezone(timezone.utc).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        result = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


class PersonalStore:
    """SQLite source of truth for governed personal assistant data."""

    SCHEMA_VERSION = 5
    _UPDATABLE_FIELDS = frozenset(
        {
            "record_key",
            "title",
            "summary",
            "data",
            "source",
            "source_ref",
            "confidence",
            "sensitivity",
            "data_category",
            "access_policy",
            "valid_from",
            "expires_at",
            "user_locked",
            "allow_auto_update",
        }
    )

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
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS personal_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS personal_records (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    record_key TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    data_json TEXT NOT NULL DEFAULT '{}',
                    source TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 1.0
                        CHECK(confidence >= 0.0 AND confidence <= 1.0),
                    sensitivity TEXT NOT NULL DEFAULT 'personal',
                    data_category TEXT NOT NULL DEFAULT 'general',
                    access_policy TEXT NOT NULL DEFAULT 'standard',
                    status TEXT NOT NULL DEFAULT 'active',
                    valid_from TEXT,
                    valid_to TEXT,
                    expires_at TEXT,
                    last_confirmed_at TEXT,
                    user_locked INTEGER NOT NULL DEFAULT 0,
                    allow_auto_update INTEGER NOT NULL DEFAULT 1,
                    supersedes_id TEXT,
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    forgotten_at TEXT,
                    FOREIGN KEY (supersedes_id) REFERENCES personal_records(id)
                        ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_personal_records_type_status
                    ON personal_records(entity_type, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_personal_records_key
                    ON personal_records(entity_type, record_key, status);
                CREATE INDEX IF NOT EXISTS idx_personal_records_expiry
                    ON personal_records(status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_personal_records_source
                    ON personal_records(source, source_ref);
                CREATE TABLE IF NOT EXISTS personal_record_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (record_id) REFERENCES personal_records(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_personal_revision_record
                    ON personal_record_revisions(record_id, id);

                CREATE TABLE IF NOT EXISTS personal_memory_evidence (
                    id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    statement TEXT NOT NULL,
                    statement_hash TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0
                        CHECK(confidence >= 0.0 AND confidence <= 1.0),
                    observed_at TEXT NOT NULL,
                    FOREIGN KEY (record_id) REFERENCES personal_records(id)
                        ON DELETE CASCADE,
                    UNIQUE(record_id, source, source_ref, statement_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_personal_evidence_record
                    ON personal_memory_evidence(record_id, observed_at DESC);
                """
            )
            columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(personal_records)").fetchall()
            }
            if "data_category" not in columns:
                db.execute(
                    "ALTER TABLE personal_records ADD COLUMN data_category TEXT NOT NULL DEFAULT 'general'"
                )
            if "access_policy" not in columns:
                db.execute(
                    "ALTER TABLE personal_records ADD COLUMN access_policy TEXT NOT NULL DEFAULT 'standard'"
                )
            if "valid_to" not in columns:
                db.execute("ALTER TABLE personal_records ADD COLUMN valid_to TEXT")
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_personal_records_governance
                ON personal_records(entity_type, data_category, access_policy, status)
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
                raise RuntimeError("personal store is closed")
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

    def create_record(
        self,
        *,
        entity_type: PersonalEntityType,
        title: str,
        summary: str,
        data: dict[str, Any],
        source: RecordSource,
        record_key: str = "",
        confidence: float = 1.0,
        sensitivity: SensitivityLevel = SensitivityLevel.PERSONAL,
        data_category: DataCategory = DataCategory.GENERAL,
        access_policy: AccessPolicy = AccessPolicy.STANDARD,
        valid_from: str | None = None,
        expires_at: str | None = None,
        user_locked: bool = False,
        allow_auto_update: bool = True,
        supersedes_id: str | None = None,
        actor: str = "user",
    ) -> PersonalRecord:
        with self._transaction() as db:
            return self._insert_record(
                db,
                entity_type=entity_type,
                title=title,
                summary=summary,
                data=data,
                source=source,
                record_key=record_key,
                confidence=confidence,
                sensitivity=sensitivity,
                data_category=data_category,
                access_policy=access_policy,
                valid_from=valid_from,
                expires_at=expires_at,
                user_locked=user_locked,
                allow_auto_update=allow_auto_update,
                supersedes_id=supersedes_id,
                actor=actor,
            )

    def _insert_record(
        self,
        db: sqlite3.Connection,
        *,
        entity_type: PersonalEntityType,
        title: str,
        summary: str,
        data: dict[str, Any],
        source: RecordSource,
        record_key: str,
        confidence: float,
        sensitivity: SensitivityLevel,
        data_category: DataCategory,
        access_policy: AccessPolicy,
        valid_from: str | None,
        expires_at: str | None,
        user_locked: bool,
        allow_auto_update: bool,
        supersedes_id: str | None,
        actor: str,
    ) -> PersonalRecord:
        record_id = f"pr_{uuid.uuid4().hex}"
        now = _now()
        db.execute(
            """
            INSERT INTO personal_records(
                id, entity_type, record_key, title, summary, data_json,
                source, source_ref, confidence, sensitivity, data_category,
                access_policy, status,
                valid_from, expires_at, user_locked, allow_auto_update,
                supersedes_id, revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                record_id,
                entity_type.value,
                record_key,
                title,
                summary,
                _json_dump(data),
                source.source,
                source.source_ref,
                confidence,
                sensitivity.value,
                data_category.value,
                access_policy.value,
                _normalize_timestamp(valid_from),
                _normalize_timestamp(expires_at),
                int(user_locked),
                int(allow_auto_update),
                supersedes_id,
                now,
                now,
            ),
        )
        record = self._require_record(db, record_id)
        self._insert_revision(db, record, action="created", actor=actor, reason="")
        return record

    def get_record(self, record_id: str) -> PersonalRecord | None:
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM personal_records WHERE id = ?", (record_id,)
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def find_active_by_key(
        self, entity_type: PersonalEntityType, record_key: str
    ) -> PersonalRecord | None:
        with self._transaction() as db:
            row = db.execute(
                """
                SELECT * FROM personal_records
                WHERE entity_type = ? AND record_key = ? AND status = 'active'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (entity_type.value, record_key),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def list_records(
        self,
        *,
        entity_type: PersonalEntityType | None = None,
        statuses: Sequence[RecordStatus] | None = None,
        limit: int = 100,
    ) -> list[PersonalRecord]:
        actual_limit = max(1, min(int(limit), 1000))
        clauses: list[str] = []
        values: list[Any] = []
        if entity_type is not None:
            clauses.append("entity_type = ?")
            values.append(entity_type.value)
        selected_statuses = list(statuses or [RecordStatus.ACTIVE])
        if selected_statuses:
            placeholders = ",".join("?" for _ in selected_statuses)
            clauses.append(f"status IN ({placeholders})")
            values.extend(status.value for status in selected_statuses)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(actual_limit)
        with self._transaction() as db:
            rows = db.execute(
                f"SELECT * FROM personal_records {where} ORDER BY updated_at DESC LIMIT ?",
                values,
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def update_record(
        self,
        record_id: str,
        *,
        changes: dict[str, Any],
        actor: str,
        reason: str = "",
        automatic: bool = False,
    ) -> PersonalRecord:
        unknown = set(changes) - self._UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"unsupported personal record fields: {sorted(unknown)}")
        if not changes:
            record = self.get_record(record_id)
            if record is None:
                raise ValueError(f"personal record not found: {record_id}")
            return record
        with self._transaction() as db:
            current = self._require_record(db, record_id)
            self._assert_mutable(current, automatic=automatic)
            assignments: list[str] = []
            values: list[Any] = []
            for key, value in changes.items():
                column = "data_json" if key == "data" else key
                if key == "data":
                    value = _json_dump(value)
                elif key == "sensitivity" and isinstance(value, SensitivityLevel):
                    value = value.value
                elif key == "data_category" and isinstance(value, DataCategory):
                    value = value.value
                elif key == "access_policy" and isinstance(value, AccessPolicy):
                    value = value.value
                elif key in {"user_locked", "allow_auto_update"}:
                    value = int(bool(value))
                elif key == "confidence":
                    value = float(value)
                    if not 0.0 <= value <= 1.0:
                        raise ValueError("confidence must be between 0 and 1")
                elif key in {"valid_from", "expires_at"}:
                    value = _normalize_timestamp(value)
                assignments.append(f"{column} = ?")
                values.append(value)
            assignments.extend(["revision = revision + 1", "updated_at = ?"])
            values.extend([_now(), record_id])
            db.execute(
                f"UPDATE personal_records SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            updated = self._require_record(db, record_id)
            self._insert_revision(
                db, updated, action="updated", actor=actor, reason=reason
            )
            return updated

    def confirm_record(
        self, record_id: str, *, actor: str = "user"
    ) -> PersonalRecord:
        with self._transaction() as db:
            current = self._require_record(db, record_id)
            self._assert_mutable(current, automatic=False)
            now = _now()
            db.execute(
                """
                UPDATE personal_records
                SET last_confirmed_at = ?, confidence = 1.0,
                    revision = revision + 1, updated_at = ?
                WHERE id = ?
                """,
                (now, now, record_id),
            )
            confirmed = self._require_record(db, record_id)
            self._insert_revision(
                db, confirmed, action="confirmed", actor=actor, reason=""
            )
            return confirmed

    def supersede_record(
        self,
        record_id: str,
        *,
        replacement: dict[str, Any],
        actor: str,
        reason: str = "",
    ) -> PersonalRecord:
        with self._transaction() as db:
            current = self._require_record(db, record_id)
            self._assert_mutable(current, automatic=False)
            now = _now()
            valid_boundary = (
                _normalize_timestamp(replacement.get("valid_from")) or now
            )
            db.execute(
                """
                UPDATE personal_records
                SET status = 'superseded', valid_to = COALESCE(valid_to, ?),
                    revision = revision + 1, updated_at = ?
                WHERE id = ?
                """,
                (valid_boundary, now, record_id),
            )
            superseded = self._require_record(db, record_id)
            self._insert_revision(
                db, superseded, action="superseded", actor=actor, reason=reason
            )
            source = RecordSource(
                str(replacement.get("source", current.source.source)),
                str(replacement.get("source_ref", current.source.source_ref)),
            )
            return self._insert_record(
                db,
                entity_type=PersonalEntityType(
                    replacement.get("entity_type", current.entity_type)
                ),
                record_key=str(replacement.get("record_key", current.record_key)),
                title=str(replacement.get("title", current.title)),
                summary=str(replacement.get("summary", current.summary)),
                data=dict(replacement.get("data", current.data)),
                source=source,
                confidence=float(replacement.get("confidence", current.confidence)),
                sensitivity=SensitivityLevel(
                    replacement.get("sensitivity", current.sensitivity)
                ),
                data_category=DataCategory(
                    replacement.get("data_category", current.data_category)
                ),
                access_policy=AccessPolicy(
                    replacement.get("access_policy", current.access_policy)
                ),
                valid_from=valid_boundary,
                expires_at=replacement.get("expires_at", current.expires_at),
                user_locked=bool(replacement.get("user_locked", current.user_locked)),
                allow_auto_update=bool(
                    replacement.get("allow_auto_update", current.allow_auto_update)
                ),
                supersedes_id=current.id,
                actor=actor,
            )

    def forget_record(
        self,
        record_id: str,
        *,
        actor: str = "user",
        reason: str = "",
        purge_content: bool = True,
    ) -> PersonalRecord:
        with self._transaction() as db:
            current = self._require_record(db, record_id)
            if current.status == RecordStatus.FORGOTTEN:
                return current
            now = _now()
            if purge_content:
                db.execute(
                    """
                    UPDATE personal_records
                    SET status = 'forgotten', title = '[forgotten]', summary = '',
                        data_json = '{}', source_ref = '', revision = revision + 1,
                        valid_to = COALESCE(valid_to, ?),
                        updated_at = ?, forgotten_at = ?
                    WHERE id = ?
                    """,
                    (now, now, now, record_id),
                )
                db.execute(
                    """
                    UPDATE personal_record_revisions
                    SET snapshot_json = '{"redacted":true}'
                    WHERE record_id = ?
                    """,
                    (record_id,),
                )
                db.execute(
                    "DELETE FROM personal_memory_evidence WHERE record_id = ?",
                    (record_id,),
                )
            else:
                db.execute(
                    """
                    UPDATE personal_records
                    SET status = 'forgotten', revision = revision + 1,
                        valid_to = COALESCE(valid_to, ?),
                        updated_at = ?, forgotten_at = ?
                    WHERE id = ?
                    """,
                    (now, now, now, record_id),
                )
            forgotten = self._require_record(db, record_id)
            self._insert_revision(
                db, forgotten, action="forgotten", actor=actor, reason=reason
            )
            return forgotten

    def expire_due(
        self, *, actor: str = "system", now: str | None = None
    ) -> list[str]:
        current = _normalize_timestamp(now) or _now()
        with self._transaction() as db:
            rows = db.execute(
                """
                SELECT id FROM personal_records
                WHERE status = 'active' AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (current,),
            ).fetchall()
            expired_ids = [str(row["id"]) for row in rows]
            for record_id in expired_ids:
                db.execute(
                    """
                    UPDATE personal_records
                    SET status = 'expired', valid_to = COALESCE(valid_to, ?),
                        revision = revision + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (current, current, record_id),
                )
                record = self._require_record(db, record_id)
                self._insert_revision(
                    db,
                    record,
                    action="expired",
                    actor=actor,
                    reason="validity window ended",
                )
            return expired_ids

    def hard_delete_record(self, record_id: str) -> bool:
        with self._transaction() as db:
            row = db.execute(
                "SELECT id FROM personal_records WHERE id = ?", (record_id,)
            ).fetchone()
            if row is None:
                return False
            db.execute("DELETE FROM personal_records WHERE id = ?", (record_id,))
            return True

    def list_revisions(self, record_id: str) -> list[RecordRevision]:
        with self._transaction() as db:
            self._require_record(db, record_id)
            rows = db.execute(
                """
                SELECT * FROM personal_record_revisions
                WHERE record_id = ? ORDER BY id
                """,
                (record_id,),
            ).fetchall()
            return [
                RecordRevision(
                    id=int(row["id"]),
                    record_id=str(row["record_id"]),
                    revision=int(row["revision"]),
                    action=str(row["action"]),
                    actor=str(row["actor"]),
                    reason=str(row["reason"]),
                    snapshot=_json_load(row["snapshot_json"]),
                    created_at=str(row["created_at"]),
                )
                for row in rows
            ]

    def list_lineage(
        self, record_id: str, *, limit: int = 100
    ) -> list[PersonalRecord]:
        """Return all temporal versions for the logical fact containing record_id."""

        actual_limit = max(1, min(int(limit), 1000))
        with self._transaction() as db:
            current = self._require_record(db, record_id)
            rows = db.execute(
                """
                SELECT * FROM personal_records
                WHERE entity_type = ? AND record_key = ?
                ORDER BY COALESCE(valid_from, created_at), created_at, id
                LIMIT ?
                """,
                (current.entity_type.value, current.record_key, actual_limit),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def add_memory_evidence(
        self,
        record_id: str,
        *,
        source: RecordSource,
        statement: str,
        confidence: float,
        observed_at: str | None = None,
    ) -> tuple[MemoryEvidence, bool]:
        normalized_statement = statement.strip()
        if not normalized_statement:
            raise ValueError("memory evidence statement must not be empty")
        actual_confidence = max(0.0, min(1.0, float(confidence)))
        actual_observed_at = _normalize_timestamp(observed_at) or _now()
        statement_hash = hashlib.sha256(
            normalized_statement.casefold().encode("utf-8")
        ).hexdigest()
        with self._transaction() as db:
            self._require_record(db, record_id)
            existing = db.execute(
                """
                SELECT * FROM personal_memory_evidence
                WHERE record_id = ? AND source = ? AND source_ref = ?
                    AND statement_hash = ?
                """,
                (
                    record_id,
                    source.source,
                    source.source_ref,
                    statement_hash,
                ),
            ).fetchone()
            if existing is not None:
                return self._row_to_evidence(existing), False
            evidence_id = f"pe_{uuid.uuid4().hex}"
            db.execute(
                """
                INSERT INTO personal_memory_evidence(
                    id, record_id, source, source_ref, statement,
                    statement_hash, confidence, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    record_id,
                    source.source,
                    source.source_ref,
                    normalized_statement,
                    statement_hash,
                    actual_confidence,
                    actual_observed_at,
                ),
            )
            row = db.execute(
                "SELECT * FROM personal_memory_evidence WHERE id = ?",
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("failed to persist memory evidence")
            return self._row_to_evidence(row), True

    def list_memory_evidence(
        self, record_id: str, *, limit: int = 100
    ) -> list[MemoryEvidence]:
        actual_limit = max(1, min(int(limit), 1000))
        with self._transaction() as db:
            self._require_record(db, record_id)
            rows = db.execute(
                """
                SELECT * FROM personal_memory_evidence
                WHERE record_id = ?
                ORDER BY observed_at, id
                LIMIT ?
                """,
                (record_id, actual_limit),
            ).fetchall()
            return [self._row_to_evidence(row) for row in rows]

    @staticmethod
    def _assert_mutable(record: PersonalRecord, *, automatic: bool) -> None:
        if record.status != RecordStatus.ACTIVE:
            raise ValueError(f"personal record is not active: {record.status.value}")
        if automatic and (record.user_locked or not record.allow_auto_update):
            raise PermissionError("personal record requires explicit user update")

    def _require_record(
        self, db: sqlite3.Connection, record_id: str
    ) -> PersonalRecord:
        row = db.execute(
            "SELECT * FROM personal_records WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"personal record not found: {record_id}")
        return self._row_to_record(row)

    def _insert_revision(
        self,
        db: sqlite3.Connection,
        record: PersonalRecord,
        *,
        action: str,
        actor: str,
        reason: str,
    ) -> None:
        db.execute(
            """
            INSERT INTO personal_record_revisions(
                record_id, revision, action, actor, reason, snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.revision,
                action,
                actor,
                reason,
                _json_dump(record.to_dict()),
                _now(),
            ),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PersonalRecord:
        return PersonalRecord(
            id=str(row["id"]),
            entity_type=PersonalEntityType(row["entity_type"]),
            record_key=str(row["record_key"]),
            title=str(row["title"]),
            summary=str(row["summary"]),
            data=_json_load(row["data_json"]),
            source=RecordSource(str(row["source"]), str(row["source_ref"])),
            confidence=float(row["confidence"]),
            sensitivity=SensitivityLevel(row["sensitivity"]),
            data_category=DataCategory(row["data_category"]),
            access_policy=AccessPolicy(row["access_policy"]),
            status=RecordStatus(row["status"]),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            expires_at=row["expires_at"],
            last_confirmed_at=row["last_confirmed_at"],
            user_locked=bool(row["user_locked"]),
            allow_auto_update=bool(row["allow_auto_update"]),
            supersedes_id=row["supersedes_id"],
            revision=int(row["revision"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            forgotten_at=row["forgotten_at"],
        )

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row) -> MemoryEvidence:
        return MemoryEvidence(
            id=str(row["id"]),
            record_id=str(row["record_id"]),
            source=RecordSource(str(row["source"]), str(row["source_ref"])),
            statement=str(row["statement"]),
            confidence=float(row["confidence"]),
            observed_at=str(row["observed_at"]),
        )
