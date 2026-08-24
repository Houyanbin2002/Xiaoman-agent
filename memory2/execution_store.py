from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import json
import sqlite3
import threading

from core.memory.execution import (
    ExecutionAuthority,
    ExecutionLifecycleStatus,
    ExecutionMemoryKind,
    ExecutionMemoryState,
    ExecutionScope,
    ExecutionScopeKind,
    ExecutionVerificationStatus,
    apply_execution_outcome,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_memory_state (
    item_id                 TEXT PRIMARY KEY,
    execution_kind          TEXT NOT NULL DEFAULT 'procedure',
    scope_kind              TEXT NOT NULL DEFAULT 'global',
    workspace_id            TEXT NOT NULL DEFAULT '',
    project_id              TEXT NOT NULL DEFAULT '',
    tool_name               TEXT NOT NULL DEFAULT '',
    plugin_name             TEXT NOT NULL DEFAULT '',
    platform                TEXT NOT NULL DEFAULT '',
    environment_fingerprint TEXT NOT NULL DEFAULT '',
    version_key             TEXT NOT NULL DEFAULT '',
    version_value           TEXT NOT NULL DEFAULT '',
    verification_status     TEXT NOT NULL DEFAULT 'candidate',
    authority               TEXT NOT NULL DEFAULT 'learned',
    lifecycle_status        TEXT NOT NULL DEFAULT 'proposed',
    user_locked             INTEGER NOT NULL DEFAULT 0,
    extraction_confidence   REAL NOT NULL DEFAULT 0.0,
    success_count           INTEGER NOT NULL DEFAULT 0,
    failure_count           INTEGER NOT NULL DEFAULT 0,
    last_verified_at        TEXT,
    expires_at              TEXT,
    evidence_json           TEXT NOT NULL DEFAULT '[]',
    metadata_json           TEXT NOT NULL DEFAULT '{}',
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_execution_memory_scope
    ON execution_memory_state (scope_kind, workspace_id, project_id, tool_name, plugin_name);
CREATE INDEX IF NOT EXISTS ix_execution_memory_verification
    ON execution_memory_state (verification_status, updated_at);
"""


class ExecutionMemoryRepository:
    """Persistence boundary for Agent execution experience.

    The semantic text remains in ``memory_items`` so the existing vector index can
    be reused. Applicability and reliability live here and never share personal
    memory reinforcement counters.
    """

    def __init__(self, db: sqlite3.Connection, lock: threading.RLock) -> None:
        self._db = db
        self._lock = lock
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._ensure_columns()
            self._db.commit()

    def _ensure_columns(self) -> None:
        columns = {
            str(row[1])
            for row in self._db.execute(
                "PRAGMA table_info(execution_memory_state)"
            ).fetchall()
        }
        additions = {
            "authority": "TEXT NOT NULL DEFAULT 'learned'",
            "lifecycle_status": "TEXT NOT NULL DEFAULT 'proposed'",
            "user_locked": "INTEGER NOT NULL DEFAULT 0",
            "extraction_confidence": "REAL NOT NULL DEFAULT 0.0",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                self._db.execute(
                    f"ALTER TABLE execution_memory_state ADD COLUMN {name} {sql_type}"
                )
        self._db.execute(
            "UPDATE execution_memory_state SET lifecycle_status='active' "
            "WHERE verification_status='verified' AND lifecycle_status='proposed' "
            "AND success_count >= 2"
        )

    def upsert(self, state: ExecutionMemoryState) -> None:
        with self._lock:
            exists = self._db.execute(
                "SELECT 1 FROM memory_items WHERE id=? LIMIT 1",
                (state.item_id,),
            ).fetchone()
            if exists is None:
                raise ValueError(
                    f"execution memory item does not exist: {state.item_id}"
                )
            self._upsert(state)
            self._db.commit()

    def _upsert(
        self,
        state: ExecutionMemoryState,
        *,
        created_at: str | None = None,
    ) -> None:
        now = _now_iso()
        self._db.execute(
            """
            INSERT INTO execution_memory_state (
                item_id, execution_kind, scope_kind, workspace_id, project_id,
                tool_name, plugin_name, platform, environment_fingerprint,
                version_key, version_value, verification_status, authority,
                lifecycle_status, user_locked, extraction_confidence, success_count,
                failure_count, last_verified_at, expires_at, evidence_json,
                metadata_json, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(item_id) DO UPDATE SET
                execution_kind=excluded.execution_kind,
                scope_kind=excluded.scope_kind,
                workspace_id=excluded.workspace_id,
                project_id=excluded.project_id,
                tool_name=excluded.tool_name,
                plugin_name=excluded.plugin_name,
                platform=excluded.platform,
                environment_fingerprint=excluded.environment_fingerprint,
                version_key=excluded.version_key,
                version_value=excluded.version_value,
                verification_status=excluded.verification_status,
                authority=excluded.authority,
                lifecycle_status=excluded.lifecycle_status,
                user_locked=excluded.user_locked,
                extraction_confidence=excluded.extraction_confidence,
                success_count=excluded.success_count,
                failure_count=excluded.failure_count,
                last_verified_at=excluded.last_verified_at,
                expires_at=excluded.expires_at,
                evidence_json=excluded.evidence_json,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                state.item_id,
                state.kind.value,
                state.scope.kind.value,
                state.scope.workspace_id,
                state.scope.project_id,
                state.scope.tool_name,
                state.scope.plugin_name,
                state.scope.platform,
                state.scope.environment_fingerprint,
                state.scope.version_key,
                state.scope.version_value,
                state.verification_status.value,
                state.authority.value,
                state.lifecycle_status.value,
                int(state.user_locked),
                state.extraction_confidence,
                state.success_count,
                state.failure_count,
                _datetime_iso(state.last_verified_at),
                _datetime_iso(state.expires_at),
                json.dumps(state.evidence_refs, ensure_ascii=False),
                json.dumps(dict(state.metadata), ensure_ascii=False),
                created_at or now,
                now,
            ),
        )

    def get(self, item_id: str) -> ExecutionMemoryState | None:
        with self._lock:
            row = self._db.execute(
                f"SELECT {_STATE_COLUMNS} FROM execution_memory_state WHERE item_id=?",
                (item_id,),
            ).fetchone()
        return _state_from_row(row) if row is not None else None

    def list(
        self,
        *,
        include_inactive: bool = False,
        limit: int = 500,
    ) -> list[dict[str, object]]:
        where = "" if include_inactive else "WHERE m.status='active'"
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT m.id, m.summary, m.source_ref, m.status, m.created_at,
                       m.updated_at, m.extra_json, {_PREFIXED_STATE_COLUMNS}
                FROM memory_items m
                JOIN execution_memory_state e ON e.item_id=m.id
                {where}
                ORDER BY m.updated_at DESC, m.id ASC
                LIMIT ?
                """,
                (max(1, min(int(limit), 5000)),),
            ).fetchall()
        return [
            {
                "id": str(row[0]),
                "summary": str(row[1] or ""),
                "source_ref": str(row[2] or ""),
                "status": str(row[3] or ""),
                "created_at": str(row[4] or ""),
                "updated_at": str(row[5] or ""),
                "extra_json": _json_object(row[6]),
                "execution": _state_from_row(row[7:]),
            }
            for row in rows
        ]

    def record_outcome(
        self,
        item_id: str,
        *,
        success: bool,
        evidence_ref: str = "",
        verified_at: datetime | None = None,
    ) -> ExecutionMemoryState:
        with self._lock:
            state = self.get(item_id)
            if state is None:
                raise ValueError(f"execution memory state does not exist: {item_id}")
            updated = apply_execution_outcome(
                state,
                success=success,
                evidence_ref=evidence_ref,
                verified_at=verified_at,
            )
            self._upsert(updated)
            self._db.commit()
            return updated

    def mark_superseded(self, item_ids: Sequence[str]) -> None:
        clean = tuple(
            dict.fromkeys(str(item).strip() for item in item_ids if str(item).strip())
        )
        if not clean:
            return
        placeholders = ",".join("?" for _ in clean)
        with self._lock:
            self._db.execute(
                f"""
                UPDATE execution_memory_state
                SET verification_status=?, lifecycle_status=?, updated_at=?
                WHERE item_id IN ({placeholders})
                """,
                (
                    ExecutionVerificationStatus.SUPERSEDED.value,
                    ExecutionLifecycleStatus.SUPERSEDED.value,
                    _now_iso(),
                    *clean,
                ),
            )
            self._db.commit()

    def suspend(self, item_ids: Sequence[str]) -> None:
        clean = tuple(
            dict.fromkeys(str(item).strip() for item in item_ids if str(item).strip())
        )
        if not clean:
            return
        placeholders = ",".join("?" for _ in clean)
        with self._lock:
            self._db.execute(
                f"UPDATE execution_memory_state "
                f"SET lifecycle_status=?, updated_at=? "
                f"WHERE item_id IN ({placeholders})",
                (ExecutionLifecycleStatus.SUSPENDED.value, _now_iso(), *clean),
            )
            self._db.commit()

    def delete(self, item_ids: Sequence[str]) -> None:
        clean = tuple(
            dict.fromkeys(str(item).strip() for item in item_ids if str(item).strip())
        )
        if not clean:
            return
        placeholders = ",".join("?" for _ in clean)
        with self._lock:
            self._db.execute(
                f"DELETE FROM execution_memory_state WHERE item_id IN ({placeholders})",
                clean,
            )


_STATE_COLUMNS = """
item_id, execution_kind, scope_kind, workspace_id, project_id,
tool_name, plugin_name, platform, environment_fingerprint,
version_key, version_value, verification_status, success_count,
authority, lifecycle_status, user_locked, extraction_confidence,
failure_count, last_verified_at, expires_at, evidence_json, metadata_json
""".strip()

_PREFIXED_STATE_COLUMNS = ", ".join(
    f"e.{column.strip()}" for column in _STATE_COLUMNS.replace("\n", " ").split(",")
)


def _state_from_row(row: Sequence[object]) -> ExecutionMemoryState:
    evidence = json.loads(str(row[20] or "[]"))
    metadata = json.loads(str(row[21] or "{}"))
    return ExecutionMemoryState(
        item_id=str(row[0]),
        kind=ExecutionMemoryKind(str(row[1])),
        scope=ExecutionScope(
            kind=ExecutionScopeKind(str(row[2])),
            workspace_id=str(row[3] or ""),
            project_id=str(row[4] or ""),
            tool_name=str(row[5] or ""),
            plugin_name=str(row[6] or ""),
            platform=str(row[7] or ""),
            environment_fingerprint=str(row[8] or ""),
            version_key=str(row[9] or ""),
            version_value=str(row[10] or ""),
        ),
        verification_status=ExecutionVerificationStatus(str(row[11])),
        authority=ExecutionAuthority(str(row[13])),
        lifecycle_status=ExecutionLifecycleStatus(str(row[14])),
        user_locked=bool(row[15]),
        extraction_confidence=_float(row[16]),
        success_count=_int(row[12]),
        failure_count=_int(row[17]),
        last_verified_at=_parse_datetime(row[18]),
        expires_at=_parse_datetime(row[19]),
        evidence_refs=tuple(str(item) for item in evidence if str(item).strip()),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _json_object(raw: object) -> dict[str, object]:
    if not raw:
        return {}
    value = json.loads(str(raw))
    return value if isinstance(value, dict) else {}


def _parse_datetime(raw: object) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _datetime_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    if not isinstance(value, (str, int, float)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
