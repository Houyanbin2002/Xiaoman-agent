from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from core.tracing.models import TraceEvent, TraceRecord


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class TraceStore:
    """Small append-only trace ledger for the single-process desktop runtime."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    flow TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parent_trace_id TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_traces_session
                    ON traces(session_key, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_traces_updated
                    ON traces(updated_at DESC);

                CREATE TABLE IF NOT EXISTS trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_span_id TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms INTEGER,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(trace_id) REFERENCES traces(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_trace_events_trace
                    ON trace_events(trace_id, id);
                """)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("trace store is closed")
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

    def start_trace(
        self,
        *,
        trace_id: str,
        flow: str,
        session_key: str,
        title: str,
        parent_trace_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord:
        now = _now()
        with self._transaction() as db:
            row = db.execute(
                "SELECT metadata_json FROM traces WHERE id = ?", (trace_id,)
            ).fetchone()
            merged = dict(_load(row["metadata_json"], {}) if row else {})
            merged.update(metadata or {})
            db.execute(
                """
                INSERT INTO traces(
                    id, flow, session_key, title, status, parent_trace_id,
                    metadata_json, started_at, updated_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    flow = traces.flow,
                    session_key = CASE WHEN traces.session_key = '' THEN excluded.session_key ELSE traces.session_key END,
                    title = CASE WHEN traces.title = '' THEN excluded.title ELSE traces.title END,
                    status = 'running',
                    parent_trace_id = CASE WHEN traces.parent_trace_id = '' THEN excluded.parent_trace_id ELSE traces.parent_trace_id END,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    finished_at = NULL
                """,
                (
                    trace_id,
                    flow,
                    session_key,
                    title.strip()[:180],
                    parent_trace_id,
                    _dump(merged),
                    now,
                    now,
                ),
            )
        return self.require_trace(trace_id)

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord | None:
        now = _now()
        with self._transaction() as db:
            row = db.execute(
                "SELECT metadata_json FROM traces WHERE id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return None
            merged = dict(_load(row["metadata_json"], {}))
            merged.update(metadata or {})
            db.execute(
                "UPDATE traces SET status = ?, metadata_json = ?, updated_at = ?, finished_at = ? WHERE id = ?",
                (status, _dump(merged), now, now, trace_id),
            )
        return self.get_trace(trace_id)

    def append_event(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str,
        category: str,
        name: str,
        status: str,
        started_at: str,
        finished_at: str | None,
        duration_ms: int | None,
        summary: str,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        with self._transaction() as db:
            cursor = db.execute(
                """
                INSERT INTO trace_events(
                    trace_id, span_id, parent_span_id, category, name, status,
                    started_at, finished_at, duration_ms, summary, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    span_id,
                    parent_span_id,
                    category,
                    name,
                    status,
                    started_at,
                    finished_at,
                    duration_ms,
                    summary.strip()[:500],
                    _dump(payload or {}),
                ),
            )
            db.execute(
                "UPDATE traces SET updated_at = ? WHERE id = ?",
                (_now(), trace_id),
            )
            event_id = int(cursor.lastrowid)
        return self.require_event(event_id)

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT t.*, COUNT(e.id) AS event_count
                FROM traces t LEFT JOIN trace_events e ON e.trace_id = t.id
                WHERE t.id = ? GROUP BY t.id
                """,
                (trace_id,),
            ).fetchone()
        return self._row_to_trace(row) if row is not None else None

    def require_trace(self, trace_id: str) -> TraceRecord:
        trace = self.get_trace(trace_id)
        if trace is None:
            raise ValueError(f"trace not found: {trace_id}")
        return trace

    def list_traces(
        self,
        *,
        session_key: str | None = None,
        limit: int = 100,
    ) -> list[TraceRecord]:
        where = "WHERE t.session_key = ?" if session_key else ""
        params: tuple[Any, ...] = (
            (session_key, max(1, min(limit, 500)))
            if session_key
            else (max(1, min(limit, 500)),)
        )
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT t.*, COUNT(e.id) AS event_count
                FROM traces t LEFT JOIN trace_events e ON e.trace_id = t.id
                {where}
                GROUP BY t.id ORDER BY t.started_at DESC LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def list_events(self, trace_id: str) -> list[TraceEvent]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM trace_events WHERE trace_id = ? ORDER BY started_at, id",
                (trace_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def require_event(self, event_id: int) -> TraceEvent:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM trace_events WHERE id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"trace event not found: {event_id}")
        return self._row_to_event(row)

    @staticmethod
    def _row_to_trace(row: sqlite3.Row) -> TraceRecord:
        return TraceRecord(
            id=str(row["id"]),
            flow=str(row["flow"]),
            session_key=str(row["session_key"]),
            title=str(row["title"]),
            status=str(row["status"]),
            parent_trace_id=str(row["parent_trace_id"] or ""),
            started_at=str(row["started_at"]),
            updated_at=str(row["updated_at"]),
            finished_at=str(row["finished_at"]) if row["finished_at"] else None,
            metadata=dict(_load(row["metadata_json"], {})),
            event_count=int(row["event_count"] if "event_count" in row.keys() else 0),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> TraceEvent:
        return TraceEvent(
            id=int(row["id"]),
            trace_id=str(row["trace_id"]),
            span_id=str(row["span_id"]),
            parent_span_id=str(row["parent_span_id"] or ""),
            category=str(row["category"]),
            name=str(row["name"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            finished_at=str(row["finished_at"]) if row["finished_at"] else None,
            duration_ms=(
                int(row["duration_ms"]) if row["duration_ms"] is not None else None
            ),
            summary=str(row["summary"]),
            payload=dict(_load(row["payload_json"], {})),
        )
