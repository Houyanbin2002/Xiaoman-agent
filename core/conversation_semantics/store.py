from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import SemanticBatchPayload


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _batch_id(analysis_version: str, message_ids: list[str]) -> str:
    material = json.dumps(
        [analysis_version, *message_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "semantic_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class PreparedSemanticBatch:
    batch_id: str
    session_key: str
    channel: str
    chat_id: str
    analysis_version: str
    message_ids: tuple[str, ...]
    end_seq: int
    context_consolidate_through: int
    payload: SemanticBatchPayload
    user_message_ids: tuple[str, ...] = ()
    execution_episode_ids: tuple[str, ...] = ()
    execution_tool_names: tuple[str, ...] = ()

    def to_event(self) -> ConversationSemanticBatchCommitted:
        return ConversationSemanticBatchCommitted(
            batch_id=self.batch_id,
            session_key=self.session_key,
            channel=self.channel,
            chat_id=self.chat_id,
            analysis_version=self.analysis_version,
            message_ids=self.message_ids,
            end_seq=self.end_seq,
            context_consolidate_through=self.context_consolidate_through,
            payload=self.payload,
            user_message_ids=self.user_message_ids,
            execution_episode_ids=self.execution_episode_ids,
            execution_tool_names=self.execution_tool_names,
        )


class ConversationSemanticStore:
    """Durable prepared/delivered state for shared conversation analysis."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._closed = False
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversation_semantic_state (
                    session_key TEXT PRIMARY KEY,
                    analyzed_through_seq INTEGER NOT NULL DEFAULT -1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_semantic_batches (
                    batch_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    analysis_version TEXT NOT NULL,
                    message_ids_json TEXT NOT NULL,
                    user_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    execution_episode_ids_json TEXT NOT NULL DEFAULT '[]',
                    execution_tool_names_json TEXT NOT NULL DEFAULT '[]',
                    end_seq INTEGER NOT NULL,
                    context_consolidate_through INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('prepared', 'delivered')),
                    created_at TEXT NOT NULL,
                    delivered_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_semantic_batches_delivery
                    ON conversation_semantic_batches(status, created_at);

                CREATE TABLE IF NOT EXISTS conversation_semantic_deliveries (
                    batch_id TEXT NOT NULL,
                    consumer_id TEXT NOT NULL,
                    delivered_at TEXT NOT NULL,
                    PRIMARY KEY (batch_id, consumer_id)
                );
                """)
            columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(conversation_semantic_batches)"
                ).fetchall()
            }
            if "user_message_ids_json" not in columns:
                self._conn.execute(
                    "ALTER TABLE conversation_semantic_batches "
                    "ADD COLUMN user_message_ids_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "execution_episode_ids_json" not in columns:
                self._conn.execute(
                    "ALTER TABLE conversation_semantic_batches "
                    "ADD COLUMN execution_episode_ids_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "execution_tool_names_json" not in columns:
                self._conn.execute(
                    "ALTER TABLE conversation_semantic_batches "
                    "ADD COLUMN execution_tool_names_json TEXT NOT NULL DEFAULT '[]'"
                )
            self._conn.commit()

    def pending_cursor(self, session_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT analyzed_through_seq FROM conversation_semantic_state "
                "WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return int(row["analyzed_through_seq"]) if row is not None else -1

    def prepare(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        analysis_version: str,
        message_ids: list[str],
        end_seq: int,
        context_consolidate_through: int,
        payload: SemanticBatchPayload,
        user_message_ids: list[str] | None = None,
        execution_episode_ids: list[str] | None = None,
        execution_tool_names: list[str] | None = None,
    ) -> PreparedSemanticBatch:
        if not message_ids:
            raise ValueError("message_ids must not be empty")
        batch_id = _batch_id(analysis_version, message_ids)
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO conversation_semantic_batches (
                    batch_id, session_key, channel, chat_id, analysis_version,
                    message_ids_json, user_message_ids_json,
                    execution_episode_ids_json, execution_tool_names_json, end_seq,
                    context_consolidate_through, payload_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?)
                """,
                (
                    batch_id,
                    session_key,
                    channel,
                    chat_id,
                    analysis_version,
                    json.dumps(message_ids, ensure_ascii=False),
                    json.dumps(user_message_ids or [], ensure_ascii=False),
                    json.dumps(execution_episode_ids or [], ensure_ascii=False),
                    json.dumps(execution_tool_names or [], ensure_ascii=False),
                    int(end_seq),
                    int(context_consolidate_through),
                    json.dumps(payload.to_mapping(), ensure_ascii=False),
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM conversation_semantic_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"failed to prepare semantic batch {batch_id}")
        return self._row_to_batch(row)

    def list_undelivered(self) -> list[PreparedSemanticBatch]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM conversation_semantic_batches "
                "WHERE status = 'prepared' ORDER BY created_at, batch_id"
            ).fetchall()
        return [self._row_to_batch(row) for row in rows]

    def delivered_consumers(self, batch_id: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT consumer_id FROM conversation_semantic_deliveries "
                "WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()
        return {str(row["consumer_id"]) for row in rows}

    def mark_consumer_delivered(self, batch_id: str, consumer_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO conversation_semantic_deliveries "
                "(batch_id, consumer_id, delivered_at) VALUES (?, ?, ?)",
                (batch_id, consumer_id, _now_iso()),
            )
            self._conn.commit()

    def mark_delivered(self, batch_id: str) -> None:
        now = _now_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT session_key, end_seq FROM conversation_semantic_batches "
                "WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if row is None:
                raise KeyError(batch_id)
            session_key = str(row["session_key"])
            end_seq = int(row["end_seq"])
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE conversation_semantic_batches "
                    "SET status = 'delivered', delivered_at = COALESCE(delivered_at, ?) "
                    "WHERE batch_id = ?",
                    (now, batch_id),
                )
                self._conn.execute(
                    """
                    INSERT INTO conversation_semantic_state (
                        session_key, analyzed_through_seq, updated_at
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(session_key) DO UPDATE SET
                        analyzed_through_seq = MAX(
                            conversation_semantic_state.analyzed_through_seq,
                            excluded.analyzed_through_seq
                        ),
                        updated_at = excluded.updated_at
                    """,
                    (session_key, end_seq, now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    @staticmethod
    def _row_to_batch(row: sqlite3.Row) -> PreparedSemanticBatch:
        message_ids = json.loads(str(row["message_ids_json"]))
        user_message_ids = json.loads(str(row["user_message_ids_json"] or "[]"))
        execution_episode_ids = json.loads(
            str(row["execution_episode_ids_json"] or "[]")
        )
        execution_tool_names = json.loads(str(row["execution_tool_names_json"] or "[]"))
        payload = SemanticBatchPayload.from_mapping(
            json.loads(str(row["payload_json"]))
        )
        return PreparedSemanticBatch(
            batch_id=str(row["batch_id"]),
            session_key=str(row["session_key"]),
            channel=str(row["channel"]),
            chat_id=str(row["chat_id"]),
            analysis_version=str(row["analysis_version"]),
            message_ids=tuple(str(item) for item in message_ids),
            end_seq=int(row["end_seq"]),
            context_consolidate_through=int(row["context_consolidate_through"]),
            payload=payload,
            user_message_ids=tuple(str(item) for item in user_message_ids),
            execution_episode_ids=tuple(str(item) for item in execution_episode_ids),
            execution_tool_names=tuple(str(item) for item in execution_tool_names),
        )
