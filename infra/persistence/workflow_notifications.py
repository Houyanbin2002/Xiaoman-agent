"""Durable notification attempts in the existing workflow database.

Sender acceptance is not a read receipt. An interrupted send is deliberately
ambiguous: only a user may authorize another attempt in that case.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any


class WorkflowNotifications:
    def __init__(
        self,
        transaction: Callable[[], AbstractContextManager[sqlite3.Connection]],
    ) -> None:
        self._transaction = transaction
        with self._transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS workflow_notifications (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
                    source_key TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    target_status TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    retry_limit INTEGER NOT NULL DEFAULT 3,
                    version INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL,
                    lease_until REAL,
                    error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(workflow_id, source_key)
                );
                CREATE INDEX IF NOT EXISTS idx_workflow_notification_status
                    ON workflow_notifications(status, next_attempt_at);
            """)

    @staticmethod
    def _current(db: sqlite3.Connection, row: Any) -> bool:
        workflow = db.execute(
            "SELECT * FROM workflows WHERE id = ?", (row["workflow_id"],)
        ).fetchone()
        if workflow is None or workflow["status"] == "cancelled":
            return False
        if (workflow["channel"], workflow["chat_id"]) != (
            row["channel"],
            row["chat_id"],
        ):
            return False
        if row["step_id"]:
            step = db.execute(
                "SELECT status, updated_at FROM workflow_steps WHERE workflow_id=? AND id=?",
                (row["workflow_id"], row["step_id"]),
            ).fetchone()
            return bool(
                step
                and step["status"] == "waiting"
                and step["updated_at"] == row["source_version"]
            )
        return (
            workflow["status"] == row["target_status"]
            and str(workflow["revision"]) == row["source_version"]
        )

    def ensure(
        self,
        *,
        workflow_id: str,
        step_id: str,
        source_version: str,
        target_status: str,
        channel: str,
        chat_id: str,
        message: str,
    ) -> str:
        key = f"{step_id or 'terminal'}:{target_status}:{source_version}"
        now = time.time()
        with self._transaction() as db:
            db.execute(
                """INSERT OR IGNORE INTO workflow_notifications
                (id,workflow_id,source_key,step_id,source_version,target_status,
                 channel,chat_id,message,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    uuid.uuid4().hex,
                    workflow_id,
                    key,
                    step_id,
                    source_version,
                    target_status,
                    channel,
                    chat_id,
                    message,
                    now,
                    now,
                ),
            )
            return str(
                db.execute(
                    "SELECT id FROM workflow_notifications WHERE workflow_id=? AND source_key=?",
                    (workflow_id, key),
                ).fetchone()["id"]
            )

    def recover_expired(self) -> None:
        with self._transaction() as db:
            db.execute(
                """UPDATE workflow_notifications SET status='unknown',
                error='发送中断或回执超时，结果未知；请核对消息后决定是否补发。',
                next_attempt_at=NULL, lease_until=NULL, version=version+1, updated_at=?
                WHERE status='sending' AND lease_until <= ?""",
                (time.time(), time.time()),
            )

    def due_ids(self) -> list[str]:
        with self._transaction() as db:
            return [
                str(row["id"])
                for row in db.execute(
                    """SELECT id FROM workflow_notifications WHERE status IN ('pending','failed')
                AND attempts < retry_limit AND coalesce(next_attempt_at,0) <= ?
                ORDER BY updated_at LIMIT 20""",
                    (time.time(),),
                ).fetchall()
            ]

    def claim(self, notification_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self._transaction() as db:
            # Serialize read/validate/claim across processes, not only asyncio tasks.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM workflow_notifications WHERE id=?", (notification_id,)
            ).fetchone()
            if not row or row["status"] not in ("pending", "failed"):
                return None
            if not self._current(db, row):
                db.execute(
                    "UPDATE workflow_notifications SET status='cancelled', "
                    "version=version+1, updated_at=? WHERE id=?",
                    (now, notification_id),
                )
                return None
            if (
                row["attempts"] >= row["retry_limit"]
                or (row["next_attempt_at"] or 0) > now
            ):
                return None
            db.execute(
                """UPDATE workflow_notifications SET status='sending', attempts=attempts+1,
                version=version+1, lease_until=?, next_attempt_at=NULL, updated_at=? WHERE id=?""",
                (now + 300, now, notification_id),
            )
            return dict(
                db.execute(
                    "SELECT * FROM workflow_notifications WHERE id=?",
                    (notification_id,),
                ).fetchone()
            )

    def is_current_attempt(self, notification_id: str, version: int) -> bool:
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM workflow_notifications WHERE id=?", (notification_id,)
            ).fetchone()
            return bool(
                row
                and row["status"] == "sending"
                and row["version"] == version
                and self._current(db, row)
            )

    @staticmethod
    def _mark_notified(db: sqlite3.Connection, row: Any) -> None:
        if not WorkflowNotifications._current(db, row):
            return
        # Do not change source revision/timestamps: delivery is not a new task plan.
        if row["step_id"]:
            db.execute(
                "UPDATE workflow_steps SET notified_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE workflow_id=? AND id=?",
                (row["workflow_id"], row["step_id"]),
            )
        else:
            db.execute(
                "UPDATE workflows SET notified_status=? WHERE id=?",
                (row["target_status"], row["workflow_id"]),
            )

    def finish(
        self, notification_id: str, version: int, *, status: str, error: str = ""
    ) -> None:
        if status not in ("accepted", "failed", "unknown"):
            raise ValueError("无效的通知结果")
        now = time.time()
        with self._transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM workflow_notifications WHERE id=?", (notification_id,)
            ).fetchone()
            if not row or row["status"] != "sending" or row["version"] != version:
                return
            retry_at = (
                now + (30 if row["attempts"] == 1 else 120)
                if status == "failed" and row["attempts"] < row["retry_limit"]
                else None
            )
            db.execute(
                """UPDATE workflow_notifications SET status=?, error=?,
                       next_attempt_at=?, lease_until=NULL, version=version+1, updated_at=? WHERE id=?""",
                (status, error[:2000], retry_at, now, notification_id),
            )
            if status == "accepted":
                self._mark_notified(db, row)

    def list_for_workflow(self, workflow_id: str) -> list[dict[str, Any]]:
        with self._transaction() as db:
            rows = db.execute(
                "SELECT * FROM workflow_notifications WHERE workflow_id=? "
                "ORDER BY created_at DESC LIMIT 50",
                (workflow_id,),
            ).fetchall()
            return [{**dict(row), "is_current": self._current(db, row)} for row in rows]

    def resolve(
        self,
        workflow_id: str,
        notification_id: str,
        *,
        expected_version: int,
        action: str,
        confirm_unknown: bool = False,
    ) -> None:
        with self._transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM workflow_notifications WHERE id=? AND workflow_id=?",
                (notification_id, workflow_id),
            ).fetchone()
            if not row or row["version"] != expected_version:
                raise ValueError("通知状态已变化，请刷新后重试")
            if not self._current(db, row) or row["status"] not in ("failed", "unknown"):
                raise ValueError("仅当前失败或结果未知的通知可处理，不会重新执行任务")
            if action == "received":
                self._mark_notified(db, row)
                db.execute(
                    "UPDATE workflow_notifications SET status='confirmed', error='', "
                    "next_attempt_at=NULL, version=version+1, updated_at=? WHERE id=?",
                    (time.time(), notification_id),
                )
            elif action == "retry":
                if row["status"] == "unknown" and not confirm_unknown:
                    raise ValueError("通知可能已经送达，须确认接受重复通知风险后补发")
                db.execute(
                    """UPDATE workflow_notifications SET status='pending', error='',
                           retry_limit=attempts+3, next_attempt_at=NULL,
                           version=version+1, updated_at=? WHERE id=?""",
                    (time.time(), notification_id),
                )
            else:
                raise ValueError("无效的通知处理动作")
