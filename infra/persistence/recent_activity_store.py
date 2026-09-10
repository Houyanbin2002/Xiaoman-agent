"""Current activity projection; history is retained separately for recall.

Uses the existing consolidation database. No fuzzy deletion, and no automatic
promotion of legacy prose into active user obligations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.conversation_semantics.models import RecentActivityCandidate

_OPEN = {"planned", "active"}
_CLOSED = {"completed", "cancelled", "dismissed"}
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _title_key(title: str) -> str:
    return " ".join(title.casefold().split())


class RecentActivityStore:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db = self._connect()
        try:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS recent_activities (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, title_key TEXT NOT NULL,
                    summary TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL,
                    session_key TEXT NOT NULL, source_refs TEXT NOT NULL,
                    updated_at TEXT NOT NULL, expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_recent_activity_title ON recent_activities(title_key);
                CREATE TABLE IF NOT EXISTS recent_activity_updates (
                    source_ref TEXT PRIMARY KEY, activity_id TEXT NOT NULL,
                    outcome TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recent_activity_sources (
                    source_ref TEXT NOT NULL, activity_id TEXT NOT NULL,
                    PRIMARY KEY(source_ref, activity_id)
                );
                INSERT OR IGNORE INTO recent_activity_sources
                    SELECT 'message:' || j.value, a.id
                    FROM recent_activities a, json_each(a.source_refs) j
                    WHERE json_valid(a.source_refs);
            """)
            db.commit()
        finally:
            db.close()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self._path), timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def apply(
        self, entries: list[RecentActivityCandidate], *, batch_id: str, session_key: str
    ) -> None:
        now = _now()
        db = self._connect()
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                for index, item in enumerate(entries):
                    source_ref = f"{batch_id}:activity:{index}"
                    if db.execute(
                        "SELECT 1 FROM recent_activity_updates WHERE source_ref=?",
                        (source_ref,),
                    ).fetchone():
                        continue
                    activity_id, outcome = self._apply_one(
                        db, item, session_key, source_ref, now
                    )
                    if outcome not in {"applied", "history_only"}:
                        logger.info(
                            "recent activity update skipped batch=%s outcome=%s",
                            batch_id,
                            outcome,
                        )
                    db.execute(
                        "INSERT INTO recent_activity_updates VALUES (?,?,?,?)",
                        (source_ref, activity_id, outcome, now.isoformat()),
                    )
                    if outcome == "applied":
                        db.executemany(
                            "INSERT OR IGNORE INTO recent_activity_sources VALUES (?,?)",
                            [
                                ("update:" + source_ref, activity_id),
                                *(
                                    ("message:" + ref, activity_id)
                                    for ref in item.source_message_ids
                                ),
                            ],
                        )
        finally:
            db.close()

    def _apply_one(
        self,
        db: sqlite3.Connection,
        item: RecentActivityCandidate,
        session_key: str,
        source_ref: str,
        now: datetime,
    ) -> tuple[str, str]:
        # Legacy summaries stay in history, not the current-focus projection.
        if item.status not in _OPEN | _CLOSED or not item.title:
            return "", "history_only"
        row = None
        if item.activity_id:
            row = db.execute(
                "SELECT * FROM recent_activities WHERE id=?", (item.activity_id,)
            ).fetchone()
            if row is None or item.expected_revision != row["revision"]:
                return item.activity_id, "stale_or_unknown_target"
        else:
            matches = db.execute(
                "SELECT * FROM recent_activities WHERE title_key=?",
                (_title_key(item.title),),
            ).fetchall()
            if matches:
                # The analyzer must select the visible ID/revision, not mutate
                # an existing entity based only on an invented paraphrase.
                return "", "target_required"
            if item.status in _CLOSED:
                return "", "unresolved_closure"
        if (
            row
            and (row["status"] in _CLOSED or row["expires_at"] <= now.isoformat())
            and item.status in _OPEN
            and not item.reopen
        ):
            return str(row["id"]), "explicit_reopen_required"
        evidence_time = now
        if item.occurred_at:
            try:
                occurred = datetime.fromisoformat(
                    item.occurred_at.replace("Z", "+00:00")
                )
                if occurred.tzinfo is not None:
                    evidence_time = min(now, occurred.astimezone(timezone.utc))
            except ValueError:
                pass
        # Processing an old backlog is not evidence that the user is still
        # working on it today. Never extend freshness merely due to delivery.
        expires = evidence_time + timedelta(days=14)
        if item.expires_at:
            try:
                requested = datetime.fromisoformat(
                    item.expires_at.replace("Z", "+00:00")
                )
                if requested.tzinfo is not None:
                    expires = min(
                        requested.astimezone(timezone.utc), now + timedelta(days=90)
                    )
            except ValueError:
                pass
        activity_id = (
            str(row["id"])
            if row
            else "activity_" + hashlib.sha256(source_ref.encode()).hexdigest()[:24]
        )
        db.execute(
            """INSERT INTO recent_activities VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, status=excluded.status,
                   revision=excluded.revision, session_key=excluded.session_key,
                   source_refs=excluded.source_refs, updated_at=excluded.updated_at, expires_at=excluded.expires_at""",
            (
                activity_id,
                item.title,
                _title_key(item.title),
                item.summary,
                item.status,
                int(row["revision"]) + 1 if row else 1,
                session_key,
                json.dumps(item.source_message_ids, ensure_ascii=False),
                now.isoformat(),
                expires.isoformat(),
            ),
        )
        return activity_id, "applied"

    def recall_states(self, refs: list[str]) -> dict[str, list[dict[str, object]]]:
        """Resolve exact evidence IDs in batches, not the bounded UI snapshot.

        All prior evidence remains linked after a revision or process restart.
        Rejected/stale updates never gain a lifecycle association.
        """
        refs = list(dict.fromkeys(refs))
        result: dict[str, list[dict[str, object]]] = {}
        if not refs:
            return result
        now = _now().isoformat()
        db = self._connect()
        try:
            for start in range(0, len(refs), 400):
                chunk = refs[start : start + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows = db.execute(
                    f"""SELECT s.source_ref, a.id, a.status, a.revision, a.expires_at
                        FROM recent_activity_sources s JOIN recent_activities a ON a.id=s.activity_id
                        WHERE s.source_ref IN ({placeholders})""",
                    chunk,
                ).fetchall()
                for row in rows:
                    state = {
                        key: row[key]
                        for key in ("id", "status", "revision", "expires_at")
                    }
                    if state["status"] in _OPEN and str(state["expires_at"]) <= now:
                        state["status"] = "expired"
                    result.setdefault(row["source_ref"], []).append(state)
            return result
        finally:
            db.close()

    def snapshot(self, *, limit: int = 30) -> list[dict[str, Any]]:
        now = _now()
        db = self._connect()
        try:
            rows = db.execute(
                """SELECT * FROM recent_activities
                WHERE (status IN ('planned','active') AND expires_at>?) OR updated_at>?
                ORDER BY updated_at DESC LIMIT ?""",
                (
                    now.isoformat(),
                    (now - timedelta(days=30)).isoformat(),
                    max(1, min(60, limit)),
                ),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "status": (
                        "expired"
                        if row["status"] in _OPEN
                        and row["expires_at"] <= now.isoformat()
                        else row["status"]
                    ),
                    "revision": row["revision"],
                    "updated_at": row["updated_at"],
                    "expires_at": row["expires_at"],
                }
                for row in rows
            ]
        finally:
            db.close()

    def render(self, *, max_entries: int = 18, max_chars: int = 4500) -> str:
        rows = self.snapshot(limit=60)
        active = [row for row in rows if row["status"] in _OPEN]
        closed = [row for row in rows if row["status"] in _CLOSED | {"expired"}][:6]
        lines = []
        labels = {
            "planned": "计划中",
            "active": "进行中",
            "completed": "已完成",
            "cancelled": "已取消",
            "dismissed": "不再关注",
            "expired": "已过期，状态待确认",
        }
        used = 0
        # State boundaries take precedence over open-topic background when bounded.
        for row in [*closed, *active]:
            text = row["summary"] if row["status"] in _OPEN else row["title"]
            line = f"- [{labels[row['status']]}] {' '.join(str(text).split())}"
            if used + len(line) > max_chars or len(lines) >= max_entries:
                continue
            lines.append(line)
            used += len(line)
        return (
            "# 近期事项状态\n\n"
            "> 只有计划中/进行中属于当前关注；已结束或过期事项不是主动联系理由。"
            "旧会话与图谱片段是历史证据，不得据此重新激活事项。"
            "这些状态不代表独立定时提醒已被取消。\n\n"
            + ("\n".join(lines) if lines else "当前没有已确认且未过期的近期事项。")
        )
