from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from core.workflow.models import (
    SUCCESS_STEP_STATUSES,
    TERMINAL_WORKFLOW_STATUSES,
    StepExecutor,
    StepKind,
    StepSpec,
    StepStatus,
    WorkflowEvent,
    WorkflowInstance,
    WorkflowStatus,
    WorkflowStep,
)

_STEP_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_SUBAGENT_PROFILES = frozenset({"research", "scripting", "general"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


class WorkflowStore:
    """SQLite-backed workflow state and append-only event ledger."""

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
            db.executescript("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL DEFAULT '',
                    context_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    error TEXT NOT NULL DEFAULT '',
                    notified_status TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workflow_steps (
                    workflow_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    depends_on_json TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    executor TEXT NOT NULL DEFAULT 'agent',
                    profile TEXT NOT NULL DEFAULT 'research',
                    allowed_tools_json TEXT NOT NULL DEFAULT '[]',
                    output_json TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 2,
                    next_run_at TEXT,
                    notified_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workflow_id, id),
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_status
                    ON workflows(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_workflow_session
                    ON workflows(session_key, updated_at);
                CREATE INDEX IF NOT EXISTS idx_workflow_step_status
                    ON workflow_steps(status, next_run_at);

                CREATE TABLE IF NOT EXISTS workflow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_events
                    ON workflow_events(workflow_id, id);
                """)
            columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(workflow_steps)").fetchall()
            }
            if "executor" not in columns:
                db.execute(
                    "ALTER TABLE workflow_steps ADD COLUMN executor TEXT NOT NULL DEFAULT 'agent'"
                )
            if "profile" not in columns:
                db.execute(
                    "ALTER TABLE workflow_steps ADD COLUMN profile TEXT NOT NULL DEFAULT 'research'"
                )
            if "allowed_tools_json" not in columns:
                db.execute(
                    "ALTER TABLE workflow_steps ADD COLUMN allowed_tools_json TEXT NOT NULL DEFAULT '[]'"
                )
            workflow_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(workflows)").fetchall()
            }
            if "trace_id" not in workflow_columns:
                db.execute(
                    "ALTER TABLE workflows ADD COLUMN trace_id TEXT NOT NULL DEFAULT ''"
                )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("workflow store is closed")
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

    def create_workflow(
        self,
        *,
        name: str,
        goal: str,
        steps: Sequence[StepSpec],
        session_key: str,
        channel: str,
        chat_id: str,
        trace_id: str = "",
        context: dict[str, Any] | None = None,
        auto_start: bool = True,
    ) -> WorkflowInstance:
        normalized = self._validate_specs(steps)
        workflow_id = uuid.uuid4().hex
        now = _now()
        status = WorkflowStatus.RUNNING if auto_start else WorkflowStatus.DRAFT
        with self._transaction() as db:
            db.execute(
                """
                INSERT INTO workflows(
                    id, name, goal, status, session_key, channel, chat_id,
                    trace_id, context_json, revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    workflow_id,
                    name.strip(),
                    goal.strip(),
                    status.value,
                    session_key,
                    channel,
                    chat_id,
                    trace_id,
                    _json_dump(context or {}),
                    now,
                    now,
                ),
            )
            for position, step in enumerate(normalized):
                db.execute(
                    """
                    INSERT INTO workflow_steps(
                        workflow_id, id, position, title, description, kind, status,
                        depends_on_json, input_json, executor, profile,
                        allowed_tools_json, max_attempts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workflow_id,
                        step.id,
                        position,
                        step.title,
                        step.description,
                        step.kind.value,
                        StepStatus.PENDING.value,
                        _json_dump(list(step.depends_on)),
                        _json_dump(step.input),
                        step.executor.value,
                        step.profile,
                        _json_dump(list(step.allowed_tools)),
                        step.max_attempts,
                        now,
                        now,
                    ),
                )
            self._insert_event(
                db,
                workflow_id,
                1,
                "workflow_created",
                {"auto_start": auto_start, "step_count": len(normalized)},
                now,
            )
        return self.require_workflow(workflow_id)

    def get_workflow(self, workflow_id: str) -> WorkflowInstance | None:
        with self._lock:
            resolved = self._resolve_id(self._db, workflow_id)
            if resolved is None:
                return None
            return self._load_workflow(self._db, resolved)

    def require_workflow(self, workflow_id: str) -> WorkflowInstance:
        workflow = self.get_workflow(workflow_id)
        if workflow is None:
            raise ValueError(f"未找到 workflow: {workflow_id}")
        return workflow

    def list_workflows(
        self,
        *,
        status: str | None = None,
        session_key: str | None = None,
        limit: int = 20,
    ) -> list[WorkflowInstance]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(WorkflowStatus(status).value)
        if session_key:
            clauses.append("session_key = ?")
            params.append(session_key)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(100, int(limit))))
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM workflows {where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
            return [self._load_workflow(self._db, str(row["id"])) for row in rows]

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete completed history without allowing active execution to vanish."""

        with self._transaction() as db:
            resolved = self._resolve_id(db, workflow_id)
            if resolved is None:
                return False
            row = db.execute(
                "SELECT status FROM workflows WHERE id = ?", (resolved,)
            ).fetchone()
            status = WorkflowStatus(str(row["status"]))
            if status not in TERMINAL_WORKFLOW_STATUSES:
                raise ValueError("运行中的任务不能删除，请先取消任务")
            db.execute("DELETE FROM workflows WHERE id = ?", (resolved,))
            return True

    def list_events(self, workflow_id: str, *, limit: int = 50) -> list[WorkflowEvent]:
        with self._lock:
            resolved = self._resolve_id(self._db, workflow_id)
            if resolved is None:
                return []
            rows = self._db.execute(
                """
                SELECT * FROM workflow_events
                WHERE workflow_id = ? ORDER BY id DESC LIMIT ?
                """,
                (resolved, max(1, min(200, int(limit)))),
            ).fetchall()
        return [
            WorkflowEvent(
                id=int(row["id"]),
                workflow_id=str(row["workflow_id"]),
                revision=int(row["revision"]),
                event_type=str(row["event_type"]),
                payload=dict(_json_load(row["payload_json"], {})),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def start_workflow(self, workflow_id: str) -> WorkflowInstance:
        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            row = db.execute(
                "SELECT status FROM workflows WHERE id = ?", (resolved,)
            ).fetchone()
            current = WorkflowStatus(str(row["status"]))
            if current in TERMINAL_WORKFLOW_STATUSES:
                raise ValueError(f"workflow 已结束，不能启动: {current.value}")
            if current == WorkflowStatus.BLOCKED:
                failed = db.execute(
                    "SELECT 1 FROM workflow_steps WHERE workflow_id = ? AND status = ? LIMIT 1",
                    (resolved, StepStatus.FAILED.value),
                ).fetchone()
                if failed is not None:
                    raise ValueError("workflow 仍有失败步骤，请先 retry")
            db.execute(
                "UPDATE workflows SET status = ?, error = '' WHERE id = ?",
                (WorkflowStatus.RUNNING.value, resolved),
            )
            self._record_event(db, resolved, "workflow_started", {})
        return self.require_workflow(resolved)

    def replan_workflow(
        self,
        workflow_id: str,
        *,
        remaining_steps: Sequence[StepSpec],
        expected_revision: int,
        reason: str,
    ) -> WorkflowInstance:
        """Replace only unresolved steps while retaining completed evidence."""

        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            workflow_row = db.execute(
                "SELECT status, revision FROM workflows WHERE id = ?", (resolved,)
            ).fetchone()
            current_status = WorkflowStatus(str(workflow_row["status"]))
            current_revision = int(workflow_row["revision"])
            if current_status in TERMINAL_WORKFLOW_STATUSES:
                raise ValueError(f"workflow 已结束，不能重规划: {current_status.value}")
            if current_revision != int(expected_revision):
                raise ValueError(
                    "workflow revision 已变化，请重新 get 后再 replan："
                    f"expected={expected_revision}, current={current_revision}"
                )

            existing = self._load_steps(db, resolved)
            if any(step.status == StepStatus.RUNNING for step in existing):
                raise ValueError(
                    "仍有 running 步骤，必须等待步骤结束或先取消后再重规划"
                )
            preserved = [
                step for step in existing if step.status in SUCCESS_STEP_STATUSES
            ]
            preserved_specs = [
                StepSpec(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                    kind=step.kind,
                    depends_on=step.depends_on,
                    max_attempts=step.max_attempts,
                    input=step.input,
                    executor=step.executor,
                    profile=step.profile,
                    allowed_tools=step.allowed_tools,
                )
                for step in preserved
            ]
            normalized = self._validate_specs([*preserved_specs, *remaining_steps])
            preserved_ids = {step.id for step in preserved}
            normalized_remaining = [
                step for step in normalized if step.id not in preserved_ids
            ]
            replaced_ids = [
                step.id for step in existing if step.id not in preserved_ids
            ]
            now = _now()
            db.execute(
                "DELETE FROM workflow_steps WHERE workflow_id = ? "
                "AND status NOT IN (?, ?)",
                (
                    resolved,
                    StepStatus.SUCCEEDED.value,
                    StepStatus.SKIPPED.value,
                ),
            )
            for position, step in enumerate(preserved):
                db.execute(
                    "UPDATE workflow_steps SET position = ?, updated_at = ? "
                    "WHERE workflow_id = ? AND id = ?",
                    (position, now, resolved, step.id),
                )
            for offset, step in enumerate(normalized_remaining, start=len(preserved)):
                db.execute(
                    """
                    INSERT INTO workflow_steps(
                        workflow_id, id, position, title, description, kind, status,
                        depends_on_json, input_json, executor, profile,
                        allowed_tools_json, max_attempts, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved,
                        step.id,
                        offset,
                        step.title,
                        step.description,
                        step.kind.value,
                        StepStatus.PENDING.value,
                        _json_dump(list(step.depends_on)),
                        _json_dump(step.input),
                        step.executor.value,
                        step.profile,
                        _json_dump(list(step.allowed_tools)),
                        step.max_attempts,
                        now,
                        now,
                    ),
                )
            next_status = (
                WorkflowStatus.DRAFT
                if current_status == WorkflowStatus.DRAFT
                else WorkflowStatus.RUNNING
            )
            db.execute(
                "UPDATE workflows SET status = ?, error = '', notified_status = NULL, "
                "updated_at = ? WHERE id = ?",
                (next_status.value, now, resolved),
            )
            if current_status != WorkflowStatus.DRAFT:
                self._refresh_workflow_status(db, resolved)
            self._record_event(
                db,
                resolved,
                "workflow_replanned",
                {
                    "base_revision": current_revision,
                    "reason": reason.strip(),
                    "preserved_step_ids": [step.id for step in preserved],
                    "replaced_step_ids": replaced_ids,
                    "remaining_step_ids": [step.id for step in normalized_remaining],
                },
            )
        return self.require_workflow(resolved)

    def cancel_workflow(
        self, workflow_id: str, *, reason: str = ""
    ) -> WorkflowInstance:
        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            now = _now()
            db.execute(
                """
                UPDATE workflow_steps SET status = ?, error = ?, updated_at = ?
                WHERE workflow_id = ? AND status NOT IN (?, ?, ?, ?)
                """,
                (
                    StepStatus.CANCELLED.value,
                    reason,
                    now,
                    resolved,
                    StepStatus.SUCCEEDED.value,
                    StepStatus.FAILED.value,
                    StepStatus.SKIPPED.value,
                    StepStatus.CANCELLED.value,
                ),
            )
            db.execute(
                "UPDATE workflows SET status = ?, error = ? WHERE id = ?",
                (WorkflowStatus.CANCELLED.value, reason, resolved),
            )
            self._record_event(db, resolved, "workflow_cancelled", {"reason": reason})
        return self.require_workflow(resolved)

    def prepare_human_steps(self) -> list[tuple[WorkflowInstance, WorkflowStep]]:
        prepared: list[tuple[str, str]] = []
        with self._transaction() as db:
            workflow_rows = db.execute(
                "SELECT id FROM workflows WHERE status IN (?, ?)",
                (WorkflowStatus.RUNNING.value, WorkflowStatus.WAITING.value),
            ).fetchall()
            for workflow_row in workflow_rows:
                workflow_id = str(workflow_row["id"])
                steps = self._load_steps(db, workflow_id)
                statuses = {step.id: step.status for step in steps}
                for step in steps:
                    if step.status != StepStatus.PENDING or step.kind == StepKind.AGENT:
                        continue
                    if not self._deps_satisfied(step, statuses):
                        continue
                    now = _now()
                    cursor = db.execute(
                        """
                        UPDATE workflow_steps
                        SET status = ?, notified_at = NULL, updated_at = ?
                        WHERE workflow_id = ? AND id = ? AND status = ?
                        """,
                        (
                            StepStatus.WAITING.value,
                            now,
                            workflow_id,
                            step.id,
                            StepStatus.PENDING.value,
                        ),
                    )
                    if cursor.rowcount == 1:
                        self._refresh_workflow_status(db, workflow_id)
                        self._record_event(
                            db,
                            workflow_id,
                            "step_waiting",
                            {"step_id": step.id, "kind": step.kind.value},
                        )
                        prepared.append((workflow_id, step.id))
        return [
            (workflow, self._require_step(workflow, step_id))
            for workflow_id, step_id in prepared
            if (workflow := self.get_workflow(workflow_id)) is not None
        ]

    def claim_workflow_steps(
        self,
        workflow_id: str,
        *,
        limit: int = 3,
    ) -> list[tuple[WorkflowInstance, WorkflowStep]]:
        """Atomically claim runnable nodes for one LangGraph workflow thread."""
        claimed: list[str] = []
        now = _now()
        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            row = db.execute(
                "SELECT status FROM workflows WHERE id = ?", (resolved,)
            ).fetchone()
            if WorkflowStatus(str(row["status"])) != WorkflowStatus.RUNNING:
                return []
            steps = self._load_steps(db, resolved)
            statuses = {step.id: step.status for step in steps}
            for step in steps:
                if len(claimed) >= max(1, int(limit)):
                    break
                if step.kind != StepKind.AGENT or step.status != StepStatus.PENDING:
                    continue
                if step.next_run_at and step.next_run_at > now:
                    continue
                if not self._deps_satisfied(step, statuses):
                    continue
                cursor = db.execute(
                    """
                    UPDATE workflow_steps
                    SET status = ?, attempt_count = attempt_count + 1,
                        next_run_at = NULL, updated_at = ?
                    WHERE workflow_id = ? AND id = ? AND status = ?
                    """,
                    (
                        StepStatus.RUNNING.value,
                        now,
                        resolved,
                        step.id,
                        StepStatus.PENDING.value,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                self._record_event(
                    db,
                    resolved,
                    "step_started",
                    {"step_id": step.id, "attempt": step.attempt_count + 1},
                )
                claimed.append(step.id)
                statuses[step.id] = StepStatus.RUNNING
        workflow = self.require_workflow(resolved)
        return [
            (workflow, self._require_step(workflow, step_id)) for step_id in claimed
        ]

    def complete_step(
        self, workflow_id: str, step_id: str, *, output: Any
    ) -> WorkflowInstance:
        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            self._require_step_row(db, resolved, step_id)
            self._complete_step_if_status(
                db,
                resolved,
                step_id,
                output=output,
                expected_status=StepStatus.RUNNING,
            )
        return self.require_workflow(resolved)

    def fail_step(
        self,
        workflow_id: str,
        step_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
    ) -> WorkflowInstance:
        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            row = self._require_step_row(db, resolved, step_id)
            attempts = int(row["attempt_count"])
            max_attempts = int(row["max_attempts"])
            retrying = attempts < max_attempts
            next_run_at = (
                datetime.now(timezone.utc)
                .__add__(timedelta(seconds=max(0.0, retry_delay_seconds)))
                .isoformat()
                if retrying
                else None
            )
            status = StepStatus.PENDING if retrying else StepStatus.FAILED
            cursor = db.execute(
                """
                UPDATE workflow_steps
                SET status = ?, error = ?, next_run_at = ?, updated_at = ?
                WHERE workflow_id = ? AND id = ? AND status = ?
                """,
                (
                    status.value,
                    error,
                    next_run_at,
                    _now(),
                    resolved,
                    step_id,
                    StepStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount == 1:
                self._refresh_workflow_status(db, resolved)
                self._record_event(
                    db,
                    resolved,
                    "step_retry_scheduled" if retrying else "step_failed",
                    {
                        "step_id": step_id,
                        "attempt": attempts,
                        "max_attempts": max_attempts,
                        "error": error,
                        "next_run_at": next_run_at,
                    },
                )
        return self.require_workflow(resolved)

    def respond_to_step(
        self, workflow_id: str, step_id: str, *, response: str
    ) -> WorkflowInstance:
        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            row = self._require_step_row(db, resolved, step_id)
            if StepKind(str(row["kind"])) != StepKind.WAIT_USER:
                raise ValueError("该步骤不是 wait_user")
            if StepStatus(str(row["status"])) != StepStatus.WAITING:
                raise ValueError("该步骤当前不在等待用户输入")
            completed = self._complete_step_if_status(
                db,
                resolved,
                step_id,
                output={"response": response},
                expected_status=StepStatus.WAITING,
            )
            if not completed:
                raise ValueError("该步骤当前不在等待用户输入")
        return self.require_workflow(resolved)

    def approve_step(
        self,
        workflow_id: str,
        step_id: str,
        *,
        approved: bool,
        note: str = "",
    ) -> WorkflowInstance:
        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            row = self._require_step_row(db, resolved, step_id)
            if StepKind(str(row["kind"])) != StepKind.APPROVAL:
                raise ValueError("该步骤不是 approval")
            if StepStatus(str(row["status"])) != StepStatus.WAITING:
                raise ValueError("该步骤当前不在等待审批")
            if approved:
                completed = self._complete_step_if_status(
                    db,
                    resolved,
                    step_id,
                    output={"approved": True, "note": note},
                    expected_status=StepStatus.WAITING,
                )
                if not completed:
                    raise ValueError("该步骤当前不在等待审批")
        if approved:
            return self.require_workflow(resolved)
        return self.cancel_workflow(resolved, reason=note or f"审批被拒绝: {step_id}")

    def retry_step(self, workflow_id: str, step_id: str) -> WorkflowInstance:
        with self._transaction() as db:
            resolved = self._require_id(db, workflow_id)
            row = self._require_step_row(db, resolved, step_id)
            if StepStatus(str(row["status"])) != StepStatus.FAILED:
                raise ValueError("只有 failed 步骤可以 retry")
            db.execute(
                """
                UPDATE workflow_steps
                SET status = ?, attempt_count = 0, error = '', next_run_at = NULL,
                    notified_at = NULL, updated_at = ?
                WHERE workflow_id = ? AND id = ?
                """,
                (StepStatus.PENDING.value, _now(), resolved, step_id),
            )
            db.execute(
                "UPDATE workflows SET error = '', notified_status = NULL WHERE id = ?",
                (resolved,),
            )
            self._refresh_workflow_status(db, resolved)
            self._record_event(
                db, resolved, "step_retry_requested", {"step_id": step_id}
            )
        return self.require_workflow(resolved)

    def recover_interrupted(self) -> int:
        recovered = 0
        with self._transaction() as db:
            rows = db.execute(
                "SELECT * FROM workflow_steps WHERE status = ?",
                (StepStatus.RUNNING.value,),
            ).fetchall()
            affected: set[str] = set()
            for row in rows:
                workflow_id = str(row["workflow_id"])
                attempts = int(row["attempt_count"])
                max_attempts = int(row["max_attempts"])
                status = (
                    StepStatus.PENDING if attempts < max_attempts else StepStatus.FAILED
                )
                db.execute(
                    """
                    UPDATE workflow_steps
                    SET status = ?, error = ?, next_run_at = NULL, updated_at = ?
                    WHERE workflow_id = ? AND id = ?
                    """,
                    (
                        status.value,
                        (
                            "进程中断，已恢复等待重试"
                            if status == StepStatus.PENDING
                            else "进程中断且重试次数已耗尽"
                        ),
                        _now(),
                        workflow_id,
                        str(row["id"]),
                    ),
                )
                affected.add(workflow_id)
                recovered += 1
            for workflow_id in affected:
                self._refresh_workflow_status(db, workflow_id)
                self._record_event(
                    db,
                    workflow_id,
                    "workflow_recovered",
                    {"interrupted_steps": recovered},
                )
        return recovered

    def list_unnotified_waiting(
        self, *, limit: int = 20
    ) -> list[tuple[WorkflowInstance, WorkflowStep]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT workflow_id, id FROM workflow_steps
                WHERE status = ? AND notified_at IS NULL
                ORDER BY updated_at LIMIT ?
                """,
                (StepStatus.WAITING.value, max(1, int(limit))),
            ).fetchall()
        result: list[tuple[WorkflowInstance, WorkflowStep]] = []
        for row in rows:
            workflow = self.get_workflow(str(row["workflow_id"]))
            if workflow is not None:
                result.append((workflow, self._require_step(workflow, str(row["id"]))))
        return result

    def mark_step_notified(self, workflow_id: str, step_id: str) -> None:
        with self._transaction() as db:
            db.execute(
                """
                UPDATE workflow_steps SET notified_at = ?, updated_at = ?
                WHERE workflow_id = ? AND id = ?
                """,
                (_now(), _now(), workflow_id, step_id),
            )

    def list_unnotified_terminal(self, *, limit: int = 20) -> list[WorkflowInstance]:
        statuses = (
            WorkflowStatus.SUCCEEDED.value,
            WorkflowStatus.BLOCKED.value,
            WorkflowStatus.FAILED.value,
        )
        with self._lock:
            rows = self._db.execute(
                """
                SELECT id FROM workflows
                WHERE status IN (?, ?, ?)
                  AND (notified_status IS NULL OR notified_status != status)
                ORDER BY updated_at LIMIT ?
                """,
                (*statuses, max(1, int(limit))),
            ).fetchall()
            return [self._load_workflow(self._db, str(row["id"])) for row in rows]

    def mark_workflow_notified(self, workflow_id: str, status: WorkflowStatus) -> None:
        with self._transaction() as db:
            db.execute(
                "UPDATE workflows SET notified_status = ?, updated_at = ? WHERE id = ?",
                (status.value, _now(), workflow_id),
            )

    def _validate_specs(self, specs: Sequence[StepSpec]) -> list[StepSpec]:
        if not specs:
            raise ValueError("workflow 至少需要一个步骤")
        if len(specs) > 32:
            raise ValueError("workflow 最多支持 32 个步骤")
        normalized: list[StepSpec] = []
        ids: set[str] = set()
        for raw in specs:
            if not _STEP_ID_RE.fullmatch(raw.id):
                raise ValueError(f"非法步骤 ID: {raw.id!r}")
            if raw.id in ids:
                raise ValueError(f"步骤 ID 重复: {raw.id}")
            if not raw.title.strip() or not raw.description.strip():
                raise ValueError(f"步骤 {raw.id} 的 title/description 不能为空")
            ids.add(raw.id)
            profile = str(raw.profile or "research").strip().lower()
            if profile not in _SUBAGENT_PROFILES:
                raise ValueError(f"步骤 {raw.id} 的 profile 非法: {profile!r}")
            normalized.append(
                StepSpec(
                    id=raw.id,
                    title=raw.title.strip(),
                    description=raw.description.strip(),
                    kind=StepKind(raw.kind),
                    depends_on=tuple(dict.fromkeys(raw.depends_on)),
                    max_attempts=max(1, min(5, int(raw.max_attempts))),
                    input=dict(raw.input),
                    executor=(
                        StepExecutor.AGENT
                        if StepKind(raw.kind) != StepKind.AGENT
                        else StepExecutor(raw.executor)
                    ),
                    profile=profile,
                    allowed_tools=tuple(
                        dict.fromkeys(
                            name
                            for item in raw.allowed_tools
                            if (name := str(item).strip())
                        )
                    ),
                )
            )
        for step in normalized:
            unknown = [dep for dep in step.depends_on if dep not in ids]
            if unknown:
                raise ValueError(f"步骤 {step.id} 依赖不存在: {unknown}")
            if step.id in step.depends_on:
                raise ValueError(f"步骤 {step.id} 不能依赖自身")
        graph = {step.id: step.depends_on for step in normalized}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            if step_id in visiting:
                raise ValueError("workflow 步骤依赖存在环")
            visiting.add(step_id)
            for dep in graph[step_id]:
                visit(dep)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in graph:
            visit(step_id)
        return normalized

    def _resolve_id(self, db: sqlite3.Connection, value: str) -> str | None:
        token = str(value or "").strip()
        if not token:
            return None
        exact = db.execute("SELECT id FROM workflows WHERE id = ?", (token,)).fetchone()
        if exact is not None:
            return str(exact["id"])
        rows = db.execute(
            "SELECT id FROM workflows WHERE id LIKE ? ORDER BY created_at DESC LIMIT 2",
            (f"{token}%",),
        ).fetchall()
        if len(rows) > 1:
            raise ValueError(f"workflow ID 前缀不唯一: {token}")
        return str(rows[0]["id"]) if rows else None

    def _require_id(self, db: sqlite3.Connection, value: str) -> str:
        resolved = self._resolve_id(db, value)
        if resolved is None:
            raise ValueError(f"未找到 workflow: {value}")
        return resolved

    def _load_workflow(
        self, db: sqlite3.Connection, workflow_id: str
    ) -> WorkflowInstance:
        row = db.execute(
            "SELECT * FROM workflows WHERE id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"未找到 workflow: {workflow_id}")
        return WorkflowInstance(
            id=str(row["id"]),
            name=str(row["name"]),
            goal=str(row["goal"]),
            status=WorkflowStatus(str(row["status"])),
            session_key=str(row["session_key"]),
            channel=str(row["channel"]),
            chat_id=str(row["chat_id"]),
            trace_id=str(row["trace_id"] or ""),
            context=dict(_json_load(row["context_json"], {})),
            revision=int(row["revision"]),
            error=str(row["error"] or ""),
            notified_status=(
                str(row["notified_status"]) if row["notified_status"] else None
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            steps=self._load_steps(db, workflow_id),
        )

    def _load_steps(
        self, db: sqlite3.Connection, workflow_id: str
    ) -> list[WorkflowStep]:
        rows = db.execute(
            "SELECT * FROM workflow_steps WHERE workflow_id = ? ORDER BY position",
            (workflow_id,),
        ).fetchall()
        return [self._row_to_step(row) for row in rows]

    @staticmethod
    def _row_to_step(row: sqlite3.Row) -> WorkflowStep:
        return WorkflowStep(
            workflow_id=str(row["workflow_id"]),
            id=str(row["id"]),
            position=int(row["position"]),
            title=str(row["title"]),
            description=str(row["description"]),
            kind=StepKind(str(row["kind"])),
            status=StepStatus(str(row["status"])),
            depends_on=tuple(str(v) for v in _json_load(row["depends_on_json"], [])),
            input=dict(_json_load(row["input_json"], {})),
            executor=StepExecutor(str(row["executor"] or StepExecutor.AGENT.value)),
            profile=str(row["profile"] or "research"),
            allowed_tools=tuple(
                str(value)
                for value in _json_load(row["allowed_tools_json"], [])
                if str(value)
            ),
            output=_json_load(row["output_json"], None),
            error=str(row["error"] or ""),
            attempt_count=int(row["attempt_count"]),
            max_attempts=int(row["max_attempts"]),
            next_run_at=str(row["next_run_at"]) if row["next_run_at"] else None,
            notified_at=str(row["notified_at"]) if row["notified_at"] else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _require_step(workflow: WorkflowInstance, step_id: str) -> WorkflowStep:
        for step in workflow.steps:
            if step.id == step_id:
                return step
        raise ValueError(f"未找到步骤: {step_id}")

    @staticmethod
    def _require_step_row(
        db: sqlite3.Connection, workflow_id: str, step_id: str
    ) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM workflow_steps WHERE workflow_id = ? AND id = ?",
            (workflow_id, step_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"未找到步骤: {step_id}")
        return row

    @staticmethod
    def _deps_satisfied(step: WorkflowStep, statuses: dict[str, StepStatus]) -> bool:
        return all(
            statuses.get(dep) in SUCCESS_STEP_STATUSES for dep in step.depends_on
        )

    def _complete_step_if_status(
        self,
        db: sqlite3.Connection,
        workflow_id: str,
        step_id: str,
        *,
        output: Any,
        expected_status: StepStatus,
    ) -> bool:
        cursor = db.execute(
            """
            UPDATE workflow_steps
            SET status = ?, output_json = ?, error = '', next_run_at = NULL,
                updated_at = ?
            WHERE workflow_id = ? AND id = ? AND status = ?
            """,
            (
                StepStatus.SUCCEEDED.value,
                _json_dump(output),
                _now(),
                workflow_id,
                step_id,
                expected_status.value,
            ),
        )
        if cursor.rowcount != 1:
            return False
        self._refresh_workflow_status(db, workflow_id)
        self._record_event(
            db,
            workflow_id,
            "step_succeeded",
            {"step_id": step_id, "output": output},
        )
        return True

    def _refresh_workflow_status(
        self, db: sqlite3.Connection, workflow_id: str
    ) -> None:
        row = db.execute(
            "SELECT status FROM workflows WHERE id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            return
        current = WorkflowStatus(str(row["status"]))
        if current in TERMINAL_WORKFLOW_STATUSES:
            return
        steps = self._load_steps(db, workflow_id)
        statuses = [step.status for step in steps]
        error = ""
        if steps and all(status in SUCCESS_STEP_STATUSES for status in statuses):
            status = WorkflowStatus.SUCCEEDED
        elif any(step.status == StepStatus.FAILED for step in steps):
            status = WorkflowStatus.BLOCKED
            error = next(
                (step.error for step in steps if step.status == StepStatus.FAILED), ""
            )
        elif any(step.status == StepStatus.RUNNING for step in steps):
            status = WorkflowStatus.RUNNING
        elif any(step.status == StepStatus.WAITING for step in steps):
            step_statuses = {step.id: step.status for step in steps}
            has_runnable_pending = any(
                step.status == StepStatus.PENDING
                and self._deps_satisfied(step, step_statuses)
                for step in steps
            )
            status = (
                WorkflowStatus.RUNNING
                if has_runnable_pending
                else WorkflowStatus.WAITING
            )
        elif any(step.status == StepStatus.PENDING for step in steps):
            status = WorkflowStatus.RUNNING
        elif statuses and all(status == StepStatus.CANCELLED for status in statuses):
            status = WorkflowStatus.CANCELLED
        else:
            status = current
        db.execute(
            "UPDATE workflows SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (status.value, error, _now(), workflow_id),
        )

    def _record_event(
        self,
        db: sqlite3.Connection,
        workflow_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        row = db.execute(
            "SELECT revision FROM workflows WHERE id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"未找到 workflow: {workflow_id}")
        revision = int(row["revision"]) + 1
        now = _now()
        db.execute(
            "UPDATE workflows SET revision = ?, updated_at = ? WHERE id = ?",
            (revision, now, workflow_id),
        )
        self._insert_event(db, workflow_id, revision, event_type, payload, now)
        return revision

    @staticmethod
    def _insert_event(
        db: sqlite3.Connection,
        workflow_id: str,
        revision: int,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        db.execute(
            """
            INSERT INTO workflow_events(
                workflow_id, revision, event_type, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (workflow_id, revision, event_type, _json_dump(payload), created_at),
        )
