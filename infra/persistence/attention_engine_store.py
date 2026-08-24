from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from core.attention._shared import utc_iso
from core.attention.actions import ActionPlan, ActionPlanStatus
from core.attention.feedback.models import AttentionFeedback
from core.attention.events import (
    CanonicalEntity,
    CanonicalEvent,
    EventStatus,
    WakePlan,
    WakeStatus,
)
from core.attention.learning.models import AttentionObservation
from core.attention.opportunities import OpportunityWindow
from core.attention.patterns import BehaviorPattern
from core.attention.policies import PolicyRule
from core.attention.signals import AttentionSignal

T = TypeVar("T")


class AttentionEngineStore:
    """SQLite repository for signals, patterns, policies, windows and plans."""

    def __init__(self, database: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._db = sqlite3.connect(
            database,
            check_same_thread=False,
            timeout=10.0,
        )
        self._db.row_factory = sqlite3.Row
        with self._transaction():
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS attention_signals_v2 (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    expires_at TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attention_signals_v2_active
                    ON attention_signals_v2(expires_at, occurred_at);

                CREATE TABLE IF NOT EXISTS attention_patterns_v2 (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    last_observed_at TEXT,
                    expires_at TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attention_patterns_v2_status
                    ON attention_patterns_v2(status, expires_at);

                CREATE TABLE IF NOT EXISTS attention_observations_v2 (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    rule_key TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attention_observations_v2_rule
                    ON attention_observations_v2(rule_key, observed_at DESC);

                CREATE TABLE IF NOT EXISTS attention_windows_v2 (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    available_from TEXT NOT NULL,
                    available_until TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attention_windows_v2_active
                    ON attention_windows_v2(status, available_from, available_until);

                CREATE TABLE IF NOT EXISTS attention_policies_v2 (
                    id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    effective_from TEXT,
                    expires_at TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attention_policies_v2_active
                    ON attention_policies_v2(enabled, priority, expires_at);

                CREATE TABLE IF NOT EXISTS attention_action_plans_v2 (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attention_plans_v2_status
                    ON attention_action_plans_v2(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS attention_feedback_v2 (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attention_feedback_v2_plan
                    ON attention_feedback_v2(plan_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS attention_entities_v2 (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_id, external_id)
                );
                CREATE INDEX IF NOT EXISTS idx_attention_entities_v2_state
                    ON attention_entities_v2(state, updated_at DESC);

                CREATE TABLE IF NOT EXISTS attention_events_v2 (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    delivery_semantics TEXT NOT NULL,
                    due_at TEXT,
                    dedupe_key TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(dedupe_key, source_version)
                );
                CREATE INDEX IF NOT EXISTS idx_attention_events_v2_active
                    ON attention_events_v2(status, due_at);

                CREATE TABLE IF NOT EXISTS attention_wake_plans_v2 (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    wake_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attention_wakes_v2_due
                    ON attention_wake_plans_v2(status, wake_at);
                """)

    def upsert_signal(self, signal: AttentionSignal) -> AttentionSignal:
        self._upsert(
            "attention_signals_v2",
            signal.id,
            {
                "kind": signal.kind,
                "domain": signal.domain,
                "occurred_at": signal.occurred_at,
                "expires_at": signal.expires_at,
                "payload_json": self._json(signal.to_dict()),
                "updated_at": utc_iso(),
            },
        )
        return signal

    def upsert_signals(
        self,
        signals: list[AttentionSignal],
    ) -> list[AttentionSignal]:
        if not signals:
            return []
        updated_at = utc_iso()
        rows = [
            (
                signal.id,
                signal.kind,
                signal.domain,
                signal.occurred_at,
                signal.expires_at,
                self._json(signal.to_dict()),
                updated_at,
            )
            for signal in signals
        ]
        with self._transaction():
            self._db.executemany(
                """
                INSERT INTO attention_signals_v2(
                    id, kind, domain, occurred_at, expires_at,
                    payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    domain = excluded.domain,
                    occurred_at = excluded.occurred_at,
                    expires_at = excluded.expires_at,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
        return signals

    def get_signal(self, signal_id: str) -> AttentionSignal | None:
        return self._get(
            "attention_signals_v2",
            signal_id,
            AttentionSignal.from_dict,
        )

    def list_active_signals(self, *, now: datetime) -> list[AttentionSignal]:
        current = utc_iso(now)
        return self._list(
            """
            SELECT payload_json FROM attention_signals_v2
            WHERE occurred_at <= ? AND (expires_at IS NULL OR expires_at >= ?)
            ORDER BY occurred_at DESC
            """,
            (current, current),
            AttentionSignal.from_dict,
        )

    def upsert_pattern(self, pattern: BehaviorPattern) -> BehaviorPattern:
        self._upsert(
            "attention_patterns_v2",
            pattern.id,
            {
                "status": pattern.status.value,
                "last_observed_at": pattern.last_observed_at,
                "expires_at": pattern.expires_at,
                "payload_json": self._json(pattern.to_dict()),
                "updated_at": utc_iso(),
            },
        )
        return pattern

    def get_pattern(self, pattern_id: str) -> BehaviorPattern | None:
        return self._get(
            "attention_patterns_v2",
            pattern_id,
            BehaviorPattern.from_dict,
        )

    def list_patterns(self) -> list[BehaviorPattern]:
        return self._list(
            "SELECT payload_json FROM attention_patterns_v2 ORDER BY updated_at DESC",
            (),
            BehaviorPattern.from_dict,
        )

    def upsert_window(self, window: OpportunityWindow) -> OpportunityWindow:
        self._upsert(
            "attention_windows_v2",
            window.id,
            {
                "status": window.status.value,
                "available_from": window.available_from,
                "available_until": window.available_until,
                "payload_json": self._json(window.to_dict()),
                "updated_at": utc_iso(),
            },
        )
        return window

    def upsert_windows(
        self,
        windows: list[OpportunityWindow],
    ) -> list[OpportunityWindow]:
        if not windows:
            return []
        updated_at = utc_iso()
        rows = [
            (
                window.id,
                window.status.value,
                window.available_from,
                window.available_until,
                self._json(window.to_dict()),
                updated_at,
            )
            for window in windows
        ]
        with self._transaction():
            self._db.executemany(
                """
                INSERT INTO attention_windows_v2(
                    id, status, available_from, available_until,
                    payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    available_from = excluded.available_from,
                    available_until = excluded.available_until,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
        return windows

    def get_window(self, window_id: str) -> OpportunityWindow | None:
        return self._get(
            "attention_windows_v2",
            window_id,
            OpportunityWindow.from_dict,
        )

    def list_active_windows(self, *, now: datetime) -> list[OpportunityWindow]:
        current = utc_iso(now)
        return self._list(
            """
            SELECT payload_json FROM attention_windows_v2
            WHERE status = 'active'
              AND available_from <= ? AND available_until >= ?
            ORDER BY available_until ASC
            """,
            (current, current),
            OpportunityWindow.from_dict,
        )

    def upsert_policy(self, policy: PolicyRule) -> PolicyRule:
        self._upsert(
            "attention_policies_v2",
            policy.id,
            {
                "enabled": int(policy.enabled),
                "priority": policy.priority,
                "effective_from": policy.effective_from,
                "expires_at": policy.expires_at,
                "payload_json": self._json(policy.to_dict()),
                "updated_at": utc_iso(),
            },
        )
        return policy

    def get_policy(self, policy_id: str) -> PolicyRule | None:
        return self._get(
            "attention_policies_v2",
            policy_id,
            PolicyRule.from_dict,
        )

    def list_policies(self) -> list[PolicyRule]:
        return self._list(
            "SELECT payload_json FROM attention_policies_v2 ORDER BY priority DESC, id",
            (),
            PolicyRule.from_dict,
        )

    def add_observation(self, observation: AttentionObservation) -> bool:
        with self._transaction():
            cursor = self._db.execute(
                """
                INSERT OR IGNORE INTO attention_observations_v2(
                    id, kind, rule_key, source_ref, observed_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.id,
                    observation.kind.value,
                    observation.rule_key,
                    observation.source_ref,
                    observation.observed_at,
                    self._json(observation.to_dict()),
                ),
            )
        return cursor.rowcount > 0

    def list_observations(
        self,
        *,
        limit: int = 100,
    ) -> list[AttentionObservation]:
        return self._list(
            """
            SELECT payload_json FROM attention_observations_v2
            ORDER BY observed_at DESC LIMIT ?
            """,
            (max(1, min(int(limit), 1000)),),
            AttentionObservation.from_dict,
        )

    def create_plan(self, plan: ActionPlan) -> ActionPlan:
        with self._transaction():
            try:
                self._db.execute(
                    """
                    INSERT INTO attention_action_plans_v2(
                        id, idempotency_key, status, capability_id, created_at,
                        expires_at, payload_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan.id,
                        plan.idempotency_key,
                        plan.status.value,
                        plan.capability_id,
                        plan.created_at,
                        plan.expires_at,
                        self._json(plan.to_dict()),
                        plan.updated_at,
                    ),
                )
            except sqlite3.IntegrityError:
                row = self._db.execute(
                    """
                    SELECT payload_json FROM attention_action_plans_v2
                    WHERE idempotency_key = ?
                    """,
                    (plan.idempotency_key,),
                ).fetchone()
                if row is None:
                    raise
                return ActionPlan.from_dict(json.loads(str(row["payload_json"])))
        return plan

    def get_plan(self, plan_id: str) -> ActionPlan | None:
        return self._get(
            "attention_action_plans_v2",
            plan_id,
            ActionPlan.from_dict,
        )

    def find_plan_by_key(self, idempotency_key: str) -> ActionPlan | None:
        with self._transaction():
            row = self._db.execute(
                """
                SELECT payload_json FROM attention_action_plans_v2
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
        return (
            ActionPlan.from_dict(json.loads(str(row["payload_json"])))
            if row is not None
            else None
        )

    def transition_plan(
        self,
        plan_id: str,
        status: ActionPlanStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> ActionPlan:
        current = self.get_plan(plan_id)
        if current is None:
            raise ValueError(f"attention action plan not found: {plan_id}")
        updated = current.transition(status, result=result, error=error)
        with self._transaction():
            self._db.execute(
                """
                UPDATE attention_action_plans_v2
                SET status = ?, payload_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    updated.status.value,
                    self._json(updated.to_dict()),
                    updated.updated_at,
                    updated.id,
                ),
            )
        return updated

    def list_plans(self, *, limit: int = 100) -> list[ActionPlan]:
        return self._list(
            """
            SELECT payload_json FROM attention_action_plans_v2
            ORDER BY created_at DESC LIMIT ?
            """,
            (max(1, min(int(limit), 1000)),),
            ActionPlan.from_dict,
        )

    def add_feedback(self, feedback: AttentionFeedback) -> AttentionFeedback:
        with self._transaction():
            self._db.execute(
                """
                INSERT INTO attention_feedback_v2(
                    id, plan_id, kind, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feedback.id,
                    feedback.plan_id,
                    feedback.kind.value,
                    feedback.created_at,
                    self._json(feedback.to_dict()),
                ),
            )
        return feedback

    def list_feedback(self, *, plan_id: str | None = None) -> list[AttentionFeedback]:
        if plan_id:
            return self._list(
                """
                SELECT payload_json FROM attention_feedback_v2
                WHERE plan_id = ? ORDER BY created_at DESC
                """,
                (plan_id,),
                AttentionFeedback.from_dict,
            )
        return self._list(
            "SELECT payload_json FROM attention_feedback_v2 ORDER BY created_at DESC",
            (),
            AttentionFeedback.from_dict,
        )

    def upsert_entity(self, entity: CanonicalEntity) -> CanonicalEntity:
        self._upsert(
            "attention_entities_v2",
            entity.id,
            {
                "source_id": entity.source_id,
                "external_id": entity.external_id,
                "kind": entity.kind,
                "state": entity.state.value,
                "source_version": entity.source_version,
                "payload_json": self._json(entity.to_dict()),
                "updated_at": entity.updated_at,
            },
        )
        return entity

    def get_entity(self, entity_id: str) -> CanonicalEntity | None:
        return self._get(
            "attention_entities_v2",
            entity_id,
            CanonicalEntity.from_dict,
        )

    def list_entities(self, *, limit: int = 1000) -> list[CanonicalEntity]:
        return self._list(
            """
            SELECT payload_json FROM attention_entities_v2
            ORDER BY updated_at DESC LIMIT ?
            """,
            (max(1, min(int(limit), 10000)),),
            CanonicalEntity.from_dict,
        )

    def upsert_event(self, event: CanonicalEvent) -> CanonicalEvent:
        with self._transaction():
            existing = self._db.execute(
                """
                SELECT payload_json FROM attention_events_v2
                WHERE dedupe_key = ? AND source_version = ?
                """,
                (event.dedupe_key, event.source_version),
            ).fetchone()
            if existing is not None:
                return CanonicalEvent.from_dict(
                    json.loads(str(existing["payload_json"]))
                )
            self._db.execute(
                """
                INSERT INTO attention_events_v2(
                    id, entity_id, source_id, kind, status,
                    delivery_semantics, due_at, dedupe_key, source_version,
                    payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    entity_id = excluded.entity_id,
                    source_id = excluded.source_id,
                    kind = excluded.kind,
                    status = excluded.status,
                    delivery_semantics = excluded.delivery_semantics,
                    due_at = excluded.due_at,
                    dedupe_key = excluded.dedupe_key,
                    source_version = excluded.source_version,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    event.id,
                    event.entity_id,
                    event.source_id,
                    event.kind,
                    event.status.value,
                    event.delivery_semantics.value,
                    event.due_at or None,
                    event.dedupe_key,
                    event.source_version,
                    self._json(event.to_dict()),
                    utc_iso(),
                ),
            )
        return event

    def get_event(self, event_id: str) -> CanonicalEvent | None:
        return self._get(
            "attention_events_v2",
            event_id,
            CanonicalEvent.from_dict,
        )

    def list_active_events(self) -> list[CanonicalEvent]:
        return self._list(
            """
            SELECT payload_json FROM attention_events_v2
            WHERE status = 'active' ORDER BY due_at, updated_at DESC
            """,
            (),
            CanonicalEvent.from_dict,
        )

    def upsert_wake(self, wake: WakePlan) -> WakePlan:
        self._upsert(
            "attention_wake_plans_v2",
            wake.id,
            {
                "event_id": wake.event_id,
                "wake_at": wake.wake_at,
                "status": wake.status.value,
                "attempt": wake.attempt,
                "max_attempts": wake.max_attempts,
                "payload_json": self._json(wake.to_dict()),
                "updated_at": wake.updated_at,
            },
        )
        return wake

    def get_wake(self, wake_id: str) -> WakePlan | None:
        return self._get(
            "attention_wake_plans_v2",
            wake_id,
            WakePlan.from_dict,
        )

    def list_pending_wakes(self) -> list[WakePlan]:
        return self._list(
            """
            SELECT payload_json FROM attention_wake_plans_v2
            WHERE status = 'pending' ORDER BY wake_at, id
            """,
            (),
            WakePlan.from_dict,
        )

    def next_wake_at(self) -> str | None:
        with self._transaction():
            row = self._db.execute(
                """
                SELECT wake_at FROM attention_wake_plans_v2
                WHERE status = 'pending' ORDER BY wake_at LIMIT 1
                """
            ).fetchone()
        return str(row["wake_at"]) if row is not None else None

    def claim_due_wakes(self, *, now: datetime, limit: int = 20) -> list[WakePlan]:
        current = utc_iso(now)
        claimed: list[WakePlan] = []
        with self._transaction():
            rows = self._db.execute(
                """
                SELECT id, payload_json FROM attention_wake_plans_v2
                WHERE status = 'pending' AND wake_at <= ? AND attempt < max_attempts
                ORDER BY wake_at, id LIMIT ?
                """,
                (current, max(1, min(int(limit), 100))),
            ).fetchall()
            for row in rows:
                wake = WakePlan.from_dict(json.loads(str(row["payload_json"])))
                updated = replace(
                    wake,
                    attempt=wake.attempt + 1,
                    status=WakeStatus.PROCESSING,
                    updated_at=current,
                )
                self._db.execute(
                    """
                    UPDATE attention_wake_plans_v2
                    SET status = ?, attempt = ?, payload_json = ?, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (
                        updated.status.value,
                        updated.attempt,
                        self._json(updated.to_dict()),
                        current,
                        updated.id,
                    ),
                )
                claimed.append(updated)
        return claimed

    def recover_processing_wakes(self) -> int:
        recovered = 0
        now = utc_iso()
        with self._transaction():
            rows = self._db.execute(
                """
                SELECT id, payload_json FROM attention_wake_plans_v2
                WHERE status = 'processing'
                """
            ).fetchall()
            for row in rows:
                wake = WakePlan.from_dict(json.loads(str(row["payload_json"])))
                updated = replace(
                    wake,
                    status=WakeStatus.PENDING,
                    updated_at=now,
                )
                self._db.execute(
                    """
                    UPDATE attention_wake_plans_v2
                    SET status = 'pending', payload_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (self._json(updated.to_dict()), now, wake.id),
                )
                recovered += 1
        return recovered

    def complete_wake(self, wake_id: str, *, decision: str) -> WakePlan:
        current = self.get_wake(wake_id)
        if current is None:
            raise ValueError(f"attention wake not found: {wake_id}")
        now = utc_iso()
        updated = replace(
            current,
            status=WakeStatus.COMPLETED,
            last_decision=decision,
            updated_at=now,
        )
        self.upsert_wake(updated)
        return updated

    def defer_wake(
        self,
        wake_id: str,
        *,
        wake_at: datetime,
        decision: str,
    ) -> WakePlan:
        current = self.get_wake(wake_id)
        if current is None:
            raise ValueError(f"attention wake not found: {wake_id}")
        status = (
            WakeStatus.DEAD
            if current.attempt >= current.max_attempts
            else WakeStatus.PENDING
        )
        now = utc_iso()
        updated = replace(
            current,
            wake_at=utc_iso(wake_at),
            status=status,
            last_decision=decision,
            updated_at=now,
        )
        self.upsert_wake(updated)
        return updated

    def close_event(
        self,
        event_id: str,
        status: EventStatus,
    ) -> CanonicalEvent:
        if status is EventStatus.ACTIVE:
            raise ValueError("close_event requires a terminal status")
        current = self.get_event(event_id)
        if current is None:
            raise ValueError(f"attention event not found: {event_id}")
        updated = replace(current, status=status)
        now = utc_iso()
        with self._transaction():
            self._db.execute(
                """
                UPDATE attention_events_v2
                SET status = ?, payload_json = ?, updated_at = ? WHERE id = ?
                """,
                (status.value, self._json(updated.to_dict()), now, event_id),
            )
            rows = self._db.execute(
                """
                SELECT id, payload_json FROM attention_wake_plans_v2
                WHERE event_id = ? AND status IN ('pending', 'processing')
                """,
                (event_id,),
            ).fetchall()
            for row in rows:
                wake = WakePlan.from_dict(json.loads(str(row["payload_json"])))
                cancelled = replace(
                    wake,
                    status=WakeStatus.CANCELLED,
                    last_decision=status.value,
                    updated_at=now,
                )
                self._db.execute(
                    """
                    UPDATE attention_wake_plans_v2
                    SET status = 'cancelled', payload_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (self._json(cancelled.to_dict()), now, wake.id),
                )
        return updated

    def close_events_for_entity(
        self,
        entity_id: str,
        status: EventStatus,
    ) -> list[CanonicalEvent]:
        active = self._list(
            """
            SELECT payload_json FROM attention_events_v2
            WHERE entity_id = ? AND status = 'active'
            """,
            (entity_id,),
            CanonicalEvent.from_dict,
        )
        return [self.close_event(event.id, status) for event in active]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()

    def _upsert(self, table: str, row_id: str, values: dict[str, Any]) -> None:
        columns = ["id", *values]
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{key} = excluded.{key}" for key in values)
        with self._transaction():
            self._db.execute(
                f"""
                INSERT INTO {table}({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET {updates}
                """,
                (row_id, *values.values()),
            )

    def _get(
        self,
        table: str,
        row_id: str,
        factory: Callable[[dict[str, Any]], T],
    ) -> T | None:
        with self._transaction():
            row = self._db.execute(
                f"SELECT payload_json FROM {table} WHERE id = ?",
                (row_id,),
            ).fetchone()
        return factory(json.loads(str(row["payload_json"]))) if row else None

    def _list(
        self,
        query: str,
        parameters: tuple[Any, ...],
        factory: Callable[[dict[str, Any]], T],
    ) -> list[T]:
        with self._transaction():
            rows = self._db.execute(query, parameters).fetchall()
        return [factory(json.loads(str(row["payload_json"]))) for row in rows]

    @staticmethod
    def _json(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _transaction(self):
        store = self

        class _Transaction:
            def __enter__(self):
                store._lock.acquire()
                if store._closed:
                    store._lock.release()
                    raise RuntimeError("attention engine store is closed")
                return store._db

            def __exit__(self, exc_type, exc, _traceback):
                try:
                    if exc_type is None:
                        store._db.commit()
                    else:
                        store._db.rollback()
                finally:
                    store._lock.release()
                return False

        return _Transaction()


__all__ = ["AttentionEngineStore"]
