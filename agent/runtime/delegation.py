"""Trusted delegation policy and durable, shared execution-token ledger.

This is an admission budget, not a billing counter. Failed/unknown calls retain
their reservation (including after a crash); successful calls settle to provider
usage. Cached input still counts as context processed, not full-price dollars.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CONTEXT_KEY = "_runtime_delegation"


class DelegationBudgetExceeded(RuntimeError):
    pass


class DelegationLedger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._memory: sqlite3.Connection | None = None

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.path), timeout=5)
        else:
            if self._memory is None:
                self._memory = sqlite3.connect(":memory:")
            db = self._memory
        try:
            with db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS budgets (id TEXT PRIMARY KEY, limit_tokens INTEGER NOT NULL, reserve_tokens INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0, max_children INTEGER NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS children (budget_id TEXT NOT NULL, child_id TEXT NOT NULL, PRIMARY KEY (budget_id, child_id))"
                )
                db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            if self.path is not None:
                db.close()

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    def create(self, key: str, *, limit: int, reserve: int, max_children: int) -> None:
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO budgets (id, limit_tokens, reserve_tokens, max_children) VALUES (?, ?, ?, ?)",
                (key, limit, reserve, max_children),
            )

    def claim_child(self, key: str, child: str) -> None:
        with self._db() as db:
            row = db.execute(
                "SELECT limit_tokens, reserve_tokens, used, max_children FROM budgets WHERE id=?",
                (key,),
            ).fetchone()
            if row is None:
                raise DelegationBudgetExceeded(
                    "找不到父任务预算，停止委派；请创建新任务。"
                )
            limit, reserve, used, maximum = row
            if used >= limit - reserve:
                raise DelegationBudgetExceeded(
                    "父子任务共享预算不足，停止新增委派，保留主 Agent 收尾空间。"
                )
            exists = db.execute(
                "SELECT 1 FROM children WHERE budget_id=? AND child_id=?", (key, child)
            ).fetchone()
            count = db.execute(
                "SELECT COUNT(*) FROM children WHERE budget_id=?", (key,)
            ).fetchone()[0]
            if not exists and count >= maximum:
                raise DelegationBudgetExceeded(
                    f"同一父任务最多创建 {maximum} 个子执行步骤（跨 Workflow 累计）。"
                )
            db.execute("INSERT OR IGNORE INTO children VALUES (?, ?)", (key, child))

    def reserve(self, key: str, tokens: int, *, child: bool) -> None:
        with self._db() as db:
            row = db.execute(
                "SELECT limit_tokens, reserve_tokens, used FROM budgets WHERE id=?",
                (key,),
            ).fetchone()
            if row is None:
                raise DelegationBudgetExceeded("共享预算记录不存在，拒绝继续消耗。")
            if child and row[2] + tokens > row[0] - row[1]:
                raise DelegationBudgetExceeded(
                    "父子任务共享 token 预算不足，本步骤未继续调用模型；主 Agent 仍可汇总已有结果。"
                )
            db.execute("UPDATE budgets SET used=used+? WHERE id=?", (tokens, key))

    def settle(self, key: str, reserved: int, actual: int) -> None:
        with self._db() as db:
            db.execute(
                "UPDATE budgets SET used=MAX(0, used+?) WHERE id=?",
                (actual - reserved, key),
            )

    def used(self, key: str) -> int:
        with self._db() as db:
            row = db.execute("SELECT used FROM budgets WHERE id=?", (key,)).fetchone()
            return int(row[0]) if row else 0


@dataclass(frozen=True)
class DelegationScope:
    ledger: DelegationLedger
    key: str
    allowed: bool = False
    child: bool = False

    def metadata(self) -> dict[str, object]:
        return {"budget_id": self.key, "allowed": self.allowed and not self.child}


current_scope: ContextVar[DelegationScope | None] = ContextVar(
    "delegation_scope", default=None
)


def new_scope(
    ledger: DelegationLedger, guard: object, *, allowed: bool = False, key: str = ""
) -> DelegationScope:
    key = key or uuid.uuid4().hex
    ledger.create(
        key,
        limit=int(getattr(guard, "delegation_total_tokens", 500_000)),
        reserve=int(getattr(guard, "delegation_reserve_tokens", 32_000)),
        max_children=int(getattr(guard, "delegation_max_children", 8)),
    )
    return DelegationScope(ledger, key, allowed)
