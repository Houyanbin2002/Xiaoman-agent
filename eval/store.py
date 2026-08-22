from __future__ import annotations

"""Local append-only store for evaluation runs and score history."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import EvalSummary


class EvalResultStore:
    """Persist scores locally so evaluation does not depend on Langfuse."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                version TEXT NOT NULL,
                total INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                pass_rate REAL NOT NULL,
                mean_reward REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS eval_case_results (
                run_id INTEGER NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
                case_id TEXT NOT NULL,
                passed INTEGER NOT NULL,
                reward REAL NOT NULL,
                trace_id TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL,
                PRIMARY KEY(run_id, case_id)
            );
            CREATE TABLE IF NOT EXISTS eval_scores (
                run_id INTEGER NOT NULL,
                case_id TEXT NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                passed INTEGER NOT NULL,
                hard INTEGER NOT NULL,
                source TEXT NOT NULL,
                reason TEXT NOT NULL,
                FOREIGN KEY(run_id, case_id) REFERENCES eval_case_results(run_id, case_id) ON DELETE CASCADE
            );
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def save(self, summary: EvalSummary) -> int:
        cursor = self._db.execute(
            "INSERT INTO eval_runs(dataset,version,total,passed,pass_rate,mean_reward) VALUES (?,?,?,?,?,?)",
            (summary.dataset, summary.version, summary.total, summary.passed, summary.pass_rate, summary.mean_reward),
        )
        run_id = int(cursor.lastrowid)
        for result in summary.results:
            self._db.execute(
                "INSERT INTO eval_case_results(run_id,case_id,passed,reward,trace_id,payload_json) VALUES (?,?,?,?,?,?)",
                (run_id, result.case_id, int(result.passed), result.reward, result.run.trace_id, json.dumps(result.to_dict(), ensure_ascii=False, default=str)),
            )
            for score in result.scores:
                self._db.execute(
                    "INSERT INTO eval_scores(run_id,case_id,name,value,passed,hard,source,reason) VALUES (?,?,?,?,?,?,?,?)",
                    (run_id, result.case_id, score.name, score.value, int(score.passed), int(score.hard), score.source, score.reason),
                )
        self._db.commit()
        return run_id

    def low_reward_cases(self, *, limit: int = 50, threshold: float = 0.6) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT run_id,case_id,reward,trace_id,payload_json FROM eval_case_results WHERE reward < ? ORDER BY run_id DESC LIMIT ?",
            (float(threshold), max(1, min(int(limit), 500))),
        ).fetchall()
        return [
            {
                "run_id": int(row["run_id"]),
                "case_id": str(row["case_id"]),
                "reward": float(row["reward"]),
                "trace_id": str(row["trace_id"]),
                "result": json.loads(str(row["payload_json"])),
                "review_status": "needs_human_review",
            }
            for row in rows
        ]
