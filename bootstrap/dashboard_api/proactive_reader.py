from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable


class ProactiveDashboardReader:
    """Read-only dashboard queries over the proactive state database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def get_overview(self) -> dict[str, Any]:
        counts = {
            "deliveries": self._count("deliveries"),
            "session_state": self._count("session_state"),
            "context_only_timestamps": self._count("context_only_timestamps"),
            "tick_logs": self._count("tick_log"),
            "tick_steps": self._count("tick_step_log"),
        }
        with self._lock:
            recent_tick = self._db.execute("""
                SELECT tick_id, session_key, started_at, finished_at, gate_exit,
                       terminal_action, skip_reason, steps_taken, drift_entered
                FROM tick_log
                ORDER BY started_at DESC
                LIMIT 1
                """).fetchone()
            last_send_at = self._db.execute("""
                SELECT sent_at
                FROM deliveries
                ORDER BY sent_at DESC
                LIMIT 1
                """).fetchone()
            result_counts_rows = self._db.execute("""
                SELECT COALESCE(terminal_action, gate_exit, 'unknown') AS bucket, COUNT(*) AS total
                FROM tick_log
                GROUP BY COALESCE(terminal_action, gate_exit, 'unknown')
                """).fetchall()
            flow_counts_rows = self._db.execute("""
                SELECT CASE WHEN drift_entered = 1 THEN 'drift' ELSE 'proactive' END AS bucket,
                       COUNT(*) AS total
                FROM tick_log
                GROUP BY CASE WHEN drift_entered = 1 THEN 'drift' ELSE 'proactive' END
                """).fetchall()
        result_counts = {
            str(row["bucket"]): int(row["total"]) for row in result_counts_rows
        }
        flow_counts = {
            str(row["bucket"]): int(row["total"]) for row in flow_counts_rows
        }
        return {
            "counts": counts,
            "result_counts": result_counts,
            "flow_counts": flow_counts,
            "last_tick_at": (
                recent_tick["started_at"] if recent_tick is not None else None
            ),
            "last_send_at": (
                last_send_at["sent_at"] if last_send_at is not None else None
            ),
            "last_skip_reason": (
                recent_tick["skip_reason"]
                if recent_tick is not None and recent_tick["terminal_action"] != "reply"
                else None
            ),
            "recent_tick": (
                self._row_to_tick_log(recent_tick) if recent_tick is not None else None
            ),
        }

    def list_deliveries(
        self,
        *,
        session_key: str = "",
        sent_from: str = "",
        sent_to: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = self._build_filters(
            ("session_key = ?", session_key),
            ("sent_at >= ?", sent_from),
            ("sent_at <= ?", sent_to),
        )
        return self._list_rows(
            table="deliveries",
            where=where,
            params=params,
            order_by="sent_at DESC, session_key ASC, delivery_key ASC",
            page=page,
            page_size=page_size,
            columns="session_key, delivery_key, sent_at",
        )

    def list_tick_logs(
        self,
        *,
        session_key: str = "",
        terminal_action: str = "",
        gate_exit: str = "",
        flow: str = "",
        started_from: str = "",
        started_to: str = "",
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "started_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        drift_only = ""
        if flow == "drift":
            drift_only = "1"
        elif flow == "proactive":
            drift_only = "0"
        safe_sort_by = (
            sort_by
            if sort_by
            in {
                "session_key",
                "started_at",
                "finished_at",
                "terminal_action",
                "gate_exit",
                "steps_taken",
                "alert_count",
                "content_count",
                "context_count",
                "drift_entered",
            }
            else "started_at"
        )
        safe_sort_order = "ASC" if str(sort_order).lower() == "asc" else "DESC"
        where, params = self._build_filters(
            ("session_key = ?", session_key),
            ("terminal_action = ?", terminal_action),
            ("gate_exit = ?", gate_exit),
            ("drift_entered = ?", drift_only),
            ("started_at >= ?", started_from),
            ("started_at <= ?", started_to),
        )
        return self._list_rows(
            table="tick_log",
            where=where,
            params=params,
            order_by=f"{safe_sort_by} {safe_sort_order}, id DESC",
            page=page,
            page_size=page_size,
            columns=(
                "tick_id, session_key, started_at, finished_at, gate_exit, "
                "terminal_action, skip_reason, steps_taken, alert_count, "
                "content_count, context_count, interesting_ids, discarded_ids, "
                "cited_ids, drift_entered, final_message, proactive_effects_json"
            ),
            row_mapper=self._row_to_tick_log,
        )

    def get_tick_log(self, tick_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT tick_id, session_key, started_at, finished_at, gate_exit,
                       terminal_action, skip_reason, steps_taken, alert_count,
                       content_count, context_count, interesting_ids, discarded_ids,
                       cited_ids, drift_entered, final_message, proactive_effects_json
                FROM tick_log
                WHERE tick_id = ?
                """,
                (tick_id,),
            ).fetchone()
        return self._row_to_tick_log(row) if row is not None else None

    def list_tick_steps(self, tick_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT step_index, phase, tool_name, tool_call_id, tool_args_json,
                       tool_result_text, terminal_action_after, skip_reason_after,
                       interesting_ids_after, discarded_ids_after, cited_ids_after,
                       final_message_after
                FROM tick_step_log
                WHERE tick_id = ?
                ORDER BY step_index ASC, id ASC
                """,
                (tick_id,),
            ).fetchall()
        return [self._row_to_tick_step(row) for row in rows]

    def _list_rows(
        self,
        *,
        table: str,
        where: str,
        params: tuple[Any, ...],
        order_by: str,
        page: int,
        page_size: int,
        columns: str,
        row_mapper: Callable[[sqlite3.Row], dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size
        with self._lock:
            total_row = self._db.execute(
                f"SELECT COUNT(*) FROM {table}{where}",
                params,
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT {columns}
                FROM {table}{where}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()
        total = int(total_row[0]) if total_row is not None else 0
        mapper = row_mapper or self._row_to_dict
        return [mapper(row) for row in rows], total

    @staticmethod
    def _build_filters(*filters: tuple[str, Any]) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []
        for clause, value in filters:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            clauses.append(clause)
            params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, tuple(params)

    def _count(self, table: str) -> int:
        with self._lock:
            row = self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {key: row[key] for key in row.keys()}

    @staticmethod
    def _decode_json_list(raw: Any) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except Exception:
            return []
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def _row_to_tick_log(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = self._row_to_dict(row)
        payload["interesting_ids"] = self._decode_json_list(
            payload.get("interesting_ids")
        )
        payload["discarded_ids"] = self._decode_json_list(payload.get("discarded_ids"))
        payload["cited_ids"] = self._decode_json_list(payload.get("cited_ids"))
        payload["proactive_effects"] = self._decode_json_object_list(
            payload.pop("proactive_effects_json", "")
        )
        payload["drift_entered"] = bool(payload.get("drift_entered"))
        return payload

    @staticmethod
    def _decode_json_object_list(raw: Any) -> list[dict[str, Any]]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except Exception:
            return []
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _row_to_tick_step(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = self._row_to_dict(row)
        payload["tool_args"] = self._decode_json_object(
            payload.pop("tool_args_json", "")
        )
        payload["interesting_ids_after"] = self._decode_json_list(
            payload.get("interesting_ids_after")
        )
        payload["discarded_ids_after"] = self._decode_json_list(
            payload.get("discarded_ids_after")
        )
        payload["cited_ids_after"] = self._decode_json_list(
            payload.get("cited_ids_after")
        )
        return payload

    @staticmethod
    def _decode_json_object(raw: Any) -> dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}
