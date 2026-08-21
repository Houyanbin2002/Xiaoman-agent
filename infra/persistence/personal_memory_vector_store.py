from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading

import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS personal_memory_embeddings (
    record_id     TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    model         TEXT NOT NULL,
    dimensions    INTEGER NOT NULL,
    embedding     TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_personal_memory_embeddings_model
    ON personal_memory_embeddings (model, dimensions);
"""


class PersonalMemoryVectorStore:
    """Small, portable vector sidecar stored inside personal.db.

    Personal memory is expected to stay small, so an exact cosine scan is simpler
    and more reliable than introducing another database service or ANN index.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def stale_ids(
        self,
        hashes: Mapping[str, str],
        *,
        model: str,
    ) -> list[str]:
        if not hashes:
            return []
        placeholders = ",".join("?" for _ in hashes)
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT record_id, content_hash, model
                FROM personal_memory_embeddings
                WHERE record_id IN ({placeholders})
                """,
                tuple(hashes),
            ).fetchall()
        current = {str(row[0]): (str(row[1]), str(row[2])) for row in rows}
        return [
            record_id
            for record_id, content_hash in hashes.items()
            if current.get(record_id) != (content_hash, model)
        ]

    def upsert_many(
        self,
        rows: Sequence[tuple[str, str, str, Sequence[float]]],
    ) -> None:
        if not rows:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._db.executemany(
                """
                INSERT INTO personal_memory_embeddings (
                    record_id, content_hash, model, dimensions, embedding, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    model=excluded.model,
                    dimensions=excluded.dimensions,
                    embedding=excluded.embedding,
                    updated_at=excluded.updated_at
                """,
                [
                    (
                        record_id,
                        content_hash,
                        model,
                        len(embedding),
                        json.dumps(list(embedding)),
                        now,
                    )
                    for record_id, content_hash, model, embedding in rows
                ],
            )
            self._db.commit()

    def delete_except(self, record_ids: set[str]) -> None:
        with self._lock:
            if not record_ids:
                self._db.execute("DELETE FROM personal_memory_embeddings")
            else:
                placeholders = ",".join("?" for _ in record_ids)
                self._db.execute(
                    f"DELETE FROM personal_memory_embeddings WHERE record_id NOT IN ({placeholders})",
                    tuple(record_ids),
                )
            self._db.commit()

    def semantic_scores(
        self,
        query: Sequence[float],
        *,
        record_ids: set[str],
        model: str,
    ) -> dict[str, float]:
        if not query or not record_ids:
            return {}
        placeholders = ",".join("?" for _ in record_ids)
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT record_id, embedding
                FROM personal_memory_embeddings
                WHERE model=? AND record_id IN ({placeholders})
                """,
                (model, *record_ids),
            ).fetchall()
        query_vector = np.asarray(query, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm <= 0.0:
            return {}
        scores: dict[str, float] = {}
        for record_id, raw_embedding in rows:
            vector = np.asarray(json.loads(str(raw_embedding)), dtype=np.float32)
            if vector.size != query_vector.size:
                continue
            denominator = float(np.linalg.norm(vector)) * query_norm
            if denominator <= 0.0:
                continue
            cosine = float(np.dot(vector, query_vector) / denominator)
            scores[str(record_id)] = max(0.0, min(1.0, cosine))
        return scores

    def close(self) -> None:
        with self._lock:
            self._db.close()
