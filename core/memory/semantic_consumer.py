from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import datetime, timezone

from agent.memory import MemoryStore
from core.conversation_semantics.events import ConversationSemanticBatchCommitted


class ConversationMemoryBatchConsumer:
    """Apply the memory-owned partitions of a shared semantic batch."""

    def __init__(
        self,
        *,
        markdown: MemoryStore,
        candidate_sink: Callable[..., object],
        get_session: Callable[[str], object],
        save_session: Callable[[object], object],
        recent_context_chars: int = 4500,
    ) -> None:
        self._markdown = markdown
        self._candidate_sink = candidate_sink
        self._get_session = get_session
        self._save_session = save_session
        self._recent_context_chars = max(1000, int(recent_context_chars))

    async def handle(self, event: ConversationSemanticBatchCommitted) -> None:
        valid_message_ids = set(event.message_ids)
        user_message_ids = set(event.user_message_ids) & valid_message_ids
        entries = [
            item
            for item in event.payload.recent_activity_entries
            if event.analysis_version != "conversation-v3"
            or bool(set(item.source_message_ids) & user_message_ids)
        ]
        if entries:
            rendered = []
            for item in entries:
                occurred_at = item.occurred_at or datetime.now(timezone.utc).isoformat()
                rendered.append(
                    f"- [{occurred_at[:16]}] [{event.session_key}] {item.summary}"
                )
            self._markdown.append_history_once(
                "\n".join(rendered),
                source_ref=event.batch_id,
                kind="recent_activity",
            )
        candidates = [
            {
                "tag": item.tag,
                "content": item.content,
                "confidence": item.confidence,
                "origin": item.origin,
                "evidence_refs": list(item.evidence_refs),
                **({"subject": item.subject} if item.subject else {}),
                **({"predicate": item.predicate} if item.predicate else {}),
                **({"value": item.value} if item.value else {}),
                **({"scope": item.scope} if item.scope else {}),
                **({"attributes": item.attributes} if item.attributes else {}),
                **({"replaces": item.replaces} if item.replaces else {}),
                **({"valid_from": item.valid_from} if item.valid_from else {}),
                **({"expires_at": item.expires_at} if item.expires_at else {}),
                **(
                    {
                        "source_message_id": item.source_message_id,
                        "_user_evidence_verified": True,
                    }
                    if item.source_message_id in user_message_ids
                    else {}
                ),
            }
            for item in event.payload.memory_candidates
        ]
        if candidates:
            result = self._candidate_sink(
                candidates,
                source_ref=event.batch_id,
                source="conversation_semantic_batch",
            )
            if inspect.isawaitable(result):
                await result
        if entries:
            self._markdown.write_recent_context(
                self._markdown.build_recent_activity_context(
                    max_chars=self._recent_context_chars
                )
            )
        # Semantic extraction owns only its durable analysis cursor.  The
        # model-context cursor is advanced exclusively by the cache-aware
        # summary compactor after the new summary has been persisted.
