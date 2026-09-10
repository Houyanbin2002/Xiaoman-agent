from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.memory.markdown import MarkdownStoreApi
from core.memory.activity import activity_entries
from core.conversation_semantics.evidence import sanitize_text
from core.conversation_semantics.models import MemoryCandidate


def _quote_evidence(
    item: MemoryCandidate, sources: dict[str, str]
) -> dict[str, object]:
    text = sources.get(item.source_message_id, "")
    quote = " ".join(item.evidence_quote.split())
    verified = bool(len(quote) >= 4 and quote in text)
    position = text.find(quote) if verified else 0
    return {
        "_evidence_quote_verified": verified,
        "evidence_quote": quote if verified else "",
        "evidence_context": text[max(0, position - 160) : position + len(quote) + 160][
            :1600
        ],
    }


class ConversationMemoryBatchConsumer:
    """Apply the memory-owned partitions of a shared semantic batch."""

    def __init__(
        self,
        *,
        markdown: MarkdownStoreApi,
        candidate_sink: Callable[..., object],
        recent_context_chars: int = 4500,
        message_source: Callable[[str], Sequence[Mapping[str, object]]] | None = None,
    ) -> None:
        self._markdown = markdown
        self._candidate_sink = candidate_sink
        self._recent_context_chars = max(1000, int(recent_context_chars))
        self._message_source = message_source

    async def handle(self, event: ConversationSemanticBatchCommitted) -> None:
        valid_message_ids = set(event.message_ids)
        user_message_ids = set(event.user_message_ids) & valid_message_ids
        rows = self._message_source(event.session_key) if self._message_source else []
        sources = {
            str(row.get("id") or ""): sanitize_text(row.get("content"), limit=5000)
            for row in rows
            if row.get("role") == "user"
            and str(row.get("id") or "") in user_message_ids
        }
        entries = activity_entries(event)
        if entries:
            self._markdown.apply_activity_updates(
                entries, batch_id=event.batch_id, session_key=event.session_key
            )
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
                **_quote_evidence(item, sources),
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
                    }
                    if item.source_message_id in user_message_ids
                    else {}
                ),
            }
            for item in event.payload.memory_candidates
            if item.evidence_refs
            and set(item.evidence_refs) <= valid_message_ids
            and (
                item.source_message_id in user_message_ids
                if item.origin in {"explicit_user", "user_correction"}
                else item.origin == "inferred_pattern"
                and len(set(item.evidence_refs) & user_message_ids) >= 2
            )
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
