from __future__ import annotations

from typing import Any

from core.conversation_semantics.events import ConversationSemanticBatchCommitted


class ConversationAttentionBatchConsumer:
    """Route only attention observations to the attention learning service."""

    def __init__(self, learning: Any) -> None:
        self._learning = learning

    def handle(self, event: ConversationSemanticBatchCommitted) -> None:
        valid_message_ids = set(event.message_ids)
        user_message_ids = set(event.user_message_ids) & valid_message_ids
        observations: list[dict[str, object]] = []
        for item in event.payload.attention_observations:
            observation = item.to_mapping()
            observation.pop("_user_evidence_verified", None)
            observation.pop("source_message_id", None)
            if item.source_message_id in user_message_ids:
                observation["source_message_id"] = item.source_message_id
                observation["_user_evidence_verified"] = True
            observations.append(observation)
        if not observations:
            return
        self._learning.ingest_many(
            observations,
            source_type="conversation",
            source_ref=event.batch_id,
            metadata={"channel": event.channel, "chat_id": event.chat_id},
            trust_user_evidence=True,
        )
