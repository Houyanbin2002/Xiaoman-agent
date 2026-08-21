from __future__ import annotations

from dataclasses import dataclass

from core.conversation_semantics.models import SemanticBatchPayload


@dataclass(frozen=True)
class ConversationSemanticBatchCommitted:
    batch_id: str
    session_key: str
    channel: str
    chat_id: str
    analysis_version: str
    message_ids: tuple[str, ...]
    end_seq: int
    context_consolidate_through: int
    payload: SemanticBatchPayload
    user_message_ids: tuple[str, ...] = ()
    execution_episode_ids: tuple[str, ...] = ()
    execution_tool_names: tuple[str, ...] = ()
