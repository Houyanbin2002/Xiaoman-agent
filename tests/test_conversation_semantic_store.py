from __future__ import annotations

from core.conversation_semantics.models import SemanticBatchPayload
from core.conversation_semantics.store import ConversationSemanticStore


def test_semantic_store_prepares_idempotent_batch(tmp_path) -> None:
    store = ConversationSemanticStore(tmp_path / "sessions.db")
    payload = SemanticBatchPayload.empty()

    first = store.prepare(
        session_key="web:1",
        channel="web",
        chat_id="1",
        analysis_version="conversation-v1",
        message_ids=["web:1:0", "web:1:1"],
        user_message_ids=["web:1:0"],
        end_seq=1,
        context_consolidate_through=0,
        payload=payload,
    )
    second = store.prepare(
        session_key="web:1",
        channel="web",
        chat_id="1",
        analysis_version="conversation-v1",
        message_ids=["web:1:0", "web:1:1"],
        user_message_ids=["web:1:0"],
        end_seq=1,
        context_consolidate_through=0,
        payload=payload,
    )

    assert second.batch_id == first.batch_id
    assert store.pending_cursor("web:1") == -1
    assert [batch.batch_id for batch in store.list_undelivered()] == [first.batch_id]
    assert first.user_message_ids == ("web:1:0",)
    assert first.to_event().user_message_ids == ("web:1:0",)

    store.mark_delivered(first.batch_id)

    assert store.pending_cursor("web:1") == 1
    assert store.list_undelivered() == []
    store.close()


def test_semantic_store_does_not_move_cursor_backwards(tmp_path) -> None:
    store = ConversationSemanticStore(tmp_path / "sessions.db")
    payload = SemanticBatchPayload.empty()
    newer = store.prepare(
        session_key="qq:7",
        channel="qq",
        chat_id="7",
        analysis_version="conversation-v1",
        message_ids=["qq:7:8", "qq:7:9"],
        end_seq=9,
        context_consolidate_through=4,
        payload=payload,
    )
    store.mark_delivered(newer.batch_id)
    older = store.prepare(
        session_key="qq:7",
        channel="qq",
        chat_id="7",
        analysis_version="conversation-v1",
        message_ids=["qq:7:4", "qq:7:5"],
        end_seq=5,
        context_consolidate_through=2,
        payload=payload,
    )

    store.mark_delivered(older.batch_id)

    assert store.pending_cursor("qq:7") == 9
    store.close()
