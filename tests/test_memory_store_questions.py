"""Tests for current MarkdownMemoryStore support-file behavior."""

from infra.persistence.markdown_memory_store import MarkdownMemoryStore


def test_append_history_once_is_idempotent_and_hidden_from_read(tmp_path):
    store = MarkdownMemoryStore(tmp_path)

    assert store.append_history_once(
        "[2026-03-08 12:00] USER: hi",
        source_ref="session@1-10",
        kind="history_entry",
    )
    assert not store.append_history_once(
        "[2026-03-08 12:01] USER: hi again",
        source_ref="session@1-10",
        kind="history_entry",
    )

    history = store.read_history()
    raw = store.history_file.read_text(encoding="utf-8")

    assert "USER: hi" in history
    assert "hi again" not in history
    assert "<!-- consolidation:session@1-10:history_entry -->" in raw
