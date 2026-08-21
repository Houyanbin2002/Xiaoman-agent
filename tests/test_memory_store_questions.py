"""Tests for current MemoryStore support-file behavior."""

from agent.memory import MemoryStore


def test_append_history_once_is_idempotent_and_hidden_from_read(tmp_path):
    store = MemoryStore(tmp_path)

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


def test_append_journal_writes_daily_file_once(tmp_path):
    store = MemoryStore(tmp_path)

    assert store.append_journal(
        "2026-03-08",
        "[2026-03-08 12:00] 用户确认需求",
        source_ref="session@1-10",
        kind="journal:2026-03-08",
    )
    assert not store.append_journal(
        "2026-03-08",
        "[2026-03-08 12:01] 重复写入",
        source_ref="session@1-10",
        kind="journal:2026-03-08",
    )

    raw = (store.journal_dir / "2026-03-08.md").read_text(encoding="utf-8")
    assert raw.startswith("# 2026-03-08")
    assert "用户确认需求" in raw
    assert "重复写入" not in raw


def test_append_journal_rejects_invalid_date_path(tmp_path):
    store = MemoryStore(tmp_path)

    assert not store.append_journal("../bad", "x")
    assert not (store.memory_dir / "bad.md").exists()
