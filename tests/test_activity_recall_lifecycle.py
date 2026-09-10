"""Real temporary persistence; no user database or external model calls."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.conversation_semantics.models import RecentActivityCandidate
from core.memory.engine import MemoryQuery
from infra.persistence.recent_activity_store import RecentActivityStore
from memory2.memorizer import Memorizer
from memory2.store import MemoryStore2
from plugins.akasha.engine import AkashaCard, AkashaMemoryEngine, _format_cards
from plugins.akasha.core import AkashaCandidate
from plugins.akasha.config import AkashaConfig
from plugins.default_memory.engine import DefaultMemoryEngine


def candidate(**kwargs):
    values = dict(
        summary="正在挑礼物",
        title="挑礼物",
        status="active",
        source_message_ids=("u1",),
    )
    return RecentActivityCandidate(**(values | kwargs))


def seed(tmp_path):
    store = RecentActivityStore(tmp_path / "activity.db")
    store.apply([candidate()], batch_id="start", session_key="s1")
    return store, store.snapshot()[0]


@pytest.mark.parametrize("status", ["completed", "cancelled", "dismissed"])
def test_old_evidence_resolves_latest_status_after_restart(tmp_path, status):
    store, row = seed(tmp_path)
    update = candidate(
        status=status,
        activity_id=row["id"],
        expected_revision=1,
        source_message_ids=("u2",),
    )
    store.apply([update], batch_id="finish", session_key="s2")
    store.apply([update], batch_id="finish", session_key="s2")
    restarted = RecentActivityStore(tmp_path / "activity.db")
    result = restarted.recall_states(
        ["message:u1", "message:u2", "update:start:activity:0"]
    )
    assert len(result) == 3
    assert all(
        states[0]["status"] == status and states[0]["revision"] == 2
        for states in result.values()
    )


def test_stale_update_does_not_associate_evidence_or_reopen(tmp_path):
    store, row = seed(tmp_path)
    store.apply(
        [candidate(status="completed", activity_id=row["id"], expected_revision=1)],
        batch_id="done",
        session_key="s",
    )
    store.apply(
        [
            candidate(
                activity_id=row["id"],
                expected_revision=1,
                source_message_ids=("stale",),
            )
        ],
        batch_id="stale",
        session_key="s",
    )
    assert store.recall_states(["message:stale", "update:stale:activity:0"]) == {}
    assert store.recall_states(["message:u1"])["message:u1"][0]["status"] == "completed"


def test_expiration_and_lookup_not_limited_to_ui_window(tmp_path):
    store = RecentActivityStore(tmp_path / "activity.db")
    now = datetime.now(timezone.utc)
    store.apply(
        [candidate(occurred_at=(now - timedelta(days=20)).isoformat())],
        batch_id="old",
        session_key="s",
    )
    for i in range(65):
        store.apply(
            [candidate(title=f"other-{i}", source_message_ids=(f"other-{i}",))],
            batch_id=f"b{i}",
            session_key="s",
        )
    assert len(store.snapshot(limit=60)) == 60
    assert store.recall_states(["message:u1"])["message:u1"][0]["status"] == "expired"


class Embedder:
    async def embed(self, text):
        return [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_structured_write_filter_history_and_pending_projection(tmp_path):
    activities, row = seed(tmp_path)
    db = MemoryStore2(tmp_path / "memory.db", vec_dim=3)
    try:
        memorizer = Memorizer(db, Embedder())
        engine = DefaultMemoryEngine.__new__(DefaultMemoryEngine)
        engine._activity_resolver = activities.recall_states
        args = dict(
            history_entry="正在挑礼物",
            behavior_updates=[],
            source_ref="start:activity:0",
            scope_channel="web",
            scope_chat_id="1",
            activity_update_ref="update:start:activity:0",
            happened_at="2026-09-01T00:00:00+00:00",
        )
        await memorizer.save_from_consolidation(**args)
        await memorizer.save_from_consolidation(**args)
        items = db.list_by_type("event")
        assert len(items) == 1
        assert items[0]["happened_at"] == args["happened_at"]
        assert len(engine._resolve_activity_hits(items, current_context=True)) == 1
        activities.apply(
            [candidate(status="completed", activity_id=row["id"], expected_revision=1)],
            batch_id="done",
            session_key="s",
        )
        assert engine._resolve_activity_hits(items, current_context=True) == []
        history = engine._resolve_activity_hits(items, current_context=False)
        assert len(history) == 1 and "已完成" in history[0]["summary"]
        pending = [
            {
                **items[0],
                "extra_json": {"activity_update_ref": "update:not-applied:activity:0"},
            }
        ]
        assert engine._resolve_activity_hits(pending, current_context=True) == []
        assert (
            "状态未确认"
            in engine._resolve_activity_hits(pending, current_context=False)[0][
                "summary"
            ]
        )
        assert db.list_by_type("event")[0]["summary"] == "正在挑礼物"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_identical_summaries_do_not_merge_distinct_activity_evidence(tmp_path):
    db = MemoryStore2(tmp_path / "memory.db", vec_dim=3)
    try:
        memorizer = Memorizer(db, Embedder())
        for i in range(2):
            await memorizer.save_from_consolidation(
                history_entry="准备报告",
                behavior_updates=[],
                source_ref=f"b:activity:{i}",
                scope_channel="web",
                scope_chat_id="1",
                activity_update_ref=f"update:b:activity:{i}",
            )
        assert len(db.list_by_type("event")) == 2
    finally:
        db.close()


@pytest.mark.parametrize("lane", ["dense", "ripple", "activation"])
def test_graph_filters_before_limit_keeps_explicit_history(tmp_path, monkeypatch, lane):
    store, row = seed(tmp_path)
    store.apply(
        [candidate(status="completed", activity_id=row["id"], expected_revision=1)],
        batch_id="done",
        session_key="s",
    )
    cards = {
        "ended": AkashaCard(
            "ended",
            json.dumps(["u1"]),
            "正在挑礼物",
            "我帮你看看",
            "2026-09-01",
            0.99,
            lane,
            {},
        ),
        "other": AkashaCard(
            "other",
            json.dumps(["other"]),
            "学习 Python",
            "一起学习",
            "2026-09-02",
            0.8,
            lane,
            {},
        ),
    }
    monkeypatch.setattr(
        "plugins.akasha.engine._load_turn_card", lambda path, key, **kw: cards[key]
    )
    engine = AkashaMemoryEngine.__new__(AkashaMemoryEngine)
    engine._session_db_path = tmp_path / "unused.db"
    engine._akasha_config = SimpleNamespace(assistant_preview_chars=100)
    engine._activity_resolver = store.recall_states
    hits = [(key, card.score, lane, {}) for key, card in cards.items()]
    automatic = engine._cards_from_keys(hits, limit=1, current_context=True)
    assert [card.key for card in automatic] == ["other"]
    history = engine._cards_from_keys(hits, limit=2, current_context=False)
    assert len(history) == 2
    assert "已完成" in _format_cards("历史", history)
    assert history[0].user_message == cards["ended"].user_message


def test_mixed_message_requires_all_linked_activities_open(tmp_path, monkeypatch):
    store, row = seed(tmp_path)
    store.apply([candidate(title="写报告")], batch_id="second", session_key="s")
    store.apply(
        [candidate(status="cancelled", activity_id=row["id"], expected_revision=1)],
        batch_id="cancel",
        session_key="s",
    )
    states = store.recall_states(["message:u1"])["message:u1"]
    from core.memory.activity import current_activity_evidence

    assert len(states) == 2
    assert not current_activity_evidence(states)
    store.apply(
        [candidate(activity_id=row["id"], expected_revision=2, reopen=True)],
        batch_id="reopen",
        session_key="s",
    )
    assert current_activity_evidence(store.recall_states(["message:u1"])["message:u1"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent, count", [("context", 0), ("interest", 0), ("answer", 1)]
)
async def test_public_graph_query_distinguishes_automatic_and_explicit(
    tmp_path, monkeypatch, intent, count
):
    store, row = seed(tmp_path)
    store.apply(
        [candidate(status="completed", activity_id=row["id"], expected_revision=1)],
        batch_id="done",
        session_key="s",
    )
    card = AkashaCard(
        "old",
        json.dumps(["u1"]),
        "正在挑礼物",
        "帮你挑",
        "2026-09-01",
        0.9,
        "dense",
        {},
    )
    monkeypatch.setattr(
        "plugins.akasha.engine._load_turn_card", lambda *args, **kwargs: card
    )
    engine = AkashaMemoryEngine.__new__(AkashaMemoryEngine)
    engine._session_db_path = tmp_path / "unused.db"
    engine._akasha_config = AkashaConfig(
        assistant_preview_chars=100, dense_top_k=2, ripple_top_k=2
    )
    engine._activity_resolver = store.recall_states
    engine._embedder = Embedder()
    hit = AkashaCandidate("old", "dense", 0, 0.9, 0, 0, 1, 1, 0, 0.9)
    engine._retrieve = lambda *args, **kwargs: SimpleNamespace(
        dense_items=[hit],
        ripple_items=[],
        activation_items=[],
        trace=SimpleNamespace(seed_count=1, pool_count=1),
    )
    result = await engine.query(
        MemoryQuery(
            text="礼物",
            intent=intent,
            effect="read_only",
            timestamp=datetime.now(timezone.utc),
        )
    )
    assert len(result.records) == count
    if count:
        assert "已完成" in result.records[0].summary
        assert "已完成" in result.raw["items"][0]["summary"]
