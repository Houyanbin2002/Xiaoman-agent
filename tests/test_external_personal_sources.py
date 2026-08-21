from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from agent.integrations.notion_personal_source import NotionPersonalSourceAdapter
from agent.integrations.mcp_personal_source import McpPersonalSourceAdapter
from agent.integrations.rss_personal_source import RssPersonalSourceAdapter
from core.personal.models import PersonalEntityType, RecordSource
from core.personal.service import PersonalDataService
from core.personal.sources.models import ExternalSourceItem
from core.personal.sources.service import ExternalSourceSyncService
from core.personal.today import PersonalTodayService
from core.attention.providers.personal import PersonalRecordSignalProvider
from infra.persistence.external_source_store import ExternalSourceStore
from infra.persistence.personal_store import PersonalStore


class _FakeAdapter:
    def __init__(self, items: list[ExternalSourceItem]) -> None:
        self.items = items

    async def fetch(self, _subscription):
        return list(self.items)


class _FakeTools:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return json.dumps(self.payload, ensure_ascii=False)

    def get_document(self, name: str) -> Any:
        if name != "mcp_gmail__search_threads":
            return None
        return type(
            "Document",
            (),
            {"source_type": "mcp", "source_name": "gmail"},
        )()


def _mapping() -> dict[str, Any]:
    return {
        "title": "任务",
        "summary": "备注",
        "status": "状态",
        "due": "date:到期:start",
        "priority": "优先级别",
        "category": "类别",
        "archive": "存档",
        "completed_values": ["已完成"],
        "archived_values": ["__YES__"],
    }


def test_external_source_store_deduplicates_subscription(tmp_path: Path) -> None:
    store = ExternalSourceStore(tmp_path / "personal.db")
    first = store.create_subscription(
        provider="notion",
        server_name="notion",
        name="每日待办",
        resource_url="collection://tasks",
        entity_type=PersonalEntityType.COMMITMENT,
        mapping=_mapping(),
        poll_interval_minutes=15,
    )
    second = store.create_subscription(
        provider="notion",
        server_name="notion",
        name="重复名称不会重复订阅",
        resource_url="collection://tasks",
        entity_type=PersonalEntityType.COMMITMENT,
        mapping=_mapping(),
        poll_interval_minutes=30,
    )

    assert second.id == first.id
    assert len(store.list_subscriptions()) == 1
    store.close()


@pytest.mark.asyncio
async def test_notion_adapter_maps_rows_to_canonical_commitments() -> None:
    tools = _FakeTools(
        {
            "results": [
                {
                    "id": "page-1",
                    "url": "https://www.notion.so/page-1",
                    "任务": "完成报告",
                    "备注": "整理最终结论",
                    "状态": "进行中",
                    "date:到期:start": "2026-07-16T15:00:00+08:00",
                    "优先级别": "高优先级",
                    "类别": "工作",
                    "存档": "__NO__",
                },
                {
                    "id": "page-archived",
                    "url": "https://www.notion.so/page-archived",
                    "任务": "旧事项",
                    "存档": "__YES__",
                },
            ]
        }
    )
    store = ExternalSourceStore(Path(":memory:"))
    subscription = store.create_subscription(
        provider="notion",
        server_name="notion",
        name="每日待办",
        resource_url="collection://tasks",
        entity_type=PersonalEntityType.COMMITMENT,
        mapping=_mapping(),
        poll_interval_minutes=15,
    )

    items = await NotionPersonalSourceAdapter(tools).fetch(subscription)

    assert len(items) == 1
    assert items[0].external_id == "page-1"
    assert items[0].data["state"] == "open"
    assert items[0].data["priority"] == "high"
    assert items[0].data["due_at"] == "2026-07-16T15:00:00+08:00"
    assert tools.calls[0][0] == "mcp_notion__notion-query-data-sources"
    store.close()


@pytest.mark.asyncio
async def test_generic_mcp_adapter_only_reads_explicit_subscription() -> None:
    tools = _FakeTools(
        {
            "threads": [
                {
                    "id": "thread-1",
                    "subject": "项目周报",
                    "snippet": "请在今天下班前确认",
                    "from": "leader@example.com",
                }
            ]
        }
    )
    store = ExternalSourceStore(Path(":memory:"))
    subscription = store.create_subscription(
        provider="mcp",
        server_name="gmail",
        name="重要未读邮件",
        resource_url="mcp://gmail/important-unread",
        entity_type=PersonalEntityType.MONITOR_OBSERVATION,
        mapping={
            "tool_name": "mcp_gmail__search_threads",
            "arguments": {"query": "is:unread from:leader@example.com"},
            "items_path": "threads",
            "fields": {
                "id": "id",
                "title": "subject",
                "summary": "snippet",
            },
            "data": {
                "sender": "from",
                "state": {"const": "open"},
            },
        },
        poll_interval_minutes=15,
    )

    items = await McpPersonalSourceAdapter(tools).fetch(subscription)

    assert len(items) == 1
    assert items[0].external_id == "thread-1"
    assert items[0].title == "项目周报"
    assert items[0].data["sender"] == "leader@example.com"
    assert items[0].data["external_server"] == "gmail"
    assert tools.calls == [
        (
            "mcp_gmail__search_threads",
            {"query": "is:unread from:leader@example.com"},
        )
    ]
    store.close()


def test_rss_adapter_keeps_history_silent_and_marks_future_items_new() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Example</title>
      <item><guid>old</guid><title>旧内容</title><link>https://example.com/old</link><pubDate>Sat, 01 Jan 2000 00:00:00 GMT</pubDate></item>
      <item><guid>new</guid><title>新内容</title><description>&lt;b&gt;更新摘要&lt;/b&gt;</description><link>https://example.com/new</link><pubDate>Thu, 01 Jan 2099 00:00:00 GMT</pubDate></item>
    </channel></rss>"""
    store = ExternalSourceStore(Path(":memory:"))
    subscription = store.create_subscription(
        provider="rss",
        server_name="rss",
        name="公开动态",
        resource_url="https://example.com/feed.xml",
        entity_type=PersonalEntityType.MONITOR_OBSERVATION,
        mapping={"domain": "interest", "notify_initial": False},
        poll_interval_minutes=5,
    )

    items = asyncio.run(RssPersonalSourceAdapter(lambda _url: feed).fetch(subscription))

    assert [item.external_id for item in items] == ["old", "new"]
    assert items[0].data["attention_signal"]["enabled"] is False
    assert items[1].data["attention_signal"]["enabled"] is True
    assert items[1].summary == "更新摘要"
    store.close()


def test_rss_adapter_reports_xcancel_whitelist_page() -> None:
    feed = """<rss><channel><title>RSS reader not yet whitelisted!</title>
    <item><title>Please contact the operator</title></item></channel></rss>"""
    store = ExternalSourceStore(Path(":memory:"))
    subscription = store.create_subscription(
        provider="rss",
        server_name="rss",
        name="X 用户",
        resource_url="https://xcancel.com/example/rss",
        entity_type=PersonalEntityType.MONITOR_OBSERVATION,
        mapping={},
        poll_interval_minutes=5,
    )

    with pytest.raises(ValueError, match="白名单"):
        asyncio.run(RssPersonalSourceAdapter(lambda _url: feed).fetch(subscription))
    store.close()


@pytest.mark.asyncio
async def test_sync_updates_canonical_record_without_duplicates(tmp_path: Path) -> None:
    database = tmp_path / "personal.db"
    data = PersonalDataService(PersonalStore(database))
    sources = ExternalSourceStore(database)
    subscription = sources.create_subscription(
        provider="notion",
        server_name="notion",
        name="每日待办",
        resource_url="collection://tasks",
        entity_type=PersonalEntityType.COMMITMENT,
        mapping=_mapping(),
        poll_interval_minutes=15,
    )
    adapter = _FakeAdapter(
        [
            ExternalSourceItem.build(
                external_id="page-1",
                title="完成报告",
                summary="第一版",
                data={"state": "open", "due_at": "2026-07-16T15:00:00+08:00"},
                source_ref="https://www.notion.so/page-1",
            )
        ]
    )
    service = ExternalSourceSyncService(
        store=sources,
        personal_data=data,
        adapters={"notion": adapter},
    )

    first = await service.sync_one(subscription.id)
    adapter.items = [
        ExternalSourceItem.build(
            external_id="page-1",
            title="完成报告",
            summary="第二版",
            data={"state": "completed", "due_at": "2026-07-16T15:00:00+08:00"},
            source_ref="https://www.notion.so/page-1",
        )
    ]
    second = await service.sync_one(subscription.id)

    records = data.list(entity_type=PersonalEntityType.COMMITMENT)
    assert (first.created, first.updated) == (1, 0)
    assert (second.created, second.updated) == (0, 1)
    assert len(records) == 1
    assert records[0].summary == "第二版"
    assert records[0].data["state"] == "completed"
    assert records[0].source == RecordSource("notion", "https://www.notion.so/page-1")
    sources.close()
    data.close()


def test_today_service_filters_by_local_date_and_keeps_overdue_open_items(
    tmp_path: Path,
) -> None:
    data = PersonalDataService(PersonalStore(tmp_path / "personal.db"))
    for title, due_at, state in (
        ("今天完成", "2026-07-16T09:00:00+08:00", "open"),
        ("已经逾期", "2026-07-15T09:00:00+08:00", "open"),
        ("明天处理", "2026-07-17T09:00:00+08:00", "open"),
        ("今天已完成", "2026-07-16T08:00:00+08:00", "completed"),
    ):
        data.create(
            entity_type=PersonalEntityType.COMMITMENT,
            record_key=title,
            title=title,
            summary=title,
            data={"title": title, "due_at": due_at, "state": state},
            source=RecordSource("dashboard", "test"),
        )

    result = PersonalTodayService(data).get(
        local_date="2026-07-16",
        timezone_name="Asia/Shanghai",
        now=datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc),
    )

    assert [record.title for record in result.records] == ["已经逾期", "今天完成"]
    assert result.overdue_count == 1
    assert result.counts["commitment"] == 2
    data.close()


def test_overdue_open_commitment_remains_attention_candidate(tmp_path: Path) -> None:
    data = PersonalDataService(PersonalStore(tmp_path / "personal.db"))
    data.create(
        entity_type=PersonalEntityType.COMMITMENT,
        record_key="overdue",
        title="仍未完成的逾期事项",
        summary="需要重新安排",
        data={
            "state": "open",
            "due_at": "2026-07-10T09:00:00+08:00",
            "priority": "high",
        },
        source=RecordSource("notion", "https://www.notion.so/overdue"),
    )

    signals = PersonalRecordSignalProvider(data).collect(
        datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)
    )

    assert len(signals) == 1
    assert signals[0].urgency == 1.0
    assert signals[0].source.type == "personal_record"
    data.close()
