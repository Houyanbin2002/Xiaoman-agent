from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from proactive_v2.modules_source import AttentionGatewaySource


@pytest.mark.asyncio
async def test_attention_gateway_reads_only_canonical_personal_source() -> None:
    personal = MagicMock()
    personal.alert_fn = AsyncMock(return_value=[{"event_id": "plan-1"}])
    personal.context_fn = AsyncMock(return_value=[{"kind": "personal_state"}])
    source = AttentionGatewaySource(content_limit=5, personal_source=personal)
    deps = source.build_gateway_deps(web_fetch_tool=None, max_chars=123)

    assert await deps.alert_fn() == [{"event_id": "plan-1"}]
    assert await deps.feed_fn(limit=1) == []
    assert await deps.context_fn() == [{"kind": "personal_state"}]
    assert deps.content_limit == 5
    assert deps.max_chars == 123
    personal.alert_fn.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_attention_gateway_priority_probe_uses_personal_engine() -> None:
    personal = MagicMock()
    personal.has_priority_signal.return_value = True
    source = AttentionGatewaySource(content_limit=5, personal_source=personal)
    now = datetime(2026, 7, 16, 8, 0, tzinfo=timezone.utc)

    assert await source.has_priority_signal(now)
    personal.has_priority_signal.assert_called_once_with(now)


@pytest.mark.asyncio
async def test_attention_gateway_ack_completes_action_plan() -> None:
    personal = MagicMock()
    source = AttentionGatewaySource(content_limit=5, personal_source=personal)

    await source.alert_ack_fn("attention:plan-1")
    await source.ack_fn("ignored:item", 720)

    personal.complete_action_plan.assert_called_once_with("plan-1")
