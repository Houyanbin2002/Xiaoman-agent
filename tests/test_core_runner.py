"""核心运行器测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from typing import Any, cast

import pytest

from agent.core.runner import CoreRunner, CoreRunnerDeps
from bus.events import InboundMessage, OutboundMessage


@pytest.mark.asyncio
async def test_core_runner_routes_passive_message_to_agent_core():
    runner = CoreRunner(
        CoreRunnerDeps(
            agent_core=cast(
                Any,
                SimpleNamespace(
                    process=AsyncMock(
                        return_value=OutboundMessage(
                            channel="cli",
                            chat_id="1",
                            content="final",
                        )
                    ),
                    pipeline=SimpleNamespace(),
                ),
            ),
        )
    )
    msg = InboundMessage(channel="cli", sender="hua", chat_id="1", content="hi")

    out = await runner.process(msg, "cli:1")

    assert out.content == "final"
    runner._agent_core.process.assert_awaited_once_with(
        msg,
        "cli:1",
        dispatch_outbound=True,
    )
