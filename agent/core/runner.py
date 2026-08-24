from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bus.events import InboundMessage, OutboundMessage


class AgentCorePort(Protocol):
    async def process(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage: ...


@dataclass
class CoreRunnerDeps:
    agent_core: AgentCorePort


class CoreRunner:
    """Route passive messages through the application-level AgentCore."""

    def __init__(self, deps: CoreRunnerDeps) -> None:
        self._agent_core = deps.agent_core

    async def process(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        return await self._agent_core.process(
            msg,
            key,
            dispatch_outbound=dispatch_outbound,
        )
