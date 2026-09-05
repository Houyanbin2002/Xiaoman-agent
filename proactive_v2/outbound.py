from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from agent.turns.outbound import OutboundDispatch, PushToolOutboundPort
from proactive_v2.state import ProactiveStateStore

logger = logging.getLogger(__name__)


class ProactiveOutboundPort:
    """Delivery guard on the existing push path, not a second task executor."""

    def __init__(
        self,
        push_tool: Any,
        *,
        state: ProactiveStateStore,
        allowed: Callable[[OutboundDispatch], bool],
        dedupe_hours: int,
    ) -> None:
        self._push = push_tool
        self._state = state
        self._allowed = allowed
        self._dedupe_hours = dedupe_hours
        self._generation = 0

    def cancel_pending(self) -> None:
        self._generation += 1

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        session = str(outbound.metadata.get("session_key") or "")
        key = str(outbound.metadata.get("delivery_key") or "")
        if not session or not key:
            logger.warning("proactive send rejected: missing delivery identity")
            return False
        generation = self._generation
        token = uuid.uuid4().hex
        claimed = False
        outcome = "unknown"

        def before_send(_: OutboundDispatch) -> bool:
            nonlocal claimed
            if generation != self._generation or not self._allowed(outbound):
                return False
            if not claimed:
                claimed = self._state.claim_delivery_attempt(
                    session,
                    key,
                    token,
                    dedupe_hours=self._dedupe_hours,
                )
            return claimed

        try:
            async with asyncio.timeout(60):
                result = await PushToolOutboundPort(
                    self._push,
                    before_send=before_send,
                ).dispatch_result(outbound)
            if result.success:
                outcome = "accepted"
            elif not result.sent_parts and not result.unknown_parts:
                outcome = "failed"
            # Partial sends and ambiguous remote errors require reconciliation;
            # retrying the complete message could duplicate already accepted parts.
            return result.success
        except TimeoutError:
            logger.warning("proactive send timed out; outcome is unknown: %s", key)
            return False
        finally:
            if claimed:
                self._state.finish_delivery_attempt(session, key, token, outcome)
                logger.info("proactive delivery %s: %s", key, outcome)
