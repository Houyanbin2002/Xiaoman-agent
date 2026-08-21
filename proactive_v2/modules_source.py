from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from proactive_v2.gateway import GatewayDeps

logger = logging.getLogger(__name__)


class AttentionGatewaySource:
    """Expose canonical personal facts to the proactive turn pipeline.

    External systems are synchronized into ``personal.db`` before this layer.
    This keeps OAuth/transport concerns out of the attention decision loop and
    gives Today, memory consumers and proactive decisions one shared fact set.
    """

    def __init__(self, *, content_limit: int, personal_source: Any | None) -> None:
        self._content_limit = content_limit
        self._personal_source = personal_source

    def build_gateway_deps(
        self,
        *,
        web_fetch_tool: object | None,
        max_chars: int,
    ) -> GatewayDeps:
        return GatewayDeps(
            alert_fn=self.alert_fn,
            feed_fn=self.feed_fn,
            context_fn=self.context_fn,
            web_fetch_tool=web_fetch_tool,
            max_chars=max_chars,
            content_limit=self._content_limit,
        )

    async def alert_fn(self) -> list[dict[str, object]]:
        if self._personal_source is None:
            return []
        try:
            return list(await self._personal_source.alert_fn())
        except Exception as exc:
            logger.warning("[proactive] personal attention fetch failed: %s", exc)
            return []

    async def has_priority_signal(self, now: datetime) -> bool:
        if self._personal_source is None:
            return False
        try:
            return bool(self._personal_source.has_priority_signal(now))
        except Exception as exc:
            logger.warning("[proactive] personal priority probe failed: %s", exc)
            return False

    async def feed_fn(self, limit: int = 5) -> list[dict[str, object]]:
        del limit
        return []

    async def context_fn(self) -> list[dict[str, object]]:
        if self._personal_source is None:
            return []
        try:
            return list(await self._personal_source.context_fn())
        except Exception as exc:
            logger.warning("[proactive] personal state fetch failed: %s", exc)
            return []

    async def ack_fn(self, compound_key: str, ttl_hours: int) -> None:
        del compound_key, ttl_hours

    async def alert_ack_fn(self, compound_key: str) -> None:
        if self._personal_source is None:
            return
        prefix, separator, plan_id = str(compound_key).partition(":")
        if separator and prefix == "attention" and plan_id:
            self._personal_source.complete_action_plan(plan_id)


__all__ = ["AttentionGatewaySource"]
