from __future__ import annotations

from agent.plugins import Plugin

from .channel import WeComChannel
from .config import WeComConfig


class WeComPlugin(Plugin):
    name = "wecom"
    ConfigModel = WeComConfig

    async def initialize(self) -> None:
        config = self.context.config
        self._channel = (
            WeComChannel(config)
            if isinstance(config, WeComConfig)
            and config.enabled
            and config.bot_id.strip()
            and config.secret.strip()
            else None
        )

    def channels(self) -> list[object]:
        return [self._channel] if self._channel is not None else []
