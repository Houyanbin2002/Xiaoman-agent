from __future__ import annotations

from agent.plugins import Plugin

from .channel import QQBotChannel
from .config import QQBotConfig


class QQBotPlugin(Plugin):
    name = "qqbot"
    ConfigModel = QQBotConfig

    async def initialize(self) -> None:
        config = self.context.config
        self._channel = (
            QQBotChannel(config)
            if isinstance(config, QQBotConfig)
            and config.enabled
            and config.app_id.strip()
            and config.client_secret.strip()
            else None
        )

    def channels(self) -> list[object]:
        return [self._channel] if self._channel is not None else []
