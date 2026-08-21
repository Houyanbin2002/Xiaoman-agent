from __future__ import annotations

from agent.plugins import Plugin
from infra.security import LocalSecretStore

from .channel import WeixinChannel
from .config import WeixinConfig


class WeixinPlugin(Plugin):
    name = "weixin"
    ConfigModel = WeixinConfig

    async def initialize(self) -> None:
        config = self.context.config
        workspace = self.context.workspace
        self._channel = None
        if not isinstance(config, WeixinConfig) or not config.enabled or workspace is None:
            return
        credentials = await LocalSecretStore(workspace).get_bundle("weixin")
        account_id = config.account_id.strip() or credentials.get("account_id", "")
        token = credentials.get("token", "")
        base_url = credentials.get("base_url", "") or config.base_url
        if not account_id or not token:
            return
        data_dir = self.context.data_dir or (self.context.plugin_dir / ".data")
        self._channel = WeixinChannel(
            config,
            account_id=account_id,
            token=token,
            base_url=base_url,
            data_dir=data_dir,
        )

    def channels(self) -> list[object]:
        return [self._channel] if self._channel is not None else []
