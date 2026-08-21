from __future__ import annotations

from pydantic import BaseModel, Field


class WeComConfig(BaseModel):
    enabled: bool = True
    bot_id: str = Field(default="", max_length=256)
    secret: str = Field(default="", max_length=1024)
    allow_from: list[str] = Field(default_factory=list)
    websocket_url: str = Field(
        default="wss://openws.work.weixin.qq.com",
        max_length=1000,
    )
