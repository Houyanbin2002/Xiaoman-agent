from __future__ import annotations

from pydantic import BaseModel, Field


class WeixinConfig(BaseModel):
    enabled: bool = True
    account_id: str = Field(default="", max_length=256)
    base_url: str = Field(
        default="https://ilinkai.weixin.qq.com",
        max_length=1000,
    )
    allow_from: list[str] = Field(default_factory=list)
    group_allow_from: list[str] = Field(default_factory=list)
    groups_enabled: bool = False
