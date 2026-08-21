from __future__ import annotations

from pydantic import BaseModel, Field


class QQBotGroupRule(BaseModel):
    group_openid: str = Field(min_length=1, max_length=256)
    allow_from: list[str] = Field(default_factory=list)
    require_at: bool = True
    allow_proactive: bool = False


class QQBotConfig(BaseModel):
    enabled: bool = True
    app_id: str = Field(default="", max_length=128)
    client_secret: str = Field(default="", max_length=1024)
    allow_from: list[str] = Field(default_factory=list)
    groups: list[QQBotGroupRule] = Field(default_factory=list)
    sandbox: bool = False
    markdown_support: bool = True
