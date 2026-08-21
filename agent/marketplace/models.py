from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

MarketplaceKind = Literal["skill", "mcp"]
MarketplaceInstallMode = Literal["direct", "configure", "oauth", "unsupported"]


@dataclass(frozen=True)
class MarketplaceField:
    name: str
    label: str
    required: bool = False
    secret: bool = False
    placeholder: str = ""

    def public(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "required": self.required,
            "secret": self.secret,
            "placeholder": self.placeholder,
        }


@dataclass(frozen=True)
class MarketplaceItem:
    id: str
    kind: MarketplaceKind
    name: str
    description: str
    provider: str
    source_url: str = ""
    version: str = ""
    icon_url: str = ""
    install_count: int | None = None
    verified: bool = False
    deprecated: bool = False
    installed: bool = False
    install_mode: MarketplaceInstallMode = "unsupported"
    configuration_fields: tuple[MarketplaceField, ...] = ()
    unsupported_reason: str = "当前版本暂不支持自动安装"
    install_spec: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def with_installed(self, installed: bool) -> "MarketplaceItem":
        return replace(self, installed=installed)

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "provider": self.provider,
            "source_url": self.source_url,
            "version": self.version,
            "icon_url": self.icon_url,
            "install_count": self.install_count,
            "verified": self.verified,
            "deprecated": self.deprecated,
            "installed": self.installed,
            "install_mode": self.install_mode,
            "configuration_fields": [
                item.public() for item in self.configuration_fields
            ],
            "unsupported_reason": self.unsupported_reason,
        }
