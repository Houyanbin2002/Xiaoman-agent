from __future__ import annotations

from agent.mcp.catalog import CATALOG

from .models import MarketplaceField, MarketplaceItem


class CuratedMcpProvider:
    """Product-ready MCP entries whose setup needs richer local knowledge."""

    def __init__(self) -> None:
        self._items = {item.id: item for item in _curated_items()}

    def search(self, query: str = "", limit: int = 20) -> list[MarketplaceItem]:
        needle = query.strip().casefold()
        rows = [
            item
            for item in self._items.values()
            if not needle
            or needle
            in f"{item.id} {item.name} {item.provider} {item.description}".casefold()
        ]
        return rows[: max(0, min(limit, 100))]

    def get(self, item_id: str) -> MarketplaceItem | None:
        return self._items.get(item_id)

    def refresh(self) -> list[MarketplaceItem]:
        return list(self._items.values())


def _curated_items() -> list[MarketplaceItem]:
    return [
        _item(
            "markitdown",
            source_url="https://github.com/microsoft/markitdown",
            install_mode="direct",
            install_spec={"catalog_id": "markitdown"},
        ),
        _item(
            "notion",
            source_url="https://developers.notion.com/docs/mcp",
            install_mode="oauth",
            install_spec={
                "transport": "streamable_http",
                "url": "https://mcp.notion.com/mcp",
                "auth_type": "oauth",
            },
        ),
        _item(
            "gmail",
            source_url=str(CATALOG["gmail"].configuration["docs_url"]),
            install_mode="configure",
            configuration_fields=(
                MarketplaceField(
                    name="oauth_client_id",
                    label="OAuth Client ID",
                    required=True,
                    placeholder="Google Cloud Web OAuth Client ID",
                ),
                MarketplaceField(
                    name="oauth_client_secret",
                    label="OAuth Client Secret",
                    required=True,
                    secret=True,
                    placeholder="安全保存到系统凭据库",
                ),
            ),
            install_spec={
                "transport": "streamable_http",
                "url": "https://gmailmcp.googleapis.com/mcp/v1",
                "auth_type": "oauth",
            },
        ),
        _item(
            "github",
            source_url="https://github.com/github/github-mcp-server",
            install_mode="configure",
            configuration_fields=(
                MarketplaceField(
                    name="bearer_token",
                    label="GitHub Personal Access Token",
                    required=True,
                    secret=True,
                    placeholder="安全保存到系统凭据库，不写入配置文件",
                ),
            ),
            install_spec={
                "transport": "streamable_http",
                "url": "https://api.githubcopilot.com/mcp/",
                "auth_type": "bearer",
            },
        ),
        _item(
            "obsidian",
            source_url="https://github.com/bitbonsai/mcpvault",
            install_mode="configure",
            configuration_fields=(
                MarketplaceField(
                    name="vault_path",
                    label="Vault 路径",
                    required=True,
                    placeholder="D:\\Notes\\My Vault",
                ),
            ),
            install_spec={
                "transport": "stdio",
                "registry": "npm",
                "package": "@bitbonsai/mcpvault",
                "version": "latest",
                "argument_fields": ["vault_path"],
            },
        ),
    ]


def _item(
    item_id: str,
    *,
    source_url: str,
    install_mode: str,
    install_spec: dict[str, object],
    configuration_fields: tuple[MarketplaceField, ...] = (),
) -> MarketplaceItem:
    entry = CATALOG[item_id]
    return MarketplaceItem(
        id=item_id,
        kind="mcp",
        name=entry.name,
        description=entry.description,
        provider=entry.provider,
        source_url=source_url,
        verified=True,
        install_mode=install_mode,  # type: ignore[arg-type]
        configuration_fields=configuration_fields,
        unsupported_reason="",
        install_spec=install_spec,
    )
