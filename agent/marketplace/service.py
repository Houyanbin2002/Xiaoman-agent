from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .models import MarketplaceItem, MarketplaceKind


class MarketplaceProvider(Protocol):
    def search(self, query: str, limit: int) -> list[MarketplaceItem]: ...

    def get(self, item_id: str) -> MarketplaceItem | None: ...

    def refresh(self) -> object: ...


class MarketplaceService:
    def __init__(
        self,
        skill_provider: MarketplaceProvider,
        mcp_provider: MarketplaceProvider,
        *,
        installed_skills: Callable[[], set[str]] | None = None,
        installed_mcp: Callable[[], set[str]] | None = None,
    ) -> None:
        self.skill_provider = skill_provider
        self.mcp_provider = mcp_provider
        self._installed_skills = installed_skills or set
        self._installed_mcp = installed_mcp or set

    def search(
        self, kind: MarketplaceKind, query: str = "", limit: int = 20
    ) -> list[MarketplaceItem]:
        provider = self._provider(kind)
        installed = self._installed(kind)
        return [
            item.with_installed(_is_installed(item, installed))
            for item in provider.search(query, limit)
        ]

    def get(self, kind: MarketplaceKind, item_id: str) -> MarketplaceItem | None:
        item = self._provider(kind).get(item_id)
        if item is None:
            return None
        return item.with_installed(_is_installed(item, self._installed(kind)))

    def refresh(self, kind: MarketplaceKind | None = None) -> None:
        if kind is None:
            self.skill_provider.refresh()
            self.mcp_provider.refresh()
            return
        self._provider(kind).refresh()

    def _provider(self, kind: MarketplaceKind) -> MarketplaceProvider:
        if kind == "skill":
            return self.skill_provider
        if kind == "mcp":
            return self.mcp_provider
        raise ValueError("市场类型必须是 skill 或 mcp")

    def _installed(self, kind: MarketplaceKind) -> set[str]:
        return self._installed_skills() if kind == "skill" else self._installed_mcp()


def _is_installed(item: MarketplaceItem, installed: set[str]) -> bool:
    candidates = {item.id, item.name, item.id.rsplit("/", 1)[-1]}
    normalized = {name.casefold() for name in installed}
    return any(candidate.casefold() in normalized for candidate in candidates)


class CombinedMarketplaceProvider:
    """Combine providers in priority order and hide lower-priority duplicates."""

    def __init__(self, *providers: MarketplaceProvider) -> None:
        self._providers = providers

    def search(self, query: str, limit: int) -> list[MarketplaceItem]:
        rows: list[MarketplaceItem] = []
        identities: set[str] = set()
        last_error: OSError | RuntimeError | None = None
        for provider in self._providers:
            try:
                provider_rows = provider.search(query, limit)
            except (OSError, RuntimeError) as exc:
                last_error = exc
                continue
            for item in provider_rows:
                identity = item.name.casefold()
                if identity in identities:
                    continue
                identities.add(identity)
                rows.append(item)
                if len(rows) >= limit:
                    return rows
            normalized_query = query.strip().casefold()
            if normalized_query and any(
                normalized_query
                in {item.id.casefold(), item.name.casefold(), item.provider.casefold()}
                for item in provider_rows
            ):
                return rows
        if not rows and last_error is not None:
            raise last_error
        return rows

    def get(self, item_id: str) -> MarketplaceItem | None:
        for provider in self._providers:
            item = provider.get(item_id)
            if item is not None:
                return item
        return None

    def refresh(self) -> None:
        for provider in self._providers:
            provider.refresh()
