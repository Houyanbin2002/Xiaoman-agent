"""Unified discovery and installation primitives for Skills and MCP tools."""

from .installer import MarketplaceInstaller, MarketplaceInstallResult
from .models import MarketplaceField, MarketplaceItem
from .service import MarketplaceService

__all__ = [
    "MarketplaceField",
    "MarketplaceInstaller",
    "MarketplaceInstallResult",
    "MarketplaceItem",
    "MarketplaceService",
]
