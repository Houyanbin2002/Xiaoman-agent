"""Unified discovery for Xiaoman's executable and instructional capabilities."""

from agent.capabilities.catalog import (
    CapabilityCatalog,
    CapabilityMatch,
    CapabilityRecord,
)
from agent.capabilities.router import CapabilityRoute, CapabilityRouter

__all__ = [
    "CapabilityCatalog",
    "CapabilityMatch",
    "CapabilityRecord",
    "CapabilityRoute",
    "CapabilityRouter",
]
