from __future__ import annotations

import threading

from core.attention.actions.models import ActionCapability


class ActionCapabilityRegistry:
    """Runtime registry for action manifests contributed by tools and MCPs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, ActionCapability] = {}

    def register(self, capability: ActionCapability, *, replace: bool = False) -> None:
        with self._lock:
            if capability.id in self._items and not replace:
                raise ValueError(
                    f"action capability already registered: {capability.id}"
                )
            self._items[capability.id] = capability

    def unregister(self, capability_id: str) -> bool:
        with self._lock:
            return self._items.pop(capability_id, None) is not None

    def get(self, capability_id: str) -> ActionCapability | None:
        with self._lock:
            return self._items.get(capability_id)

    def list(self) -> list[ActionCapability]:
        with self._lock:
            return sorted(self._items.values(), key=lambda item: item.id)


__all__ = ["ActionCapabilityRegistry"]
