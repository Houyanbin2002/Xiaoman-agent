from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import keyring


class LocalSecretStore:
    """Workspace-scoped credentials backed by the operating system keyring."""

    _SERVICE = "xiaoman.channels"

    def __init__(self, workspace: Path) -> None:
        resolved = str(workspace.resolve(strict=False)).casefold().encode("utf-8")
        self._namespace = hashlib.sha256(resolved).hexdigest()[:16]

    async def get_bundle(self, name: str) -> dict[str, str]:
        try:
            raw = await asyncio.to_thread(
                keyring.get_password,
                self._SERVICE,
                self._account(name),
            )
        except keyring.errors.KeyringError:
            return {}
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return (
            {str(key): str(value) for key, value in payload.items()}
            if isinstance(payload, dict)
            else {}
        )

    async def set_bundle(self, name: str, values: dict[str, str]) -> None:
        account = self._account(name)
        if not values:
            await self.delete(name)
            return
        raw = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        await asyncio.to_thread(
            keyring.set_password,
            self._SERVICE,
            account,
            raw,
        )

    async def delete(self, name: str) -> None:
        try:
            await asyncio.to_thread(
                keyring.delete_password,
                self._SERVICE,
                self._account(name),
            )
        except keyring.errors.PasswordDeleteError:
            pass

    def _account(self, name: str) -> str:
        return f"{self._namespace}:{name}"
