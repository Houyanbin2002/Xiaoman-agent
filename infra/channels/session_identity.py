from __future__ import annotations

from typing import Any

from session.manager import SessionManager


async def remember_channel_session(
    session_manager: SessionManager,
    *,
    session_key: str,
    channel: str,
    chat_id: str,
    sender_id: str,
    title: str,
    chat_type: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the small, channel-neutral identity envelope used by the UI.

    Channel-specific identifiers stay in ``metadata``.  The common fields let
    the dashboard group and label gateway conversations without knowing each
    provider's wire protocol.
    """

    session = session_manager.get_or_create(session_key)
    next_metadata: dict[str, Any] = {
        "gateway_channel": channel,
        "gateway_chat_id": chat_id,
        "external_identity": sender_id,
        "chat_type": chat_type,
    }
    if title and not str(session.metadata.get("title") or "").strip():
        next_metadata["title"] = title
    if metadata:
        next_metadata.update(metadata)
    changed = False
    for key, value in next_metadata.items():
        if session.metadata.get(key) == value:
            continue
        session.metadata[key] = value
        changed = True
    if changed:
        await session_manager.save_async(session)
