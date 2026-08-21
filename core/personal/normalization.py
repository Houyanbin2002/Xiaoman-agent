from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from core.personal.models import PersonalEntityType, normalize_payload


_DATE_PATTERN = re.compile(r"(?P<date>20\d{2}-\d{2}-\d{2})")
_PERIOD_ALIASES = {
    "上午": "morning",
    "早上": "morning",
    "中午": "noon",
    "下午": "afternoon",
    "傍晚": "evening",
    "晚上": "evening",
}
_SCENE_ALIASES = {
    "any": "any",
    "neutral": "neutral",
    "home": "home",
    "leaving": "leaving",
    "bedtime": "bedtime",
    "travel": "travel",
    "planning": "any",
    "work": "any",
}


def normalize_personal_payload(
    entity_type: PersonalEntityType,
    data: Any,
    *,
    title: str = "",
    summary: str = "",
) -> Any:
    """Normalize shared personal-domain fields at the write boundary.

    Chat tools, workflows and dashboard forms all write through
    ``PersonalDataService``. Keeping canonical task fields here prevents each
    entry point from growing its own slightly different conversion rules.
    """

    payload = normalize_payload(data)
    if entity_type != PersonalEntityType.COMMITMENT or not isinstance(
        payload, Mapping
    ):
        return payload
    return _normalize_commitment(dict(payload), title=title, summary=summary)


def _normalize_commitment(
    payload: dict[str, Any], *, title: str, summary: str
) -> dict[str, Any]:
    payload["state"] = {
        "done": "completed",
        "pending": "open",
        "todo": "open",
    }.get(str(payload.get("state") or "open").lower(), str(payload.get("state") or "open").lower())
    if payload["state"] not in {"open", "completed", "cancelled"}:
        payload["state"] = "open"
    payload["priority"] = {
        "medium": "normal",
    }.get(str(payload.get("priority") or "normal").lower(), str(payload.get("priority") or "normal").lower())
    if payload["priority"] not in {"urgent", "high", "normal", "low"}:
        payload["priority"] = "normal"
    payload["energy"] = {
        "normal": "medium",
    }.get(str(payload.get("energy") or "medium").lower(), str(payload.get("energy") or "medium").lower())
    if payload["energy"] not in {"low", "medium", "high"}:
        payload["energy"] = "medium"
    try:
        progress = float(payload.get("progress", 0.0))
    except (TypeError, ValueError):
        progress = 0.0
    payload["progress"] = min(1.0, max(0.0, progress))

    contexts = payload.get("contexts")
    if isinstance(contexts, str):
        contexts = [contexts]
    if not isinstance(contexts, list):
        contexts = []
    payload["contexts"] = list(
        dict.fromkeys(
            _SCENE_ALIASES[value]
            for item in contexts
            if (value := str(item).strip().lower()) in _SCENE_ALIASES
        )
    ) or ["any"]

    next_action = str(
        payload.get("next_action")
        or payload.get("action")
        or summary
        or title
    ).strip()
    payload["next_action"] = next_action

    estimated = payload.get("estimated_minutes")
    if estimated not in (None, ""):
        try:
            minutes = int(float(estimated))
        except (TypeError, ValueError):
            payload.pop("estimated_minutes", None)
        else:
            if minutes > 0:
                payload["estimated_minutes"] = minutes
            else:
                payload.pop("estimated_minutes", None)

    _normalize_due_window(payload, summary, title)
    return payload


def _normalize_due_window(payload: dict[str, Any], *fallback_texts: str) -> None:
    """Preserve date/period precision without inventing an exact clock time."""

    if payload.get("due_at"):
        return
    explicit_due_text = str(
        payload.get("due_text")
        or payload.get("deadline")
        or payload.get("deadline_text")
        or ""
    ).strip()
    candidates = [explicit_due_text, *(str(item).strip() for item in fallback_texts)]
    match = None
    source_text = ""
    for candidate in candidates:
        match = _DATE_PATTERN.search(candidate)
        if match:
            source_text = candidate
            break
    if match is None:
        return
    if not payload.get("due_date"):
        payload["due_date"] = match.group("date")
    period_label = ""
    if not payload.get("due_period"):
        for label, value in _PERIOD_ALIASES.items():
            if label in source_text:
                payload["due_period"] = value
                period_label = label
                break
    if not payload.get("due_text"):
        payload["due_text"] = " ".join(
            item for item in (match.group("date"), period_label) if item
        )
