from __future__ import annotations

from collections.abc import Mapping, Sequence

import json_repair

from core.conversation_semantics.models import SemanticBatchPayload
from core.conversation_semantics.explicit_candidates import (
    extract_explicit_candidates,
)
from core.conversation_semantics.prompt import (
    SEMANTIC_SYSTEM_PROMPT,
    build_semantic_batch_prompt,
)
from core.llm import LLMProvider


class ConversationSemanticAnalyzer:
    ANALYSIS_VERSION = "conversation-v3"

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        max_tokens: int = 1800,
        analysis_version: str = ANALYSIS_VERSION,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max(600, int(max_tokens))
        self.ANALYSIS_VERSION = str(analysis_version or self.ANALYSIS_VERSION)

    async def analyze(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> SemanticBatchPayload:
        response = await self._provider.chat(
            messages=[
                {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
                {"role": "user", "content": build_semantic_batch_prompt(messages)},
            ],
            tools=[],
            model=self._model,
            max_tokens=self._max_tokens,
            disable_thinking=True,
        )
        parsed = json_repair.loads(str(response.content or "{}"))
        payload = dict(parsed) if isinstance(parsed, Mapping) else {}
        # This conservative fallback runs inside the same background semantic
        # batch.  It never writes memory directly and therefore preserves the
        # single governed persistence path.
        deterministic = extract_explicit_candidates(messages)
        for key, items in deterministic.items():
            existing = payload.get(key)
            merged = list(existing) if isinstance(existing, list) else []
            merged.extend(items)
            payload[key] = _dedupe_candidates(key, merged)
        return SemanticBatchPayload.from_mapping(payload)


def _dedupe_candidates(key: str, items: list[object]) -> list[object]:
    """Deduplicate one batch without allowing two writes to the same slot.

    Explicit deterministic candidates are appended after model candidates, so
    a same-message/same-slot memory uses last-wins. Other partitions preserve
    their original first-wins ordering.
    """

    seen: set[tuple[str, str, str]] = set()
    result: list[object] = []
    for item in items:
        if not isinstance(item, Mapping):
            result.append(item)
            continue
        source = str(item.get("source_message_id") or "")
        if key == "memory_candidates":
            attributes = item.get("attributes")
            slot = (
                str(attributes.get("preference_key") or "")
                if isinstance(attributes, Mapping)
                else ""
            )
            identity = (
                str(item.get("tag") or ""),
                source,
                slot or str(item.get("content") or ""),
            )
        else:
            identity = (
                str(item.get("type") or ""),
                source,
                str(item.get("statement") or ""),
            )
        if identity in seen:
            if key == "memory_candidates":
                for index, existing in enumerate(result):
                    if not isinstance(existing, Mapping):
                        continue
                    existing_attributes = existing.get("attributes")
                    existing_slot = (
                        str(existing_attributes.get("preference_key") or "")
                        if isinstance(existing_attributes, Mapping)
                        else ""
                    )
                    existing_identity = (
                        str(existing.get("tag") or ""),
                        str(existing.get("source_message_id") or ""),
                        existing_slot or str(existing.get("content") or ""),
                    )
                    if existing_identity == identity:
                        result[index] = item
                        break
            continue
        seen.add(identity)
        result.append(item)
    return result
