from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from core.personal.memory_scope import memory_boundary, preference_slot

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
    ANALYSIS_VERSION = "conversation-v4"

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        max_tokens: int = 1800,
        analysis_version: str = ANALYSIS_VERSION,
        activity_context_provider: Callable[[], list[dict[str, object]]] | None = None,
    ) -> None:
        self._provider = provider
        self._activity_context_provider = activity_context_provider
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
                {
                    "role": "user",
                    "content": build_semantic_batch_prompt(
                        messages,
                        existing_activities=(
                            self._activity_context_provider()
                            if self._activity_context_provider
                            else []
                        ),
                    ),
                },
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
            if key == "memory_candidates":
                merged = _merge_memory_fallback(merged, items)
            else:
                merged.extend(items)
            payload[key] = _dedupe_candidates(key, merged)
        return SemanticBatchPayload.from_mapping(payload)


def _merge_memory_fallback(
    model_items: list[object],
    fallback_items: list[dict[str, object]],
) -> list[object]:
    """Fill missing slots without inventing a second, broader scope.

    A fallback has no semantic scope parser. If the model already extracted
    the same source/slot for the same subject with a narrower boundary, keep
    that candidate and don't append a global rule. Distinct scenes and subjects
    remain independent; model-supplied origin labels are not authority.
    """
    normalized: list[object] = []
    for item in model_items:
        if not isinstance(item, Mapping):
            normalized.append(item)
            continue
        item = dict(item)
        item["subject"], item["scope"] = memory_boundary(
            item.get("subject"), item.get("scope")
        )
        attrs = item.get("attributes")
        attributes = dict(attrs) if isinstance(attrs, Mapping) else {}
        slot = preference_slot(item, attributes)
        if slot:
            item["attributes"] = {**attributes, "preference_key": slot}
        normalized.append(item)
    for fallback in fallback_items:
        attrs = fallback.get("attributes")
        slot = (
            str(attrs.get("preference_key") or "") if isinstance(attrs, Mapping) else ""
        )
        subject, scope = memory_boundary(fallback.get("subject"), fallback.get("scope"))
        source = fallback.get("source_message_id")
        if scope:
            # A narrowly recognized literal condition must not become global
            # merely because the model omitted its boundary.
            for item in normalized:
                if (
                    isinstance(item, dict)
                    and item.get("source_message_id") == source
                    and isinstance(item.get("attributes"), Mapping)
                    and item["attributes"].get("preference_key") == slot
                    and item.get("subject") == subject
                    and not item.get("scope")
                ):
                    item["scope"] = scope
        scoped_match = any(
            isinstance(item, Mapping)
            and item.get("source_message_id") == source
            and isinstance(item.get("attributes"), Mapping)
            and item["attributes"].get("preference_key") == slot
            and item.get("subject") == subject
            and item.get("scope") != scope
            for item in normalized
        )
        if not scoped_match:
            normalized.append({**fallback, "subject": subject, "scope": scope})
    return normalized


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
                slot or str(item.get("tag") or ""),
                source,
                repr(
                    (
                        memory_boundary(item.get("subject"), item.get("scope")),
                        "" if slot else str(item.get("content") or ""),
                    )
                ),
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
                        existing_slot or str(existing.get("tag") or ""),
                        str(existing.get("source_message_id") or ""),
                        repr(
                            (
                                memory_boundary(
                                    existing.get("subject"), existing.get("scope")
                                ),
                                (
                                    ""
                                    if existing_slot
                                    else str(existing.get("content") or "")
                                ),
                            )
                        ),
                    )
                    if existing_identity == identity:
                        result[index] = item
                        break
            continue
        seen.add(identity)
        result.append(item)
    return result
