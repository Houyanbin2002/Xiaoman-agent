from __future__ import annotations

from collections.abc import Mapping, Sequence

import json_repair

from core.conversation_semantics.models import SemanticBatchPayload
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
        return SemanticBatchPayload.from_mapping(parsed)
