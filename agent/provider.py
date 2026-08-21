"""Deprecated import surface for external extensions.

Runtime code uses ``core.llm`` contracts and bootstrap constructs the concrete
adapter from ``infra.providers``. This module remains only for third-party
plugins that have not migrated their imports yet.
"""

from core.llm import ContentSafetyError, ContextLengthError, LLMResponse, ToolCall
from infra.providers.llm_provider import LLMProvider

__all__ = [
    "ContentSafetyError",
    "ContextLengthError",
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
]
