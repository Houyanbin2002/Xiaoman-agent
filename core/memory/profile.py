from __future__ import annotations

from typing import Protocol



class MarkdownSupportApi(Protocol):
    def read_self(self) -> str: ...

    def read_recent_context(self) -> str: ...


class CanonicalLongTermMemory(Protocol):
    """The single governed source used to render long-term prompt memory."""

    def render_prompt(self) -> str: ...


class UnifiedMemoryProfile:
    """Prompt profile backed by governed memory plus Markdown support files.

    Once a canonical source is bound, long-term memory is read directly from
    the governed store. SELF, recent context and history remain Markdown
    support documents because they are not long-term user facts.
    """

    def __init__(self, markdown: MarkdownSupportApi) -> None:
        self._markdown = markdown
        self._canonical: CanonicalLongTermMemory | None = None

    def bind_canonical(self, source: CanonicalLongTermMemory) -> None:
        self._canonical = source

    def read_long_term(self) -> str:
        source = self._canonical
        return source.render_prompt().strip() if source is not None else ""

    def read_self(self) -> str:
        return self._markdown.read_self()

    def read_recent_context(self) -> str:
        return self._markdown.read_recent_context()

    def get_memory_context(self) -> str:
        content = self.read_long_term().strip()
        return f"## Long-term Memory\n{content}" if content else ""
