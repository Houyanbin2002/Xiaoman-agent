from __future__ import annotations

from dataclasses import dataclass
import re


_CITATION_MARKER_RE = re.compile(
    r"[ \t]*§cited:\[[^\]\r\n§]*\]§[ \t]*",
    re.IGNORECASE,
)


@dataclass
class ResponseMetadata:
    raw_text: str


@dataclass
class ParsedResponse:
    clean_text: str
    metadata: ResponseMetadata


def parse_response(
    raw_text: str,
    *,
    tool_chain: list[dict[str, object]],
) -> ParsedResponse:
    clean_text = _CITATION_MARKER_RE.sub("", raw_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
    return ParsedResponse(
        clean_text=clean_text,
        metadata=ResponseMetadata(raw_text=raw_text),
    )
