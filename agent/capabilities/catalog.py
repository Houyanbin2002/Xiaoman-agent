from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent.skills import SkillsLoader
    from agent.tools.registry import ToolRegistry

CapabilityKind = Literal["tool", "skill"]

_ASCII_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_QUERY_STOP_TERMS = {
    "一下",
    "为什么",
    "什么",
    "可以",
    "如何",
    "帮我",
    "怎么",
    "当前",
    "现在",
    "这个",
    "那个",
    "进行",
    "需要",
}


@dataclass(frozen=True)
class CapabilityRecord:
    """A small, indexable view of one capability.

    It deliberately contains metadata only. Tool schemas and SKILL.md bodies stay in
    their owning subsystems and are loaded only after a match is selected.
    """

    kind: CapabilityKind
    name: str
    description: str
    source_type: str
    source_name: str = ""
    available: bool = True
    always_on: bool = False
    risk: str = "read-only"
    search_hint: str = ""
    parameter_names: tuple[str, ...] = ()
    when_to_use: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class CapabilityMatch:
    record: CapabilityRecord
    score: float
    matched_terms: tuple[str, ...]
    exact_name: bool = False

    def as_result(self) -> dict[str, object]:
        record = self.record
        result: dict[str, object] = {
            "kind": record.kind,
            "name": record.name,
            "summary": record.description[:160],
            "why_matched": (
                ["名称:精确匹配"]
                if self.exact_name
                else [f"相关词:{term}" for term in self.matched_terms[:5]]
            ),
            "source": {
                "type": record.source_type,
                "name": record.source_name,
            },
            "available": record.available,
        }
        if record.kind == "tool":
            result.update(risk=record.risk, always_on=record.always_on)
        elif not record.available:
            result["availability"] = "依赖未满足，暂不可加载"
        return result


class CapabilityCatalog:
    """Build and search a live catalog spanning tools, MCP services and skills.

    The catalog is rebuilt from lightweight metadata on every search. This is
    intentional: installing or removing a Skill/MCP server becomes visible
    immediately and cannot leave a stale global index behind.
    """

    def __init__(
        self,
        tools: "ToolRegistry",
        skills: "SkillsLoader | None" = None,
    ) -> None:
        self._tools = tools
        self._skills = skills

    def records(self) -> list[CapabilityRecord]:
        get_documents = getattr(self._tools, "get_documents", None)
        documents = get_documents() if callable(get_documents) else []
        records = [
            CapabilityRecord(
                kind="tool",
                name=doc.name,
                description=doc.description,
                source_type=doc.source_type,
                source_name=doc.source_name,
                always_on=doc.always_on,
                risk=doc.risk,
                search_hint=doc.search_hint or "",
                parameter_names=doc.parameter_names,
            )
            for doc in documents
            if doc.name != "tool_search"
        ]
        if self._skills is None:
            return records
        for skill in self._skills.list_skill_records(filter_unavailable=False):
            records.append(
                CapabilityRecord(
                    kind="skill",
                    name=skill.name,
                    description=skill.description,
                    source_type=skill.source,
                    source_name=skill.source_id,
                    available=skill.available,
                    search_hint=skill.display_name,
                    when_to_use=skill.when_to_use,
                )
            )
        return records

    def get(self, kind: CapabilityKind, name: str) -> CapabilityRecord | None:
        return next(
            (
                record
                for record in self.records()
                if record.kind == kind and record.name == name
            ),
            None,
        )

    def counts(self) -> dict[str, int]:
        result = {"tool": 0, "skill": 0, "mcp": 0, "system": 0}
        for record in self.records():
            result[record.kind] += 1
            if record.source_type == "mcp":
                result["mcp"] += 1
            elif record.source_type == "plugin":
                result["system"] += 1
        return result

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        kinds: set[CapabilityKind] | None = None,
        allowed_risk: list[str] | None = None,
        excluded_tool_names: set[str] | None = None,
        include_unavailable: bool = True,
    ) -> list[CapabilityMatch]:
        query = query.strip()
        if not query:
            return []
        records = [
            record
            for record in self.records()
            if (kinds is None or record.kind in kinds)
            and (include_unavailable or record.available)
            and not (
                record.kind == "tool"
                and excluded_tool_names
                and record.name in excluded_tool_names
            )
            and not (
                record.kind == "tool"
                and allowed_risk
                and record.risk not in set(allowed_risk)
            )
        ]
        if not records:
            return []

        query_lower = query.lower()
        query_terms = [
            term for term in _tokenize(query) if term not in _QUERY_STOP_TERMS
        ]
        if not query_terms:
            return []

        weighted_docs = [_weighted_tokens(record) for record in records]
        document_frequency: Counter[str] = Counter()
        for tokens in weighted_docs:
            document_frequency.update(set(tokens))

        avg_len = sum(len(tokens) for tokens in weighted_docs) / len(weighted_docs)
        matches: list[CapabilityMatch] = []
        for record, tokens in zip(records, weighted_docs, strict=True):
            name_lower = record.name.lower()
            exact_name = query_lower == name_lower or query_lower == f"${name_lower}"
            frequencies = Counter(tokens)
            score = _bm25_score(
                query_terms,
                frequencies,
                document_frequency,
                document_count=len(records),
                doc_length=len(tokens),
                average_length=avg_len,
            )
            if exact_name:
                score += 30.0
            elif name_lower in query_lower or query_lower in name_lower:
                score += 10.0

            phrase_haystack = " ".join(
                part
                for part in (
                    record.description,
                    record.when_to_use,
                    record.search_hint,
                )
                if part
            ).lower()
            if len(query_lower) >= 2 and query_lower in phrase_haystack:
                score += 6.0

            doc_terms = set(tokens)
            matched = tuple(
                sorted(set(query_terms) & doc_terms, key=lambda x: (-len(x), x))
            )
            if (score > 0 and matched) or exact_name:
                matches.append(
                    CapabilityMatch(
                        record=record,
                        score=round(score, 4),
                        matched_terms=matched,
                        exact_name=exact_name,
                    )
                )

        matches.sort(
            key=lambda item: (
                -item.score,
                0 if item.record.kind == "tool" else 1,
                item.record.name,
            )
        )
        return matches[: max(1, top_k)]


def _tokenize(text: str) -> list[str]:
    lowered = text.lower().replace("_", " ").replace("-", " ").replace(":", " ")
    tokens = _ASCII_WORD.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        if len(run) <= 8:
            tokens.append(run)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return [token for token in tokens if token]


def _weighted_tokens(record: CapabilityRecord) -> list[str]:
    tokens: list[str] = []
    tokens.extend(_tokenize(record.name) * 5)
    tokens.extend(_tokenize(record.search_hint) * 3)
    tokens.extend(_tokenize(record.description) * 2)
    tokens.extend(_tokenize(record.when_to_use) * 2)
    for parameter in record.parameter_names:
        tokens.extend(_tokenize(parameter) * 2)
    tokens.extend(_tokenize(record.source_type))
    tokens.extend(_tokenize(record.source_name))
    return tokens


def _bm25_score(
    query_terms: list[str],
    frequencies: Counter[str],
    document_frequency: Counter[str],
    *,
    document_count: int,
    doc_length: int,
    average_length: float,
) -> float:
    score = 0.0
    k1 = 1.5
    b = 0.75
    normalizer = k1 * (1 - b + b * doc_length / max(1.0, average_length))
    for term in set(query_terms):
        frequency = frequencies.get(term, 0)
        if not frequency:
            continue
        frequency_in_docs = document_frequency.get(term, 0)
        inverse_frequency = math.log(
            1 + (document_count - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5)
        )
        score += inverse_frequency * frequency * (k1 + 1) / (frequency + normalizer)
    return score
