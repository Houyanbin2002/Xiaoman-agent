from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from core.personal.models import MemoryData, MemoryKind, PersonalRecord


_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCTUATION = re.compile(r"[\s,，.。!！?？;；:：]+$")


class MemorySemanticRelation(StrEnum):
    SAME = "same"
    REFINE = "refine"
    CONTRADICT = "contradict"
    INDEPENDENT = "independent"


@dataclass(frozen=True)
class MemoryIdentity:
    record_key: str
    quality: str
    subject: str = ""
    predicate: str = ""
    value: str = ""
    scope: str = ""

    @property
    def is_strong(self) -> bool:
        return self.quality in {"structured", "explicit"}


@dataclass(frozen=True)
class MemoryReconciliation:
    identity: MemoryIdentity
    relation: MemorySemanticRelation
    reason: str


def normalize_memory_text(value: object) -> str:
    """Normalize text for deterministic identity checks without losing negation."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = _WHITESPACE.sub(" ", text).strip()
    return _TRAILING_PUNCTUATION.sub("", text)


class MemoryReconciler:
    """Deterministic identity and relation rules for governed personal memory.

    The model may propose structured facts, but this class owns identity and
    overwrite semantics. It intentionally does not use fuzzy similarity for a
    destructive decision.
    """

    def identity(
        self,
        memory: MemoryData,
        *,
        summary: str,
        supplied_key: str = "",
    ) -> MemoryIdentity:
        explicit_key = supplied_key.strip()
        if explicit_key:
            return MemoryIdentity(
                record_key=explicit_key,
                quality="explicit",
                subject=normalize_memory_text(memory.subject),
                predicate=normalize_memory_text(memory.predicate),
                value=normalize_memory_text(memory.value),
                scope=normalize_memory_text(memory.scope),
            )

        subject = normalize_memory_text(memory.subject)
        predicate = normalize_memory_text(memory.predicate)
        value = normalize_memory_text(memory.value)
        scope = normalize_memory_text(memory.scope)
        if subject and predicate and value:
            material = json.dumps(
                [
                    subject,
                    predicate,
                    scope,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
            return MemoryIdentity(
                record_key=f"memory:slot:{digest}",
                quality="structured",
                subject=subject,
                predicate=predicate,
                value=value,
                scope=scope,
            )

        normalized_summary = normalize_memory_text(summary)
        digest = hashlib.sha256(normalized_summary.encode("utf-8")).hexdigest()[:16]
        return MemoryIdentity(
            record_key=f"memory:{memory.kind.value}:{digest}",
            quality="weak",
        )

    def classify(
        self,
        existing: PersonalRecord | None,
        *,
        candidate: MemoryData,
        identity: MemoryIdentity,
        explicit_replaces: bool = False,
    ) -> MemoryReconciliation:
        if existing is None:
            return MemoryReconciliation(
                identity=identity,
                relation=MemorySemanticRelation.INDEPENDENT,
                reason="no_active_fact_in_slot",
            )

        existing_identity = self._identity_from_record(existing)
        existing_content = normalize_memory_text(
            existing.data.get("content") or existing.summary
        )
        candidate_content = normalize_memory_text(candidate.content)
        same_slot = (
            identity.record_key == existing.record_key
            and identity.is_strong
            and existing_identity.is_strong
        )
        if same_slot:
            if (
                identity.value
                and existing_identity.value
                and identity.value == existing_identity.value
            ):
                old_attributes = self._normalized_attributes(
                    existing.data.get("attributes")
                )
                new_attributes = self._normalized_attributes(candidate.attributes)
                if any(
                    key in new_attributes and new_attributes[key] != old_value
                    for key, old_value in old_attributes.items()
                ):
                    return MemoryReconciliation(
                        identity=identity,
                        relation=MemorySemanticRelation.CONTRADICT,
                        reason="structured_attribute_conflict",
                    )
                relation = (
                    MemorySemanticRelation.REFINE
                    if old_attributes.items() < new_attributes.items()
                    else MemorySemanticRelation.SAME
                )
                return MemoryReconciliation(
                    identity=identity,
                    relation=relation,
                    reason=(
                        "same_fact_with_additional_attributes"
                        if relation == MemorySemanticRelation.REFINE
                        else "same_structured_fact"
                    ),
                )
            if existing_content and existing_content == candidate_content:
                return MemoryReconciliation(
                    identity=identity,
                    relation=MemorySemanticRelation.SAME,
                    reason="same_normalized_content",
                )
            return MemoryReconciliation(
                identity=identity,
                relation=MemorySemanticRelation.CONTRADICT,
                reason="same_fact_slot_with_different_value",
            )

        if existing_content and existing_content == candidate_content:
            return MemoryReconciliation(
                identity=identity,
                relation=MemorySemanticRelation.SAME,
                reason="same_normalized_content",
            )

        if explicit_replaces:
            return MemoryReconciliation(
                identity=identity,
                relation=MemorySemanticRelation.CONTRADICT,
                reason="explicit_replacement_target",
            )

        return MemoryReconciliation(
            identity=identity,
            relation=MemorySemanticRelation.INDEPENDENT,
            reason="different_fact_slot",
        )

    @staticmethod
    def _identity_from_record(record: PersonalRecord) -> MemoryIdentity:
        data = record.data
        return MemoryIdentity(
            record_key=record.record_key,
            quality=str(data.get("identity_quality") or "weak"),
            subject=normalize_memory_text(data.get("subject")),
            predicate=normalize_memory_text(data.get("predicate")),
            value=normalize_memory_text(data.get("value")),
            scope=normalize_memory_text(data.get("scope")),
        )

    @staticmethod
    def _normalized_attributes(raw: object) -> dict[str, str]:
        if not isinstance(raw, Mapping):
            return {}
        return {
            normalize_memory_text(key): normalize_memory_text(value)
            for key, value in raw.items()
            if normalize_memory_text(key)
        }


def normalized_memory_kind(value: MemoryKind) -> MemoryKind:
    return MemoryKind.HISTORICAL_EVENT if value == MemoryKind.EPISODE else value
