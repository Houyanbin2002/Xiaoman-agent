from __future__ import annotations

"""Versioned JSONL dataset helpers and hard-case mining utilities."""

import json
from pathlib import Path
from typing import Iterable

from .models import EvalCase


def load_cases(path: str | Path) -> list[EvalCase]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    cases: list[EvalCase] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {source}:{line_number}") from exc
        case = EvalCase.from_dict(value)
        if not case.case_id:
            raise ValueError(f"missing case_id at {source}:{line_number}")
        cases.append(case)
    return cases


def write_cases(path: str | Path, cases: Iterable[EvalCase]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(case.to_dict(), ensure_ascii=False, separators=(",", ":"))
        for case in cases
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def mine_hard_cases(
    results: Iterable[dict],
    *,
    min_reward: float = 0.6,
    tags: set[str] | None = None,
) -> list[dict]:
    """Select de-identified candidate records for human review.

    This function intentionally does not promote candidates automatically. A
    reviewer must turn a candidate into a golden case before it enters CI.
    """
    candidates: list[dict] = []
    for item in results:
        if float(item.get("reward", 1.0)) >= min_reward and item.get("passed", True):
            continue
        if tags and not tags.intersection(set(item.get("tags", ()) or ())):
            continue
        candidates.append(
            {
                "source_case_id": item.get("case_id", ""),
                "reason": item.get("error") or "low_reward_or_failed",
                "run": item.get("run", {}),
                "review_status": "needs_human_review",
            }
        )
    return candidates
