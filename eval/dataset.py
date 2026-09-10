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
    seen: set[str] = set()
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
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
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
        turns = case.metadata.get("turns", [])
        if not isinstance(turns, list) or any(
            not isinstance(t, dict) or not str(t.get("input", "")).strip()
            for t in turns
        ):
            raise ValueError(f"invalid scenario turns: {case.case_id}")
        from .scorers import canonical_rubric

        rubric = canonical_rubric(case)
        ids = [c.criterion_id for c in rubric]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate criterion IDs: {case.case_id}")
        acceptance = case.metadata.get("acceptance", {})
        for key in ("objective_checks", "blocked_checks", "delivery_checks"):
            if any(name not in ids for name in acceptance.get(key, [])):
                raise ValueError(f"unknown acceptance check: {case.case_id}.{key}")
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
        if (
            float(item.get("reward", 1.0)) >= min_reward
            and item.get("passed", True)
            and item.get("assessment", {}).get("accepted", True)
        ):
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
