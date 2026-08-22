from __future__ import annotations

"""Turn failed batch cases into actionable optimization buckets."""

from collections import Counter
from typing import Iterable

from .models import EvalCase, EvalSummary


def failure_hotspots(cases: Iterable[EvalCase], summary: EvalSummary) -> dict[str, object]:
    """Group failed/low-reward cases by slice, tag and declared failure mode."""
    case_by_id = {case.case_id: case for case in cases}
    failed = [result for result in summary.results if not result.passed or result.reward < 0.8]
    slices: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    cases_out: list[dict[str, object]] = []
    for result in failed:
        case = case_by_id.get(result.case_id)
        if case is None:
            continue
        slice_name = str(case.metadata.get("slice") or (case.tags[0] if case.tags else "unknown"))
        slices[slice_name] += 1
        tags.update(case.tags)
        failure_modes = [str(item) for item in case.metadata.get("failure_modes", ())]
        requires = [str(item) for item in case.metadata.get("requires", ())]
        modes.update(failure_modes)
        cases_out.append(
            {
                "case_id": result.case_id,
                "slice": slice_name,
                "reward": result.reward,
                "failure_modes": failure_modes,
                "requires": requires,
                "fixture_dependent": bool(requires),
                "failed_scores": [score.name for score in result.scores if not score.passed],
                "error": result.error,
            }
        )
    return {
        "failed_or_low_reward": len(failed),
        "hot_slices": slices.most_common(),
        "hot_tags": tags.most_common(),
        "hot_failure_modes": modes.most_common(),
        "cases": cases_out,
    }


def write_hotspot_report(path: str, data: dict[str, object]) -> None:
    import json
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
