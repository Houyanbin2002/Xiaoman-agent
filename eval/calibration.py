from __future__ import annotations

"""Judge calibration against reviewed labels; drafts never count as human truth."""

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import load_cases
from .models import AgentRun, EvalCase


def calibrate(cases: list[EvalCase], judge: Any) -> dict[str, Any]:
    rows = []
    confusion = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }
    for case in cases:
        labels = case.metadata["calibration"]
        try:
            scores = judge(case, AgentRun.from_value(case.replay))
            criterion = case.rubric[0]
            actual = scores[criterion.criterion_id] >= criterion.threshold
            expected = bool(labels["expected_pass"])
            reviewed = labels.get("review_status") == "human_reviewed"
            if reviewed:
                name = ("true_" if actual == expected else "false_") + (
                    "positive" if actual else "negative"
                )
                confusion[name] += 1
            rows.append(
                {
                    "case_id": case.case_id,
                    "label_source": labels.get("review_status", "draft"),
                    "expected_pass": expected,
                    "judge_pass": actual,
                    "agreement": actual == expected,
                    "scores": dict(scores),
                    "reasons": getattr(scores, "reasons", {}),
                }
            )
        except Exception as exc:
            rows.append(
                {"case_id": case.case_id, "error": f"{type(exc).__name__}: {exc}"}
            )
    total = sum(confusion.values())
    return {
        "cases": rows,
        "human_reviewed_assessed": total,
        "confusion_human_only": confusion,
        "human_agreement_rate": (
            (confusion["true_positive"] + confusion["true_negative"]) / total
            if total
            else None
        ),
        "draft_agreement_count": sum(
            bool(row.get("agreement"))
            for row in rows
            if row.get("label_source") != "human_reviewed"
        ),
        "note": "Draft labels are author hypotheses, not human calibration. Review disagreements before changing Agent prompts.",
    }


def main() -> int:
    from agent.config import Config
    from .judge import build_judge_from_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/datasets/judge_calibration_v1.jsonl")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--report", default="eval/reports/judge-calibration.json")
    args = parser.parse_args()
    judge = build_judge_from_config(Config.load(args.config))
    report = calibrate(load_cases(args.dataset), judge)
    report["judge_model"] = judge.model
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"calibration report: {path}; human reviewed: {report['human_reviewed_assessed']}"
    )
    return 1 if any(row.get("error") for row in report["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
