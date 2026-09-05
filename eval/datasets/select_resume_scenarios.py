from __future__ import annotations

"""Select the fixed cross-slice scenario set used by the resume benchmark."""

import argparse

from eval.dataset import load_cases, write_cases

CASE_IDS = (
    "memory.preference.language",
    "memory.correction.language",
    "memory.correction.project",
    "execution.recovery.config",
    "execution.recovery.search",
    "execution.recovery.fetch",
    "workflow.resume.report",
    "workflow.resume.migration",
    "workflow.resume.research",
    "proactive.policy.dnd",
    "proactive.policy.deadline",
    "proactive.policy.feedback",
    "context.compaction.tool_result",
    "context.compaction.memory",
    "context.compaction.cache",
    "safety.policy.delete",
    "safety.policy.credential",
    "safety.policy.payment",
    "schedule.intent.one_off",
    "schedule.intent.daily",
    "schedule.intent.timezone",
    "retrieval.history.recent_project",
    "retrieval.history.last_decision",
    "retrieval.history.avoid_noise",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="eval/datasets/regression_v1_judge.jsonl",
    )
    parser.add_argument(
        "--output",
        default="eval/datasets/resume_real_scenarios_v1.jsonl",
    )
    args = parser.parse_args(argv)
    by_id = {case.case_id: case for case in load_cases(args.source)}
    missing = [case_id for case_id in CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError(f"source dataset is missing resume case(s): {missing}")
    write_cases(args.output, [by_id[case_id] for case_id in CASE_IDS])
    print(f"wrote {len(CASE_IDS)} resume scenarios: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
