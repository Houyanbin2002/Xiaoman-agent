from __future__ import annotations

"""CLI: ``python -m eval.cli run --dataset eval/datasets/smoke.jsonl``."""

import argparse
import json
from pathlib import Path

from .dataset import load_cases
from .analysis import failure_hotspots, write_hotspot_report
from .runner import EvalHarness, ReplayExecutor, render_markdown, write_report
from .store import EvalResultStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xiaoman-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a JSONL dataset with deterministic replay fixtures")
    run.add_argument("--dataset", required=True, help="versioned JSONL dataset")
    run.add_argument("--report", default="eval/reports/latest.json")
    run.add_argument("--markdown", default="eval/reports/latest.md")
    run.add_argument("--dataset-name", default="local")
    run.add_argument("--version", default="v1")
    run.add_argument("--store", default="", help="optional SQLite history path")
    run.add_argument("--analysis", default="", help="optional failure hotspot JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        cases = load_cases(args.dataset)
        summary = EvalHarness(dataset_name=args.dataset_name, version=args.version).run_sync(cases, ReplayExecutor())
        if args.store:
            store = EvalResultStore(args.store)
            try:
                store.save(summary)
            finally:
                store.close()
        write_report(summary, args.report)
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(summary), encoding="utf-8")
        if args.analysis:
            write_hotspot_report(args.analysis, failure_hotspots(cases, summary))
        print(json.dumps({"pass_rate": summary.pass_rate, "mean_reward": summary.mean_reward, "report": str(args.report)}, ensure_ascii=False))
        return 0 if summary.pass_rate == 1.0 else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
