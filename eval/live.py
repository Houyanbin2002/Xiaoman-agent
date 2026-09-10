from __future__ import annotations

"""Run the versioned dataset against a real, isolated AgentLoop.

This command is intentionally separate from ``eval.cli`` so CI remains offline
by default. The workspace is isolated from the user's normal Xiaoman workspace.
"""

import argparse
import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path

from agent.config import Config
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources

from .analysis import failure_hotspots, write_hotspot_report
from .dataset import load_cases
from .experiment import aggregate_repeats, manifest, select_split
from .models import AgentRun, CaseResult
from .outcomes import acceptance_exit_code
from .fixtures import LiveEvalFixtureManager
from .judge import build_judge_from_config
from .publishers import LangfuseScorePublisher, publish_best_effort
from .runner import EvalHarness, ProcessDirectExecutor, render_markdown, write_report
from .store import EvalResultStore


async def run_live(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    if args.repeats < 1 or args.case_timeout <= 0 or args.limit < 0:
        raise ValueError("repeats/timeout must be positive; limit must be nonnegative")
    selected_ids = set(args.case or ())
    if selected_ids:
        missing = selected_ids - {c.case_id for c in cases}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
        cases = [c for c in cases if c.case_id in selected_ids]
    cases = select_split(cases, args.split, args.holdout_percent)
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("selection contains no cases")
    config = Config.load(args.config)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    # New experiment root prevents memory/activity/scheduler leakage, even
    # between repeats. Keep raw evidence on disk instead of deleting it.
    root = Path(tempfile.mkdtemp(prefix="experiment-", dir=workspace))
    judge = (
        build_judge_from_config(config, model=args.judge_model) if args.judge else None
    )
    run_manifest = manifest(
        cases,
        config,
        split=args.split,
        repeats=args.repeats,
        judge_model=judge.model if judge else "disabled",
        timeout=args.case_timeout,
    )
    run_manifest["workspace"] = str(root)
    harness = EvalHarness(
        dataset_name=args.dataset_name, version=args.version, judge=judge
    )
    summaries = []
    store = EvalResultStore(args.store) if args.store else None
    try:
        for repeat in range(1, args.repeats + 1):
            results = []
            for index, case in enumerate(cases, 1):
                print(
                    f"[repeat {repeat}/{args.repeats}] {index}/{len(cases)} {case.case_id}",
                    flush=True,
                )
                case_root = root / f"r{repeat}" / f"case-{index}"
                case_root.mkdir(parents=True)
                http_resources = SharedHttpResources()
                runtime = None
                try:
                    runtime = build_core_runtime(config, case_root, http_resources)

                    async def settle(_session_key: str) -> None:
                        await runtime.event_bus.drain()
                        semantics = runtime.conversation_semantics
                        if semantics is not None:
                            await semantics.batcher.drain()
                        await runtime.event_bus.drain()

                    executor = ProcessDirectExecutor(
                        runtime.loop,
                        trace_store=runtime.trace_store,
                        session_prefix=args.session_prefix,
                        settle=settle,
                        memory_runtime=runtime.memory_runtime,
                        fixture_manager=LiveEvalFixtureManager(runtime),
                    )

                    async def timed_executor(current):
                        return await asyncio.wait_for(
                            executor(current), timeout=args.case_timeout
                        )

                    result = await harness.run_case(case, timed_executor)
                except Exception as exc:
                    # Setup failure is evidence of untested capability, not a pass.
                    result = CaseResult(
                        case.case_id,
                        case.title,
                        False,
                        0.0,
                        (),
                        AgentRun(status="error"),
                        error=f"setup: {type(exc).__name__}: {exc}",
                    )
                finally:
                    try:
                        if runtime is not None:
                            await runtime.stop()
                    finally:
                        await http_resources.aclose()
                results.append(result)
                print(
                    f"  accepted={result.assessment.get('accepted', False)} "
                    f"objective={result.assessment.get('objective_outcome', 'unassessed')}",
                    flush=True,
                )
                # Incremental artifact survives interruption; incomplete batch is explicit.
                partial = harness.summarize(cases[:index], tuple(results))
                write_report(
                    replace(
                        partial,
                        manifest={
                            **run_manifest,
                            "partial": index < len(cases),
                            "repeat": repeat,
                        },
                    ),
                    Path(args.report).with_suffix(f".r{repeat}.json"),
                )
            current = replace(
                harness.summarize(cases, tuple(results)),
                manifest={**run_manifest, "repeat": repeat},
            )
            summaries.append(current)
            path = Path(args.markdown).with_suffix(f".r{repeat}.md")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_markdown(current), encoding="utf-8")
            if store is not None:
                store.save(current)
        summary = aggregate_repeats(cases, summaries, run_manifest)
        write_report(summary, args.report)
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(summary), encoding="utf-8")
        if args.analysis:
            expanded = [
                replace(c, case_id=f"{c.case_id}::r{n}")
                for n in range(1, args.repeats + 1)
                for c in cases
            ]
            write_hotspot_report(args.analysis, failure_hotspots(expanded, summary))
    finally:
        if store is not None:
            store.close()
    if args.publish_langfuse:
        langfuse_config = config.observability.langfuse
        if not langfuse_config.public_key or not langfuse_config.secret_key:
            raise RuntimeError(
                "--publish-langfuse requires LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY"
            )
        from langfuse import Langfuse

        client = Langfuse(
            public_key=langfuse_config.public_key,
            secret_key=langfuse_config.secret_key,
            base_url=langfuse_config.base_url,
            environment=langfuse_config.environment,
            sample_rate=langfuse_config.sample_rate,
            flush_at=langfuse_config.flush_at,
            flush_interval=langfuse_config.flush_interval_seconds,
            debug=langfuse_config.debug,
        )
        try:
            errors = publish_best_effort(summary, [LangfuseScorePublisher(client)])
            client.flush()
        finally:
            shutdown = getattr(client, "shutdown", None)
            if callable(shutdown):
                shutdown()
        if errors:
            raise RuntimeError("Langfuse score publish failed: " + "; ".join(errors))

    print(
        f"live eval: {int(summary.metrics['rubric_acceptance_cases'])}/{summary.total} accepted; "
        f"{len(cases)} unique scenarios; report={args.report}"
    )
    return acceptance_exit_code(summary, args.gate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xiaoman-eval-live",
        description="run personal-assistant cases against an isolated real AgentLoop",
    )
    parser.add_argument("--dataset", default="eval/datasets/regression_v1.jsonl")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--workspace", default="data/eval-live-workspace")
    parser.add_argument("--dataset-name", default="personal-assistant-live")
    parser.add_argument("--version", default="regression-v1-live")
    parser.add_argument("--session-prefix", default="eval-live")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="run one case id; repeat to select multiple cases",
    )
    parser.add_argument("--tag", default="", help="run only cases containing this tag")
    parser.add_argument(
        "--holdout-percent",
        type=int,
        default=20,
        help="holdout percentage for cases without explicit split (1-99)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeat the same cases; repeated reports use .rN suffixes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="number of cases (default 6); use 0 for all selected cases",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="score explicit evaluator=llm Rubric criteria with the configured fast model",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="optional OpenAI-compatible judge model override",
    )
    parser.add_argument(
        "--publish-langfuse",
        action="store_true",
        help="publish Rubric scores to Langfuse; requires configured Langfuse credentials",
    )
    parser.add_argument("--report", default="eval/reports/live.json")
    parser.add_argument("--markdown", default="eval/reports/live.md")
    parser.add_argument("--analysis", default="eval/reports/live-hotspots.json")
    parser.add_argument("--store", default="data/eval-live.sqlite")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--case-timeout", type=float, default=300)
    parser.add_argument("--gate", choices=["acceptance", "hard"], default="acceptance")
    args = parser.parse_args(argv)
    return asyncio.run(run_live(args))


if __name__ == "__main__":
    raise SystemExit(main())
