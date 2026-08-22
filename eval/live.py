from __future__ import annotations

"""Run the versioned dataset against a real, isolated AgentLoop.

This command is intentionally separate from ``eval.cli`` so CI remains offline
by default. The workspace is isolated from the user's normal Xiaoman workspace.
"""

import argparse
import asyncio
from pathlib import Path

from agent.config import Config
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources

from .analysis import failure_hotspots, write_hotspot_report
from .dataset import load_cases
from .fixtures import LiveEvalFixtureManager
from .judge import build_judge_from_config
from .publishers import LangfuseScorePublisher, publish_best_effort
from .runner import EvalHarness, ProcessDirectExecutor, render_markdown, write_report
from .store import EvalResultStore


async def run_live(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    selected_cases = set(args.case or ())
    if selected_cases:
        cases = [case for case in cases if case.case_id in selected_cases]
        missing = sorted(selected_cases - {case.case_id for case in cases})
        if missing:
            raise ValueError(f"unknown --case id(s): {missing}")
    if args.tag:
        cases = [case for case in cases if args.tag in case.tags]
    if args.limit:
        cases = cases[: max(1, int(args.limit))]
    config = Config.load(args.config)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    http_resources = SharedHttpResources()
    runtime = build_core_runtime(config, workspace, http_resources)
    try:
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
        judge = build_judge_from_config(config, model=args.judge_model) if args.judge else None
        summary = await EvalHarness(
            dataset_name=args.dataset_name,
            version=args.version,
            judge=judge,
        ).run(cases, executor)
        write_report(summary, args.report)
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(summary), encoding="utf-8")
        if args.analysis:
            write_hotspot_report(args.analysis, failure_hotspots(cases, summary))
        if args.store:
            store = EvalResultStore(args.store)
            try:
                store.save(summary)
            finally:
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
            f"live eval finished: {summary.passed}/{summary.total} passed, "
            f"reward={summary.mean_reward:.3f}, report={args.report}"
        )
        return 0 if summary.pass_rate == 1.0 else 1
    finally:
        await runtime.stop()
        await http_resources.aclose()


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
    parser.add_argument("--limit", type=int, default=6, help="number of cases; omit to run the whole dataset")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="score explicit evaluator=llm Rubric criteria with the configured fast model",
    )
    parser.add_argument("--judge-model", default="", help="optional OpenAI-compatible judge model override")
    parser.add_argument(
        "--publish-langfuse",
        action="store_true",
        help="publish Rubric scores to Langfuse; requires configured Langfuse credentials",
    )
    parser.add_argument("--report", default="eval/reports/live.json")
    parser.add_argument("--markdown", default="eval/reports/live.md")
    parser.add_argument("--analysis", default="eval/reports/live-hotspots.json")
    parser.add_argument("--store", default="data/eval-live.sqlite")
    args = parser.parse_args(argv)
    return asyncio.run(run_live(args))


if __name__ == "__main__":
    raise SystemExit(main())
