from __future__ import annotations

"""Frozen selection, reproducibility metadata and repeat-aware reporting."""

import hashlib
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import EvalCase, EvalSummary
from .runner import EvalHarness


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def select_split(
    cases: list[EvalCase], split: str, percent: int = 20
) -> list[EvalCase]:
    if split not in {"dev", "holdout", "all"} or not 1 <= percent <= 99:
        raise ValueError("split must be dev/holdout/all; holdout percent must be 1..99")
    selected = []
    for case in cases:
        # Keep paraphrases of a scenario together. Explicit frozen split takes priority.
        family = case.metadata.get("family", case.case_id)
        assigned = case.metadata.get("split") or (
            "holdout" if int(digest(family)[:8], 16) % 100 < percent else "dev"
        )
        if assigned not in {"dev", "holdout"}:
            raise ValueError(f"invalid split for {case.case_id}")
        if split == "all" or assigned == split:
            selected.append(case)
    return selected


def manifest(
    cases: list[EvalCase],
    config: Any,
    *,
    split: str,
    repeats: int,
    judge_model: str,
    timeout: float,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    code_files = [
        path
        for folder in (
            "eval",
            "agent",
            "core",
            "prompts",
            "bootstrap",
            "infra",
            "memory2",
            "plugins",
            "proactive_v2",
            "session",
            "bus",
        )
        for path in (root / folder).rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    # Never serialize credentials/endpoints; fingerprints are non-reversible.
    config_value = (
        config.to_dict() if callable(getattr(config, "to_dict", None)) else vars(config)
    )
    return {
        "schema_version": 2,
        "git_revision": revision,
        "code_sha256": digest(
            {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(code_files)
            }
        ),
        "dataset_sha256": digest([c.to_dict() for c in cases]),
        "config_sha256": digest(config_value),
        "case_ids": [c.case_id for c in cases],
        "case_hashes": {c.case_id: digest(c.to_dict()) for c in cases},
        "model": config.model,
        "judge_model": judge_model,
        "judge_sha256": hashlib.sha256(
            (root / "eval/judge.py").read_bytes()
        ).hexdigest(),
        "split": split,
        "repeats": repeats,
        "case_timeout_seconds": timeout,
        "coverage_level": "kernel_integration",
        "isolation": "fresh_runtime_per_case_per_repeat",
        "scope_note": "Real model + kernel, not browser/Gateway delivery or process-crash recovery.",
    }


def aggregate_repeats(
    cases: list[EvalCase], summaries: list[EvalSummary], run_manifest: dict[str, Any]
) -> EvalSummary:
    expanded, results = [], []
    stability = {}
    for number, summary in enumerate(summaries, 1):
        if [r.case_id for r in summary.results] != [c.case_id for c in cases]:
            raise ValueError("repeat case set/order changed")
        for case, result in zip(cases, summary.results):
            trial_id = f"{case.case_id}::r{number}"
            expanded.append(replace(case, case_id=trial_id))
            results.append(replace(result, case_id=trial_id))
            stability.setdefault(case.case_id, []).append(
                bool(result.assessment.get("accepted"))
            )
    if not summaries:
        raise ValueError("no repeats")
    summary = EvalHarness(
        dataset_name=summaries[0].dataset, version=summaries[0].version
    ).summarize(expanded, tuple(results))
    total, accepted = summary.total, int(summary.metrics["rubric_acceptance_cases"])
    # Descriptive only: repeated trials are correlated and this is not population quality.
    p, z = accepted / total, 1.96
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    metrics = {
        **summary.metrics,
        "unique_cases": float(len(cases)),
        "repeat_count": float(len(summaries)),
        "all_repeats_accepted_cases": float(sum(all(v) for v in stability.values())),
        "unstable_cases": float(sum(any(v) and not all(v) for v in stability.values())),
        "acceptance_trial_rate": p,
        "acceptance_wilson95_low_descriptive": max(0, center - spread),
        "acceptance_wilson95_high_descriptive": min(1, center + spread),
    }
    for name in ("latency_ms",):
        values = sorted(
            getattr(r.run, name) for r in results if getattr(r.run, name) is not None
        )
        if values:
            metrics[f"{name}_p50"] = values[math.ceil(0.5 * len(values)) - 1]
            metrics[f"{name}_p95"] = values[math.ceil(0.95 * len(values)) - 1]
    return replace(
        summary,
        metrics=metrics,
        manifest={**run_manifest, "trial_acceptance": stability},
    )
