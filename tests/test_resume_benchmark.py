from __future__ import annotations

import asyncio

from agent.config import Config
from eval.resume_benchmark import (
    benchmark_cache_breakpoint,
    benchmark_embedding_singleflight,
    benchmark_execution_guard,
)


def test_resume_cache_benchmark_uses_scoped_and_lossless_metrics() -> None:
    result = benchmark_cache_breakpoint(Config.load("config.example.toml"))

    assert result["scenarios"] == 30
    assert result["under_50k_rate"] == 1.0
    assert result["cold_head_tail_retention"] == 1.0
    assert result["recent_within_cap_exact_retention"] == 1.0
    assert result["recent_oversize_head_tail_retention"] == 1.0
    assert result["durable_state_unchanged_rate"] == 1.0
    assert result["provider_cache_hit_rate"] is None


def test_resume_guard_benchmark_keeps_legitimate_long_tasks() -> None:
    result = benchmark_execution_guard(Config.load("config.example.toml"))

    assert result["scenarios"] == 30
    assert result["dangerous_or_stuck_cases_blocked"] == 1.0
    assert result["legitimate_long_task_false_stop_rate"] == 0.0


def test_resume_singleflight_benchmark_counts_remote_requests() -> None:
    result = asyncio.run(benchmark_embedding_singleflight())

    assert result["baseline_http_calls"] == 2
    assert result["singleflight_http_calls"] == 1
    assert result["duplicate_request_reduction"] == 0.5
    assert result["vectors_equal"] is True
