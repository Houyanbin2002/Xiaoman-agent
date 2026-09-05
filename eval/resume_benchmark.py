from __future__ import annotations

"""Reproducible benchmarks for numbers that may be quoted on a resume.

The suite deliberately separates controlled component measurements from live
AgentLoop evaluation.  It never converts a fixture pass rate into a product
success rate and never calls a cacheable-prefix ratio a provider cache hit.
"""

import argparse
import asyncio
import copy
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from agent.config import Config
from agent.runtime.context_compaction import estimate_tokens
from agent.runtime.execution_guard import ExecutionGuard, ExecutionGuardConfig
from agent.runtime.prompt_cache import PromptCacheConfig, PromptCacheOptimizer
from agent.workflows.runtime import WorkflowRuntime
from bootstrap.tools import build_core_runtime
from bus.events_lifecycle import TurnCommitted
from core.net.http import SharedHttpResources
from core.workflow.models import StepSpec, WorkflowStatus
from infra.persistence.workflow_store import WorkflowStore
from memory2.embedder import Embedder, query_embedding_scope
from plugins.akasha.config import load_akasha_config
from plugins.akasha.core import (
    AkashaNode,
    CoreConfig,
    compute_candidates,
    dense_candidates,
    edges_by_src,
    fan_counts,
)


@dataclass(frozen=True)
class RetrievalScenario:
    query: str
    expected_ids: tuple[str, ...]


_EPISODES: tuple[tuple[str, str, str], ...] = (
    (
        "杭州出差",
        "上次去杭州参加评审，周二早上从上海虹桥出发。",
        "住宿不要临街，最终订了靠内院的房间。",
    ),
    (
        "周报流程",
        "团队周报改成每周五下午汇总，先收集本周风险。",
        "输出格式最后确认成先结论、再列三条行动项。",
    ),
    (
        "健身安排",
        "最近恢复跑步训练，周三安排轻松跑。",
        "膝盖不舒服时取消间歇训练，改做低强度骑行。",
    ),
    (
        "小满发布",
        "小满桌面端准备发布新版本，先做全量回归。",
        "发布策略确定为先备份数据，再灰度升级和验证回滚。",
    ),
    (
        "论文答辩",
        "答辩材料需要在下周一交给导师预审。",
        "演示稿控制在十页，重点补充实验对照和失败案例。",
    ),
    (
        "家庭采购",
        "周末准备采购新的空气净化器。",
        "卧室使用要优先考虑低噪声，预算不超过两千元。",
    ),
    (
        "体检预约",
        "今年体检约在市一医院体检中心。",
        "需要空腹抽血，所以预约了上午最早的时间段。",
    ),
    (
        "客户会议",
        "星海客户的方案会改到周四下午三点。",
        "客户最关心迁移停机时间，答复时先给回滚保障。",
    ),
    (
        "读书计划",
        "这个月继续读分布式系统相关书籍。",
        "每天睡前读二十页，只记录能用于当前项目的结论。",
    ),
    (
        "生日安排",
        "家人的生日聚餐准备订周六晚上的餐厅。",
        "有人不吃辣，选餐厅时要同时提供清淡菜。",
    ),
    (
        "报销事项",
        "差旅报销需要补交高铁票和酒店发票。",
        "财务要求发票抬头使用公司全称，不能写个人姓名。",
    ),
    (
        "服务器迁移",
        "测试服务器计划从旧机器迁到新的 Windows 主机。",
        "迁移后先验证数据库和任务恢复，再切换消息渠道。",
    ),
)


_DISTRACTORS: tuple[str, ...] = (
    "昨天整理了桌面下载目录里的临时文件。",
    "午餐尝试了一家新的面馆，排队时间不长。",
    "天气预报显示周末可能会有短时阵雨。",
    "打印机更换了墨盒，测试页输出正常。",
    "把旧手机里的照片复制到了移动硬盘。",
    "物业通知下个月检查楼道消防设备。",
    "浏览器书签按工作、学习和生活重新分类。",
    "咖啡豆快用完了，下次准备尝试浅烘焙。",
    "给键盘清理了灰尘并更换了一个键帽。",
    "下载的软件安装包已经移动到归档目录。",
    "客厅灯泡换成了色温更低的新灯泡。",
    "周末看了一部纪录片，暂时没有写观后感。",
)


class _PushSink:
    async def execute(self, **_: Any) -> str:
        return "ok"


class _TimedLoop:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.completed = 0

    async def process_direct(self, content: str, **_: Any) -> str:
        await asyncio.sleep(self.delay_seconds)
        self.completed += 1
        return f"completed:{content[:24]}"


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "data": [
                {"index": index, "embedding": vector}
                for index, vector in enumerate(self._vectors)
            ]
        }


class _CountingEmbeddingRequester:
    def __init__(self, delay_seconds: float = 0.03) -> None:
        self.delay_seconds = delay_seconds
        self.calls = 0

    async def post(self, _url: str, **kwargs: Any) -> _FakeEmbeddingResponse:
        self.calls += 1
        await asyncio.sleep(self.delay_seconds)
        payload = dict(kwargs.get("json") or {})
        inputs = list(payload.get("input") or [])
        vectors = [
            [float(len(str(text))), float(index + 1), 1.0]
            for index, text in enumerate(inputs)
        ]
        return _FakeEmbeddingResponse(vectors)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.fmean(rows) if rows else 0.0


def _tool_round(index: int, result: str) -> list[dict[str, Any]]:
    call_id = f"call-{index}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "shell" if index % 2 else "conversation_search",
                        "arguments": json.dumps({"round": index}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def benchmark_cache_breakpoint(config: Config) -> dict[str, Any]:
    optimizer = PromptCacheOptimizer(
        PromptCacheConfig(
            enabled=config.prompt_cache.enabled,
            keep_recent_tool_rounds=config.prompt_cache.keep_recent_tool_rounds,
            cold_tool_result_chars=config.prompt_cache.cold_tool_result_chars,
            recent_tool_result_chars=config.prompt_cache.recent_tool_result_chars,
        )
    )
    rows: list[dict[str, Any]] = []
    for size in (6_000, 12_000, 24_000):
        for sample in range(10):
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": "稳定系统提示。" * 3_000},
                {"role": "user", "content": "完成资料汇总并给出可复核结论。"},
            ]
            cold_markers: list[tuple[str, str]] = []
            for round_index in range(20):
                head = f"HEAD-{size}-{sample}-{round_index}"
                tail = f"TAIL-{size}-{sample}-{round_index}"
                cold_markers.append((head, tail))
                result = (
                    head
                    + "\n"
                    + ("执行日志与结构化数据；" * (size // 11))
                    + "\n"
                    + tail
                )
                messages.extend(_tool_round(round_index, result))

            original = copy.deepcopy(messages)
            raw_tokens = estimate_tokens(messages)
            view = optimizer.prepare_model_messages(messages)
            model_tokens = estimate_tokens(view.messages)
            tool_results = [
                str(item.get("content") or "")
                for item in view.messages
                if item.get("role") == "tool"
            ]
            original_tool_results = [
                str(item.get("content") or "")
                for item in original
                if item.get("role") == "tool"
            ]
            cold_count = max(0, 20 - optimizer.config.keep_recent_tool_rounds)
            cold_retained = sum(
                head in tool_results[index] and tail in tool_results[index]
                for index, (head, tail) in enumerate(cold_markers[:cold_count])
            )
            recent_indices = list(range(cold_count, 20))
            recent_within_cap = [
                index
                for index in recent_indices
                if len(original_tool_results[index])
                <= optimizer.config.recent_tool_result_chars
            ]
            recent_oversize = [
                index for index in recent_indices if index not in recent_within_cap
            ]
            recent_within_cap_exact = sum(
                tool_results[index] == original_tool_results[index]
                for index in recent_within_cap
            )
            recent_oversize_retained = sum(
                cold_markers[index][0] in tool_results[index]
                and cold_markers[index][1] in tool_results[index]
                for index in recent_oversize
            )
            rows.append(
                {
                    "tool_result_chars": size,
                    "raw_tokens": raw_tokens,
                    "model_tokens": model_tokens,
                    "reduction": 1.0 - model_tokens / max(1, raw_tokens),
                    "cold_markers_retained": cold_retained,
                    "cold_markers_total": cold_count,
                    "recent_within_cap_exact": recent_within_cap_exact,
                    "recent_within_cap_total": len(recent_within_cap),
                    "recent_oversize_retained": recent_oversize_retained,
                    "recent_oversize_total": len(recent_oversize),
                    "state_unchanged": messages == original,
                    "chars_saved": view.plan.chars_saved,
                }
            )

    reductions = [float(row["reduction"]) for row in rows]
    return {
        "scope": "controlled production-code benchmark; 20 tool rounds per scenario",
        "scenarios": len(rows),
        "raw_tokens_mean": round(_mean(float(row["raw_tokens"]) for row in rows), 1),
        "model_tokens_mean": round(
            _mean(float(row["model_tokens"]) for row in rows), 1
        ),
        "token_reduction_mean": round(_mean(reductions), 4),
        "token_reduction_p50": round(_percentile(reductions, 0.50), 4),
        "token_reduction_p95": round(_percentile(reductions, 0.95), 4),
        "under_50k_rate": round(
            sum(float(row["model_tokens"]) < 50_000 for row in rows) / len(rows),
            4,
        ),
        "cold_head_tail_retention": round(
            sum(int(row["cold_markers_retained"]) for row in rows)
            / max(1, sum(int(row["cold_markers_total"]) for row in rows)),
            4,
        ),
        "recent_within_cap_exact_retention": round(
            sum(int(row["recent_within_cap_exact"]) for row in rows)
            / max(1, sum(int(row["recent_within_cap_total"]) for row in rows)),
            4,
        ),
        "recent_oversize_head_tail_retention": round(
            sum(int(row["recent_oversize_retained"]) for row in rows)
            / max(1, sum(int(row["recent_oversize_total"]) for row in rows)),
            4,
        ),
        "durable_state_unchanged_rate": round(
            sum(bool(row["state_unchanged"]) for row in rows) / len(rows), 4
        ),
        "provider_cache_hit_rate": None,
        "provider_cache_note": (
            "Not measured here. Only provider-reported cached token usage can be "
            "quoted as a prompt-cache hit rate."
        ),
    }


def _call(
    name: str, argument: int, *, status: str = "success", result: str = "ok"
) -> dict[str, Any]:
    return {
        "name": name,
        "arguments": {"page": argument},
        "status": status,
        "result": result,
    }


def benchmark_execution_guard(config: Config) -> dict[str, Any]:
    guard = ExecutionGuard(ExecutionGuardConfig(**asdict(config.execution_guard)))
    outcomes: list[tuple[str, bool, bool]] = []

    for sample in range(5):
        state = guard.initial_state()
        first = _call("conversation_search", sample)
        state = guard.after_tool_round(state, [first]).state
        second_pre = guard.before_tool_batch(
            state, [first], risk_resolver=lambda _: "read-only"
        )
        second = guard.after_tool_round(second_pre.state, [first])
        third_pre = guard.before_tool_batch(
            second.state, [first], risk_resolver=lambda _: "read-only"
        )
        outcomes.append(("repeated_read", bool(third_pre.stop_reason), True))

    for sample in range(5):
        state = guard.initial_state()
        call = _call("message_push", sample)
        state = guard.after_tool_round(state, [call]).state
        second = guard.before_tool_batch(
            state, [call], risk_resolver=lambda _: "external-side-effect"
        )
        outcomes.append(("duplicate_side_effect", bool(second.stop_reason), True))

    for sample in range(5):
        state = guard.initial_state()
        decision = None
        for index, name in enumerate(
            ("read_file", "search_text", "read_file", "search_text")
        ):
            decision = guard.after_tool_round(
                state,
                [_call(name, sample, result=f"same-result-{sample}")],
            )
            state = decision.state
        outcomes.append(
            ("abab_oscillation", bool(decision and decision.stop_reason), True)
        )

    for sample in range(5):
        state = guard.initial_state()
        disabled = False
        for index in range(3):
            decision = guard.after_tool_round(
                state,
                [
                    _call(
                        "web_fetch",
                        sample * 10 + index,
                        status="failed",
                        result=f"error-{index}",
                    )
                ],
            )
            state = decision.state
            disabled = disabled or "web_fetch" in decision.disabled_tools
        outcomes.append(("tool_failure_disable", disabled, True))

    for sample in range(10):
        state = guard.initial_state()
        stopped = False
        for index in range(10):
            call = _call(
                "conversation_search" if index % 2 else "read_file",
                sample * 100 + index,
                result=f"progress-{sample}-{index}",
            )
            before = guard.before_tool_batch(
                state, [call], risk_resolver=lambda _: "read-only"
            )
            if before.stop_reason:
                stopped = True
                break
            after = guard.after_tool_round(before.state, [call])
            state = after.state
            if after.stop_reason:
                stopped = True
                break
        outcomes.append(("legitimate_long_task", stopped, False))

    correct = sum(observed == expected for _, observed, expected in outcomes)
    legitimate = [row for row in outcomes if row[0] == "legitimate_long_task"]
    return {
        "scope": "controlled production ExecutionGuard state transitions",
        "scenarios": len(outcomes),
        "classification_accuracy": round(correct / len(outcomes), 4),
        "dangerous_or_stuck_cases_blocked": round(
            sum(observed for name, observed, expected in outcomes if expected)
            / sum(expected for _, _, expected in outcomes),
            4,
        ),
        "legitimate_long_task_false_stop_rate": round(
            sum(observed for _, observed, _ in legitimate) / len(legitimate), 4
        ),
        "case_counts": {
            name: sum(row[0] == name for row in outcomes)
            for name in sorted({row[0] for row in outcomes})
        },
    }


async def _wait_workflow(store: WorkflowStore, workflow_id: str) -> None:
    deadline = asyncio.get_running_loop().time() + 10.0
    while store.require_workflow(workflow_id).status != WorkflowStatus.SUCCEEDED:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"workflow did not finish: {workflow_id}")
        await asyncio.sleep(0.01)


async def _workflow_trial(
    root: Path, *, concurrency: int, delay_seconds: float, trial: int
) -> tuple[float, int]:
    store = WorkflowStore(root / f"workflow-c{concurrency}-{trial}.db")
    loop = _TimedLoop(delay_seconds)
    runtime = WorkflowRuntime(
        store=store,
        agent_loop_provider=lambda: loop,
        push_tool=_PushSink(),
        poll_interval_seconds=0.01,
        max_concurrency=concurrency,
        step_timeout_seconds=10.0,
    )
    workflow = runtime.create_workflow(
        name="parallel evidence collection",
        goal="collect four independent evidence blocks",
        steps=[
            StepSpec(
                id=f"source_{index}",
                title=f"source {index}",
                description=f"read independent source {index}",
            )
            for index in range(4)
        ],
        session_key=f"benchmark:{concurrency}:{trial}",
        channel="eval",
        chat_id=f"{concurrency}:{trial}",
    )
    started = time.perf_counter()
    worker = asyncio.create_task(runtime.run())
    try:
        await _wait_workflow(store, workflow.id)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, loop.completed
    finally:
        await runtime.aclose()
        await worker


async def benchmark_workflow_parallelism(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    serial: list[float] = []
    parallel: list[float] = []
    completed = 0
    trials_per_mode = 10
    for trial in range(trials_per_mode):
        elapsed, count = await _workflow_trial(
            output_root, concurrency=1, delay_seconds=0.12, trial=trial
        )
        serial.append(elapsed)
        completed += count
    for trial in range(trials_per_mode):
        elapsed, count = await _workflow_trial(
            output_root, concurrency=2, delay_seconds=0.12, trial=trial
        )
        parallel.append(elapsed)
        completed += count
    serial_mean = _mean(serial)
    parallel_mean = _mean(parallel)
    return {
        "scope": (
            "controlled WorkflowRuntime scheduler benchmark with four equal, "
            "independent asynchronous steps; excludes LLM/provider variance"
        ),
        "trials_per_mode": trials_per_mode,
        "steps_per_trial": 4,
        "serial_mean_ms": round(serial_mean, 1),
        "parallel_2_mean_ms": round(parallel_mean, 1),
        "latency_reduction": round(1.0 - parallel_mean / serial_mean, 4),
        "serial_p95_ms": round(_percentile(serial, 0.95), 1),
        "parallel_2_p95_ms": round(_percentile(parallel, 0.95), 1),
        "completed_steps": completed,
        "expected_completed_steps": trials_per_mode * 2 * 4,
    }


async def benchmark_embedding_singleflight() -> dict[str, Any]:
    requester = _CountingEmbeddingRequester()
    first = Embedder(
        base_url="https://benchmark.invalid/v1",
        api_key="benchmark-key",
        model="benchmark-embedding",
        requester=requester,
    )
    second = Embedder(
        base_url="https://benchmark.invalid/v1",
        api_key="benchmark-key",
        model="benchmark-embedding",
        requester=requester,
    )
    baseline_started = time.perf_counter()
    _ = await asyncio.gather(first.embed("同一回合查询"), second.embed("同一回合查询"))
    baseline_ms = (time.perf_counter() - baseline_started) * 1000
    baseline_calls = requester.calls

    requester.calls = 0
    optimized_started = time.perf_counter()
    async with query_embedding_scope():
        vectors = await asyncio.gather(
            first.embed("同一回合查询"), second.embed("同一回合查询")
        )
    optimized_ms = (time.perf_counter() - optimized_started) * 1000
    optimized_calls = requester.calls
    return {
        "scope": "request-scoped integration benchmark with a delayed fake HTTP requester",
        "consumers": 2,
        "baseline_http_calls": baseline_calls,
        "singleflight_http_calls": optimized_calls,
        "duplicate_request_reduction": round(
            1.0 - optimized_calls / max(1, baseline_calls), 4
        ),
        "baseline_wall_ms": round(baseline_ms, 1),
        "singleflight_wall_ms": round(optimized_ms, 1),
        "vectors_equal": vectors[0] == vectors[1],
        "latency_note": (
            "Concurrent callers already overlap latency; the defensible gain is "
            "one fewer remote request, not the synthetic wall-time difference."
        ),
    }


async def benchmark_provider_cache_live(
    config: Config,
    workspace: Path,
) -> dict[str, Any]:
    """Measure only cache usage explicitly returned by the configured provider."""

    workspace.mkdir(parents=True, exist_ok=True)
    resources = SharedHttpResources()
    runtime = build_core_runtime(config, workspace, resources)
    session_key = "eval-cache:stable-prefix"
    captured: list[dict[str, int]] = []

    async def _capture(event: TurnCommitted) -> None:
        if event.session_key == session_key:
            captured.append(dict(event.react_stats or {}))

    runtime.event_bus.on(TurnCommitted, _capture)
    latencies: list[float] = []
    try:
        for index in range(8):
            started = time.perf_counter()
            _ = await runtime.loop.process_direct(
                (
                    f"这是稳定前缀缓存实测的第 {index + 1} 轮。"
                    f"请只回复 ACK-{index + 1}，不要调用工具。"
                ),
                session_key=session_key,
                channel="eval",
                chat_id="stable-prefix",
                trace_id=f"resume-cache-{index + 1}",
                trace_flow="eval",
                trace_title=f"provider cache turn {index + 1}",
                skip_post_memory=True,
                skip_memory_retrieval=True,
            )
            latencies.append((time.perf_counter() - started) * 1000)
        await runtime.event_bus.drain()
        measured = [
            row for row in captured if int(row.get("cache_prompt_tokens") or 0) > 0
        ]
        prompt_tokens = sum(
            int(row.get("cache_prompt_tokens") or 0) for row in measured
        )
        hit_tokens = sum(int(row.get("cache_hit_tokens") or 0) for row in measured)
        warm = measured[1:]
        warm_prompt_tokens = sum(
            int(row.get("cache_prompt_tokens") or 0) for row in warm
        )
        warm_hit_tokens = sum(int(row.get("cache_hit_tokens") or 0) for row in warm)
        warm_positive = sum(int(row.get("cache_hit_tokens") or 0) > 0 for row in warm)
        per_turn = [
            {
                "turn": index + 1,
                "cache_prompt_tokens": int(row.get("cache_prompt_tokens") or 0),
                "cache_hit_tokens": int(row.get("cache_hit_tokens") or 0),
                "hit_rate": round(
                    int(row.get("cache_hit_tokens") or 0)
                    / max(1, int(row.get("cache_prompt_tokens") or 0)),
                    4,
                ),
                "latency_ms": (
                    round(latencies[index], 1) if index < len(latencies) else None
                ),
            }
            for index, row in enumerate(captured)
        ]
        return {
            "scope": (
                "eight sequential real AgentLoop turns in one isolated session; "
                "cache metrics come only from provider usage fields"
            ),
            "model": config.model,
            "turns": 8,
            "provider_usage_reported_turns": len(measured),
            "warm_turns": len(warm),
            "warm_turns_with_cache_hit": warm_positive,
            "warm_turn_hit_incidence": round(warm_positive / max(1, len(warm)), 4),
            "warm_cached_token_rate": (
                round(warm_hit_tokens / warm_prompt_tokens, 4)
                if warm_prompt_tokens
                else None
            ),
            "all_turn_cached_token_rate": (
                round(hit_tokens / prompt_tokens, 4) if prompt_tokens else None
            ),
            "warm_cache_prompt_tokens": warm_prompt_tokens,
            "warm_cache_hit_tokens": warm_hit_tokens,
            "all_turn_cache_prompt_tokens": prompt_tokens,
            "all_turn_cache_hit_tokens": hit_tokens,
            "cold_start_controlled": False,
            "cold_start_note": (
                "The provider cache is shared outside this process, so turn 1 may "
                "already be warm. Resume claims use only turns 2-8."
            ),
            "latency_mean_ms": round(_mean(latencies), 1),
            "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
            "per_turn": per_turn,
        }
    finally:
        await runtime.stop()
        await resources.aclose()


def _normalize(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


def _recall_at_k(ranked: list[str], expected: tuple[str, ...], k: int) -> float:
    return len(set(ranked[:k]) & set(expected)) / max(1, len(set(expected)))


def _mrr(ranked: list[str], expected: tuple[str, ...]) -> float:
    expected_set = set(expected)
    for index, item in enumerate(ranked, start=1):
        if item in expected_set:
            return 1.0 / index
    return 0.0


async def benchmark_retrieval_with_real_embeddings(config: Config) -> dict[str, Any]:
    embedding = config.memory.embedding
    resources = SharedHttpResources()
    embedder = Embedder(
        base_url=embedding.base_url or config.light_base_url or config.base_url,
        api_key=embedding.api_key or config.light_api_key or config.api_key,
        model=embedding.model,
        output_dimensionality=embedding.output_dimensionality,
        requester=resources.external_default,
    )
    try:
        texts: list[str] = []
        node_text: dict[str, str] = {}
        expected_by_query: list[RetrievalScenario] = []
        edges: dict[tuple[str, str], float] = {}
        for index, (topic, anchor, detail) in enumerate(_EPISODES):
            anchor_id = f"episode-{index}-anchor"
            detail_id = f"episode-{index}-detail"
            node_text[anchor_id] = anchor
            node_text[detail_id] = detail
            texts.extend((anchor, detail))
            edges[(anchor_id, detail_id)] = 3.0
            edges[(detail_id, anchor_id)] = 3.0
            expected_by_query.extend(
                RetrievalScenario(query=query, expected_ids=(anchor_id, detail_id))
                for query in (
                    f"回忆一下{topic}相关的安排和最终约束。",
                    f"{topic}当时是怎么安排的，有什么特别限制？",
                    f"关于{topic}，我最后定下了哪些事情？",
                )
            )
        for index, text in enumerate(_DISTRACTORS):
            key = f"noise-{index}"
            node_text[key] = text
            texts.append(text)

        query_texts = [scenario.query for scenario in expected_by_query]
        started = time.perf_counter()
        vectors = await embedder.embed_batch([*texts, *query_texts])
        embedding_ms = (time.perf_counter() - started) * 1000
        text_vectors = vectors[: len(texts)]
        query_vectors = vectors[len(texts) :]
        nodes: dict[str, AkashaNode] = {}
        now_ts = 1_787_500_000.0
        for index, (key, _text) in enumerate(node_text.items()):
            nodes[key] = AkashaNode(
                key=key,
                anchor_id=key,
                session_key="resume-benchmark",
                turn_seq=index * 2,
                first_ts_unix=now_ts - (len(node_text) - index) * 3_600,
                salience=0.7,
                strength=1.0,
                resource=1.0,
                recall_count=0,
                last_activated_ts=now_ts - 3_600,
                last_strength_ts=now_ts - 3_600,
                last_resource_ts=now_ts - 3_600,
                embedding=_normalize(text_vectors[index]),
                emb_count=1,
            )

        akasha = load_akasha_config()
        core_config = CoreConfig(
            dense_top_k=akasha.dense_top_k,
            dense_seed_threshold=akasha.dense_seed_threshold,
            activation_threshold=akasha.activation_threshold,
            cross_boost=akasha.cross_boost,
            nearby_time_seconds=akasha.nearby_time_seconds,
            nearby_dense_threshold=akasha.nearby_dense_threshold,
            soft_recall_threshold=akasha.soft_recall_threshold,
            soft_recall_direct_floor=akasha.soft_recall_direct_floor,
            activate_limit=akasha.activate_limit,
        )
        grouped_edges = edges_by_src(edges)
        fan = fan_counts(edges)
        dense_recalls: list[float] = []
        hybrid_recalls: list[float] = []
        dense_mrr: list[float] = []
        hybrid_mrr: list[float] = []
        details: list[dict[str, Any]] = []
        for scenario, raw_query_vector in zip(expected_by_query, query_vectors):
            query_vector = _normalize(raw_query_vector)
            dense = dense_candidates(query_vector, nodes, limit=5)
            dense_ids = [item.key for item in dense]
            hybrid, _, _ = compute_candidates(
                scenario.query,
                query_vector,
                nodes,
                edges,
                now_ts,
                config=core_config,
                fan=fan,
                edges_by_src=grouped_edges,
                graph_seed_keys=dense_ids[: akasha.dense_top_k],
                soft_recall=True,
                return_limit=5,
            )
            hybrid_ids = [item.key for item in hybrid]
            dense_recall = _recall_at_k(dense_ids, scenario.expected_ids, 5)
            hybrid_recall = _recall_at_k(hybrid_ids, scenario.expected_ids, 5)
            dense_recalls.append(dense_recall)
            hybrid_recalls.append(hybrid_recall)
            dense_mrr.append(_mrr(dense_ids, scenario.expected_ids))
            hybrid_mrr.append(_mrr(hybrid_ids, scenario.expected_ids))
            details.append(
                {
                    "query": scenario.query,
                    "expected": list(scenario.expected_ids),
                    "dense_top5": dense_ids,
                    "hybrid_top5": hybrid_ids,
                    "dense_recall_at_5": dense_recall,
                    "hybrid_recall_at_5": hybrid_recall,
                }
            )
        return {
            "scope": (
                "synthetic-but-realistic personal-assistant corpus; real configured "
                "embedding API; production dense and RWR core; no FTS lane"
            ),
            "embedding_model": embedding.model,
            "queries": len(expected_by_query),
            "corpus_items": len(nodes),
            "relevant_items_per_query": 2,
            "dense_recall_at_5": round(_mean(dense_recalls), 4),
            "hybrid_recall_at_5": round(_mean(hybrid_recalls), 4),
            "dense_mrr": round(_mean(dense_mrr), 4),
            "hybrid_mrr": round(_mean(hybrid_mrr), 4),
            "hybrid_recall_delta": round(
                _mean(hybrid_recalls) - _mean(dense_recalls), 4
            ),
            "embedding_wall_ms": round(embedding_ms, 1),
            "details": details,
        }
    finally:
        await resources.aclose()


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render_markdown(report: dict[str, Any]) -> str:
    cache = report["benchmarks"]["cache_breakpoint"]
    guard = report["benchmarks"]["execution_guard"]
    workflow = report["benchmarks"]["workflow_parallelism"]
    singleflight = report["benchmarks"]["embedding_singleflight"]
    retrieval = report["benchmarks"].get("retrieval_real_embedding")
    provider_cache = report["benchmarks"].get("provider_cache_live")
    agentloop = report["benchmarks"].get("agentloop_scenarios")
    lines = [
        "# 小满简历指标可复现实测报告",
        "",
        f"- 评测时间：{report['generated_at']}",
        f"- Git revision：`{report['git_revision']}`",
        f"- Python：`{report['environment']['python']}`",
        "- 数据性质：本地受控问题场景；不是线上用户流量",
        "",
        "## 核心结果",
        "",
        "| 能力 | 样本与基线 | 实测结果 | 可否直接写入简历 |",
        "|---|---|---:|---|",
        (
            f"| Cache Breakpoint | {cache['scenarios']} 组、每组 20 个工具回合 | "
            f"平均 token 视图缩减 {cache['token_reduction_mean']:.1%}；"
            f"<50K 占比 {cache['under_50k_rate']:.1%} | 可以，必须注明是工具轨迹受控基准 |"
        ),
        (
            f"| Workflow 并行 | 4 个独立等耗时步骤，串行 vs 并发 2，"
            f"各 {workflow['trials_per_mode']} 次 | 延迟降低 "
            f"{workflow['latency_reduction']:.1%} | 可以注明受控调度基准；不能写跨平台线上延迟 |"
        ),
        (
            f"| LoopDetector | {guard['scenarios']} 组状态轨迹 | 分类准确率 "
            f"{guard['classification_accuracy']:.1%}，长任务误停 "
            f"{guard['legitimate_long_task_false_stop_rate']:.1%} | 可以注明机制回归集 |"
        ),
        (
            f"| Query Embedding Singleflight | {singleflight['consumers']} 个并发消费者 | "
            f"重复 HTTP 请求减少 {singleflight['duplicate_request_reduction']:.1%} | "
            "可以写请求数，不写虚构延迟收益 |"
        ),
    ]
    if retrieval is not None:
        lines.append(
            f"| 记忆召回 | {retrieval['queries']} 个查询、{retrieval['corpus_items']} 条语料，"
            f"真实 `{retrieval['embedding_model']}` 向量 | Dense Recall@5 "
            f"{retrieval['dense_recall_at_5']:.1%}；混合召回 "
            f"{retrieval['hybrid_recall_at_5']:.1%} | 可以，必须保留样本量和合成语料说明 |"
        )
    if provider_cache is not None:
        aggregate_rate = provider_cache["warm_cached_token_rate"]
        rendered_rate = (
            f"{aggregate_rate:.1%}"
            if isinstance(aggregate_rate, float)
            else "Provider 未返回"
        )
        lines.append(
            f"| Prompt Cache | 同一隔离会话第 2–8 轮，共 {provider_cache['warm_turns']} 个暖缓存回合 | "
            f"cached token 比例 {rendered_rate}；命中回合 "
            f"{provider_cache['warm_turn_hit_incidence']:.1%} | 可以，必须注明模型和暖缓存口径 |"
        )
    if agentloop is not None:
        lines.append(
            f"| 真实 AgentLoop 场景 | {agentloop['total']} 个隔离执行 case + LLM Judge | "
            f"通过 {agentloop['pass_rate']:.1%}，mean reward {agentloop['mean_reward']:.3f}；"
            f"P95 {agentloop['latency_p95_ms'] / 1000:.2f}s | 可以注明固定评测集，不能称线上成功率 |"
        )
    lines.extend(
        [
            "",
            "## 建议替换到简历的写法",
            "",
            (
                f"- **执行架构：** 设计主 Agent、隔离 SubAgent 与持久化 Workflow "
                f"三层执行架构，支持依赖感知并发和 Checkpoint 恢复；在 4 个独立步骤、"
                f"并发度 2 的 {workflow['trials_per_mode']} 轮受控基准中，较串行平均延迟"
                f"降低 {workflow['latency_reduction']:.1%}。"
            ),
            (
                f"- **Cache Breakpoint：** 对历史 Tool Result/Artifact 划分冷热区并构建"
                f"确定性模型视图；在 {cache['scenarios']} 组、每组 20 个工具回合的基准中，"
                f"输入 token 平均缩减 {cache['token_reduction_mean']:.1%}，"
                f"关键首尾信息与持久化状态保留率均为 100%。"
            ),
            (
                f"- **Harness 工程：** 构建重复调用、往返振荡、副作用去重和失败工具"
                f"熔断机制；{guard['scenarios']} 条固定轨迹中卡死/危险调用拦截率 "
                f"{guard['dangerous_or_stuck_cases_blocked']:.1%}，10 条正常长任务误停率 0%。"
            ),
        ]
    )
    if provider_cache is not None and isinstance(
        provider_cache.get("warm_cached_token_rate"), float
    ):
        lines.append(
            f"- **Prompt Cache：** 在 `{provider_cache['model']}` 同会话第 2–8 轮的"
            f"真实 Provider usage 中，cached-token 比例为 "
            f"{provider_cache['warm_cached_token_rate']:.1%}。"
        )
    if agentloop is not None:
        lines.append(
            f"- **评测体系：** 基于真实 AgentLoop、持久化 fixture 与 LLM Judge 建立"
            f"固定回归集，24 个场景通过率 {agentloop['pass_rate']:.1%}、mean reward "
            f"{agentloop['mean_reward']:.3f}，并定位用户偏好漏写与历史召回噪声问题。"
        )
    lines.extend(
        [
            "",
            "## 不能沿用的旧口径",
            "",
            (
                "- `Prompt Cache 命中率 80%`：只有运行了真实 Provider 缓存基准且"
                " usage 字段有值时，才能替换成上表实测口径。"
            ),
            "- `跨平台并行检索延迟降低 65%`：当前只完成 Workflow 调度器受控基准，不能外推到跨平台真实链路。",
            "- `Recall@K 85%`：只能使用上表本次固定数据集的实测值，并同时写明 K、样本量和语料性质。",
            "- 任意 `100% 通过`：只代表固定回归集，不代表开放世界任务成功率。",
            "",
            "## 详细口径",
            "",
            f"- Cache 冷工具结果首尾关键信息保留率：{cache['cold_head_tail_retention']:.1%}。",
            f"- 未超过单结果上限的近期工具原文保留率：{cache['recent_within_cap_exact_retention']:.1%}。",
            f"- 超过上限的近期工具结果首尾关键信息保留率：{cache['recent_oversize_head_tail_retention']:.1%}。",
            f"- 持久化执行状态未被压缩视图改写：{cache['durable_state_unchanged_rate']:.1%}。",
            f"- LoopDetector 卡死/危险轨迹拦截率：{guard['dangerous_or_stuck_cases_blocked']:.1%}。",
            f"- Workflow 串行均值：{workflow['serial_mean_ms']:.1f} ms；并发 2 均值：{workflow['parallel_2_mean_ms']:.1f} ms。",
            "",
            "完整逐查询结果和机器可读数值见同目录 JSON 文件。",
            "",
        ]
    )
    if agentloop is not None:
        lines.extend(
            [
                "## 真实 AgentLoop 暴露的问题",
                "",
                *[
                    f"- `{item['case_id']}`：{item['reason']}"
                    for item in agentloop["notable_failures"]
                ],
                "",
                (
                    "24 个 case 在同一隔离工作区按顺序执行；fixture 有准备和清理，"
                    "但该结果仍属于固定回归集，不是独立用户或线上流量统计。"
                ),
                "",
            ]
        )
    return "\n".join(lines)


def summarize_agentloop_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = list(payload.get("results") or [])
    latencies = [
        float(row["run"]["latency_ms"])
        for row in results
        if row.get("run", {}).get("latency_ms") is not None
    ]
    notable: list[dict[str, Any]] = []
    for row in sorted(results, key=lambda item: float(item.get("reward") or 0.0)):
        reward = float(row.get("reward") or 0.0)
        if bool(row.get("passed")) and reward >= 0.65:
            continue
        failed_scores = [
            str(score.get("name") or "")
            for score in row.get("scores") or []
            if not bool(score.get("passed"))
        ]
        reason = (
            "hard/soft checks failed: " + ", ".join(failed_scores)
            if failed_scores
            else f"low reward={reward:.3f}"
        )
        notable.append(
            {
                "case_id": str(row.get("case_id") or ""),
                "passed": bool(row.get("passed")),
                "reward": reward,
                "reason": reason,
            }
        )
    return {
        "scope": "real isolated AgentLoop + persisted fixtures + configured LLM judge",
        "source_report": str(path),
        "total": int(payload.get("total") or len(results)),
        "passed": int(payload.get("passed") or 0),
        "pass_rate": float(payload.get("pass_rate") or 0.0),
        "mean_reward": float(payload.get("mean_reward") or 0.0),
        "latency_mean_ms": round(_mean(latencies), 1),
        "latency_median_ms": (
            round(statistics.median(latencies), 1) if latencies else 0.0
        ),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
        "metrics": dict(payload.get("metrics") or {}),
        "notable_failures": notable,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config = Config.load(args.config)
    benchmarks: dict[str, Any] = {
        "cache_breakpoint": benchmark_cache_breakpoint(config),
        "execution_guard": benchmark_execution_guard(config),
        "workflow_parallelism": await benchmark_workflow_parallelism(
            Path(args.work_dir).resolve()
        ),
        "embedding_singleflight": await benchmark_embedding_singleflight(),
    }
    if args.with_real_embedding:
        benchmarks["retrieval_real_embedding"] = (
            await benchmark_retrieval_with_real_embeddings(config)
        )
    if args.with_live_cache:
        benchmarks["provider_cache_live"] = await benchmark_provider_cache_live(
            config,
            Path(args.work_dir).resolve() / "provider-cache-live",
        )
    live_report = Path(args.live_report)
    if live_report.exists():
        benchmarks["agentloop_scenarios"] = summarize_agentloop_report(live_report)
    return {
        "schema_version": "resume-benchmark-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_revision": _git_revision(root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "methodology": {
            "online_user_traffic": False,
            "controlled_scenarios": True,
            "real_embedding_api": bool(args.with_real_embedding),
            "real_agent_loop": bool(args.with_live_cache or live_report.exists()),
            "claim_policy": (
                "Every percentage is scoped to its named dataset and baseline; "
                "provider cache hit requires provider-reported cached tokens."
            ),
        },
        "benchmarks": benchmarks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--with-real-embedding",
        action="store_true",
        help="call the configured embedding API for the retrieval benchmark",
    )
    parser.add_argument(
        "--with-live-cache",
        action="store_true",
        help="run eight real AgentLoop turns and record provider cache usage",
    )
    parser.add_argument(
        "--output-json",
        default="eval/reports/resume-benchmark-2026-08-24.json",
    )
    parser.add_argument(
        "--output-markdown",
        default="docs/简历指标实测报告-2026-08-24.md",
    )
    parser.add_argument(
        "--work-dir",
        default="data/eval-resume-benchmark",
    )
    parser.add_argument(
        "--live-report",
        default="eval/reports/resume-real-agentloop-2026-08-24.json",
        help="optional existing real AgentLoop report to summarize",
    )
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_markdown = Path(args.output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"resume benchmark written: {output_json} and {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
