from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import pytest

from bootstrap.attention import build_attention_runtime
from bootstrap.personal import build_personal_runtime
from core.attention.actions import ActionPlanStatus
from core.attention.feedback import FeedbackKind
from core.attention.patterns import BehaviorPattern, PatternSource, RecurrenceSpec
from core.attention.policies import DecisionContext
from core.attention.providers import McpAlertSignalAdapter
from core.attention.signals import (
    AttentionSignal,
    SignalProviderManifest,
    SignalSource,
)

NOW = datetime(2026, 7, 16, 8, 0, tzinfo=timezone.utc)


def _signal(
    signal_id: str,
    *,
    severity: float = 0.7,
    urgency: float = 0.7,
    domain: str = "general",
    expires_in: timedelta = timedelta(hours=2),
) -> AttentionSignal:
    return AttentionSignal.create(
        signal_id=signal_id,
        kind=f"simulation.{domain}",
        domain=domain,
        summary=f"仿真信号 {signal_id}",
        source=SignalSource("simulation", "load", signal_id),
        occurred_at=NOW,
        expires_at=NOW + expires_in,
        severity=severity,
        urgency=urgency,
        actionability=0.9,
        confidence=0.9,
        freshness=1.0,
        suggested_capabilities=("message.notify",),
        metadata={
            "goal_alignment": 0.7,
            "opportunity": {
                "scene": "neutral",
                "duration_minutes": 120,
                "available_minutes": 5,
            },
        },
    )


def _context(now: datetime = NOW, **changes) -> DecisionContext:
    values = {
        "now": now,
        "scene": "neutral",
        "permission_mode": "delegated",
    }
    values.update(changes)
    return DecisionContext(**values)


def test_recurring_window_cannot_make_a_delayed_signal_fire_early(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.store.upsert_pattern(
        BehaviorPattern.create(
            pattern_id="pat_open_now",
            kind="availability_pattern",
            scene="quiet_window",
            recurrence=RecurrenceSpec(
                timezone="UTC",
                days=("thu",),
                start="07:50",
                end="08:20",
            ),
            available_minutes=20,
            confidence=1.0,
            source=PatternSource.USER,
            user_locked=True,
        )
    )
    delayed = _signal("sig_delayed")
    delayed = AttentionSignal.from_dict(
        {
            **delayed.to_dict(),
            "metadata": {
                **delayed.metadata,
                "opportunity": {
                    "scene": "follow_up",
                    "starts_at": (NOW + timedelta(minutes=45)).isoformat(),
                    "ends_at": (NOW + timedelta(hours=2)).isoformat(),
                    "available_minutes": 5,
                },
            },
        }
    )
    runtime.engine.ingest_signal(delayed)

    assert runtime.engine.evaluate(context=_context()).plan is None
    later = runtime.engine.evaluate(context=_context(NOW + timedelta(hours=1)))
    assert later.plan is not None
    assert later.plan.signal_ids == ("sig_delayed",)
    runtime.close()


def test_each_plan_acknowledges_only_the_signal_it_actually_exposes(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.store.upsert_pattern(
        BehaviorPattern.create(
            pattern_id="pat_batch_guard",
            kind="availability_pattern",
            scene="neutral",
            recurrence=RecurrenceSpec(
                timezone="UTC",
                days=("thu",),
                start="07:50",
                end="08:20",
            ),
            available_minutes=20,
            confidence=1.0,
            source=PatternSource.USER,
            user_locked=True,
        )
    )
    runtime.engine.ingest_signal(_signal("sig_atomic_a", severity=0.92))
    runtime.engine.ingest_signal(_signal("sig_atomic_b", severity=0.9))

    first = runtime.engine.evaluate(context=_context()).plan
    assert first is not None
    assert len(first.signal_ids) == 1
    runtime.store.transition_plan(first.id, ActionPlanStatus.EXECUTING)
    runtime.store.transition_plan(first.id, ActionPlanStatus.SUCCEEDED)

    second = runtime.engine.evaluate(context=_context()).plan
    assert second is not None
    assert len(second.signal_ids) == 1
    assert second.signal_ids != first.signal_ids
    runtime.close()


@pytest.mark.asyncio
async def test_one_broken_provider_does_not_abort_other_attention_sources(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")

    def broken(_now):
        raise RuntimeError("fitness provider offline")

    runtime.providers.register(
        SignalProviderManifest(id="broken", version=1, domains=("fitness",)),
        broken,
    )
    runtime.providers.register(
        SignalProviderManifest(id="healthy", version=1, domains=("tasks",)),
        lambda _now: [_signal("sig_healthy", domain="tasks")],
    )

    collected = await runtime.engine.refresh(now=NOW)

    assert [item.id for item in collected] == ["sig_healthy"]
    assert runtime.store.get_signal("sig_healthy") is not None
    assert runtime.providers.last_failures[0].provider_id == "broken"
    runtime.close()


def test_mcp_learning_contract_forwards_pattern_and_policy_observations(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    signal = McpAlertSignalAdapter.convert(
        {
            "ack_server": "wellbeing",
            "event_id": "observation-1",
            "title": "用户在晚间更希望保持安静",
            "metadata": {
                "pattern_observation": {
                    "kind": "availability_pattern",
                    "scene": "evening_walk",
                    "recurrence": {
                        "timezone": "UTC",
                        "days": ["thu"],
                        "start": "07:50",
                        "end": "08:20",
                    },
                    "available_minutes": 20,
                },
                "policy_observation": {
                    "scope": {"action_type": "notify"},
                    "conditions": {"scene": "evening_walk"},
                    "effect": "adjust_score",
                    "score_adjustment": -0.2,
                },
            },
        },
        now=NOW,
    )

    runtime.engine.ingest_signal(signal)

    assert len(runtime.store.list_observations()) == 2
    assert len(runtime.store.list_patterns()) == 1
    assert len(runtime.store.list_policies()) == 1
    runtime.close()


def test_observation_deduplication_is_stable_when_llm_reorders_items(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    opportunity = {
        "type": "opportunity",
        "statement": "通勤有空闲",
        "confidence": 0.8,
        "scene": "commute",
        "recurrence": {
            "timezone": "UTC",
            "days": ["thu"],
            "start": "07:50",
            "end": "08:20",
        },
    }
    policy = {
        "type": "policy",
        "statement": "工作中减少打扰",
        "confidence": 0.8,
        "scope": {"action_type": "notify"},
        "conditions": {"scene": "work"},
        "effect": "adjust_score",
        "score_adjustment": -0.2,
    }

    assert len(
        runtime.learning.ingest_many(
            [opportunity, policy],
            source_type="conversation",
            source_ref="conversation:stable",
            observed_at=NOW,
        )
    ) == 2
    assert runtime.learning.ingest_many(
        [policy, opportunity],
        source_type="conversation",
        source_ref="conversation:stable",
        observed_at=NOW,
    ) == []
    assert len(runtime.store.list_observations()) == 2
    runtime.close()


def test_feedback_and_delivery_history_feed_the_next_decision_profile(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.engine.ingest_signal(_signal("sig_feedback"))
    plan = runtime.engine.evaluate(context=_context()).plan
    assert plan is not None
    runtime.store.transition_plan(plan.id, ActionPlanStatus.EXECUTING)
    runtime.store.transition_plan(plan.id, ActionPlanStatus.SUCCEEDED)
    runtime.feedback.record(
        plan_id=plan.id,
        kind=FeedbackKind.TOO_FREQUENT,
        now=NOW + timedelta(minutes=1),
    )

    attributes = runtime.feedback.decision_attributes(
        now=NOW + timedelta(minutes=2)
    )
    profile = attributes["capability_preferences"][
        "message.notify|domain:general"
    ]
    assert attributes["frequency_counts"]["message.notify"] == 1
    assert profile["historical_acceptance"] == 0.5
    assert profile["repetition_penalty"] >= 0.9
    runtime.close()


@pytest.mark.asyncio
async def test_too_frequent_feedback_quiets_normal_nudges_but_not_critical_alerts(
    tmp_path: Path,
) -> None:
    runtime = build_personal_runtime(tmp_path, None)

    first = await runtime.attention_source.alert_fn(
        now=NOW,
        external_alerts=[
            {
                "ack_server": "simulator",
                "event_id": "first",
                "title": "普通状态变化",
                "severity": "warning",
                "urgency": 0.65,
            }
        ],
    )
    assert len(first) == 1
    first_plan_id = first[0]["action_plan_id"]
    runtime.attention_source.complete_action_plan(first_plan_id, now=NOW)
    runtime.attention.feedback.record(
        plan_id=first_plan_id,
        kind=FeedbackKind.TOO_FREQUENT,
        now=NOW + timedelta(minutes=1),
    )

    normal = await runtime.attention_source.alert_fn(
        now=NOW + timedelta(minutes=2),
        external_alerts=[
            {
                "ack_server": "simulator",
                "event_id": "second-normal",
                "title": "又一个普通状态变化",
                "severity": "warning",
                "urgency": 0.65,
            }
        ],
    )
    critical = await runtime.attention_source.alert_fn(
        now=NOW + timedelta(minutes=3),
        external_alerts=[
            {
                "ack_server": "simulator",
                "event_id": "critical",
                "title": "需要立即关注的可靠风险",
                "severity": "critical",
                "urgency": 0.99,
                "confidence": 0.99,
            }
        ],
    )

    assert normal == []
    assert len(critical) == 1
    assert critical[0]["title"] == "需要立即关注的可靠风险"
    runtime.close()


@pytest.mark.asyncio
async def test_frequency_feedback_is_scoped_and_does_not_silence_other_domains(
    tmp_path: Path,
) -> None:
    runtime = build_personal_runtime(tmp_path, None)
    first = await runtime.attention_source.alert_fn(
        now=NOW,
        external_alerts=[
            {
                "ack_server": "content",
                "event_id": "content-1",
                "domain": "content",
                "title": "一条普通兴趣内容",
                "severity": "warning",
            }
        ],
    )
    assert first
    runtime.attention_source.complete_action_plan(first[0]["action_plan_id"], now=NOW)
    runtime.attention.feedback.record(
        plan_id=first[0]["action_plan_id"],
        kind=FeedbackKind.TOO_FREQUENT,
        now=NOW + timedelta(minutes=1),
    )

    fitness = await runtime.attention_source.alert_fn(
        now=NOW + timedelta(minutes=2),
        external_alerts=[
            {
                "ack_server": "fitness",
                "event_id": "fitness-1",
                "domain": "fitness",
                "title": "训练恢复状态值得看一眼",
                "severity": "warning",
                "confidence": 0.9,
            }
        ],
    )

    assert fitness
    assert fitness[0]["title"] == "训练恢复状态值得看一眼"
    runtime.close()


def test_handled_signal_does_not_starve_a_newer_important_signal(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.engine.ingest_signal(_signal("sig_old", severity=0.95, urgency=0.95))
    first = runtime.engine.evaluate(context=_context()).plan
    assert first is not None
    runtime.store.transition_plan(first.id, ActionPlanStatus.EXECUTING)
    runtime.store.transition_plan(first.id, ActionPlanStatus.SUCCEEDED)
    runtime.engine.ingest_signal(_signal("sig_new", severity=0.9, urgency=0.9))

    second = runtime.engine.evaluate(context=_context()).plan

    assert second is not None
    assert second.id != first.id
    assert "sig_new" in second.signal_ids
    runtime.close()


def test_stale_pending_plan_expires_before_the_next_evaluation(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.engine.ingest_signal(
        _signal("sig_expiring", expires_in=timedelta(minutes=10))
    )
    plan = runtime.engine.evaluate(context=_context()).plan
    assert plan is not None

    runtime.engine.evaluate(context=_context(NOW + timedelta(minutes=11)))

    assert runtime.store.get_plan(plan.id).status == ActionPlanStatus.EXPIRED
    runtime.close()


@pytest.mark.asyncio
async def test_large_mixed_signal_simulation_selects_value_without_flooding(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    domains = ("tasks", "emotion", "fitness", "sleep", "calendar", "content")
    simulated = []
    for index in range(2000):
        simulated.append(
            _signal(
                f"sig_load_{index}",
                severity=0.35 + (index % 30) / 100,
                urgency=0.3 + (index % 25) / 100,
                domain=domains[index % len(domains)],
            )
        )
    simulated.append(
        _signal(
            "sig_critical_recovery",
            severity=0.99,
            urgency=0.98,
            domain="fitness",
        )
    )
    runtime.providers.register(
        SignalProviderManifest(id="load", version=1, domains=("*",)),
        lambda _now: simulated,
    )
    started = perf_counter()
    await runtime.engine.refresh(now=NOW)

    evaluation = runtime.engine.evaluate(context=_context())
    elapsed = perf_counter() - started

    assert evaluation.plan is not None
    assert "sig_critical_recovery" in evaluation.plan.signal_ids
    assert len(runtime.store.list_plans(limit=1000)) == 1
    assert evaluation.candidate_count == 2001
    assert elapsed < 8
    runtime.close()
