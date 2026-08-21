from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bootstrap.attention import build_attention_runtime
from bootstrap.personal import build_personal_runtime
from bus.event_bus import EventBus
from core.attention.actions import (
    ActionCapability,
    ActionPlanStatus,
    ActionRisk,
)
from core.attention.actions.executor import (
    ActionExecutionService,
    ActionHandlerRegistry,
)
from core.attention.feedback import FeedbackKind
from core.attention.feedback.service import FeedbackService
from core.attention.learning import ObservationKind
from core.attention.patterns import (
    BehaviorPattern,
    PatternSource,
    PatternStatus,
    RecurrenceSpec,
)
from core.attention.policies import (
    DecisionContext,
    PolicyEffect,
    PolicyRule,
    PolicyStatus,
)
from core.attention.signals import AttentionSignal, SignalSource, SignalValence
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import SemanticBatchPayload
from core.personal.models import FollowUpTrigger, PersonalEntityType, RecordSource
from infra.persistence.attention_engine_store import AttentionEngineStore

NOW = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)


def _signal(
    *,
    kind: str = "test.actionable",
    domain: str = "personal",
    severity: float = 0.8,
    capability: str = "message.notify",
    metadata: dict | None = None,
    signal_id: str | None = None,
) -> AttentionSignal:
    return AttentionSignal.create(
        signal_id=signal_id,
        kind=kind,
        domain=domain,
        summary="值得处理的变化",
        source=SignalSource("test", "provider", "event:1"),
        occurred_at=NOW,
        expires_at=NOW + timedelta(hours=2),
        valence=SignalValence.NEGATIVE,
        severity=severity,
        urgency=0.75,
        actionability=0.9,
        confidence=0.92,
        freshness=1.0,
        suggested_capabilities=(capability,),
        metadata=metadata or {},
    )


def _context(**changes) -> DecisionContext:
    values = {
        "now": NOW,
        "scene": "neutral",
        "permission_mode": "delegated",
    }
    values.update(changes)
    return DecisionContext(**values)


def test_store_round_trips_all_governed_attention_records(tmp_path: Path) -> None:
    store = AttentionEngineStore(tmp_path / "personal.db")
    signal = store.upsert_signal(_signal(signal_id="sig_roundtrip"))
    pattern = store.upsert_pattern(
        BehaviorPattern.create(
            pattern_id="pat_commute",
            kind="availability_pattern",
            scene="commute",
            recurrence=RecurrenceSpec(
                timezone="Asia/Shanghai",
                days=("mon", "tue", "wed", "thu", "fri"),
                start="08:00",
                end="08:25",
            ),
            available_minutes=20,
            confidence=1.0,
            source=PatternSource.USER,
            user_locked=True,
        )
    )
    policy = store.upsert_policy(
        PolicyRule.create(
            policy_id="pol_test",
            effect=PolicyEffect.DENY,
            scope={"domain": "fitness"},
            source="user",
        )
    )

    assert store.get_signal(signal.id) == signal
    assert store.get_pattern(pattern.id) == pattern
    assert store.list_policies() == [policy]
    store.close()


def test_generic_pattern_observations_activate_without_scenario_code(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    observation = {
        "pattern_observation": {
            "kind": "availability_pattern",
            "scene": "custom_scene",
            "available_minutes": 17,
            "confidence": 0.9,
            "recurrence": {
                "timezone": "UTC",
                "days": ("wed",),
                "start": "09:55",
                "end": "10:20",
            },
        }
    }
    for index in range(3):
        runtime.engine.ingest_signal(
            _signal(
                signal_id=f"sig_observation_{index}",
                metadata=observation,
            )
        )

    patterns = runtime.store.list_patterns()
    assert len(patterns) == 1
    pattern = patterns[0]
    assert pattern.scene == "custom_scene"
    assert pattern.observation_count == 3
    assert pattern.status == PatternStatus.ACTIVE
    runtime.close()


def test_learning_observations_are_idempotent_and_promote_policy(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    observation = {
        "type": "policy",
        "statement": "工作时非紧急通知会打断我",
        "confidence": 0.9,
        "scope": {"action_type": "notify"},
        "conditions": {"scene": ["work"]},
        "effect": "adjust_score",
        "score_adjustment": -0.25,
    }

    first = runtime.learning.ingest_many(
        [observation],
        source_type="conversation",
        source_ref="conversation:1",
        observed_at=NOW,
    )
    duplicate = runtime.learning.ingest_many(
        [observation],
        source_type="conversation",
        source_ref="conversation:1",
        observed_at=NOW,
    )
    for index in (2, 3):
        runtime.learning.ingest_many(
            [observation],
            source_type="conversation",
            source_ref=f"conversation:{index}",
            observed_at=NOW + timedelta(days=index),
        )

    assert len(first) == 1
    assert duplicate == []
    assert len(runtime.store.list_observations()) == 3
    policy = runtime.store.list_policies()[0]
    assert policy.status == PolicyStatus.ACTIVE
    assert policy.observation_count == 3
    assert policy.user_locked is False
    runtime.close()


def test_policy_observations_share_a_slot_but_keep_distinct_variants(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    base = {
        "type": "policy",
        "statement": "工作时普通通知会影响专注",
        "confidence": 0.8,
        "scope": {"action_type": "notify"},
        "conditions": {"scene": ["work"]},
        "effect": "adjust_score",
    }

    negative = runtime.learning.ingest_many(
        [{**base, "score_adjustment": -0.25}],
        source_type="conversation",
        source_ref="conversation:negative",
        observed_at=NOW,
    )[0]
    positive = runtime.learning.ingest_many(
        [{**base, "score_adjustment": 0.2}],
        source_type="conversation",
        source_ref="conversation:positive",
        observed_at=NOW + timedelta(days=1),
    )[0]

    assert negative.rule_key == positive.rule_key
    assert negative.variant_key != positive.variant_key
    runtime.close()


def test_opportunity_identity_versions_time_without_merging_weekdays(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")

    def observation(*, start: str, days: tuple[str, ...]) -> dict:
        return {
            "type": "opportunity",
            "statement": "通勤时通常有一段空闲时间",
            "confidence": 0.8,
            "scene": "commute",
            "kind": "availability_pattern",
            "recurrence": {
                "timezone": "Asia/Shanghai",
                "days": days,
                "start": start,
                "end": "08:30",
            },
            "available_minutes": 20,
        }

    weekday_early = runtime.learning.ingest_many(
        [observation(start="08:00", days=("mon", "tue", "wed", "thu", "fri"))],
        source_type="conversation",
        source_ref="conversation:weekday-early",
        observed_at=NOW,
    )[0]
    weekday_late = runtime.learning.ingest_many(
        [observation(start="08:10", days=("mon", "tue", "wed", "thu", "fri"))],
        source_type="conversation",
        source_ref="conversation:weekday-late",
        observed_at=NOW + timedelta(days=1),
    )[0]
    weekend = runtime.learning.ingest_many(
        [observation(start="08:10", days=("sat", "sun"))],
        source_type="conversation",
        source_ref="conversation:weekend",
        observed_at=NOW + timedelta(days=2),
    )[0]

    assert weekday_early.rule_key == weekday_late.rule_key
    assert weekday_early.variant_key != weekday_late.variant_key
    assert weekday_late.rule_key != weekend.rule_key
    runtime.close()


def test_verified_user_policy_correction_suspends_the_previous_variant(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    base = {
        "type": "policy",
        "confidence": 0.98,
        "user_directive": True,
        "_user_evidence_verified": True,
        "scope": {"action_type": "notify"},
        "conditions": {"focus_active": True},
    }
    runtime.learning.ingest_many(
        [
            {
                **base,
                "statement": "我专注时不要主动通知我",
                "source_message_id": "conversation:policy:0",
                "effect": "deny",
            }
        ],
        source_type="conversation",
        source_ref="conversation:policy",
        observed_at=NOW,
        trust_user_evidence=True,
    )
    runtime.learning.ingest_many(
        [
            {
                **base,
                "statement": "更正：专注时可以通知，但请降低频率",
                "source_message_id": "conversation:policy:1",
                "effect": "adjust_score",
                "score_adjustment": -0.15,
            }
        ],
        source_type="conversation",
        source_ref="conversation:policy",
        observed_at=NOW + timedelta(days=1),
        trust_user_evidence=True,
    )

    policies = runtime.store.list_policies()
    assert len(policies) == 2
    active = [item for item in policies if item.status == PolicyStatus.ACTIVE]
    suspended = [item for item in policies if item.status == PolicyStatus.SUSPENDED]
    assert len(active) == 1
    assert len(suspended) == 1
    assert active[0].effect == PolicyEffect.ADJUST_SCORE
    assert active[0].user_locked is True
    assert active[0].metadata["supersedes_id"] == suspended[0].id
    assert suspended[0].metadata["superseded_by"] == active[0].id
    runtime.close()


def test_inferred_policy_cannot_override_a_user_locked_variant(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    direct = {
        "type": "policy",
        "statement": "我专注时不要主动通知我",
        "confidence": 0.99,
        "user_directive": True,
        "_user_evidence_verified": True,
        "source_message_id": "conversation:locked:0",
        "scope": {"action_type": "notify"},
        "conditions": {"focus_active": True},
        "effect": "deny",
    }
    runtime.learning.ingest_many(
        [direct],
        source_type="conversation",
        source_ref="conversation:locked",
        observed_at=NOW,
        trust_user_evidence=True,
    )
    inferred = {
        **direct,
        "statement": "最近专注时似乎也接受普通通知",
        "confidence": 0.95,
        "user_directive": False,
        "_user_evidence_verified": False,
        "source_message_id": "",
        "effect": "adjust_score",
        "score_adjustment": 0.25,
    }
    for index in range(3):
        runtime.learning.ingest_many(
            [inferred],
            source_type="conversation",
            source_ref=f"conversation:inferred:{index}",
            observed_at=NOW + timedelta(days=index + 1),
        )

    policies = runtime.store.list_policies()
    assert len(policies) == 2
    locked = next(item for item in policies if item.user_locked)
    learned = next(item for item in policies if not item.user_locked)
    assert locked.effect == PolicyEffect.DENY
    assert locked.status == PolicyStatus.ACTIVE
    assert locked.observation_count == 1
    assert learned.effect == PolicyEffect.ADJUST_SCORE
    assert learned.status == PolicyStatus.PROPOSED
    runtime.close()


def test_verified_opportunity_correction_versions_the_window(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    base = {
        "type": "opportunity",
        "confidence": 0.98,
        "explicit_user_statement": True,
        "_user_evidence_verified": True,
        "scene": "commute",
        "kind": "availability_pattern",
        "available_minutes": 20,
    }
    for index, start in enumerate(("08:00", "08:15")):
        runtime.learning.ingest_many(
            [
                {
                    **base,
                    "statement": f"工作日现在从{start}开始通勤",
                    "source_message_id": f"conversation:window:{index}",
                    "recurrence": {
                        "timezone": "Asia/Shanghai",
                        "days": ["mon", "tue", "wed", "thu", "fri"],
                        "start": start,
                        "end": "08:40",
                    },
                }
            ],
            source_type="conversation",
            source_ref="conversation:window",
            observed_at=NOW + timedelta(days=index),
            trust_user_evidence=True,
        )

    patterns = runtime.store.list_patterns()
    assert len(patterns) == 2
    active = next(item for item in patterns if item.status == PatternStatus.ACTIVE)
    suspended = next(
        item for item in patterns if item.status == PatternStatus.SUSPENDED
    )
    assert active.recurrence.start == "08:15"
    assert active.metadata["supersedes_id"] == suspended.id
    assert suspended.metadata["superseded_by"] == active.id
    runtime.close()


def test_explicit_user_policy_is_active_and_locked_immediately(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    accepted = runtime.learning.ingest_many(
        [
            {
                "type": "policy",
                "statement": "我专注时不要主动通知我",
                "confidence": 0.98,
                "user_directive": True,
                "source_message_id": "conversation:directive:0",
                "_user_evidence_verified": True,
                "scope": {"action_type": "notify"},
                "conditions": {"focus_active": True},
                "effect": "deny",
            }
        ],
        source_type="conversation",
        source_ref="conversation:directive",
        observed_at=NOW,
        trust_user_evidence=True,
    )

    assert len(accepted) == 1
    policy = runtime.store.list_policies()[0]
    assert policy.status == PolicyStatus.ACTIVE
    assert policy.user_locked is True
    assert policy.source == "user"
    runtime.close()


def test_unverified_user_directive_cannot_lock_an_attention_policy(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.learning.ingest_many(
        [
            {
                "type": "policy",
                "statement": "模型猜测用户专注时不想被打扰",
                "confidence": 0.98,
                "user_directive": True,
                "_user_evidence_verified": True,
                "scope": {"action_type": "notify"},
                "conditions": {"focus_active": True},
                "effect": "deny",
            }
        ],
        source_type="conversation",
        source_ref="conversation:unverified",
        observed_at=NOW,
    )

    policy = runtime.store.list_policies()[0]
    assert policy.status == PolicyStatus.PROPOSED
    assert policy.user_locked is False
    assert policy.source == "learned"
    runtime.close()


def test_user_directive_promotes_matching_inference_without_duplicate(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    base = {
        "type": "policy",
        "statement": "专注时不适合出现普通通知",
        "confidence": 0.65,
        "scope": {"action_type": "notify"},
        "conditions": {"focus_active": True},
        "effect": "deny",
    }
    runtime.learning.ingest_many(
        [base],
        source_type="conversation",
        source_ref="conversation:inference",
        observed_at=NOW,
    )
    runtime.learning.ingest_many(
        [
            {
                **base,
                "statement": "我专注时不要主动通知我",
                "confidence": 0.99,
                "user_directive": True,
                "source_message_id": "conversation:directive:1",
                "_user_evidence_verified": True,
            }
        ],
        source_type="conversation",
        source_ref="conversation:directive",
        observed_at=NOW + timedelta(days=1),
        trust_user_evidence=True,
    )

    policies = runtime.store.list_policies()
    assert len(policies) == 1
    assert policies[0].status == PolicyStatus.ACTIVE
    assert policies[0].source == "user"
    assert policies[0].user_locked is True
    assert policies[0].observation_count == 2
    runtime.close()


def test_learned_policy_decay_is_incremental_and_can_expire(tmp_path: Path) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.store.upsert_policy(
        PolicyRule.create(
            policy_id="pol_decay",
            effect=PolicyEffect.ADJUST_SCORE,
            scope={"action_type": "notify"},
            status=PolicyStatus.ACTIVE,
            confidence=0.8,
            observation_count=4,
            last_observed_at=NOW - timedelta(days=60),
            source="learned",
        )
    )

    runtime.learning.refresh_lifecycle(now=NOW)
    once = runtime.store.get_policy("pol_decay")
    assert once is not None
    assert once.confidence == pytest.approx(0.64)
    assert once.status == PolicyStatus.PROPOSED

    runtime.learning.refresh_lifecycle(now=NOW)
    repeated = runtime.store.get_policy("pol_decay")
    assert repeated is not None
    assert repeated.confidence == pytest.approx(once.confidence)

    runtime.learning.refresh_lifecycle(now=NOW + timedelta(days=150))
    expired = runtime.store.get_policy("pol_decay")
    assert expired is not None
    assert expired.status == PolicyStatus.EXPIRED
    runtime.close()


@pytest.mark.asyncio
async def test_conversation_semantic_batch_feeds_attention_learning(
    tmp_path: Path,
) -> None:
    event_bus = EventBus()
    runtime = build_personal_runtime(tmp_path, None, event_bus=event_bus)

    await event_bus.emit(
        ConversationSemanticBatchCommitted(
            batch_id="conversation:event:1",
            session_key="dashboard:chat-1",
            channel="dashboard",
            chat_id="chat-1",
            analysis_version="conversation-v1",
            message_ids=("dashboard:chat-1:0",),
            user_message_ids=("dashboard:chat-1:0",),
            end_seq=0,
            context_consolidate_through=-1,
            payload=SemanticBatchPayload.from_mapping(
                {
                    "attention_observations": [
                        {
                            "type": "opportunity",
                            "statement": "工作日八点通勤时通常有二十分钟空闲",
                            "confidence": 0.95,
                            "explicit_user_statement": True,
                            "source_message_id": "dashboard:chat-1:0",
                            "scene": "commute",
                            "kind": "availability_pattern",
                            "recurrence": {
                                "timezone": "Asia/Shanghai",
                                "days": ["mon", "tue", "wed", "thu", "fri"],
                                "start": "08:00",
                                "end": "08:20",
                            },
                            "available_minutes": 20,
                        }
                    ]
                }
            ),
        )
    )

    observations = runtime.attention.store.list_observations()
    assert len(observations) == 1
    assert observations[0].kind == ObservationKind.OPPORTUNITY
    pattern = runtime.attention.store.list_patterns()[0]
    assert pattern.status == PatternStatus.ACTIVE
    assert pattern.source == PatternSource.USER
    assert pattern.user_locked is True
    runtime.close()


def test_recurring_window_can_propose_content_without_a_trigger_signal(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.capabilities.register(
        ActionCapability(
            id="content.digest",
            name="兴趣内容摘要",
            description="按可用时长选择内容",
            provider="mcp.content",
            action_type="content",
            risk=ActionRisk.NOTIFY,
            auto_execute=True,
            supported_domains=("*",),
            supported_scenes=("commute",),
            minimum_minutes=5,
            maximum_minutes=30,
            default_minutes=15,
            can_propose_without_signal=True,
        )
    )
    runtime.store.upsert_pattern(
        BehaviorPattern.create(
            pattern_id="pat_commute",
            kind="availability_pattern",
            scene="commute",
            recurrence=RecurrenceSpec(
                timezone="UTC",
                days=("wed",),
                start="09:55",
                end="10:25",
            ),
            available_minutes=20,
            confidence=1.0,
            source=PatternSource.USER,
            user_locked=True,
        )
    )

    evaluation = runtime.engine.evaluate(context=_context(scene="commute"))

    assert evaluation.plan is not None
    assert evaluation.plan.capability_id == "content.digest"
    assert evaluation.plan.inputs["available_minutes"] == 20
    runtime.close()


def test_fitness_signal_uses_generic_event_window_and_capability(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.capabilities.register(
        ActionCapability(
            id="fitness.recovery_summary",
            name="训练恢复建议",
            description="根据健身提供者的证据生成简短恢复建议",
            provider="mcp.fitness",
            action_type="wellbeing_guidance",
            risk=ActionRisk.NOTIFY,
            auto_execute=True,
            supported_domains=("fitness",),
            supported_scenes=("post_workout",),
            minimum_minutes=1,
            maximum_minutes=15,
            default_minutes=5,
        )
    )
    runtime.engine.ingest_signal(
        _signal(
            kind="fitness.provider_assessment",
            domain="fitness",
            capability="fitness.recovery_summary",
            metadata={
                "goal_alignment": 0.85,
                "opportunity": {
                    "scene": "post_workout",
                    "duration_minutes": 60,
                    "available_minutes": 5,
                },
            },
        )
    )

    evaluation = runtime.engine.evaluate(context=_context(scene="post_workout"))

    assert evaluation.plan is not None
    assert evaluation.plan.capability_id == "fitness.recovery_summary"
    assert evaluation.plan.status == ActionPlanStatus.PROPOSED
    runtime.close()


def test_do_not_disturb_blocks_low_severity_but_allows_high_override(
    tmp_path: Path,
) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.engine.ingest_signal(_signal(severity=0.5, signal_id="sig_low"))
    blocked = runtime.engine.evaluate(
        context=_context(do_not_disturb=True, allow_high_priority=True)
    )
    assert blocked.plan is None
    assert blocked.denied_count >= 1

    runtime.engine.ingest_signal(_signal(severity=0.95, signal_id="sig_high"))
    allowed = runtime.engine.evaluate(
        context=_context(do_not_disturb=True, allow_high_priority=True)
    )
    assert allowed.plan is not None
    runtime.close()


def test_dynamic_policy_can_deny_and_then_expire(tmp_path: Path) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.engine.ingest_signal(_signal(domain="fitness"))
    runtime.store.upsert_policy(
        PolicyRule.create(
            policy_id="pol_pause_fitness",
            effect=PolicyEffect.DENY,
            scope={"domain": "fitness"},
            priority=100,
            effective_from=NOW - timedelta(hours=1),
            expires_at=NOW + timedelta(hours=1),
            source="user",
        )
    )
    blocked = runtime.engine.evaluate(context=_context())
    assert blocked.plan is None

    later = NOW + timedelta(hours=2)
    runtime.engine.ingest_signal(
        AttentionSignal.create(
            kind="fitness.new_assessment",
            domain="fitness",
            summary="新的训练评估",
            source=SignalSource("test", "fitness"),
            occurred_at=later,
            expires_at=later + timedelta(hours=1),
            severity=0.9,
            urgency=0.8,
            actionability=0.9,
            confidence=0.95,
            suggested_capabilities=("message.notify",),
        )
    )
    allowed = runtime.engine.evaluate(
        context=DecisionContext(
            now=later,
            scene="neutral",
            permission_mode="delegated",
        )
    )
    assert allowed.plan is not None
    runtime.close()


@pytest.mark.asyncio
async def test_action_execution_and_feedback_have_durable_state(tmp_path: Path) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.engine.ingest_signal(_signal())
    evaluation = runtime.engine.evaluate(context=_context())
    assert evaluation.plan is not None
    plan = evaluation.plan

    handlers = ActionHandlerRegistry()
    handlers.register("message.notify", lambda current: {"sent": current.id})
    execution = ActionExecutionService(runtime.store, handlers)
    succeeded = await execution.execute(plan.id)

    assert succeeded.status == ActionPlanStatus.SUCCEEDED
    feedback = FeedbackService(runtime.store).record(
        plan_id=plan.id,
        kind=FeedbackKind.ACCEPTED,
    )
    assert feedback.plan_id == plan.id
    assert runtime.store.list_feedback(plan_id=plan.id) == [feedback]
    runtime.close()


def test_plan_creation_is_idempotent_inside_same_window(tmp_path: Path) -> None:
    runtime = build_attention_runtime(tmp_path / "personal.db")
    runtime.engine.ingest_signal(_signal(signal_id="sig_once"))

    first = runtime.engine.evaluate(context=_context()).plan
    second = runtime.engine.evaluate(context=_context()).plan

    assert first is not None and second is not None
    assert first.id == second.id
    assert len(runtime.store.list_plans()) == 1
    runtime.close()


@pytest.mark.asyncio
async def test_personal_record_protocol_is_planned_and_acknowledged_by_v2(
    tmp_path: Path,
) -> None:
    runtime = build_personal_runtime(tmp_path, None)
    runtime.data.create(
        entity_type=PersonalEntityType.COMMITMENT,
        title="完成汇报材料",
        summary="今天完成汇报材料",
        data={
            "state": "open",
            "due_at": (NOW + timedelta(hours=3)).isoformat(),
            "progress": 0.1,
        },
        source=RecordSource("test", "commitment:report"),
    )

    alerts = await runtime.attention_source.alert_fn(now=NOW)

    assert len(alerts) == 1
    assert alerts[0]["action_plan_id"].startswith("act_")
    plan = runtime.attention.store.get_plan(alerts[0]["action_plan_id"])
    assert plan is not None and plan.status == ActionPlanStatus.PROPOSED

    await runtime.attention_source.alert_ack_fn(
        f"attention:{plan.id}",
    )

    assert (
        runtime.attention.store.get_plan(plan.id).status == ActionPlanStatus.SUCCEEDED
    )
    assert await runtime.attention_source.alert_fn(now=NOW) == []
    runtime.close()


@pytest.mark.asyncio
async def test_declared_recurring_record_advances_after_delivery(
    tmp_path: Path,
) -> None:
    runtime = build_personal_runtime(tmp_path, None)
    record = runtime.rhythm.create_follow_up(
        title="每日复盘",
        message="简单回顾今天的进展",
        reason="保持复盘节奏",
        trigger_type=FollowUpTrigger.INTERVAL,
        next_trigger_at=(NOW - timedelta(minutes=5)).isoformat(),
        interval_minutes=60,
        user_confirmed=True,
    )

    alerts = await runtime.attention_source.alert_fn(now=NOW)
    assert len(alerts) == 1

    runtime.attention_source.complete_action_plan(
        alerts[0]["action_plan_id"],
        now=NOW,
    )

    updated = runtime.data.get(record.id)
    assert updated is not None
    next_trigger = datetime.fromisoformat(updated.data["next_trigger_at"])
    assert next_trigger > NOW
    runtime.close()


@pytest.mark.asyncio
async def test_mcp_alerts_are_normalized_planned_and_acknowledged_generically(
    tmp_path: Path,
) -> None:
    runtime = build_personal_runtime(tmp_path, None)
    alerts = await runtime.attention_source.alert_fn(
        now=NOW,
        external_alerts=[
            {
                "ack_server": "fitness-provider",
                "event_id": "daily-assessment-42",
                "signal_kind": "fitness.provider_assessment",
                "domain": "fitness",
                "title": "今天的恢复需求值得关注",
                "content": "训练负荷与恢复记录存在明显变化。",
                "severity": "high",
                "urgency": 0.8,
                "confidence": 0.95,
                "opportunity": {
                    "scene": "post_workout",
                    "duration_minutes": 90,
                    "available_minutes": 5,
                },
            }
        ],
    )

    assert len(alerts) == 1
    assert alerts[0]["ack_server"] == "attention"
    plan_id = alerts[0]["action_plan_id"]
    assert runtime.attention_source.external_alert_ack_targets(plan_id) == [
        ("fitness-provider", "daily-assessment-42")
    ]

    runtime.attention_source.complete_action_plan(plan_id)

    assert (
        runtime.attention.store.get_plan(plan_id).status == ActionPlanStatus.SUCCEEDED
    )
    assert (
        await runtime.attention_source.alert_fn(
            now=NOW,
            external_alerts=[
                {
                    "ack_server": "fitness-provider",
                    "event_id": "daily-assessment-42",
                    "signal_kind": "fitness.provider_assessment",
                    "domain": "fitness",
                    "title": "今天的恢复需求值得关注",
                    "severity": "high",
                    "opportunity": {
                        "scene": "post_workout",
                        "duration_minutes": 90,
                        "available_minutes": 5,
                    },
                }
            ],
        )
        == []
    )
    runtime.close()
