from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from core.attention._shared import parse_datetime, utc_iso
from core.attention.actions import (
    ActionCandidate,
    ActionCapabilityRegistry,
    ActionPlan,
    ActionPlanStatus,
)
from core.attention.learning import AttentionLearningService
from core.attention.opportunities import OpportunityManager, OpportunityWindow
from core.attention.planning import ActionPlanner
from core.attention.policies import DecisionContext, PolicyDecision, PolicyEngine
from core.attention.ports import AttentionRepository
from core.attention.scoring import UtilityScorer
from core.attention.signals import AttentionSignal, SignalProviderRegistry
from core.tracing import record_trace_event


@dataclass(frozen=True)
class AttentionEvaluation:
    plan: ActionPlan | None
    windows: tuple[OpportunityWindow, ...]
    candidate_count: int
    denied_count: int
    below_threshold_count: int
    reason: str


class AttentionEngine:
    """Generic opportunity-to-action planner; it never executes side effects."""

    def __init__(
        self,
        *,
        repository: AttentionRepository,
        capabilities: ActionCapabilityRegistry,
        providers: SignalProviderRegistry | None = None,
        minimum_score: float = 0.38,
        opportunity_manager: OpportunityManager | None = None,
        planner: ActionPlanner | None = None,
        scorer: UtilityScorer | None = None,
        policy_engine: PolicyEngine | None = None,
        learning: AttentionLearningService | None = None,
    ) -> None:
        self.repository = repository
        self.capabilities = capabilities
        self.providers = providers
        self.minimum_score = float(minimum_score)
        self.opportunity_manager = opportunity_manager or OpportunityManager()
        self.planner = planner or ActionPlanner()
        self.scorer = scorer or UtilityScorer()
        self.policy_engine = policy_engine or PolicyEngine()
        self.learning = learning or AttentionLearningService(repository)

    def ingest_signal(self, signal: AttentionSignal) -> AttentionSignal:
        stored = self.repository.upsert_signal(signal)
        self.learning.ingest_signal(signal)
        return stored

    async def refresh(self, *, now: datetime) -> list[AttentionSignal]:
        if self.providers is None:
            return []
        signals = await self.providers.collect(now=now)
        stored = self.repository.upsert_signals(signals)
        for signal in stored:
            self.learning.ingest_signal(signal)
        return stored

    def evaluate(
        self,
        *,
        context: DecisionContext,
    ) -> AttentionEvaluation:
        now = context.now.astimezone(timezone.utc)
        self._expire_stale_plans(now=now)
        self.learning.refresh_lifecycle(now=now)
        patterns = self.repository.list_patterns()
        unavailable_signal_ids = {
            signal_id
            for plan in self.repository.list_plans(limit=1000)
            if plan.status
            in {
                ActionPlanStatus.PENDING_APPROVAL,
                ActionPlanStatus.EXECUTING,
                ActionPlanStatus.SUCCEEDED,
                ActionPlanStatus.SKIPPED,
                ActionPlanStatus.FAILED,
            }
            for signal_id in plan.signal_ids
        }
        signals = [
            signal
            for signal in self.repository.list_active_signals(now=now)
            if signal.id not in unavailable_signal_ids
        ]
        materialized = self.opportunity_manager.materialize(
            patterns=patterns,
            signals=signals,
            now=now,
            scene=context.scene,
        )
        self.repository.upsert_windows(materialized)
        windows = self.repository.list_active_windows(now=now)
        if not windows:
            return AttentionEvaluation(
                plan=None,
                windows=(),
                candidate_count=0,
                denied_count=0,
                below_threshold_count=0,
                reason="no_active_opportunity",
            )
        preferences = dict(context.attributes.get("capability_preferences") or {})
        candidates = self.planner.generate(
            signals=signals,
            windows=windows,
            capabilities=self.capabilities.list(),
            preference_features=preferences,
        )
        if not candidates:
            return AttentionEvaluation(
                plan=None,
                windows=tuple(windows),
                candidate_count=0,
                denied_count=0,
                below_threshold_count=0,
                reason="no_compatible_action",
            )
        policies = self.repository.list_policies()
        windows_by_id = {item.id: item for item in windows}
        allowed: list[tuple[ActionCandidate, PolicyDecision]] = []
        denied = 0
        below = 0
        for candidate in candidates:
            policy = self.policy_engine.evaluate(
                candidate=candidate,
                context=context,
                policies=policies,
            )
            if not policy.allowed:
                denied += 1
                continue
            scored = self.scorer.score(
                candidate,
                policy_adjustment=policy.score_adjustment,
            )
            if scored.score < self.minimum_score:
                below += 1
                continue
            allowed.append((scored, policy))
        if not allowed:
            return AttentionEvaluation(
                plan=None,
                windows=tuple(windows),
                candidate_count=len(candidates),
                denied_count=denied,
                below_threshold_count=below,
                reason="all_actions_filtered",
            )
        allowed.sort(key=lambda item: (-item[0].score, item[0].id))
        selected: tuple[ActionCandidate, PolicyDecision] | None = None
        for candidate, policy in allowed:
            candidate_window = windows_by_id[candidate.opportunity_id]
            existing = self.repository.find_plan_by_key(
                self._idempotency_key(candidate.id, candidate_window.available_from)
            )
            if existing is not None and existing.status in {
                ActionPlanStatus.SUCCEEDED,
                ActionPlanStatus.SKIPPED,
                ActionPlanStatus.EXPIRED,
                ActionPlanStatus.FAILED,
            }:
                # A handled event must not win the ranking again and starve a
                # newer, slightly lower-scored signal in the same tick.
                continue
            selected = (candidate, policy)
            break
        if selected is None:
            return AttentionEvaluation(
                plan=None,
                windows=tuple(windows),
                candidate_count=len(candidates),
                denied_count=denied,
                below_threshold_count=below,
                reason="no_new_action",
            )
        candidate, policy = selected
        window = next(item for item in windows if item.id == candidate.opportunity_id)
        approval_required = (
            policy.require_approval
            or not (
                self.capabilities.get(candidate.capability_id) or _MissingCapability()
            ).auto_execute
        )
        status = (
            ActionPlanStatus.PENDING_APPROVAL
            if approval_required
            else ActionPlanStatus.PROPOSED
        )
        approval = "required" if approval_required else "not_required"
        idempotency_key = self._idempotency_key(candidate.id, window.available_from)
        plan = ActionPlan.from_candidate(
            candidate,
            idempotency_key=idempotency_key,
            status=status,
            approval=approval,
            policy_ids=policy.matched_policy_ids,
            created_at=utc_iso(now),
            expires_at=window.available_until,
        )
        plan = self.repository.create_plan(plan)
        record_trace_event(
            category="attention",
            name="action_plan",
            summary=f"形成主动协助计划：{candidate.capability_id}",
            payload={
                "plan_id": plan.id,
                "capability_id": candidate.capability_id,
                "score": round(candidate.score, 4),
                "status": plan.status.value,
                "approval": plan.approval,
                "signal_ids": list(plan.signal_ids),
                "policy_ids": list(plan.policy_ids),
                "opportunity_id": candidate.opportunity_id,
            },
        )
        return AttentionEvaluation(
            plan=plan,
            windows=tuple(windows),
            candidate_count=len(candidates),
            denied_count=denied,
            below_threshold_count=below,
            reason="action_planned",
        )

    def _expire_stale_plans(self, *, now: datetime) -> None:
        expirable = {
            ActionPlanStatus.PROPOSED,
            ActionPlanStatus.PENDING_APPROVAL,
            ActionPlanStatus.APPROVED,
            ActionPlanStatus.DEFERRED,
        }
        for plan in self.repository.list_plans(limit=1000):
            expires = parse_datetime(plan.expires_at)
            if plan.status in expirable and expires is not None and expires < now:
                self.repository.transition_plan(plan.id, ActionPlanStatus.EXPIRED)

    @staticmethod
    def _idempotency_key(candidate_id: str, window_start: str) -> str:
        digest = hashlib.sha256(
            f"{candidate_id}|{window_start}".encode("utf-8")
        ).hexdigest()
        return f"attention:{digest}"


@dataclass(frozen=True)
class _MissingCapability:
    auto_execute: bool = False


__all__ = ["AttentionEngine", "AttentionEvaluation"]
