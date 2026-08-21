from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.attention.actions import (
    ActionCapability,
    ActionCapabilityRegistry,
    ActionRisk,
)
from core.attention.actions.executor import (
    ActionExecutionService,
    ActionHandlerRegistry,
)
from core.attention.engine import AttentionEngine
from core.attention.feedback.service import FeedbackService
from core.attention.learning import AttentionLearningService
from core.attention.signals import SignalProviderRegistry
from infra.persistence.attention_engine_store import AttentionEngineStore


@dataclass
class AttentionRuntime:
    store: AttentionEngineStore
    capabilities: ActionCapabilityRegistry
    handlers: ActionHandlerRegistry
    execution: ActionExecutionService
    providers: SignalProviderRegistry
    engine: AttentionEngine
    learning: AttentionLearningService
    feedback: FeedbackService

    def close(self) -> None:
        self.store.close()


def build_attention_runtime(database: Path) -> AttentionRuntime:
    store = AttentionEngineStore(database)
    capabilities = ActionCapabilityRegistry()
    capabilities.register(
        ActionCapability(
            id="message.notify",
            name="主动消息",
            description="通过当前主动渠道发送一条有依据的低打扰消息",
            provider="system.outbound",
            action_type="notify",
            risk=ActionRisk.NOTIFY,
            auto_execute=True,
            supported_domains=("*",),
            supported_scenes=("*",),
            minimum_minutes=1,
            maximum_minutes=60,
            default_minutes=5,
            interruption_cost=0.12,
        )
    )
    providers = SignalProviderRegistry()
    handlers = ActionHandlerRegistry()
    learning = AttentionLearningService(store)
    engine = AttentionEngine(
        repository=store,
        capabilities=capabilities,
        providers=providers,
        learning=learning,
    )
    return AttentionRuntime(
        store=store,
        capabilities=capabilities,
        handlers=handlers,
        execution=ActionExecutionService(store, handlers),
        providers=providers,
        engine=engine,
        learning=learning,
        feedback=FeedbackService(store),
    )


__all__ = ["AttentionRuntime", "build_attention_runtime"]
