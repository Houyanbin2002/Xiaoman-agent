from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bootstrap.attention import AttentionRuntime, build_attention_runtime
from bus.event_bus import EventBus
from core.attention.semantic_consumer import ConversationAttentionBatchConsumer
from core.attention.events.runtime import AttentionWakeRuntime
from core.attention.events.service import EventDrivenAttentionService
from core.attention.events.acknowledgement import EventAcknowledgementService
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.personal.events import PersonalRecordChanged
from agent.scheduler import ScheduledJobChanged
from agent.tools.personal import (
    PersonalContextTool,
    PersonalGuidanceTool,
    PersonalRecordTool,
    PersonalRhythmControlTool,
    PersonalRoutineTool,
)
from agent.tools.attention import AttentionControlTool
from agent.tools.personal_sources import PersonalSourceTool
from agent.tools.registry import ToolRegistry
from agent.workflows.personal import PersonalRoutineService
from agent.workflows.runtime import WorkflowRuntime
from core.personal.governance import MemoryGovernanceService
from core.personal.rhythm import PersonalRhythmService
from core.personal.service import PersonalDataService
from core.personal.sources.service import ExternalSourceSyncService
from core.personal.today import PersonalTodayService
from core.attention.source import PersonalAttentionSource
from core.attention.providers import PersonalRecordSignalProvider
from core.attention.signals import SignalProviderManifest
from infra.persistence.memory_governance_store import MemoryGovernanceStore
from infra.persistence.personal_automation_store import PersonalAutomationStore
from infra.persistence.personal_store import PersonalStore
from infra.persistence.external_source_store import ExternalSourceStore
from agent.integrations.notion_personal_source import NotionPersonalSourceAdapter
from agent.integrations.mcp_personal_source import McpPersonalSourceAdapter
from agent.integrations.rss_personal_source import RssPersonalSourceAdapter


@dataclass
class PersonalRuntime:
    """Owned personal-assistant services assembled by the composition root."""

    data: PersonalDataService
    automation: PersonalAutomationStore
    governance: MemoryGovernanceService
    rhythm: PersonalRhythmService
    attention_source: PersonalAttentionSource
    attention: AttentionRuntime
    external_sources: ExternalSourceSyncService
    today: PersonalTodayService
    attention_events: EventDrivenAttentionService
    attention_wakes: AttentionWakeRuntime
    acknowledgements: EventAcknowledgementService
    routines: PersonalRoutineService | None = None
    event_unsubscribers: list[Callable[[], None]] | None = None

    def close(self) -> None:
        self.attention_wakes.stop()
        for unsubscribe in reversed(self.event_unsubscribers or []):
            unsubscribe()
        self.event_unsubscribers = []
        self.attention.close()
        self.external_sources.close()
        self.governance.close()
        self.automation.close()
        self.data.close()


def build_personal_runtime(
    workspace: Path,
    workflow_runtime: WorkflowRuntime | None,
    *,
    event_bus: EventBus | None = None,
    tools: ToolRegistry | None = None,
    scheduler: Any | None = None,
    default_channel: str = "",
    default_chat_id: str = "",
) -> PersonalRuntime:
    database = workspace / "personal.db"
    data = PersonalDataService(
        PersonalStore(database),
        event_publisher=(event_bus.enqueue if event_bus is not None else None),
    )
    governance = MemoryGovernanceService(
        personal_data=data,
        conflict_store=MemoryGovernanceStore(database),
    )
    rhythm = PersonalRhythmService(data)
    attention = build_attention_runtime(database)
    attention_events = EventDrivenAttentionService(
        repository=attention.store,
        attention_engine=attention.engine,
        scheduler=scheduler,
        default_channel=default_channel,
        default_chat_id=default_chat_id,
    )
    attention_wakes = AttentionWakeRuntime(
        repository=attention.store,
        events=attention_events,
    )
    acknowledgements = EventAcknowledgementService(
        repository=attention.store,
        personal_data=data,
        feedback=attention.feedback,
        scheduler=scheduler,
    )
    external_sources = ExternalSourceSyncService(
        store=ExternalSourceStore(database),
        personal_data=data,
        adapters={
            "rss": RssPersonalSourceAdapter(),
            **(
                {
                    "mcp": McpPersonalSourceAdapter(tools),
                    "notion": NotionPersonalSourceAdapter(tools),
                }
                if tools is not None
                else {}
            ),
        },
    )
    personal_provider = PersonalRecordSignalProvider(data)
    attention.providers.register(
        SignalProviderManifest(
            id="personal-records",
            version=1,
            domains=("*",),
            refresh_minutes=5,
            source_type="personal_record",
        ),
        personal_provider.collect,
    )
    runtime = PersonalRuntime(
        data=data,
        automation=PersonalAutomationStore(database),
        governance=governance,
        rhythm=rhythm,
        attention_source=PersonalAttentionSource(
            personal_data=data,
            rhythm=rhythm,
            engine=attention.engine,
            feedback=attention.feedback,
        ),
        attention=attention,
        attention_events=attention_events,
        attention_wakes=attention_wakes,
        acknowledgements=acknowledgements,
        external_sources=external_sources,
        today=PersonalTodayService(data),
        routines=(
            PersonalRoutineService(workflow_runtime)
            if workflow_runtime is not None
            else None
        ),
    )
    if event_bus is not None:
        attention_consumer = ConversationAttentionBatchConsumer(
            runtime.attention.learning
        )
        runtime.event_unsubscribers = [
            event_bus.on(
                ConversationSemanticBatchCommitted,
                attention_consumer.handle,
            ),
            event_bus.on(
                ConversationSemanticBatchCommitted,
                attention_events.handle_semantic_batch,
            ),
            event_bus.on(
                PersonalRecordChanged,
                attention_events.handle_personal_record_changed,
            ),
            event_bus.on(
                ScheduledJobChanged,
                attention_events.handle_scheduled_job_changed,
            ),
        ]
    return runtime


def register_personal_tools(
    registry: ToolRegistry,
    runtime: PersonalRuntime,
) -> None:
    registry.register(
        PersonalSourceTool(runtime.external_sources, registry),
        always_on=False,
        risk="write",
        search_hint="MCP RSS 信号源 外部订阅 Gmail Obsidian Notion X 推文 定期同步 主动关注",
    )
    registry.register(
        AttentionControlTool(
            runtime.attention.store,
            runtime.attention.feedback,
            runtime.acknowledgements,
        ),
        always_on=False,
        risk="write",
        search_hint="机会窗口 通勤 午休 睡前 主动策略 提醒频率 为什么现在找我 主动反馈",
    )
    registry.register(
        PersonalContextTool(runtime.data),
        always_on=True,
        risk="read-only",
        search_hint="个人资料 今日计划 健康记录 签到 承诺 长期记忆",
    )
    registry.register(
        PersonalGuidanceTool(runtime.rhythm),
        always_on=True,
        risk="read-only",
        search_hint="我现在有多少时间 当前场景 专注状态 周报 月报 目标偏差 下一步推荐",
    )
    registry.register(
        PersonalRecordTool(runtime.data, runtime.governance),
        always_on=False,
        risk="write",
        search_hint="保存个人资料 承诺 健康观测 每日计划 签到 遗忘",
    )
    registry.register(
        PersonalRhythmControlTool(runtime.rhythm, runtime.attention.feedback),
        always_on=False,
        risk="write",
        search_hint="出门 回家 睡前 专注 免打扰 定期找我 主动关注 提醒反馈",
    )
    if runtime.routines is not None:
        registry.register(
            PersonalRoutineTool(runtime.routines),
            always_on=True,
            risk="write",
            search_hint="晨间简报 晚间回顾 承诺捕获 每日助理",
        )
