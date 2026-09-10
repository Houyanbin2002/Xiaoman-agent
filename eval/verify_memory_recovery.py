"""Opt-in live semantic extraction smoke test; all writes use a temporary workspace.

Run: python -m eval.verify_memory_recovery
Uses the configured provider; never starts channels or touches production memories.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent.config import Config
from bootstrap.providers import build_providers
from core.conversation_semantics.analyzer import ConversationSemanticAnalyzer
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.memory.governed import GovernedLongTermMemory
from core.memory.semantic_consumer import ConversationMemoryBatchConsumer
from core.personal.governance import MemoryGovernanceService
from core.personal.service import PersonalDataService
from infra.persistence.personal_store import PersonalStore
from infra.persistence.memory_governance_store import MemoryGovernanceStore
from infra.persistence.markdown_memory_store import MarkdownMemoryStore


async def main() -> None:
    config = Config.load("config.toml")
    main_provider, light, agent = build_providers(config)
    provider = light or main_provider
    model = config.light_model if light else config.model
    print(
        json.dumps({"model": model, "isolated_workspace": True}, ensure_ascii=False),
        flush=True,
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="xiaoman-memory-recovery-"
        ) as directory:
            path = Path(directory)
            data = PersonalDataService(PersonalStore(path / "personal.db"))
            governance = MemoryGovernanceService(
                personal_data=data,
                conflict_store=MemoryGovernanceStore(path / "personal.db"),
            )
            markdown = MarkdownMemoryStore(path)
            memory = GovernedLongTermMemory(governance=governance)
            messages: list[dict[str, object]] = []
            analyzer = ConversationSemanticAnalyzer(
                provider,
                model,
                max_tokens=2400,
                activity_context_provider=markdown.activity_snapshot,
            )
            consumer = ConversationMemoryBatchConsumer(
                markdown=markdown,
                candidate_sink=memory.ingest_candidates,
                message_source=lambda _: messages,
            )
            cases = [
                "工作时我希望你回复简洁，日常闲聊时我希望你详细解释。最近正在挑选给女朋友的生日礼物，还没买。",
                "给女朋友的生日礼物已经买好了，这件事已完成，不用再关注。工作时的回复风格更正为详细解释；闲聊时的偏好保持不变。",
                "同事说他喜欢喝咖啡，不是说我喜欢。刚才那只是转述，不要当成我的喜好。",
            ]
            try:
                for index, content in enumerate(cases):
                    message = {
                        "id": f"fixture:u{index}",
                        "seq": index,
                        "role": "user",
                        "content": content,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    messages.append(message)
                    payload = await asyncio.wait_for(
                        analyzer.analyze([message]), timeout=120
                    )
                    event = ConversationSemanticBatchCommitted(
                        batch_id=f"fixture:batch{index}",
                        session_key="fixture:memory",
                        channel="fixture",
                        chat_id="memory",
                        analysis_version="conversation-v4",
                        message_ids=(str(message["id"]),),
                        user_message_ids=(str(message["id"]),),
                        end_seq=index,
                        context_consolidate_through=-1,
                        payload=payload,
                    )
                    await consumer.handle(event)
                    print(
                        json.dumps(
                            {
                                "case": index + 1,
                                "candidates": [
                                    item.to_mapping()
                                    for item in payload.memory_candidates
                                ],
                                "activities": markdown.activity_snapshot(),
                                "active_memories": [
                                    {
                                        "content": row.data.get("content"),
                                        "scope": row.data.get("scope"),
                                        "locked": row.user_locked,
                                    }
                                    for row in governance.list_memories()
                                ],
                                "pending": [
                                    {"reason": row.reason}
                                    for row in governance.conflict_store.list_conflicts()
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            finally:
                governance.close()
                data.close()
    finally:
        for current in (main_provider, light, agent):
            if current is not None:
                await current.aclose()


if __name__ == "__main__":
    asyncio.run(main())
