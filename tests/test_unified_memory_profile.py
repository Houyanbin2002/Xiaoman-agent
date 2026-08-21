from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from core.memory.engine import MemoryEngine
from core.memory.governed import GovernedLongTermMemory
from core.memory.markdown import MarkdownMemoryStore
from core.memory.runtime import MemoryRuntime
from core.personal.governance import MemoryGovernanceService
from core.personal.models import PersonalEntityType, RecordStatus
from core.personal.service import PersonalDataService
from infra.persistence.memory_governance_store import MemoryGovernanceStore
from infra.persistence.personal_store import PersonalStore


def _services(
    tmp_path: Path,
) -> tuple[MarkdownMemoryStore, MemoryGovernanceService, GovernedLongTermMemory]:
    markdown = MarkdownMemoryStore(tmp_path)
    db_path = tmp_path / "personal.db"
    governance = MemoryGovernanceService(
        personal_data=PersonalDataService(PersonalStore(db_path)),
        conflict_store=MemoryGovernanceStore(db_path),
    )
    canonical = GovernedLongTermMemory(governance=governance)
    return markdown, governance, canonical


def _close(governance: MemoryGovernanceService) -> None:
    governance.close()
    governance.personal_data.close()


def test_runtime_reads_canonical_long_term_records(tmp_path: Path) -> None:
    markdown, governance, canonical = _services(tmp_path)
    canonical.ingest_candidates(
        [{"tag": "identity", "content": "用户长期维护小满项目"}],
        source_ref="semantic:1",
    )
    runtime = MemoryRuntime(
        markdown=cast(
            Any,
            SimpleNamespace(store=markdown, maintenance=SimpleNamespace()),
        ),
        engine=cast(MemoryEngine, SimpleNamespace()),
    )
    runtime.bind_canonical_long_term_memory(canonical)

    assert "用户长期维护小满项目" in runtime.read_long_term()
    assert "用户长期维护小满项目" in runtime.get_memory_context()
    _close(governance)


def test_correction_enters_same_governance_conflict_flow(tmp_path: Path) -> None:
    _, governance, canonical = _services(tmp_path)
    canonical.ingest_candidates(
        [{"tag": "preference", "content": "用户偏好上午锻炼"}],
        source_ref="semantic:preference",
    )

    result = canonical.ingest_candidates(
        [
            {
                "tag": "correction",
                "content": "用户现在偏好晚上锻炼",
                "replaces": "用户偏好上午锻炼",
            }
        ],
        source_ref="chat:correction",
    )

    assert result.conflicts == 1
    active = governance.list_memories(limit=10)
    assert len(active) == 1
    assert active[0].data["content"] == "用户偏好上午锻炼"
    conflicts = governance.conflict_store.list_conflicts(limit=10)
    assert conflicts[0].existing_record_id == active[0].id
    _close(governance)


def test_forgetting_record_removes_it_from_canonical_prompt(tmp_path: Path) -> None:
    _, governance, canonical = _services(tmp_path)
    canonical.ingest_candidates(
        [{"tag": "identity", "content": "用户维护小满"}],
        source_ref="semantic:identity",
    )
    record = governance.list_memories(limit=10)[0]

    forgotten = governance.forget(record.id, reason="user request")

    assert forgotten.status == RecordStatus.FORGOTTEN
    assert canonical.render_prompt() == ""
    assert (
        governance.personal_data.list(
            entity_type=PersonalEntityType.MEMORY,
            statuses=[RecordStatus.ACTIVE],
        )
        == []
    )
    _close(governance)


def test_requested_memory_tag_is_not_a_background_memory_category(
    tmp_path: Path,
) -> None:
    _, governance, canonical = _services(tmp_path)

    result = canonical.ingest_candidates(
        [
            {
                "tag": "requested_memory",
                "content": "模型声称用户要求记住这句话",
                "confidence": 0.99,
                "source_message_id": "assistant-message-id",
            }
        ],
        source_ref="semantic:untrusted",
    )

    assert result.skipped == 1
    assert governance.list_memories() == []
    _close(governance)


def test_background_extraction_rejects_requested_content_as_a_memory_kind(
    tmp_path: Path,
) -> None:
    _, governance, canonical = _services(tmp_path)

    result = canonical.ingest_candidates(
        [
            {
                "tag": "requested_memory",
                "content": "这是一段内容载荷，不是描述用户的稳定事实",
                "confidence": 1.0,
                "source_message_id": "web:1:0",
                "_user_evidence_verified": True,
            }
        ],
        source_ref="semantic:content-payload",
    )

    assert result.skipped == 1
    assert governance.list_memories() == []
    _close(governance)
