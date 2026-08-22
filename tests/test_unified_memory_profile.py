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


def test_preference_slot_merges_repeat_and_protects_user_locked_value(
    tmp_path: Path,
) -> None:
    _, governance, canonical = _services(tmp_path)
    explicit = {
        "tag": "preference",
        "content": "以后代码示例默认使用 Python",
        "confidence": 0.99,
        "origin": "explicit_user",
        "source_message_id": "web:1:0",
        "_user_evidence_verified": True,
        "subject": "用户",
        "predicate": "代码示例语言",
        "value": "Python",
        "attributes": {"preference_key": "code_language"},
    }
    created = canonical.ingest_candidates([explicit], source_ref="batch:1")
    repeated = canonical.ingest_candidates(
        [
            {
                **explicit,
                "content": "编程代码请继续给 Python 版本",
                "source_message_id": "web:2:0",
                "_user_evidence_verified": False,
                "origin": "inferred_pattern",
            }
        ],
        source_ref="batch:2",
    )
    conflict = canonical.ingest_candidates(
        [
            {
                **explicit,
                "content": "模型推测用户改用 JavaScript",
                "value": "JavaScript",
                "source_message_id": "",
                "_user_evidence_verified": False,
                "origin": "inferred_pattern",
            }
        ],
        source_ref="batch:3",
    )

    assert created.created == 1
    assert repeated.unchanged == 1
    assert conflict.conflicts == 1
    active = governance.list_memories()
    assert len(active) == 1
    assert active[0].record_key == "memory:preference:code_language"
    assert active[0].data["value"] == "Python"
    assert active[0].user_locked is True
    evidence = governance.personal_data.memory_evidence(active[0].id)
    assert {item.source.source_ref for item in evidence} == {
        "batch:1#message:web:1:0",
        "batch:2#message:web:2:0",
    }
    _close(governance)


def test_same_batch_preference_candidates_write_only_one_version(
    tmp_path: Path,
) -> None:
    _, governance, canonical = _services(tmp_path)
    common = {
        "tag": "correction",
        "confidence": 0.99,
        "origin": "user_correction",
        "source_message_id": "web:3:0",
        "_user_evidence_verified": True,
        "subject": "用户",
        "predicate": "notification_quiet_hours",
        "replaces": "晚上九点",
        "attributes": {"preference_key": "notification_quiet_hours"},
    }
    canonical.ingest_candidates(
        [
            {
                "tag": "preference",
                "content": "用户原有偏好：晚上九点",
                "confidence": 1.0,
                "origin": "explicit_user",
                "source_message_id": "fixture:old",
                "_user_evidence_verified": True,
                "subject": "用户",
                "predicate": "notification_quiet_hours",
                "value": "晚上九点",
                "attributes": {
                    "preference_key": "notification_quiet_hours"
                },
            }
        ],
        source_ref="fixture:old",
    )

    result = canonical.ingest_candidates(
        [
            {
                **common,
                "content": "用户把免打扰起点调整为 22:00",
                "value": "22:00",
            },
            {
                **common,
                "content": "免打扰时间从晚上九点改成晚上十点开始。",
                "value": "晚上十点",
            },
        ],
        source_ref="batch:3",
    )

    assert result.created == 1
    active = governance.list_memories()
    assert len(active) == 1
    assert active[0].data["value"] == "晚上十点"
    old = governance.personal_data.get(active[0].supersedes_id)
    assert old is not None
    assert old.data["value"] == "晚上九点"
    assert len(governance.personal_data.lineage(active[0].id)) == 2
    _close(governance)


def test_same_slot_candidates_coalesce_even_when_model_tag_differs(
    tmp_path: Path,
) -> None:
    _, governance, canonical = _services(tmp_path)
    canonical.ingest_candidates(
        [
            {
                "tag": "preference",
                "content": "用户原有偏好：旧项目",
                "confidence": 1.0,
                "origin": "explicit_user",
                "source_message_id": "fixture:project",
                "_user_evidence_verified": True,
                "subject": "用户",
                "predicate": "active_project",
                "value": "旧项目",
                "attributes": {"preference_key": "active_project"},
            }
        ],
        source_ref="fixture:project",
    )

    result = canonical.ingest_candidates(
        [
            {
                "tag": "project_context",
                "content": "用户当前主要关注 Xiaoman 项目。",
                "confidence": 0.95,
                "origin": "explicit_user",
                "source_message_id": "web:project:0",
                "_user_evidence_verified": True,
                "subject": "用户",
                "predicate": "active_project_focus",
                "value": "Xiaoman",
                "attributes": {"preference_key": "active_project"},
            },
            {
                "tag": "correction",
                "content": "旧项目已经结束，当前主要关注 Xiaoman 项目。",
                "confidence": 0.98,
                "origin": "user_correction",
                "source_message_id": "web:project:0",
                "_user_evidence_verified": True,
                "subject": "用户",
                "predicate": "active_project",
                "value": "Xiaoman 项目",
                "replaces": "旧项目",
                "attributes": {"preference_key": "active_project"},
            },
        ],
        source_ref="batch:project",
    )

    assert result.created == 1
    active = governance.list_memories()
    assert len(active) == 1
    assert active[0].data["value"] == "Xiaoman 项目"
    old = governance.personal_data.get(active[0].supersedes_id)
    assert old is not None
    assert old.data["value"] == "旧项目"
    assert len(governance.personal_data.lineage(active[0].id)) == 2
    _close(governance)
