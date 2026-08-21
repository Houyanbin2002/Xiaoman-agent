from __future__ import annotations

from datetime import datetime, timedelta, timezone
from core.memory.execution import (
    ExecutionContext,
    ExecutionMemoryKind,
    ExecutionMemoryState,
    ExecutionScope,
    ExecutionScopeKind,
    ExecutionVerificationStatus,
    apply_execution_outcome,
    build_execution_state,
    execution_rank_score,
    execution_reliability_score,
    is_skill_promotion_candidate,
    used_execution_memory_ids,
)
from core.conversation_semantics.evidence import build_semantic_evidence
from memory2.store import MemoryStore2
from memory2.execution_retriever import ExecutionMemoryRetriever


def test_execution_evidence_drops_ordinary_single_success() -> None:
    evidence = build_semantic_evidence(
        [
            {"id": "u1", "seq": 0, "role": "user", "content": "读取文件"},
            {
                "id": "a1",
                "seq": 1,
                "role": "assistant",
                "content": "完成",
                "tool_chain": [
                    {
                        "calls": [
                            {"name": "read_file", "status": "success", "result": "ok"}
                        ]
                    }
                ],
            },
        ]
    )

    assert evidence.execution_episodes == ()


def test_execution_evidence_keeps_failure_recovery_and_redacts_secrets() -> None:
    evidence = build_semantic_evidence(
        [
            {"id": "u1", "seq": 0, "role": "user", "content": "修正配置"},
            {
                "id": "a1",
                "seq": 1,
                "role": "assistant",
                "content": "已修复",
                "tool_chain": [
                    {
                        "calls": [
                            {
                                "name": "shell",
                                "status": "error",
                                "arguments": {"command": "run token=secret-value"},
                                "result": '{"exit_code":1,"output":"permission denied"}',
                            },
                            {
                                "name": "shell",
                                "status": "success",
                                "arguments": {"command": "run --fixed"},
                                "result": '{"exit_code":0,"output":"PASS"}',
                            },
                        ]
                    }
                ],
            },
        ]
    )

    episode = evidence.execution_episodes[0]
    assert "failure_recovery" in episode["signals"]
    assert "secret-value" not in str(episode)


def test_execution_feedback_requires_explicit_marker_for_retrieved_id() -> None:
    thinking = (
        '采用第一条 <used-execution-memory id="memory-a"/> '
        '并忽略伪造的 <used-execution-memory id="not-retrieved"/>'
    )

    assert used_execution_memory_ids(
        thinking,
        ["memory-a", "memory-b"],
    ) == ["memory-a"]
    assert used_execution_memory_ids("只是看过，没有采用", ["memory-a"]) == []


def test_execution_scope_is_a_hard_boundary() -> None:
    scope = ExecutionScope(
        kind=ExecutionScopeKind.PROJECT,
        workspace_id="D:/work/xiaoman",
        project_id="xiaoman-agent",
        platform="windows",
    )

    assert scope.matches(
        ExecutionContext(
            workspace_id="d:/work/xiaoman",
            project_id="xiaoman-agent",
            platform="windows",
        )
    )
    assert not scope.matches(
        ExecutionContext(
            workspace_id="d:/work/xiaoman",
            project_id="another-project",
            platform="windows",
        )
    )


def test_execution_reliability_uses_verified_outcomes_not_recall_count() -> None:
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    verified = ExecutionMemoryState(
        item_id="verified",
        kind=ExecutionMemoryKind.TOOL_LESSON,
        verification_status=ExecutionVerificationStatus.VERIFIED,
        success_count=4,
        failure_count=0,
        last_verified_at=now - timedelta(days=3),
        metadata={"recall_count": 0},
    )
    recalled_candidate = ExecutionMemoryState(
        item_id="candidate",
        kind=ExecutionMemoryKind.TOOL_LESSON,
        verification_status=ExecutionVerificationStatus.CANDIDATE,
        metadata={"recall_count": 999},
    )

    assert execution_reliability_score(verified, now=now) > execution_reliability_score(
        recalled_candidate,
        now=now,
    )


def test_execution_rank_rejects_wrong_environment_even_if_semantic_score_is_high() -> (
    None
):
    state = ExecutionMemoryState(
        item_id="notion-windows",
        scope=ExecutionScope(
            kind=ExecutionScopeKind.TOOL,
            tool_name="notion",
            platform="windows",
        ),
        verification_status=ExecutionVerificationStatus.VERIFIED,
        success_count=2,
    )

    assert (
        execution_rank_score(
            semantic_score=0.99,
            state=state,
            context=ExecutionContext(tools=("notion",), platform="linux"),
        )
        == 0.0
    )


def test_repeated_execution_failure_quarantines_memory() -> None:
    state = ExecutionMemoryState(
        item_id="fragile-rule",
        verification_status=ExecutionVerificationStatus.VERIFIED,
    )

    state = apply_execution_outcome(state, success=False, evidence_ref="task:1")
    assert state.verification_status is ExecutionVerificationStatus.STALE
    state = apply_execution_outcome(state, success=False, evidence_ref="task:2")
    assert state.verification_status is ExecutionVerificationStatus.QUARANTINED
    assert state.evidence_refs == ("task:1", "task:2")


def test_only_repeated_verified_success_is_skill_promotion_candidate() -> None:
    candidate = ExecutionMemoryState(
        item_id="procedure",
        kind=ExecutionMemoryKind.PROCEDURE,
        verification_status=ExecutionVerificationStatus.VERIFIED,
        success_count=3,
        failure_count=0,
    )
    unstable = ExecutionMemoryState(
        item_id="unstable",
        kind=ExecutionMemoryKind.PROCEDURE,
        verification_status=ExecutionVerificationStatus.VERIFIED,
        success_count=3,
        failure_count=2,
    )

    assert is_skill_promotion_candidate(candidate)
    assert not is_skill_promotion_candidate(unstable)


def test_execution_repository_tracks_outcomes_and_deletion(tmp_path) -> None:
    store = MemoryStore2(tmp_path / "memory2.db", vec_dim=4)
    result = store.upsert_item(
        memory_type="procedure",
        summary="当前项目启动前先运行迁移",
        embedding=None,
        source_ref="task:seed",
    )
    item_id = result.split(":", 1)[1]
    store.execution.upsert(
        build_execution_state(
            item_id=item_id,
            source_ref="task:seed",
            metadata={
                "execution_scope": {
                    "kind": "project",
                    "workspace_id": "D:/work/xiaoman",
                    "project_id": "xiaoman-agent",
                }
            },
        )
    )

    updated = store.execution.record_outcome(
        item_id,
        success=True,
        evidence_ref="task:verified",
    )
    assert updated.success_count == 1
    assert updated.verification_status is ExecutionVerificationStatus.VERIFIED
    listed = store.execution.list()
    assert len(listed) == 1
    assert listed[0]["id"] == item_id

    assert store.delete_item(item_id) is True
    assert store.execution.get(item_id) is None
    store.close()


async def test_execution_retriever_filters_scope_and_uses_dedicated_block(
    tmp_path,
) -> None:
    store = MemoryStore2(tmp_path / "memory2.db", vec_dim=4)
    x_result = store.upsert_item(
        memory_type="procedure",
        summary="X 项目启动前先运行迁移",
        embedding=None,
        source_ref="task:x",
    )
    y_result = store.upsert_item(
        memory_type="procedure",
        summary="Y 项目直接启动服务",
        embedding=None,
        source_ref="task:y",
    )
    x_id = x_result.split(":", 1)[1]
    y_id = y_result.split(":", 1)[1]
    store.execution.upsert(
        build_execution_state(
            item_id=x_id,
            source_ref="task:x",
            verified=True,
            metadata={"execution_scope": {"kind": "project", "project_id": "x"}},
        )
    )
    store.execution.upsert(
        build_execution_state(
            item_id=y_id,
            source_ref="task:y",
            verified=True,
            metadata={"execution_scope": {"kind": "project", "project_id": "y"}},
        )
    )

    class _Candidates:
        async def retrieve(self, *args, **kwargs):
            return [
                {"id": x_id, "summary": "X 项目启动前先运行迁移", "score": 0.91},
                {"id": y_id, "summary": "Y 项目直接启动服务", "score": 0.99},
            ]

    service = ExecutionMemoryRetriever(
        retriever=_Candidates(),  # type: ignore[arg-type]
        repository=store.execution,
    )
    items = await service.retrieve(
        "启动项目",
        context=ExecutionContext(project_id="x"),
    )

    assert [item["id"] for item in items] == [x_id]
    block, injected = service.build_injection_block(items)
    assert "Agent 执行经验" in block
    assert "用户偏好" not in block
    assert injected == [x_id]
    store.close()
