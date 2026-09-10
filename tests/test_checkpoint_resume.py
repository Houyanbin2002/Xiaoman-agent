from agent.runtime.checkpoint_resume import reconcile_checkpoint


def test_reconcile_checkpoint_replaces_stale_request_and_marks_unknown_call():
    state = {
        "messages": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "旧任务"},
        ],
        "pending_tool_calls": [
            {"id": "call-1", "name": "message_push", "arguments": {"text": "old"}}
        ],
        "pending_tool_index": 0,
        "pending_tool_results": [],
        "pending_content": "准备发送",
        "tool_chain": [],
    }
    resumed = reconcile_checkpoint(
        state,
        [{"role": "system", "content": "new system"}, {"role": "user", "content": "新任务"}],
        request_text="新任务",
        channel="dashboard",
        chat_id="chat-2",
        permission_mode="read_only",
        delegation_allowed=False,
    )
    assert resumed["messages"][0]["content"] == "new system"
    assert resumed["messages"][-1]["content"] == "新任务"
    assert resumed["pending_tool_calls"] == []
    assert resumed["tool_chain"][-1]["calls"][0]["status"] == "interrupted_unknown"
    assert "dashboard" in resumed["messages"][-2]["content"]
    assert "read_only" in resumed["messages"][-2]["content"]
