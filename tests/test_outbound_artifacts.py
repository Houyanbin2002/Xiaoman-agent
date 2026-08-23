from __future__ import annotations

from pathlib import Path

from agent.artifacts import discover_outbound_artifacts, requests_file_delivery
from prompts.agent import build_current_session_prompt


def test_artifact_discovery_requires_delivery_intent_and_workspace(tmp_path: Path) -> None:
    artifact = tmp_path / "ppt" / "科研答辩模板.pptx"
    artifact.parent.mkdir()
    artifact.write_bytes(b"pptx")
    outside = tmp_path.parent / "private.txt"
    outside.write_text("private", encoding="utf-8")

    assert requests_file_delivery("做一个 PPT 发给我") is True
    assert discover_outbound_artifacts(
        user_request="做一个 PPT 发给我",
        reply=f"已生成：`{artifact}`，可以下载。",
        workspace=tmp_path,
    ) == [str(artifact.resolve())]
    assert discover_outbound_artifacts(
        user_request="告诉我文件在哪里",
        reply=f"`{artifact}`",
        workspace=tmp_path,
    ) == []
    assert discover_outbound_artifacts(
        user_request="把文件发送给我",
        reply=f"`{outside}`",
        workspace=tmp_path,
    ) == []


def test_artifact_discovery_does_not_redeliver_successful_message_push(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"pdf")
    chain = [
        {
            "calls": [
                {
                    "name": "message_push",
                    "arguments": {"file": str(artifact)},
                    "result": "文件 'report.pdf' 已发送",
                    "status": "success",
                }
            ]
        }
    ]
    assert discover_outbound_artifacts(
        user_request="把报告发给我",
        reply=f"已经发好了：`{artifact}`",
        tool_chain=chain,
        workspace=tmp_path,
    ) == []


def test_current_session_prompt_is_channel_authoritative() -> None:
    prompt = build_current_session_prompt(channel="dashboard", chat_id="chat-1")
    assert "网页 Dashboard 当前聊天" in prompt
    assert "channel=dashboard" in prompt
    assert "chat_id=chat-1" in prompt
    assert "优先级高于历史对话" in prompt
