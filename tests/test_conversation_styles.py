from __future__ import annotations

from pathlib import Path

import pytest

from agent.context import ContextBuilder
from agent.conversation_styles import ConversationStyleService


class _Memory:
    def get_memory_context(self) -> str:
        return ""

    def read_self(self) -> str:
        return ""

    def read_recent_context(self) -> str:
        return ""


def test_conversation_style_defaults_and_persists(tmp_path: Path) -> None:
    service = ConversationStyleService(tmp_path)

    assert service.active_id == "balanced"
    assert service.snapshot()["styles"][0]["name"] == "自然"
    assert "instruction" not in service.snapshot()["styles"][0]

    selected = service.set_active("professional")

    assert selected.name == "专业"
    assert ConversationStyleService(tmp_path).active_id == "professional"
    assert "只控制面向用户的表达方式" in service.prompt()


def test_conversation_style_rejects_unknown_value(tmp_path: Path) -> None:
    service = ConversationStyleService(tmp_path)

    with pytest.raises(ValueError, match="未知对话风格"):
        service.set_active("invented")

    assert service.active_id == "balanced"


def test_context_builder_injects_selected_style_as_cacheable_block(
    tmp_path: Path,
) -> None:
    builder = ContextBuilder(tmp_path, memory=_Memory())  # type: ignore[arg-type]
    builder.conversation_styles.set_active("candid")

    result = builder._build_system_prompt_result()
    style_section = next(
        section
        for section in result.system_sections
        if section.name == "conversation_style"
    )

    assert style_section.is_static is True
    assert "“直率”风格" in style_section.content
    assert "本轮明确要求为准" in style_section.content
    assert any(item.name == "conversation_style" for item in result.debug_breakdown)


def test_invalid_persisted_style_falls_back_to_balanced(tmp_path: Path) -> None:
    (tmp_path / "conversation_style.json").write_text(
        '{"active_style":"removed-style"}',
        encoding="utf-8",
    )

    assert ConversationStyleService(tmp_path).active_id == "balanced"
