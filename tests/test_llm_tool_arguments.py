from __future__ import annotations

import pytest

from core.llm import ToolArgumentsDecodeError
from infra.providers.llm_provider import _decode_tool_arguments


def test_decode_tool_arguments_accepts_exactly_one_object():
    assert _decode_tool_arguments(
        '{"name":"task","steps":[]}',
        tool_name="task_create",
        call_id="call-1",
    ) == {"name": "task", "steps": []}


def test_decode_tool_arguments_classifies_trailing_json():
    with pytest.raises(ToolArgumentsDecodeError) as caught:
        _decode_tool_arguments(
            '{"name":"task"}{"extra":true}',
            tool_name="task_create",
            call_id="call-1",
        )

    error = caught.value
    assert error.tool_name == "task_create"
    assert "多余内容或第二个 JSON" in error.reason
    assert error.raw_arguments == '{"name":"task"}{"extra":true}'


def test_decode_tool_arguments_rejects_non_object_top_level():
    with pytest.raises(ToolArgumentsDecodeError, match="顶层必须是 JSON 对象"):
        _decode_tool_arguments(
            "[]",
            tool_name="task_create",
            call_id="call-1",
        )
