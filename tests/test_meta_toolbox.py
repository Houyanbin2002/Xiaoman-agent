from typing import Any, cast
import pytest
from agent.tools.filesystem import ListDirTool, ReadFileTool
from agent.tools.meta import (
    META_TOOLBOX_NAMES,
    register_common_meta_tools,
    register_memory_meta_tools,
)
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from bootstrap.toolsets.meta import CommonMetaToolsetProvider
from bootstrap.toolsets.protocol import ToolsetDeps
from core.memory.engine import MemoryToolProfile, MemoryToolSpec


class _MemoryEngineStub:
    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile(
            memorize=MemoryToolSpec(
                description="test",
                parameters={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
                risk="write",
            ),
            recall=MemoryToolSpec(
                description="test",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            forget=MemoryToolSpec(
                description="test",
                parameters={
                    "type": "object",
                    "properties": {
                        "ids": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["ids"],
                },
                risk="write",
            ),
        )

    async def query(self, request):
        raise NotImplementedError

    async def mutate(self, request):
        raise NotImplementedError

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None

    async def execute(self, **kwargs):
        return ""


def test_register_meta_tool_helpers_mark_expected_tools_always_on():
    tools = ToolRegistry()
    readonly_tools = {
        "web_search": WebSearchTool(),
        "web_fetch": WebFetchTool(requester=cast(Any, object())),
        "read_file": ReadFileTool(),
        "list_dir": ListDirTool(),
    }

    push_tool = register_common_meta_tools(
        tools,
        readonly_tools,
        session_store=object(),
    )
    register_memory_meta_tools(
        tools,
        cast(Any, _MemoryEngineStub()),
    )

    always_on = tools.get_always_on_names()
    assert isinstance(push_tool, MessagePushTool)
    assert set(META_TOOLBOX_NAMES) <= always_on
    assert "memorize" not in always_on
    assert not tools.has_tool("memorize")
    assert "reinforce_memory" not in always_on
    assert not tools.has_tool("reinforce_memory")


def test_meta_toolbox_does_not_register_direct_long_term_memory_writes():
    assert "memorize" not in META_TOOLBOX_NAMES
def test_common_meta_toolset_registers_load_skill(tmp_path):
    tools = ToolRegistry()
    readonly_tools = {
        "web_search": WebSearchTool(),
        "web_fetch": WebFetchTool(requester=cast(Any, object())),
        "read_file": ReadFileTool(),
        "list_dir": ListDirTool(),
    }

    result = CommonMetaToolsetProvider(readonly_tools).register(
        tools,
        ToolsetDeps(
            config=None,
            workspace=tmp_path,
            session_store=object(),
        ),
    )

    assert tools.has_tool("load_skill")
    assert "load_skill" in result.always_on_names


def test_register_memory_meta_tools_rejects_duplicate_names():
    tools = ToolRegistry()

    register_memory_meta_tools(tools, cast(Any, _MemoryEngineStub()))

    with pytest.raises(ValueError, match="重复注册"):
        register_memory_meta_tools(tools, cast(Any, _MemoryEngineStub()))


def test_register_memory_meta_tools_rejects_invalid_custom_name():
    class _BadMemoryEngineStub(_MemoryEngineStub):
        def tool_profile(self) -> MemoryToolProfile:
            return MemoryToolProfile(
                tools=(
                    MemoryToolSpec(
                        name="bad-name",
                        description="test",
                        parameters={"type": "object", "properties": {}, "required": []},
                    ),
                )
            )

    tools = ToolRegistry()

    with pytest.raises(ValueError, match="非法"):
        register_memory_meta_tools(tools, cast(Any, _BadMemoryEngineStub()))
