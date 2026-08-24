from __future__ import annotations

from dataclasses import dataclass

from agent.tools.base import Tool


@dataclass
class ToolMeta:
    risk: str = "read-only"
    always_on: bool = False
    search_hint: str | None = None


@dataclass
class ToolDocument:
    """工具的索引态视图，供注册表和搜索后端共同使用。"""

    name: str
    description: str
    risk: str
    always_on: bool
    search_hint: str | None
    source_type: str
    source_name: str
    parameter_names: tuple[str, ...] = ()

    @classmethod
    def from_tool_and_meta(
        cls,
        tool: Tool,
        meta: ToolMeta,
        source_type: str = "builtin",
        source_name: str = "",
    ) -> ToolDocument:
        return cls(
            name=tool.name,
            description=tool.description,
            risk=meta.risk,
            always_on=meta.always_on,
            search_hint=meta.search_hint,
            source_type=source_type,
            source_name=source_name,
            parameter_names=tuple(
                str(name)
                for name in ((tool.parameters or {}).get("properties", {}) or {}).keys()
            ),
        )
