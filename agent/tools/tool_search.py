import json
from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool
from agent.tools.registry import _META_TOOLS

if TYPE_CHECKING:
    from agent.skills import SkillsLoader
    from agent.tools.registry import ToolRegistry


class ToolSearchTool(Tool):
    """在工具目录中搜索可用工具，帮助模型发现并解锁需要的工具。

    调用此工具后，匹配到的工具将在本轮对话中解锁，可直接调用。
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        skills: "SkillsLoader | None" = None,
    ) -> None:
        self._registry = registry
        self._skills = skills
        if skills is None:
            self._catalog = None
        else:
            from agent.capabilities.catalog import CapabilityCatalog

            self._catalog = CapabilityCatalog(registry, skills)

    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return (
            "在实时能力目录中搜索内置工具、MCP 工具和 Skill。"
            "匹配到的工具会立即解锁；匹配到的 Skill 需调用 load_skill 读取。\n\n"
            "调用时机：\n"
            "- 需要某类功能，但不知道工具名称 → 必须调用\n"
            "- 知道工具名且已可见 → 直接调用，不要先搜索\n"
            "- 知道工具名但不可见 → 用 select: 前缀精确加载（见下）\n"
            "- 收到'工具不存在'错误 → 必须调用，用错误中的建议关键词搜索\n"
            "- 纯对话/推理，不涉及工具能力 → 不调用\n\n"
            "查询形式：\n"
            "- \"select:工具名\" → 精确加载已知工具，支持逗号分隔多个：\"select:A,B,C\"\n"
            "- \"关键词\" → 模糊搜索，例如：\"定时提醒\"、\"RSS订阅管理\"、\"Fitbit健康数据\"\n\n"
            "正确流程：tool_search(query) → 工具直接调用；Skill 先 load_skill 再按其流程执行"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索查询。两种形式：\n"
                        "1. \"select:工具名\" 精确加载（支持逗号分隔多个）\n"
                        "2. 关键词描述功能，例如：\"定时任务\"、\"文件读取\"、\"订阅管理\""
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "关键词搜索时返回的最大能力数量，默认 5，最大 10",
                    "default": 5,
                },
                "allowed_risk": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["read-only", "write", "external-side-effect"],
                    },
                    "description": "允许的风险等级，不填则不过滤。read-only=只读，write=写操作，external-side-effect=外部副作用",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        excluded_names: set[str] | list[str] | tuple[str, ...] | None = None,
        **_: Any,
    ) -> str:
        excluded: set[str] = (
            set(excluded_names) if excluded_names is not None else set()
        )

        query = (query or "").strip()
        if not query:
            return json.dumps(
                {
                    "matched": [],
                    "unlocked": [],
                    "already_loaded": [],
                    "tip": "query 不能为空，请描述你需要的功能",
                },
                ensure_ascii=False,
            )

        # ── select: 精确加载路径 ──────────────────────────────────────────
        if query.lower().startswith("select:"):
            return self._handle_select(
                query[7:],
                allowed_risk=allowed_risk,
                excluded_names=excluded,
            )

        # ── 关键词搜索路径 ────────────────────────────────────────────────
        top_k = min(max(1, int(top_k)), 10)
        if self._catalog is None:
            results = self._registry.search(
                query=query,
                top_k=top_k,
                allowed_risk=allowed_risk,
                excluded_names=excluded,
            )
        else:
            results = [
                match.as_result()
                for match in self._catalog.search(
                    query,
                    top_k=top_k,
                    allowed_risk=allowed_risk,
                    excluded_tool_names=excluded | _META_TOOLS,
                )
            ]
        if not results:
            return json.dumps(
                {
                    "matched": [],
                    "unlocked": [],
                    "already_loaded": [],
                    "tip": "没有找到匹配能力，请换个功能描述重试",
                },
                ensure_ascii=False,
            )
        unlocked = [
            item["name"]
            for item in results
            if item.get("kind", "tool") == "tool"
            and isinstance(item.get("name"), str)
            and item["name"]
        ]
        skills = [
            item
            for item in results
            if item.get("kind") == "skill"
        ]
        return json.dumps(
            {
                "matched": results,
                "unlocked": unlocked,
                "skills": skills,
                "already_loaded": [],
                "next_action": (
                    "unlocked 中的工具 schema 已加载，可直接调用；skills 中的能力"
                    "先调用 load_skill(skill=名称) 读取说明。不要再次 tool_search 同一能力。"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    def _handle_select(
        self,
        names_str: str,
        *,
        allowed_risk: list[str] | None = None,
        excluded_names: set[str] | None = None,
    ) -> str:
        """处理 select:A,B,C 精确加载路径。

        与 search() 使用相同的过滤语义：
        - excluded_names 中的工具已可见，无需加载（返回 tip 提示直接调用）
        - allowed_risk 不为空时，风险等级不符的工具不返回
        """
        requested = [n.strip() for n in names_str.split(",") if n.strip()]
        if not requested:
            return json.dumps(
                {
                    "matched": [],
                    "unlocked": [],
                    "already_loaded": [],
                    "tip": "select: 后面需要提供工具名",
                },
                ensure_ascii=False,
            )

        excluded: set[str] = set(_META_TOOLS)
        if excluded_names:
            excluded.update(excluded_names)
        risk_filter = set(allowed_risk) if allowed_risk else None

        already_loaded: list[str] = []
        found: list[str] = []
        skill_found: list[dict[str, object]] = []
        missing: list[str] = []
        risk_blocked: list[str] = []

        for name in requested:
            if name in excluded:
                already_loaded.append(name)
            elif self._registry.has_tool(name):
                doc = self._registry.get_document(name)
                if risk_filter and doc and doc.risk not in risk_filter:
                    risk_blocked.append(name)
                else:
                    found.append(name)
            elif self._catalog is not None and (
                skill := self._catalog.get("skill", name)
            ) is not None:
                skill_found.append(
                    {
                        "kind": "skill",
                        "name": skill.name,
                        "summary": skill.description[:160],
                        "available": skill.available,
                        "source": {
                            "type": skill.source_type,
                            "name": skill.source_name,
                        },
                        "why_matched": ["名称:精确匹配"],
                    }
                )
            else:
                missing.append(name)

        matched = self._registry.get_schemas_as_doc_results(found) + skill_found
        result: dict[str, Any] = {
            "matched": matched,
            "unlocked": found,
            "skills": skill_found,
            "already_loaded": already_loaded,
        }
        if found or skill_found:
            result["next_action"] = (
                "unlocked 中的工具可直接调用；skills 中的能力先调用 load_skill。"
            )

        tip_parts: list[str] = []
        if already_loaded:
            tip_parts.append(f"已加载可直接调用: {', '.join(already_loaded)}")
        if missing:
            tip_parts.append(
                f"未找到工具: {', '.join(missing)}，请用关键词搜索确认正确名称"
            )
        if risk_blocked:
            tip_parts.append(
                f"风险等级不符（allowed_risk={allowed_risk}）: {', '.join(risk_blocked)}"
            )
        if tip_parts:
            result["tip"] = "; ".join(tip_parts)

        return json.dumps(result, ensure_ascii=False, indent=2)
