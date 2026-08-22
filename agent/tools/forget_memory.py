from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool
from core.memory.engine import MemoryMutation, MemoryToolSpec

if TYPE_CHECKING:
    from core.memory.engine import MemoryWriteApi


class ForgetMemoryTool(Tool):
    name = "forget_memory"
    description = "由当前 memory engine 的 tool_profile 注入工具描述。"
    parameters = {
        "type": "object",
        "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["ids"],
    }

    def __init__(
        self,
        memory: "MemoryWriteApi",
        spec: MemoryToolSpec,
    ) -> None:
        self._memory = memory
        self._spec = spec
        self.description = self._spec.description
        self.parameters = self._spec.parameters

    async def execute(self, ids: list[str], **context: Any) -> str:
        request = str(context.get("current_user_message") or "").strip()
        correction = re.search(
            r"(改为|改成|更正|纠正|不要再|不再默认|以后.+(?:用|按)|"
            r"作废.+(?:改为|改成)|已经结束.+当前主要关注|规则取消.+只在)",
            request,
            re.IGNORECASE,
        )
        explicit_erasure = re.search(
            r"(忘记|删除|清除|彻底移除|隐私删除)",
            request,
            re.IGNORECASE,
        )
        if correction and not explicit_erasure:
            return json.dumps(
                {
                    "accepted": False,
                    "status": "deferred_to_background_governance",
                    "message": (
                        "这是记忆纠错而不是隐私删除；旧值与新值将由回合后的"
                        "后台冲突治理建立版本血缘。"
                    ),
                },
                ensure_ascii=False,
            )
        clean_ids = _clean_ids(ids)
        if not clean_ids:
            return _render_forget_result(clean_ids, [], [], [])

        result = await self._memory.mutate(
            MemoryMutation(kind="forget", ids=tuple(clean_ids))
        )
        return _render_forget_result(
            clean_ids,
            result.affected_ids,
            result.missing_ids,
            result.items,
        )


def _clean_ids(ids: list[str]) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for raw in ids or []:
        item_id = str(raw).strip()
        if item_id and item_id not in seen:
            seen.add(item_id)
            clean.append(item_id)
    return clean


def _render_forget_result(
    requested_ids: list[str],
    affected_ids: list[str],
    missing_ids: list[str],
    items: list[dict[str, object]],
) -> str:
    return json.dumps(
        {
            "requested_ids": requested_ids,
            "superseded_ids": affected_ids,
            "missing_ids": missing_ids,
            "count": len(affected_ids),
            "items": items,
        },
        ensure_ascii=False,
    )
