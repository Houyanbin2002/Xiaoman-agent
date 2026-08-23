from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from infra.persistence.json_store import atomic_save_json, load_json


@dataclass(frozen=True)
class ConversationStyle:
    id: str
    name: str
    description: str
    instruction: str

    def public(self) -> dict[str, str]:
        payload = asdict(self)
        payload.pop("instruction", None)
        return payload


CONVERSATION_STYLES: tuple[ConversationStyle, ...] = (
    ConversationStyle(
        id="balanced",
        name="自然",
        description="根据问题自动调节详略与语气",
        instruction=(
            "保持自然、清楚、克制；简单问题直接回答，复杂任务再展开必要结构。"
            "结合用户本轮语气，但不要刻意表演某种人格。"
        ),
    ),
    ConversationStyle(
        id="concise",
        name="简洁",
        description="先给结论，只保留必要信息",
        instruction=(
            "优先用最短的完整回答解决问题。先给结论，省略过程复述、背景铺垫和重复总结；"
            "只有确实影响执行时才补充步骤或注意事项。"
        ),
    ),
    ConversationStyle(
        id="warm",
        name="温和",
        description="耐心、有共情，但不说空话",
        instruction=(
            "语气温和、耐心，先准确回应用户的处境与真实需求；可以表达理解，"
            "但不使用模板化安慰、过度鼓励或黏腻措辞。"
        ),
    ),
    ConversationStyle(
        id="professional",
        name="专业",
        description="结构清楚，措辞严谨，依据明确",
        instruction=(
            "采用专业、准确、结构清楚的表达。区分事实、判断与建议，必要时说明依据、"
            "边界和取舍；不堆砌术语，不把不确定内容写成结论。"
        ),
    ),
    ConversationStyle(
        id="candid",
        name="直率",
        description="明确判断，直接指出问题与取舍",
        instruction=(
            "表达直接、坦率、有判断。发现前提不成立、方案代价过高或方向有问题时明确指出；"
            "批评方案而不是攻击用户，并给出可执行的替代建议。"
        ),
    ),
    ConversationStyle(
        id="lively",
        name="活泼",
        description="轻松有趣，表达更有节奏",
        instruction=(
            "表达轻松、有节奏，可以适度使用幽默和生动比喻；不能牺牲准确性，"
            "不能使用表情符号，也不要为了活泼增加无关内容。"
        ),
    ),
)

_STYLE_BY_ID = {style.id: style for style in CONVERSATION_STYLES}
DEFAULT_CONVERSATION_STYLE = "balanced"


class ConversationStyleService:
    """Own the global user-selected response style for every main Agent turn."""

    def __init__(self, workspace: Path) -> None:
        self._path = workspace / "conversation_style.json"
        self._lock = RLock()
        self._active_id = self._load_active_id()

    @property
    def active_id(self) -> str:
        with self._lock:
            return self._active_id

    @property
    def active(self) -> ConversationStyle:
        return _STYLE_BY_ID[self.active_id]

    def set_active(self, style_id: str) -> ConversationStyle:
        normalized = str(style_id or "").strip().lower()
        style = _STYLE_BY_ID.get(normalized)
        if style is None:
            raise ValueError(f"未知对话风格: {style_id}")
        with self._lock:
            atomic_save_json(
                self._path,
                {"version": 1, "active_style": style.id},
                domain="conversation_style",
            )
            self._active_id = style.id
        return style

    def prompt(self) -> str:
        return self.prompt_state()[1]

    def prompt_state(self) -> tuple[str, str]:
        with self._lock:
            style = _STYLE_BY_ID[self._active_id]
        return style.id, self._render_prompt(style)

    @staticmethod
    def _render_prompt(style: ConversationStyle) -> str:
        return (
            "## 当前对话风格\n\n"
            f"用户在界面中选择了“{style.name}”风格。{style.instruction}\n"
            "这只控制面向用户的表达方式，不改变事实标准、工具路由、权限、安全规则或任务完成标准。\n"
            "若用户在当前消息中明确指定语气、格式或详略，以本轮明确要求为准；"
            "不要声称或暗示自己切换了模型、能力或身份。"
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_style": self.active_id,
            "styles": [style.public() for style in CONVERSATION_STYLES],
        }

    def _load_active_id(self) -> str:
        payload = load_json(
            self._path,
            default={},
            domain="conversation_style",
        )
        if not isinstance(payload, dict):
            return DEFAULT_CONVERSATION_STYLE
        value = str(payload.get("active_style") or "").strip().lower()
        return value if value in _STYLE_BY_ID else DEFAULT_CONVERSATION_STYLE


__all__ = [
    "CONVERSATION_STYLES",
    "DEFAULT_CONVERSATION_STYLE",
    "ConversationStyle",
    "ConversationStyleService",
]
