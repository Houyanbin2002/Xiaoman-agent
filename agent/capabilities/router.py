from __future__ import annotations

from dataclasses import dataclass

from agent.capabilities.catalog import CapabilityCatalog, CapabilityMatch


@dataclass(frozen=True)
class CapabilityRoute:
    active_skills: tuple[str, ...] = ()
    preloaded_tools: tuple[str, ...] = ()
    candidates: tuple[CapabilityMatch, ...] = ()

    def prompt(self) -> str:
        if not (self.active_skills or self.preloaded_tools or self.candidates):
            return ""
        lines = ["【本轮能力路由（由实时目录召回，尚未执行任何操作）】"]
        if self.active_skills:
            lines.append(f"已加载 Skill: {', '.join(self.active_skills)}")
        if self.preloaded_tools:
            lines.append(f"已预载工具: {', '.join(self.preloaded_tools)}")
        remaining = [
            match
            for match in self.candidates
            if match.record.name not in {*self.active_skills, *self.preloaded_tools}
        ][:4]
        if remaining:
            rendered = "; ".join(
                f"{match.record.kind}:{match.record.name} — {match.record.description[:60]}"
                for match in remaining
            )
            lines.append(f"其他候选: {rendered}")
        lines.append(
            "Skill 提供工作方法；工具执行真实操作。候选不等于已执行，涉及写入或外部副作用仍按权限策略处理。"
        )
        return "\n".join(lines)


class CapabilityRouter:
    """Select a small turn-scoped working set from the live catalog."""

    def __init__(self, catalog: CapabilityCatalog) -> None:
        self._catalog = catalog

    def route(
        self,
        query: str,
        *,
        explicit_skills: list[str] | None = None,
        visible_tools: set[str] | None = None,
        max_tools: int = 3,
    ) -> CapabilityRoute:
        explicit = list(dict.fromkeys(explicit_skills or []))
        visible = visible_tools or set()
        matches = self._catalog.search(
            query,
            top_k=12,
            excluded_tool_names=visible,
            include_unavailable=True,
        )

        active_skills = list(explicit)
        for match in matches:
            if match.record.kind != "skill" or not match.record.available:
                continue
            if match.record.name in active_skills:
                continue
            if _high_confidence_skill(match):
                active_skills.append(match.record.name)
            if len(active_skills) >= len(explicit) + 2:
                break

        preloaded_tools: list[str] = []
        for match in matches if max_tools > 0 else []:
            record = match.record
            if record.kind != "tool" or record.always_on or record.name in visible:
                continue
            if not _useful_tool_match(match):
                continue
            preloaded_tools.append(record.name)
            if len(preloaded_tools) >= max_tools:
                break

        return CapabilityRoute(
            active_skills=tuple(active_skills),
            preloaded_tools=tuple(preloaded_tools),
            candidates=tuple(matches[:8]),
        )


def _high_confidence_skill(match: CapabilityMatch) -> bool:
    if match.exact_name:
        return True
    # Two independent terms avoids activating a workflow on one generic word.
    return match.score >= 4.0 and len(match.matched_terms) >= 2


def _useful_tool_match(match: CapabilityMatch) -> bool:
    return match.exact_name or (match.score >= 0.8 and len(match.matched_terms) >= 2)
