"""Shared read boundary: historical evidence is not a current obligation."""

from collections.abc import Callable

from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.models import RecentActivityCandidate

ActivityStates = dict[str, list[dict[str, object]]]
ActivityResolver = Callable[[list[str]], ActivityStates]


def activity_entries(
    event: ConversationSemanticBatchCommitted,
) -> list[RecentActivityCandidate]:
    """Keep the same validated order for projection and episodic source IDs."""
    valid = set(event.message_ids)
    users = set(event.user_message_ids) & valid
    return [
        item
        for item in event.payload.recent_activity_entries
        if item.source_message_ids
        and set(item.source_message_ids) <= valid
        and set(item.source_message_ids) & users
    ]


def activity_states(
    refs: list[str], resolved: ActivityStates
) -> list[dict[str, object]]:
    states = {
        str(state["id"]): state for ref in refs for state in resolved.get(ref, [])
    }
    return list(states.values())


def current_activity_evidence(states: list[dict[str, object]]) -> bool:
    # A mixed-topic turn is indivisible evidence. Conservatively omit it from
    # automatic context if any linked activity ended; explicit recall keeps it.
    return all(state.get("status") in {"active", "planned"} for state in states)


def activity_label(states: list[dict[str, object]]) -> str:
    labels = {
        "active": "进行中",
        "planned": "计划中",
        "completed": "已完成",
        "cancelled": "已取消",
        "dismissed": "不再关注",
        "expired": "已过期",
    }
    return "；".join(
        f"{state['id']}:{labels.get(str(state.get('status')), '状态未知')}"
        for state in states
    )
