"""Token-watermark conversation compaction with cache-stable summary epochs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence


SUMMARY_METADATA_KEY = "context_summary"
SUMMARY_VERSION = "context-summary-v1"


@dataclass(frozen=True)
class ContextCompactionConfig:
    """Controls proactive conversation summarization.

    Token counts are conservative estimates because OpenAI-compatible providers
    do not expose a tokenizer before the request is sent.  The watermarks use
    hysteresis so a newly compacted session does not immediately compact again.
    """

    enabled: bool = True
    trigger_tokens: int = 200_000
    target_tokens: int = 100_000
    keep_recent_tokens: int = 40_000
    summary_max_tokens: int = 4_096
    chunk_tokens: int = 24_000
    max_history_messages: int = 2_000

    def normalized(self) -> "ContextCompactionConfig":
        trigger = max(8_000, int(self.trigger_tokens))
        target = min(trigger - 1_000, max(4_000, int(self.target_tokens)))
        keep_recent = min(target - 1_000, max(2_000, int(self.keep_recent_tokens)))
        return ContextCompactionConfig(
            enabled=bool(self.enabled),
            trigger_tokens=trigger,
            target_tokens=target,
            keep_recent_tokens=keep_recent,
            summary_max_tokens=max(256, int(self.summary_max_tokens)),
            chunk_tokens=max(4_000, int(self.chunk_tokens)),
            max_history_messages=max(100, int(self.max_history_messages)),
        )


@dataclass(frozen=True)
class ContextSummaryState:
    summary: str
    summarized_through: int
    epoch: int
    source_digest: str
    updated_at: str
    version: str = SUMMARY_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: object) -> "ContextSummaryState | None":
        if not isinstance(metadata, Mapping):
            return None
        raw = metadata.get(SUMMARY_METADATA_KEY)
        if not isinstance(raw, Mapping):
            return None
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            return None
        try:
            summarized_through = max(0, int(raw.get("summarized_through") or 0))
            epoch = max(1, int(raw.get("epoch") or 1))
        except (TypeError, ValueError):
            return None
        return cls(
            summary=summary,
            summarized_through=summarized_through,
            epoch=epoch,
            source_digest=str(raw.get("source_digest") or ""),
            updated_at=str(raw.get("updated_at") or ""),
            version=str(raw.get("version") or SUMMARY_VERSION),
        )


@dataclass(frozen=True)
class CompactionBoundary:
    start_index: int
    end_index: int
    cold_messages: tuple[dict[str, Any], ...]
    recent_messages: tuple[dict[str, Any], ...]
    cold_tokens: int
    recent_tokens: int


def estimate_tokens(value: object) -> int:
    """Return a deliberately conservative tokenizer-independent estimate."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return max(1, len(encoded) // 3)


def select_compaction_boundary(
    messages: Sequence[dict[str, Any]],
    *,
    start_index: int,
    keep_recent_tokens: int,
    protect_recent_tool_rounds: int = 3,
) -> CompactionBoundary | None:
    """Select a complete-turn boundary while retaining the largest safe suffix.

    A boundary is placed immediately before a user message, so an assistant
    tool call and its persisted tool results always stay in the same side of
    the split.  If one completed turn alone exceeds the recent budget, that
    turn may be summarized as a whole; a message is never split structurally.
    """

    total = len(messages)
    start = min(total, max(0, int(start_index)))
    if start >= total:
        return None
    active = list(messages[start:])
    if not active:
        return None

    # New user messages are legal turn boundaries.  len(active) is legal only
    # when the active prefix ends in an assistant message (a committed turn).
    user_boundaries = [
        index
        for index in range(1, len(active))
        if str(active[index].get("role") or "") == "user"
    ]
    candidates = list(user_boundaries)
    if str(active[-1].get("role") or "") == "assistant":
        candidates.append(len(active))
    candidates = sorted(set(candidates))
    if not candidates:
        return None

    # Always retain the latest user turn.  Also retain the configured number
    # of recent tool-bearing turns so Cache Breakpoint and conversation
    # compaction agree on the hot suffix.
    protected_starts: list[int] = []
    current_user_start = 0 if str(active[0].get("role") or "") == "user" else -1
    for index, message in enumerate(active):
        if str(message.get("role") or "") == "user":
            current_user_start = index
        if message.get("tool_chain") and current_user_start >= 0:
            protected_starts.append(current_user_start)
    latest_user_start = max(
        (
            index
            for index, message in enumerate(active)
            if str(message.get("role") or "") == "user"
        ),
        default=0,
    )
    protected_start = latest_user_start
    tool_rounds = max(0, int(protect_recent_tool_rounds))
    if tool_rounds and protected_starts:
        protected_start = min(
            protected_start,
            protected_starts[max(0, len(protected_starts) - tool_rounds)],
        )

    keep_budget = max(1, int(keep_recent_tokens))
    selected: int | None = None
    # Earliest eligible boundary retains the largest recent suffix under the
    # configured budget and therefore maximizes conversational continuity.
    for candidate in candidates:
        recent_tokens = (
            estimate_tokens(active[candidate:]) if candidate < len(active) else 0
        )
        if recent_tokens <= keep_budget:
            selected = candidate
            break
    if selected is None:
        # The latest complete turn is itself larger than the recent budget.
        # Summarize complete older turns and never cut through its structure.
        selected = candidates[-1]
    selected = min(selected, protected_start)

    cold = active[:selected]
    if not cold:
        return None
    recent = active[selected:]
    return CompactionBoundary(
        start_index=start,
        end_index=start + selected,
        cold_messages=tuple(cold),
        recent_messages=tuple(recent),
        cold_tokens=estimate_tokens(cold),
        recent_tokens=estimate_tokens(recent) if recent else 0,
    )


def summary_source_digest(
    prior_summary: str,
    messages: Sequence[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {"prior_summary": prior_summary, "messages": list(messages)},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_summary_state(
    *,
    summary: str,
    summarized_through: int,
    previous_epoch: int,
    source_digest: str,
) -> ContextSummaryState:
    return ContextSummaryState(
        summary=summary.strip(),
        summarized_through=max(0, int(summarized_through)),
        epoch=max(0, int(previous_epoch)) + 1,
        source_digest=source_digest,
        updated_at=datetime.now().astimezone().isoformat(),
    )


def write_summary_state(metadata: object, state: ContextSummaryState) -> dict[str, Any]:
    copied = dict(metadata) if isinstance(metadata, Mapping) else {}
    copied[SUMMARY_METADATA_KEY] = state.to_metadata()
    return copied


def render_summary_evidence(messages: Sequence[dict[str, Any]]) -> str:
    """Render persisted turns, including tool trajectories, for summarization."""

    blocks: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        message_id = str(message.get("id") or "")
        timestamp = str(message.get("timestamp") or "")
        header = f"[message {index} role={role}"
        if message_id:
            header += f" id={message_id}"
        if timestamp:
            header += f" time={timestamp}"
        header += "]"
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        parts = [header, content]
        tool_chain = message.get("tool_chain") or []
        if tool_chain:
            parts.append(
                "[tool_trajectory]\n"
                + json.dumps(
                    tool_chain, ensure_ascii=False, sort_keys=True, default=str
                )
            )
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def chunk_summary_evidence(text: str, *, chunk_tokens: int) -> list[str]:
    """Split evidence into bounded chunks without sending the full history once."""

    max_chars = max(3_000, int(chunk_tokens) * 3)
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        split = remaining.rfind("\n\n[message ", 0, max_chars)
        if split < max_chars // 2:
            split = max_chars
        chunks.append(remaining[:split])
        remaining = remaining[split:].lstrip()
    return chunks
