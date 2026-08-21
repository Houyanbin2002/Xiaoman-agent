from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|password|passwd|secret|token|cookie)"
    r"\s*[:=]\s*([^\s,;\"']+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+\-/]+=*")
_CONSTRAINT_RE = re.compile(
    r"(?i)(permission|denied|forbidden|unauthori[sz]ed|requires?|required|"
    r"missing|unsupported|not supported|version|platform|windows|linux|macos|"
    r"权限|拒绝|必须|必填|缺少|不支持|版本|平台)"
)
_ERROR_STATUSES = frozenset(
    {"error", "failed", "failure", "denied", "blocked", "cancelled", "interrupted"}
)
_ARGUMENT_KEYS = frozenset(
    {
        "description",
        "path",
        "file",
        "target",
        "cmd",
        "command",
        "query",
        "url",
        "package",
        "plugin",
        "version",
        "platform",
    }
)


@dataclass(frozen=True)
class SemanticEvidenceBatch:
    """Bounded, sanitized evidence passed to the shared semantic analyzer."""

    conversation_evidence: tuple[dict[str, object], ...]
    execution_episodes: tuple[dict[str, object], ...]

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item.get("episode_id") or "")
            for item in self.execution_episodes
            if str(item.get("episode_id") or "")
        )

    @property
    def execution_tool_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(call.get("tool") or "").strip().lower()
                for episode in self.execution_episodes
                for call in (
                    episode.get("calls")
                    if isinstance(episode.get("calls"), list)
                    else []
                )
                if isinstance(call, Mapping) and str(call.get("tool") or "").strip()
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "analysis_version": "conversation-v3",
            "conversation_evidence": list(self.conversation_evidence),
            "execution_episodes": list(self.execution_episodes),
        }


def build_semantic_evidence(
    messages: Sequence[Mapping[str, object]],
    *,
    max_messages: int = 20,
    max_episodes: int = 6,
) -> SemanticEvidenceBatch:
    """Create separate conversation and execution evidence views.

    Session messages remain the durable source of truth. This function only
    produces a bounded view and never forwards a raw tool chain to the model.
    """

    selected = list(messages)[-max(2, int(max_messages)) :]
    conversation: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    latest_user: Mapping[str, object] | None = None

    for message in selected:
        role = str(message.get("role") or "").strip().lower()
        message_id = str(message.get("id") or "").strip()
        if role not in {"user", "assistant"} or not message_id:
            continue
        if role == "user":
            latest_user = message
        conversation.append(
            {
                "id": message_id,
                "seq": message.get("seq"),
                "role": role,
                "content": sanitize_text(
                    message.get("content"), limit=5000 if role == "user" else 1400
                ),
                "timestamp": sanitize_text(message.get("timestamp"), limit=80),
            }
        )
        if role != "assistant":
            continue
        episode = _execution_episode(message, latest_user=latest_user)
        if episode is not None:
            episodes.append(episode)

    return SemanticEvidenceBatch(
        conversation_evidence=tuple(conversation),
        execution_episodes=tuple(episodes[-max(1, int(max_episodes)) :]),
    )


def sanitize_text(value: object, *, limit: int) -> str:
    text = str(value or "")
    text = _SECRET_RE.sub(r"\1=<redacted>", text)
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def _execution_episode(
    message: Mapping[str, object],
    *,
    latest_user: Mapping[str, object] | None,
) -> dict[str, object] | None:
    raw_chain = message.get("tool_chain")
    if not isinstance(raw_chain, list):
        return None
    raw_calls = [
        call
        for group in raw_chain
        if isinstance(group, Mapping)
        for call in (group.get("calls") if isinstance(group.get("calls"), list) else [])
        if isinstance(call, Mapping)
    ][:12]
    if not raw_calls:
        return None

    calls = [_compact_call(call) for call in raw_calls]
    failures = [
        index for index, call in enumerate(calls) if call["outcome"] != "success"
    ]
    successes = [
        index for index, call in enumerate(calls) if call["outcome"] == "success"
    ]
    signals: list[str] = []
    if failures and any(success > failures[0] for success in successes):
        signals.append("failure_recovery")
    if len(failures) >= 2:
        signals.append("repeated_failure")
    if any(call["outcome"] in {"blocked", "denied"} for call in calls):
        signals.append("permission_or_capability_constraint")
    if any(_CONSTRAINT_RE.search(str(call.get("result") or "")) for call in calls):
        signals.append("environment_constraint")
    unique_tools = {str(call.get("tool") or "") for call in calls}
    if len(calls) >= 3 and len(unique_tools) >= 2 and not failures:
        signals.append("verified_multistep")

    # Ordinary one-off successes never enter the semantic model.
    if not signals:
        return None

    message_id = str(message.get("id") or "")
    episode_id = (
        "episode_" + hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:24]
    )
    extra = message.get("extra")
    retrieval = extra.get("memory_retrieval") if isinstance(extra, Mapping) else None
    return {
        "episode_id": episode_id,
        "trace_ref": message_id,
        "user_message_id": str((latest_user or {}).get("id") or ""),
        "task_intent": sanitize_text((latest_user or {}).get("content"), limit=1200),
        "assistant_outcome": sanitize_text(message.get("content"), limit=700),
        "signals": list(dict.fromkeys(signals)),
        "calls": calls,
        "retrieved_execution_memory_ids": _string_ids(
            retrieval.get("execution_memory_ids")
            if isinstance(retrieval, Mapping)
            else None
        ),
        "used_execution_memory_ids": _string_ids(
            retrieval.get("used_execution_memory_ids")
            if isinstance(retrieval, Mapping)
            else None
        ),
    }


def _compact_call(call: Mapping[str, object]) -> dict[str, object]:
    status = str(call.get("status") or "").strip().lower()
    result = call.get("result")
    outcome = _call_outcome(status=status, result=result)
    arguments = call.get("final_arguments") or call.get("arguments")
    return {
        "tool": sanitize_text(call.get("name"), limit=120),
        "outcome": outcome,
        "arguments": _compact_arguments(arguments),
        "result": sanitize_text(_result_signal(result), limit=320),
    }


def _call_outcome(*, status: str, result: object) -> str:
    if status in {"blocked", "denied"}:
        return status
    if status in _ERROR_STATUSES:
        return "failure"
    parsed = _json_value(result)
    if isinstance(parsed, Mapping):
        exit_code = parsed.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return "failure"
        if parsed.get("success") is False or parsed.get("interrupted") is True:
            return "failure"
    text = str(result or "").strip()
    if re.match(r"(?i)^(error|failed|failure|错误|失败)\s*[:：]", text):
        return "failure"
    return "success" if status == "success" or result is not None else "unknown"


def _compact_arguments(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, object] = {}
    for key, raw in value.items():
        name = str(key).strip()
        if name not in _ARGUMENT_KEYS:
            continue
        compact[name] = sanitize_text(raw, limit=260)
    return compact


def _result_signal(value: object) -> str:
    parsed = _json_value(value)
    if isinstance(parsed, Mapping):
        parts: list[str] = []
        if "exit_code" in parsed:
            parts.append(f"exit_code={parsed.get('exit_code')}")
        if parsed.get("interrupted"):
            parts.append("interrupted=true")
        output = parsed.get("output") or parsed.get("error") or parsed.get("message")
        if output:
            parts.append(str(output))
        return "; ".join(parts)
    return str(parsed or "")


def _json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text.startswith(("{", "[")):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _string_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


__all__ = ["SemanticEvidenceBatch", "build_semantic_evidence", "sanitize_text"]
