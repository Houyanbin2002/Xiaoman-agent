from __future__ import annotations

"""Deterministic assertions over observed evidence, not the Agent's claims."""

import json
import re
from collections.abc import Mapping
from typing import Any

from .models import AgentRun, EvalCase, Score


def contains(actual: Any, wanted: Any) -> bool:
    if isinstance(wanted, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and contains(actual[key], value)
            for key, value in wanted.items()
        )
    if isinstance(wanted, list):
        return isinstance(actual, list) and all(
            any(contains(item, target) for item in actual) for target in wanted
        )
    return type(actual) is type(wanted) and actual == wanted


def decoded(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def check_evidence(spec: Mapping[str, Any], run: AgentRun) -> bool:
    source = spec.get("source")
    if source == "turn":
        turns = run.metadata.get("turn_runs", [])
        index = int(spec["index"])
        if index < 0 or index >= len(turns):
            return False
        return check_evidence(spec["assert"], AgentRun.from_value(turns[index]))
    if source == "state":
        return contains(run.state, spec["contains"])
    if source == "response_json":
        # Permit a JSON code fence, but no recovery from arbitrary prose or wrong facts.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", run.response.strip())
        return contains(decoded(text), spec["contains"])
    if source == "response_regex":
        return re.search(str(spec["pattern"]), run.response) is not None
    if source == "memory_count":
        records = run.state.get("new_memory_records")
        if not isinstance(records, list):
            return False
        return (
            sum(contains(record, spec.get("where", {})) for record in records)
            == spec["count"]
        )
    if source == "tool":
        found = []
        for tool in run.tools:
            if tool.name != spec["name"]:
                continue
            if "status" in spec:
                wanted = spec["status"]
                allowed = wanted if isinstance(wanted, list) else [wanted]
                if tool.status not in allowed:
                    continue
                if tool.status in {"success", "completed"} and tool.error:
                    continue
            if "arguments" in spec and not contains(tool.arguments, spec["arguments"]):
                continue
            if "output" in spec and not contains(decoded(tool.output), spec["output"]):
                continue
            if spec.get("nonempty_output") and tool.output in (None, "", {}, []):
                continue
            found.append(tool)
        if "count" in spec:
            return len(found) == spec["count"]
        return bool(found)
    raise ValueError(f"unsupported evidence source: {source}")


def score_evidence(case: EvalCase, run: AgentRun) -> list[Score]:
    scores = []
    for name, spec in case.expected.get("evidence", {}).items():
        ok = check_evidence(spec, run)
        scores.append(
            Score(
                name,
                float(ok),
                ok,
                hard=bool(spec.get("hard", True)),
                reason=f"{spec['source']} evidence {'matched' if ok else 'missing or mismatched'}",
                source="deterministic_evidence",
            )
        )
    return scores
