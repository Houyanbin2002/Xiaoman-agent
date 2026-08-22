from __future__ import annotations

"""OpenAI-compatible LLM-as-Judge adapter for the Rubric evaluator."""

import json
import re
from collections.abc import Mapping
from typing import Any

from agent.config_models import Config

from .models import AgentRun, EvalCase


_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


class RubricJudgeError(RuntimeError):
    """The judge endpoint returned an unusable response."""


class OpenAICompatibleRubricJudge:
    """Score explicit ``evaluator=llm`` Rubric criteria with a chat model.

    The adapter intentionally uses the same OpenAI-compatible endpoint and
    credentials as Xiaoman's configured models. It is synchronous because the
    current Rubric contract is synchronous; the evaluation runner executes it
    outside user request handling.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 90.0,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("LLM Judge requires a non-empty API key")
        if not model.strip():
            raise ValueError("LLM Judge requires a model name")
        self.model = model.strip()
        self._base_url = base_url or ""
        if client is not None:
            self._client = client
        else:
            # Import lazily so the repository's optional-dependency test stub
            # does not make deterministic evaluator tests depend on OpenAI.
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url or None,
                timeout=max(1.0, float(timeout_s)),
                max_retries=1,
            )

    @classmethod
    def from_config(cls, config: Config, *, model: str = "") -> "OpenAICompatibleRubricJudge":
        return cls(
            api_key=config.light_api_key or config.api_key,
            base_url=config.light_base_url or config.base_url or "",
            model=model or config.light_model or config.model,
        )

    def __call__(self, case: EvalCase, run: AgentRun) -> Mapping[str, float]:
        criteria = [criterion for criterion in case.rubric if criterion.evaluator == "llm"]
        if not criteria:
            return {}

        criterion_payload = [
            {
                "id": criterion.criterion_id,
                "description": criterion.description,
                "threshold": criterion.threshold,
            }
            for criterion in criteria
        ]
        request = {
            "case_id": case.case_id,
            "title": case.title,
            "user_input": case.input,
            "criteria": criterion_payload,
            "agent_response": run.response[:12000],
            "tool_calls": [tool.to_dict() for tool in run.tools][:30],
            "state": _truncate_json(run.state, 12000),
            "memory_events": _truncate_json(list(run.memory_events), 12000),
            "status": run.status,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 Xiaoman 的严格评测器。只根据给定的用户请求、Rubric、"
                    "Agent 输出、工具轨迹和状态评分。每个 criterion 输出 0 到 1 的"
                    "连续分数：1 表示完全满足，0 表示完全不满足。不要因为措辞流畅"
                    "而忽略工具、安全、状态或事实错误。只返回 JSON："
                    "{\"scores\": {\"criterion_id\": 0.0}, "
                    "\"reasons\": {\"criterion_id\": \"简短原因\"}}。"
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request_kwargs = {
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": max(512, min(2048, 384 * len(criteria))),
                    "response_format": {"type": "json_object"},
                    "messages": messages,
                }
                # DashScope's DeepSeek/Qwen-compatible endpoint supports this
                # switch; don't send vendor-specific fields to plain OpenAI.
                if "dashscope" in self._base_url.casefold():
                    request_kwargs["extra_body"] = {"enable_thinking": False}
                response = self._client.chat.completions.create(**request_kwargs)
            except Exception as exc:  # pragma: no cover - live endpoint path
                last_error = RubricJudgeError(
                    f"judge request failed: {type(exc).__name__}: {exc}"
                )
                continue
            content = (
                str(getattr(response.choices[0].message, "content", None) or "")
                if response.choices
                else ""
            )
            try:
                payload = _parse_json_object(content)
                scores = payload.get("scores", payload)
                if not isinstance(scores, Mapping):
                    raise RubricJudgeError(
                        "judge response does not contain a scores object"
                    )
                normalized: dict[str, float] = {}
                for criterion in criteria:
                    raw = scores.get(criterion.criterion_id)
                    if raw is None:
                        raise RubricJudgeError(
                            "judge response missing score for criterion "
                            f"{criterion.criterion_id!r}"
                        )
                    try:
                        normalized[criterion.criterion_id] = max(
                            0.0,
                            min(1.0, float(raw)),
                        )
                    except (TypeError, ValueError) as exc:
                        raise RubricJudgeError(
                            "judge score for "
                            f"{criterion.criterion_id!r} is not numeric"
                        ) from exc
                return normalized
            except RubricJudgeError as exc:
                last_error = exc
                if attempt < 2:
                    messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "上一次输出无法解析："
                                f"{exc}。请重新评分，只返回符合约定结构的 JSON，"
                                "不要输出 Markdown 或额外说明。"
                            ),
                        },
                    ]

        if last_error is not None:
            raise last_error
        raise RubricJudgeError("judge returned no usable response")


def build_judge_from_config(config: Config, *, model: str = "") -> OpenAICompatibleRubricJudge:
    """Build a judge from the configured fast model, falling back to main."""

    return OpenAICompatibleRubricJudge.from_config(config, model=model)


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    match = _FENCED_JSON.search(text)
    if match:
        text = match.group(1).strip()
    if not text:
        raise RubricJudgeError("judge returned an empty response")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise RubricJudgeError("judge response is not valid JSON")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RubricJudgeError("judge response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RubricJudgeError("judge response JSON must be an object")
    return value


def _truncate_json(value: Any, limit: int) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= limit:
        return value
    return encoded[:limit] + "…"


__all__ = [
    "OpenAICompatibleRubricJudge",
    "RubricJudgeError",
    "build_judge_from_config",
]
