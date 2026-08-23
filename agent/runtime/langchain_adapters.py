"""Adapters from Xiaoman providers/tools to LangChain core interfaces."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Self, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from agent.runtime.model_step import run_model_step
from agent.runtime.prompt_cache import tool_schema_fingerprint
from agent.tools.registry import ToolRegistry

ContentDelta = Callable[[dict[str, str]], Awaitable[None]]


class ProviderChatModel(BaseChatModel):
    """Expose the existing provider contract as a LangChain chat model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # LLMProvider is a structural Protocol, so Pydantic cannot build an
    # isinstance validator for it. Runtime validation happens in run_model_step.
    provider: Any = Field(exclude=True)
    model_name: str
    max_output_tokens: int = 8192
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "passive"
    iteration: int = 1
    purpose: str = "reasoning"
    cache_metadata: dict[str, Any] = Field(default_factory=dict)
    on_content_delta: ContentDelta | None = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "xiaoman-provider"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "source": self.source}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Self:
        del tool_choice, kwargs
        schemas = [convert_to_openai_tool(tool) for tool in tools]
        cache_metadata = {
            **self.cache_metadata,
            "tool_schema_count": len(schemas),
            "tool_schema_hash": tool_schema_fingerprint(schemas),
        }
        return cast(
            Self,
            self.model_copy(
                update={
                    "tool_schemas": schemas,
                    "cache_metadata": cache_metadata,
                }
            ),
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, kwargs
        raise RuntimeError("ProviderChatModel is async-only; use ainvoke/aagenerate")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, kwargs
        raw_messages = convert_to_openai_messages(messages)
        response = await run_model_step(
            self.provider,
            messages=cast(list[dict[str, Any]], raw_messages),
            tools=self.tool_schemas,
            model=self.model_name,
            max_tokens=self.max_output_tokens,
            tool_choice="auto",
            on_content_delta=self.on_content_delta,
            source=self.source,
            iteration=self.iteration,
            purpose=self.purpose,
            cache_metadata=self.cache_metadata,
        )
        message = AIMessage(
            content=response.content or "",
            tool_calls=[
                {
                    "name": call.name,
                    "args": dict(call.arguments),
                    "id": call.id,
                    "type": "tool_call",
                }
                for call in response.tool_calls
            ],
            additional_kwargs=dict(response.provider_fields or {}),
            response_metadata={
                "thinking": response.thinking,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "cache_prompt_tokens": response.cache_prompt_tokens,
                "cache_hit_tokens": response.cache_hit_tokens,
                "finish_reason": response.finish_reason,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def registry_tools_as_langchain(
    registry: ToolRegistry,
    names: Sequence[str] | set[str] | None = None,
) -> list[BaseTool]:
    """Build interoperable LangChain tools while retaining registry execution."""
    ordered = (
        registry.get_registered_order(set(names))
        if isinstance(names, set)
        else list(names) if names is not None else registry.get_registered_order()
    )
    adapted: list[BaseTool] = []
    for name in ordered:
        tool = registry.get_tool(name)
        if tool is None:
            continue

        def _executor_for(tool_name: str) -> Callable[..., Awaitable[Any]]:
            async def _execute(**kwargs: Any) -> Any:
                return await registry.execute(tool_name, kwargs)

            return _execute

        schema = registry.get_schemas([name])[0]["function"]

        adapted.append(
            StructuredTool.from_function(
                coroutine=_executor_for(name),
                name=tool.name,
                description=tool.description,
                args_schema=schema["parameters"],
            )
        )
    return adapted
