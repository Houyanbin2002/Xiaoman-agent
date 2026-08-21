from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from bootstrap.runtime_models import RuntimeModelService, RuntimeModelUpdate


class _Provider:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def reconfigure(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    async def aclose(self) -> None:
        return None


class _Response:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {
            "data": [
                {"id": "qwen-plus", "owned_by": "dashscope"},
                {"id": "deepseek-v4-flash", "owned_by": "dashscope"},
            ]
        }


def _runtime() -> tuple[RuntimeModelService, SimpleNamespace, dict[str, Any]]:
    config = SimpleNamespace(
        model="main-old",
        provider="openai",
        base_url="https://example.com/v1",
        api_key="main-key",
        system_prompt="system",
        extra_body={},
        dev_mode=False,
        light_model="fast-old",
        light_provider="openai",
        light_base_url="https://example.com/v1",
        light_api_key="fast-key",
        agent_model="agent-old",
        agent_provider="openai",
        agent_base_url="https://example.com/v1",
        agent_api_key="agent-key",
        vl_model="vision-old",
        vl_provider="openai",
        vl_base_url="https://example.com/v1",
        vl_api_key="vision-key",
        multimodal=True,
        memory=SimpleNamespace(
            embedding=SimpleNamespace(
                model="embed-old",
                base_url="https://example.com/v1",
                api_key="embed-key",
                output_dimensionality=1024,
            )
        ),
    )
    calls: dict[str, Any] = {
        "loop": MagicMock(),
        "maintenance": MagicMock(),
        "plugin_llm": MagicMock(),
        "executor": MagicMock(),
        "engine": MagicMock(),
        "proactive": MagicMock(),
        "optimizer": MagicMock(),
    }
    requester = SimpleNamespace(get=MagicMock())
    requester.get.side_effect = lambda *_args, **_kwargs: _async_value(_Response())
    runtime = RuntimeModelService(
        config=config,
        main_provider=_Provider(),  # type: ignore[arg-type]
        light_provider=_Provider(),  # type: ignore[arg-type]
        agent_provider=_Provider(),  # type: ignore[arg-type]
        vl_provider=_Provider(),  # type: ignore[arg-type]
        agent_loop=calls["loop"],
        memory_runtime=SimpleNamespace(
            engine=calls["engine"],
            markdown=SimpleNamespace(
                maintenance=calls["maintenance"],
            ),
        ),
        plugin_manager=SimpleNamespace(llm=calls["plugin_llm"]),
        workflow_runtime=SimpleNamespace(subagent_executor=calls["executor"]),
        tools=SimpleNamespace(get_tool=lambda _name: None),
        workspace=Path("."),
        http_resources=SimpleNamespace(external_default=requester),
    )
    runtime.bind_background_services(
        proactive_loop=calls["proactive"],
        memory_optimizer=calls["optimizer"],
    )
    calls["requester"] = requester
    return runtime, config, calls


async def _async_value(value: Any) -> Any:
    return value


@pytest.mark.asyncio
async def test_runtime_model_slots_update_their_declared_consumers() -> None:
    runtime, config, calls = _runtime()

    result = await runtime.apply(
        "main",
        RuntimeModelUpdate(
            model="main-new",
            provider="openai",
            base_url="https://new.example.com/v1",
            api_key="new-key",
        ),
    )
    await runtime.apply(
        "fast",
        RuntimeModelUpdate(
            model="fast-new",
            provider="openai",
            base_url="https://fast.example.com/v1",
            api_key="fast-new-key",
        ),
    )
    await runtime.apply(
        "agent",
        RuntimeModelUpdate(
            model="agent-new",
            provider="openai",
            base_url="https://agent.example.com/v1",
            api_key="agent-new-key",
        ),
    )

    assert result["hot_reloaded"] is True
    assert result["restart_required"] is False
    assert config.model == "main-new"
    assert config.light_model == "fast-new"
    assert config.agent_model == "agent-new"
    calls["loop"].reconfigure_models.assert_any_call(
        provider=runtime._main_provider,
        model="main-new",
    )
    calls["loop"].reconfigure_models.assert_any_call(
        light_provider=runtime._light_provider,
        light_model="fast-new",
    )
    calls["executor"].reconfigure.assert_called_once_with(
        provider=runtime._agent_provider,
        model="agent-new",
    )
    calls["plugin_llm"].reconfigure.assert_called_once_with(
        provider=runtime._main_provider,
        model="main-new",
    )
    calls["proactive"].reconfigure_model.assert_called_once_with(
        provider=runtime._main_provider,
        model="main-new",
    )


@pytest.mark.asyncio
async def test_runtime_memory_model_reconfigures_akasha_in_place() -> None:
    runtime, _config, calls = _runtime()

    await runtime.apply(
        "memory",
        RuntimeModelUpdate(
            model="text-embedding-v4",
            provider="dashscope",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="embed-new-key",
            output_dimensionality=1024,
        ),
    )

    calls["engine"].reconfigure_embedding.assert_called_once_with(
        model="text-embedding-v4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="embed-new-key",
        output_dimensionality=1024,
    )


@pytest.mark.asyncio
async def test_model_catalog_uses_current_connection_and_normalizes_models() -> None:
    runtime, _config, calls = _runtime()

    result = await runtime.fetch_catalog("main")

    assert [item["id"] for item in result["items"]] == [
        "deepseek-v4-flash",
        "qwen-plus",
    ]
    args, kwargs = calls["requester"].get.call_args
    assert args[0] == "https://example.com/v1/models"
    assert kwargs["headers"]["Authorization"] == "Bearer main-key"
