from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from agent.tools.vision import ReadImageVisionTool
from core.net.http import RequestBudget
from infra.providers.llm_provider import LLMProvider

from .providers import _sanitize_extra_body


@dataclass(frozen=True)
class RuntimeModelUpdate:
    model: str
    provider: str
    base_url: str
    api_key: str
    output_dimensionality: int | None = None


class ModelCatalogError(RuntimeError):
    pass


class RuntimeModelService:
    """Own model-slot routing, remote catalogs, and in-process configuration swaps."""

    def __init__(
        self,
        *,
        config: Any,
        main_provider: LLMProvider,
        light_provider: LLMProvider | None,
        agent_provider: LLMProvider | None,
        vl_provider: LLMProvider | None,
        agent_loop: Any,
        memory_runtime: Any,
        plugin_manager: Any | None,
        workflow_runtime: Any | None,
        tools: Any,
        workspace: Any,
        http_resources: Any,
        core_runtime: Any | None = None,
    ) -> None:
        self.config = config
        self._main_provider = main_provider
        self._light_provider = light_provider
        self._agent_provider = agent_provider
        self._vl_provider = vl_provider
        self._agent_loop = agent_loop
        self._memory_runtime = memory_runtime
        self._plugin_manager = plugin_manager
        self._workflow_runtime = workflow_runtime
        self._tools = tools
        self._workspace = workspace
        self._http_resources = http_resources
        self._core_runtime = core_runtime
        self._proactive_loop: Any | None = None
        self._memory_optimizer: Any | None = None
        self._update_lock = asyncio.Lock()

    def bind_background_services(
        self,
        *,
        proactive_loop: Any | None,
        memory_optimizer: Any | None,
    ) -> None:
        self._proactive_loop = proactive_loop
        self._memory_optimizer = memory_optimizer

    async def aclose(self) -> None:
        seen: set[int] = set()
        for provider in (
            self._main_provider,
            self._light_provider,
            self._agent_provider,
            self._vl_provider,
        ):
            if provider is None or id(provider) in seen:
                continue
            seen.add(id(provider))
            close = getattr(provider, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

    async def apply(self, slot: str, update: RuntimeModelUpdate) -> dict[str, Any]:
        _validate_base_url(update.base_url)
        async with self._update_lock:
            if slot == "memory":
                self._apply_memory(update)
            elif slot in {"main", "fast", "agent", "vision"}:
                self._apply_generation_model(slot, update)
            else:
                raise ValueError("未知模型槽位")
        return {
            "saved": True,
            "hot_reloaded": True,
            "restart_required": False,
            "slot": slot,
            "model": update.model,
        }

    async def fetch_catalog(
        self,
        slot: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        current = self.current_connection(slot)
        target_url = str(base_url or current.base_url).strip().rstrip("/")
        key = str(api_key or current.api_key).strip()
        _validate_base_url(target_url)
        if not key:
            raise ModelCatalogError("当前模型接口没有可用的 API Key。")
        response = await self._http_resources.external_default.get(
            f"{target_url}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout_s=20.0,
            budget=RequestBudget(total_timeout_s=25.0),
        )
        if response.status_code >= 400:
            detail = _remote_error_detail(response)
            raise ModelCatalogError(
                f"模型目录获取失败（HTTP {response.status_code}）：{detail}"
            )
        items = _normalize_model_catalog(response.json())
        if not items:
            raise ModelCatalogError("接口返回成功，但没有发现可选择的模型。")
        return {"items": items, "total": len(items), "base_url": target_url}

    def current_connection(self, slot: str) -> RuntimeModelUpdate:
        config = self.config
        if slot == "main":
            return RuntimeModelUpdate(
                model=str(getattr(config, "model", "") or ""),
                provider=str(getattr(config, "provider", "") or "openai"),
                base_url=str(getattr(config, "base_url", "") or ""),
                api_key=str(getattr(config, "api_key", "") or ""),
            )
        if slot == "fast":
            return RuntimeModelUpdate(
                model=str(getattr(config, "light_model", "") or getattr(config, "model", "")),
                provider=str(getattr(config, "light_provider", "") or getattr(config, "provider", "openai")),
                base_url=str(getattr(config, "light_base_url", "") or getattr(config, "base_url", "") or ""),
                api_key=str(getattr(config, "light_api_key", "") or getattr(config, "api_key", "") or ""),
            )
        if slot == "agent":
            return RuntimeModelUpdate(
                model=str(getattr(config, "agent_model", "") or getattr(config, "model", "")),
                provider=str(getattr(config, "agent_provider", "") or getattr(config, "provider", "openai")),
                base_url=str(getattr(config, "agent_base_url", "") or getattr(config, "base_url", "") or ""),
                api_key=str(getattr(config, "agent_api_key", "") or getattr(config, "api_key", "") or ""),
            )
        if slot == "vision":
            return RuntimeModelUpdate(
                model=str(getattr(config, "vl_model", "") or getattr(config, "model", "")),
                provider=str(getattr(config, "vl_provider", "") or getattr(config, "provider", "openai")),
                base_url=str(getattr(config, "vl_base_url", "") or getattr(config, "base_url", "") or ""),
                api_key=str(getattr(config, "vl_api_key", "") or getattr(config, "api_key", "") or ""),
            )
        if slot == "memory":
            embedding = config.memory.embedding
            return RuntimeModelUpdate(
                model=str(embedding.model or ""),
                provider="dashscope" if "dashscope.aliyuncs.com" in str(embedding.base_url) else "custom",
                base_url=str(embedding.base_url or getattr(config, "light_base_url", "") or getattr(config, "base_url", "") or ""),
                api_key=str(embedding.api_key or getattr(config, "light_api_key", "") or getattr(config, "api_key", "") or ""),
                output_dimensionality=embedding.output_dimensionality,
            )
        raise ValueError("未知模型槽位")

    def _apply_generation_model(self, slot: str, update: RuntimeModelUpdate) -> None:
        provider = self._provider_for_slot(slot)
        if provider is None:
            provider = self._new_provider(slot, update)
            self._set_provider_for_slot(slot, provider)
        else:
            provider.reconfigure(
                api_key=update.api_key,
                base_url=update.base_url,
                provider_name=update.provider,
                system_prompt=("" if slot == "vision" else self.config.system_prompt),
                extra_body=self._extra_body_for_slot(slot, update.base_url),
            )

        if slot == "main":
            self.config.model = update.model
            self.config.provider = update.provider
            self.config.base_url = update.base_url
            self.config.api_key = update.api_key
            self._agent_loop.reconfigure_models(provider=provider, model=update.model)
            self._reconfigure_main_consumers(provider, update.model)
        elif slot == "fast":
            self.config.light_model = update.model
            self.config.light_provider = update.provider
            self.config.light_base_url = update.base_url
            self.config.light_api_key = update.api_key
            self._agent_loop.reconfigure_models(
                light_provider=provider,
                light_model=update.model,
            )
        elif slot == "agent":
            self.config.agent_model = update.model
            self.config.agent_provider = update.provider
            self.config.agent_base_url = update.base_url
            self.config.agent_api_key = update.api_key
            executor = getattr(self._workflow_runtime, "subagent_executor", None)
            if executor is not None and hasattr(executor, "reconfigure"):
                executor.reconfigure(provider=provider, model=update.model)
        else:
            self.config.vl_model = update.model
            self.config.vl_provider = update.provider
            self.config.vl_base_url = update.base_url
            self.config.vl_api_key = update.api_key
            self._reconfigure_vision_tool(provider, update.model)

    def _apply_memory(self, update: RuntimeModelUpdate) -> None:
        engine = self._memory_runtime.engine
        if not hasattr(engine, "reconfigure_embedding"):
            raise RuntimeError("当前记忆引擎不支持嵌入模型热更新。")
        engine.reconfigure_embedding(
            model=update.model,
            base_url=update.base_url,
            api_key=update.api_key,
            output_dimensionality=update.output_dimensionality,
        )

    def _reconfigure_main_consumers(self, provider: LLMProvider, model: str) -> None:
        plugin_llm = getattr(self._plugin_manager, "llm", None)
        if plugin_llm is not None and hasattr(plugin_llm, "reconfigure"):
            plugin_llm.reconfigure(provider=provider, model=model)
        if not str(getattr(self.config, "agent_model", "") or "").strip():
            executor = getattr(self._workflow_runtime, "subagent_executor", None)
            if executor is not None and hasattr(executor, "reconfigure"):
                executor.reconfigure(provider=provider, model=model)
        if self._proactive_loop is not None:
            self._proactive_loop.reconfigure_model(provider=provider, model=model)
    def _reconfigure_vision_tool(self, provider: LLMProvider, model: str) -> None:
        tool = self._tools.get_tool("read_image_vision")
        if tool is not None and hasattr(tool, "reconfigure"):
            tool.reconfigure(provider=provider, model=model)
            return
        if not bool(getattr(self.config, "multimodal", True)):
            self._tools.register(
                ReadImageVisionTool(
                    vl_provider=provider,
                    vl_model=model,
                    allowed_dir=self._workspace,
                ),
                always_on=True,
                risk="read-only",
                search_hint="看图 识图 图片内容 视觉识别 VL",
            )

    def _provider_for_slot(self, slot: str) -> LLMProvider | None:
        return {
            "main": self._main_provider,
            "fast": self._light_provider,
            "agent": self._agent_provider,
            "vision": self._vl_provider,
        }[slot]

    def _set_provider_for_slot(self, slot: str, provider: LLMProvider) -> None:
        attr = {
            "main": "_main_provider",
            "fast": "_light_provider",
            "agent": "_agent_provider",
            "vision": "_vl_provider",
        }[slot]
        setattr(self, attr, provider)
        if self._core_runtime is not None:
            core_attr = {
                "main": "provider",
                "fast": "light_provider",
                "agent": "agent_provider",
                "vision": "vl_provider",
            }[slot]
            setattr(self._core_runtime, core_attr, provider)

    def _new_provider(self, slot: str, update: RuntimeModelUpdate) -> LLMProvider:
        return LLMProvider(
            api_key=update.api_key,
            base_url=update.base_url,
            system_prompt="" if slot == "vision" else self.config.system_prompt,
            extra_body=self._extra_body_for_slot(slot, update.base_url),
            provider_name=update.provider,
            force_disable_thinking=slot == "fast",
            payload_snapshot_enabled=bool(getattr(self.config, "dev_mode", False)),
        )

    def _extra_body_for_slot(self, slot: str, base_url: str) -> dict[str, Any]:
        if slot == "main":
            return _sanitize_extra_body(
                base_url=base_url,
                extra_body=dict(getattr(self.config, "extra_body", {}) or {}),
            )
        if slot == "fast":
            return _sanitize_extra_body(
                base_url=base_url,
                extra_body={"enable_thinking": False},
            )
        return _sanitize_extra_body(base_url=base_url, extra_body={})


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(str(base_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("模型服务地址必须是有效的 HTTP(S) URL。")


def _normalize_model_catalog(payload: object) -> list[dict[str, str]]:
    rows: object = payload
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("models") or payload.get("items") or []
    if not isinstance(rows, list):
        return []
    found: dict[str, dict[str, str]] = {}
    for row in rows:
        if isinstance(row, str):
            model_id = row.strip()
            owner = ""
        elif isinstance(row, dict):
            model_id = str(row.get("id") or row.get("name") or "").strip()
            owner = str(row.get("owned_by") or row.get("provider") or "").strip()
        else:
            continue
        if model_id:
            found[model_id] = {"id": model_id, "owned_by": owner}
    return [found[key] for key in sorted(found, key=str.casefold)]


def _remote_error_detail(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        return str(getattr(response, "text", "远端接口拒绝了请求"))[:300]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:300]
        return str(payload.get("message") or payload.get("detail") or payload)[:300]
    return str(payload)[:300]
