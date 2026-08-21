from __future__ import annotations

from typing import Any, cast

from fastapi import FastAPI, HTTPException

from ..contracts import DashboardRuntimeServices
from ..model_settings import memory_model_row, save_memory_model_settings
from bootstrap.runtime_models import ModelCatalogError, RuntimeModelUpdate

from ..schemas import ModelCatalogPayload, ModelUpdatePayload
from ..support import load_config, model_row, save_config


def register_model_routes(
    app: FastAPI,
    services: DashboardRuntimeServices,
) -> None:
    @app.get("/api/dashboard/control/models")
    def list_models() -> list[dict[str, Any]]:
        config = services.config
        config_data = load_config(services.config_path)
        rows = [
            model_row(
                "main",
                "主模型",
                config.model,
                config.provider,
                config.base_url,
                config.api_key,
                "小满主对话、工具决策与通用后台生成",
            ),
            model_row(
                "fast",
                "快速模型",
                config.light_model or config.model,
                getattr(config, "light_provider", "") or config.provider,
                config.light_base_url or config.base_url,
                config.light_api_key or config.api_key,
                "近期语境压缩与低成本辅助处理",
            ),
            model_row(
                "agent",
                "Agent 执行模型",
                config.agent_model or config.model,
                getattr(config, "agent_provider", "") or config.provider,
                config.agent_base_url or config.base_url,
                config.agent_api_key or config.api_key,
                "子任务与工作流中的独立 Agent 执行",
            ),
            model_row(
                "vision",
                "视觉模型",
                config.vl_model,
                getattr(config, "vl_provider", "") or config.provider,
                config.vl_base_url or config.base_url,
                config.vl_api_key or config.api_key,
                "主模型不支持多模态时的图片理解",
            ),
        ]
        rows.append(memory_model_row(config_data, config))
        return rows

    @app.post("/api/dashboard/control/models/{slot}/catalog")
    async def list_model_catalog(
        slot: str,
        payload: ModelCatalogPayload,
    ) -> dict[str, Any]:
        runtime = services.runtime_models
        if runtime is None:
            raise HTTPException(status_code=503, detail="模型运行时尚未就绪。")
        try:
            return cast(
                dict[str, Any],
                await runtime.fetch_catalog(
                    slot,
                    base_url=payload.base_url,
                    api_key=payload.api_key,
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelCatalogError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.patch("/api/dashboard/control/models/{slot}")
    async def update_model(slot: str, payload: ModelUpdatePayload) -> dict[str, Any]:
        config_data = load_config(services.config_path)
        runtime = services.runtime_models
        try:
            current = runtime.current_connection(slot) if runtime is not None else _current_connection(services.config, slot)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        update = RuntimeModelUpdate(
            model=payload.model.strip(),
            provider=str(payload.provider or current.provider).strip(),
            base_url=str(payload.base_url or current.base_url).strip().rstrip("/"),
            api_key=str(payload.api_key or current.api_key).strip(),
            output_dimensionality=(
                payload.output_dimensionality or current.output_dimensionality
                if slot == "memory"
                else None
            ),
        )
        if slot == "memory":
            save_memory_model_settings(config_data, payload)
            result = await runtime.apply(slot, update) if runtime is not None else _restart_fallback(slot, update.model)
            save_config(services.config_path, config_data)
            return result

        section_map = {
            "main": "main",
            "fast": "fast",
            "agent": "agent",
            "vision": "vl",
        }
        section = section_map.get(slot)
        if section is None:
            raise HTTPException(status_code=404, detail="未知模型槽位")
        llm = cast(dict[str, Any], config_data.setdefault("llm", {}))
        target = cast(dict[str, Any], llm.setdefault(section, {}))
        target["model"] = payload.model.strip()
        if payload.provider:
            target["provider"] = payload.provider.strip()
        if payload.base_url is not None:
            target["base_url"] = payload.base_url.strip()
        if payload.api_key:
            target["api_key"] = payload.api_key.strip()
        result = await runtime.apply(slot, update) if runtime is not None else _restart_fallback(slot, update.model)
        save_config(services.config_path, config_data)
        return result


def _restart_fallback(slot: str, model: str) -> dict[str, Any]:
    return {
        "saved": True,
        "hot_reloaded": False,
        "restart_required": True,
        "slot": slot,
        "model": model,
    }


def _current_connection(config: Any, slot: str) -> RuntimeModelUpdate:
    if slot == "main":
        return RuntimeModelUpdate(config.model, config.provider, config.base_url or "", config.api_key)
    if slot == "fast":
        return RuntimeModelUpdate(
            config.light_model or config.model,
            getattr(config, "light_provider", "") or config.provider,
            config.light_base_url or config.base_url or "",
            config.light_api_key or config.api_key,
        )
    if slot == "agent":
        return RuntimeModelUpdate(
            config.agent_model or config.model,
            getattr(config, "agent_provider", "") or config.provider,
            config.agent_base_url or config.base_url or "",
            config.agent_api_key or config.api_key,
        )
    if slot == "vision":
        return RuntimeModelUpdate(
            config.vl_model or config.model,
            getattr(config, "vl_provider", "") or config.provider,
            config.vl_base_url or config.base_url or "",
            config.vl_api_key or config.api_key,
        )
    if slot == "memory":
        embedding = config.memory.embedding
        base_url = embedding.base_url or config.light_base_url or config.base_url or ""
        return RuntimeModelUpdate(
            embedding.model,
            "dashscope" if "dashscope.aliyuncs.com" in base_url else "custom",
            base_url,
            embedding.api_key or config.light_api_key or config.api_key,
            embedding.output_dimensionality,
        )
    raise ValueError("未知模型槽位")
