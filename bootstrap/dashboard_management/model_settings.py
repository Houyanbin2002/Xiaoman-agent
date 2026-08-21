from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlparse

from fastapi import HTTPException

from .schemas import ModelUpdatePayload

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_MEMORY_ENGINE = "akasha"


def memory_model_row(
    config_data: dict[str, Any],
    runtime_config: Any,
) -> dict[str, Any]:
    """Expose the memory embedding model through the common model catalog."""

    memory = _mapping(config_data.get("memory"))
    embedding = _mapping(memory.get("embedding"))
    runtime_memory = getattr(runtime_config, "memory", None)
    runtime_embedding = getattr(runtime_memory, "embedding", None)
    base_url = str(
        embedding.get("base_url")
        or getattr(runtime_embedding, "base_url", "")
        or DASHSCOPE_BASE_URL
    ).strip()
    model = str(
        embedding.get("model")
        or getattr(runtime_embedding, "model", "")
        or DEFAULT_EMBEDDING_MODEL
    ).strip()
    dimension = int(
        embedding.get("output_dimensionality")
        or getattr(runtime_embedding, "output_dimensionality", None)
        or DEFAULT_EMBEDDING_DIMENSION
    )
    explicit_key = str(embedding.get("api_key") or "").strip()
    return {
        "slot": "memory",
        "kind": "embedding",
        "label": "记忆嵌入模型",
        "model": model,
        "provider": _provider_from_url(base_url),
        "base_url": base_url,
        "api_key_configured": _has_effective_api_key(
            explicit_key=explicit_key,
            runtime_embedding=runtime_embedding,
            runtime_config=runtime_config,
        ),
        "engine": str(
            memory.get("engine")
            or getattr(runtime_memory, "engine", "")
            or DEFAULT_MEMORY_ENGINE
        ),
        "output_dimensionality": dimension,
        "usage": "Akasha 的向量写入与语义召回",
        "hot_reload": True,
    }


def save_memory_model_settings(
    config_data: dict[str, Any],
    payload: ModelUpdatePayload,
) -> dict[str, Any]:
    """Persist embedding credentials while keeping local memory services on."""

    base_url = str(payload.base_url or "").strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="嵌入服务地址必须是有效的 HTTP(S) URL。",
        )
    provider = str(payload.provider or _provider_from_url(base_url)).strip()
    if provider == "dashscope" and "dashscope.aliyuncs.com" not in parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="DashScope 预设需要使用阿里云百炼服务地址。",
        )
    dimension = payload.output_dimensionality or DEFAULT_EMBEDDING_DIMENSION
    memory = cast(dict[str, Any], config_data.setdefault("memory", {}))
    memory["enabled"] = True
    memory["engine"] = DEFAULT_MEMORY_ENGINE
    embedding = cast(dict[str, Any], memory.setdefault("embedding", {}))
    embedding["model"] = payload.model.strip()
    embedding["base_url"] = base_url
    embedding["output_dimensionality"] = dimension
    if payload.api_key and payload.api_key.strip():
        embedding["api_key"] = payload.api_key.strip()
    return config_data


def _has_effective_api_key(
    *,
    explicit_key: str,
    runtime_embedding: Any,
    runtime_config: Any,
) -> bool:
    if explicit_key:
        return True
    if str(getattr(runtime_embedding, "api_key", "") or "").strip():
        return True
    if str(getattr(runtime_config, "light_api_key", "") or "").strip():
        return True
    return bool(str(getattr(runtime_config, "api_key", "") or "").strip())


def _provider_from_url(base_url: str) -> str:
    return "dashscope" if "dashscope.aliyuncs.com" in base_url else "custom"


def _mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}
