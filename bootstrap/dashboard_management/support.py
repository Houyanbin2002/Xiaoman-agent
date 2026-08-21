from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import toml
from fastapi import HTTPException

from agent.skills import SkillsLoader
from core.workflow.models import StepStatus

from .contracts import DashboardRuntimeServices


def require_workflow_runtime(services: DashboardRuntimeServices) -> Any:
    if services.workflow_runtime is None:
        raise HTTPException(status_code=503, detail="Task Runtime 未启用")
    return services.workflow_runtime


def require_personal_data(services: DashboardRuntimeServices) -> Any:
    if services.personal_data is None:
        raise HTTPException(status_code=503, detail="个人数据服务未启用")
    return services.personal_data


def require_memory_governance(services: DashboardRuntimeServices) -> Any:
    if services.memory_governance is None:
        raise HTTPException(status_code=503, detail="记忆治理服务未启用")
    return services.memory_governance


def require_personal_rhythm(services: DashboardRuntimeServices) -> Any:
    if services.personal_rhythm is None:
        raise HTTPException(status_code=503, detail="个人节奏服务未启用")
    return services.personal_rhythm


def require_attention_runtime(services: DashboardRuntimeServices) -> Any:
    if services.attention_runtime is None:
        raise HTTPException(status_code=503, detail="注意力与行动引擎未启用")
    return services.attention_runtime


def model_row(
    slot: str,
    label: str,
    model: str,
    provider: str,
    base_url: str | None,
    api_key: str,
    usage: str,
) -> dict[str, Any]:
    return {
        "slot": slot,
        "kind": "chat",
        "label": label,
        "model": model,
        "provider": provider,
        "base_url": base_url or "",
        "api_key_configured": bool(api_key),
        "usage": usage,
        "hot_reload": True,
    }


def channel_rows(
    config: Any, plugin_manager: Any | None = None
) -> list[dict[str, Any]]:
    telegram = config.channels.telegram
    plugins = getattr(config, "plugins", {})
    qqbot = plugins.get("qqbot", {})
    weixin = plugins.get("weixin", {})
    wecom = plugins.get("wecom", {})
    runtime_channels = {
        str(getattr(channel, "name", "")): channel
        for channel in (plugin_manager.channels if plugin_manager is not None else [])
    }

    def runtime_status(channel_id: str) -> tuple[bool, str]:
        channel = runtime_channels.get(channel_id)
        if channel is None:
            return False, ""
        snapshot = getattr(channel, "status_snapshot", None)
        if not callable(snapshot):
            return True, ""
        status = snapshot()
        return bool(status.get("connected")), str(status.get("detail") or "")

    qq_connected, qq_runtime_detail = runtime_status("qqbot")
    weixin_connected, weixin_runtime_detail = runtime_status("weixin")
    wecom_connected, wecom_runtime_detail = runtime_status("wecom")
    qq_configured = bool(str(qqbot.get("app_id") or "").strip()) and bool(
        str(qqbot.get("client_secret") or "").strip()
    )
    wecom_configured = bool(str(wecom.get("bot_id") or "").strip()) and bool(
        str(wecom.get("secret") or "").strip()
    )
    weixin_configured = bool(str(weixin.get("account_id") or "").strip())
    return [
        {
            "id": "dashboard",
            "label": "Web 控制台",
            "configured": True,
            "connected": True,
            "detail": "当前浏览器实时聊天",
            "allow_from": [],
            "kind": "local",
        },
        {
            "id": "ipc",
            "label": "本地 IPC",
            "configured": True,
            "connected": True,
            "detail": config.channels.socket,
            "allow_from": [],
            "kind": "local",
        },
        {
            "id": "telegram",
            "label": "Telegram",
            "configured": telegram is not None and bool(telegram.token),
            "connected": telegram is not None and bool(telegram.token),
            "detail": telegram.channel_name if telegram else "尚未配置",
            "allow_from": telegram.allow_from if telegram else [],
            "kind": "gateway",
            "fields": {
                "token": {"value": "", "configured": bool(telegram and telegram.token)},
            },
            "docs_url": "https://core.telegram.org/bots/tutorial",
        },
        {
            "id": "qqbot",
            "label": "QQ 官方机器人",
            "configured": qq_configured,
            "connected": qq_connected,
            "detail": qq_runtime_detail
            or (
                "保存后重启即可连接"
                if qq_configured
                else "使用 QQ 开放平台 AppID / AppSecret"
            ),
            "allow_from": _string_items(qqbot.get("allow_from")),
            "kind": "gateway",
            "fields": {
                "app_id": {
                    "value": str(qqbot.get("app_id") or ""),
                    "configured": bool(qqbot.get("app_id")),
                },
                "client_secret": {
                    "value": "",
                    "configured": bool(qqbot.get("client_secret")),
                },
            },
            "docs_url": "https://bot.q.qq.com/wiki/develop/api-v2/",
            "docs_label": "官方文档",
        },
        {
            "id": "weixin",
            "label": "微信（个人）",
            "configured": weixin_configured,
            "connected": weixin_connected,
            "detail": weixin_runtime_detail
            or (
                "保存后重启即可连接"
                if weixin_configured
                else "扫码创建腾讯 iLink 机器人身份"
            ),
            "allow_from": _string_items(weixin.get("allow_from")),
            "kind": "gateway",
            "fields": {
                "account_id": {
                    "value": str(weixin.get("account_id") or ""),
                    "configured": weixin_configured,
                },
            },
            "docs_url": "https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/messaging/weixin.md",
            "docs_label": "Hermes 接入说明",
        },
        {
            "id": "wecom",
            "label": "企业微信智能机器人",
            "configured": wecom_configured,
            "connected": wecom_connected,
            "detail": wecom_runtime_detail
            or (
                "保存后重启即可连接"
                if wecom_configured
                else "官方长连接，无需公网回调地址"
            ),
            "allow_from": _string_items(wecom.get("allow_from")),
            "kind": "gateway",
            "fields": {
                "bot_id": {
                    "value": str(wecom.get("bot_id") or ""),
                    "configured": bool(wecom.get("bot_id")),
                },
                "secret": {"value": "", "configured": bool(wecom.get("secret"))},
            },
            "docs_url": "https://github.com/WecomTeam/aibot-node-sdk",
            "docs_label": "官方 SDK",
        },
    ]


def skill_rows(services: DashboardRuntimeServices) -> list[dict[str, Any]]:
    records = SkillsLoader(services.workspace).list_skill_records(
        filter_unavailable=False
    )
    rows: list[dict[str, Any]] = []
    for item in records:
        origin = (
            "standalone"
            if item.source == "installed"
            else "system" if item.source == "plugin" else item.source
        )
        source_label = {
            "installed": "独立安装",
            "builtin": "内置",
            "workspace": "工作区",
            "plugin": "系统组件",
        }.get(item.source, item.source)
        rows.append(
            {
                "name": item.name,
                "display_name": item.display_name,
                "source": item.source,
                "source_id": item.source_id,
                "origin": origin,
                "source_label": source_label,
                "provider_id": item.name if item.source == "installed" else "",
                "provider_name": "",
                "can_uninstall": item.source == "installed",
                "description": item.description,
                "when_to_use": item.when_to_use,
                "always": item.always,
                "available": item.available,
                "missing": item.missing,
            }
        )
    return rows


def _string_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in cast(list[object], value) if isinstance(item, str)]


def schedule_row(job: Any) -> dict[str, Any]:
    row = asdict(job)
    for key, value in list(row.items()):
        if isinstance(value, datetime):
            row[key] = value.isoformat()
    return row


def workflow_rows(services: DashboardRuntimeServices) -> list[dict[str, Any]]:
    if services.workflow_runtime is None:
        return []
    result: list[dict[str, Any]] = []
    for workflow in services.workflow_runtime.store.list_workflows(limit=100):
        succeeded = sum(step.status == StepStatus.SUCCEEDED for step in workflow.steps)
        result.append(
            {
                **workflow.to_dict(include_steps=False),
                "short_id": workflow.id[:8],
                "step_count": len(workflow.steps),
                "completed_steps": succeeded,
                "waiting_steps": [
                    step.id
                    for step in workflow.steps
                    if step.status == StepStatus.WAITING
                ],
                "waiting_actions": [
                    {
                        "id": step.id,
                        "title": step.title,
                        "description": step.description,
                        "kind": step.kind.value,
                    }
                    for step in workflow.steps
                    if step.status == StepStatus.WAITING
                ],
                "failed_steps": [
                    step.id
                    for step in workflow.steps
                    if step.status == StepStatus.FAILED
                ],
                "failed_actions": [
                    {
                        "id": step.id,
                        "title": step.title,
                        "description": step.description,
                        "error": step.error,
                    }
                    for step in workflow.steps
                    if step.status == StepStatus.FAILED
                ],
            }
        )
    return result


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail="config.toml 不存在")
    try:
        return cast(dict[str, Any], toml.load(path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"配置读取失败：{exc}") from exc


def save_config(path: Path, data: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".dashboard.tmp")
    try:
        temp_path.write_text(toml.dumps(data), encoding="utf-8")
        temp_path.replace(path)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"配置保存失败：{exc}") from exc
