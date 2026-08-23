from __future__ import annotations

import base64
import asyncio
import io
import secrets
import time
from typing import Any, cast
from urllib.parse import quote, urlsplit

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from agent.marketplace import MarketplaceInstaller, MarketplaceService
from agent.marketplace.curated_mcp_provider import CuratedMcpProvider
from agent.marketplace.mcp_registry_provider import McpRegistryProvider
from agent.marketplace.service import CombinedMarketplaceProvider
from agent.marketplace.skills_provider import SkillsCliProvider
from agent.mcp.catalog import (
    install_catalog_server,
    list_catalog,
    uninstall_catalog_runtime,
)
from agent.skill_packages import install_skill_from_git, uninstall_skill
from agent.skills import SkillsLoader
from agent.tools.schedule import ScheduleTool
from core.net.http import RequestBudget, get_default_http_requester
from infra.security import LocalSecretStore
from agent.config_models import TelegramChannelConfig

from ..contracts import DashboardRuntimeServices
from ..schemas import (
    ChannelUpdatePayload,
    ConversationStyleUpdatePayload,
    MarketplaceInstallPayload,
    McpCreatePayload,
    ScheduleCreatePayload,
    SkillInstallPayload,
)

_WEIXIN_BASE_URL = "https://ilinkai.weixin.qq.com"
_WEIXIN_CLIENT_VERSION = str((2 << 16) | (2 << 8))
_weixin_qr_flows: dict[str, dict[str, Any]] = {}
from ..support import (
    channel_rows,
    load_config,
    save_config,
    schedule_row,
    skill_rows,
    workflow_rows,
)


def default_marketplace_service(
    services: DashboardRuntimeServices,
) -> MarketplaceService:
    return MarketplaceService(
        skill_provider=SkillsCliProvider(),
        mcp_provider=CombinedMarketplaceProvider(
            CuratedMcpProvider(),
            McpRegistryProvider(),
        ),
        installed_skills=lambda: {
            str(row["name"]) for row in skill_rows(services)
        },
        installed_mcp=lambda: {
            str(row["name"]) for row in services.mcp_registry.snapshot()
        },
    )


def register_system_routes(
    app: FastAPI,
    services: DashboardRuntimeServices,
) -> None:
    marketplace = default_marketplace_service(services)
    marketplace_installer = MarketplaceInstaller(marketplace, services.mcp_registry)

    @app.get("/api/dashboard/control/gateway/status")
    def gateway_status() -> dict[str, Any]:
        return {
            "status": "running",
            "instance_id": services.gateway_instance_id,
            "restart_supported": services.gateway_restart is not None,
        }

    @app.post("/api/dashboard/control/gateway/restart", status_code=202)
    async def restart_gateway(background_tasks: BackgroundTasks) -> dict[str, Any]:
        if services.gateway_restart is None:
            raise HTTPException(
                status_code=409,
                detail="当前运行方式不支持从界面重启网关",
            )

        async def request_after_response() -> None:
            await asyncio.sleep(0.15)
            services.gateway_restart()

        background_tasks.add_task(request_after_response)
        return {
            "accepted": True,
            "instance_id": services.gateway_instance_id,
        }

    @app.get("/api/dashboard/control/overview")
    def control_overview() -> dict[str, Any]:
        configured_engine = services.config.memory.engine or "akasha"
        runtime_description = services.memory_admin.describe()
        runtime_engine = runtime_description.name
        runtime_matches_config = (
            runtime_engine == configured_engine
            or str(runtime_description.notes.get("configured_as") or "")
            == configured_engine
        )
        memory_status = (
            "healthy"
            if services.config.memory.enabled and runtime_matches_config
            else "pending_restart"
            if services.config.memory.enabled
            else "disabled"
        )
        workflows = workflow_rows(services)
        schedules = services.scheduler.list_jobs()
        skills = SkillsLoader(services.workspace).list_skill_records(
            filter_unavailable=False
        )
        tool_names = services.tools.get_registered_names()
        mcp_rows = services.mcp_registry.snapshot()
        channels = channel_rows(services.config, services.plugin_manager)
        personal_records = (
            services.personal_data.list(limit=1000)
            if services.personal_data is not None
            else []
        )
        return {
            "status": "online",
            "assistant": "小满",
            "model": services.config.model,
            "provider": services.config.provider,
            "memory_enabled": bool(services.config.memory.enabled),
            "memory_engine": configured_engine,
            "memory_status": memory_status,
            "counts": {
                "tools": len(tool_names),
                "mcp_servers": len(mcp_rows),
                "skills": len(skills),
                "channels": len(channels),
                "workflows_active": sum(
                    row["status"] in {"running", "waiting", "blocked"}
                    for row in workflows
                ),
                "tasks_active": sum(
                    row["status"] in {"running", "waiting", "blocked"}
                    for row in workflows
                ),
                "schedules": len(schedules),
                "personal_records": len(personal_records),
            },
            "channels": channels,
        }

    @app.get("/api/dashboard/control/channels")
    def list_channels() -> list[dict[str, Any]]:
        return channel_rows(services.config, services.plugin_manager)

    @app.patch("/api/dashboard/control/channels/{channel}")
    def update_channel(channel: str, payload: ChannelUpdatePayload) -> dict[str, Any]:
        if channel not in {"telegram", "qqbot", "weixin", "wecom"}:
            raise HTTPException(status_code=404, detail="该频道暂不支持配置写入")
        config_data = load_config(services.config_path)
        section_name = "channels" if channel == "telegram" else "plugins"
        section = cast(dict[str, Any], config_data.setdefault(section_name, {}))
        target = cast(dict[str, Any], section.setdefault(channel, {}))
        target["enabled"] = payload.enabled
        target["allow_from"] = [
            item.strip() for item in payload.allow_from if item.strip()
        ]
        if channel == "telegram" and payload.token:
            target["token"] = payload.token.strip()
        if channel == "qqbot":
            if payload.app_id is not None:
                target["app_id"] = payload.app_id.strip()
            if payload.client_secret:
                target["client_secret"] = payload.client_secret.strip()
        if channel == "wecom":
            if payload.bot_id is not None:
                target["bot_id"] = payload.bot_id.strip()
            if payload.secret:
                target["secret"] = payload.secret.strip()
        if payload.enabled:
            required_fields = {
                "telegram": (("token", "Bot Token"),),
                "qqbot": (("app_id", "AppID"), ("client_secret", "AppSecret")),
                "wecom": (("bot_id", "Bot ID"), ("secret", "Secret")),
            }.get(channel, ())
            missing = [
                label
                for field_name, label in required_fields
                if not str(target.get(field_name) or "").strip()
            ]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"请先填写完整的 {channel} 配置：{'、'.join(missing)}",
                )
        channels = cast(dict[str, Any], config_data.get("channels") or {})
        channels.pop("qq", None)
        channels.pop("qqbot", None)
        save_config(services.config_path, config_data)
        if channel == "telegram":
            services.config.channels.telegram = (
                TelegramChannelConfig(
                    token=str(target.get("token") or ""),
                    allow_from=[str(item) for item in target.get("allow_from") or []],
                    channel_name=str(target.get("channel_name") or "telegram"),
                )
                if payload.enabled
                else None
            )
        else:
            runtime_plugins = getattr(services.config, "plugins", None)
            if not isinstance(runtime_plugins, dict):
                runtime_plugins = {}
                services.config.plugins = runtime_plugins
            runtime_plugins.setdefault(channel, {}).update(target)
        updated = next(
            row
            for row in channel_rows(services.config, services.plugin_manager)
            if row["id"] == channel
        )
        return {"saved": True, "restart_required": True, "channel": updated}

    @app.post("/api/dashboard/control/channels/weixin/qr")
    async def start_weixin_qr_login() -> dict[str, object]:
        payload = await _weixin_get(
            _WEIXIN_BASE_URL,
            "ilink/bot/get_bot_qrcode?bot_type=3",
        )
        qrcode = str(payload.get("qrcode") or "")
        scan_content = str(payload.get("qrcode_img_content") or qrcode)
        if not qrcode or not scan_content:
            raise HTTPException(status_code=502, detail="微信未返回可用二维码")
        flow_id = secrets.token_urlsafe(18)
        _weixin_qr_flows[flow_id] = {
            "qrcode": qrcode,
            "base_url": _WEIXIN_BASE_URL,
            "created_at": time.monotonic(),
        }
        return {
            "flow_id": flow_id,
            "status": "wait",
            "image": _qr_data_url(scan_content),
        }

    @app.get("/api/dashboard/control/channels/weixin/qr/{flow_id}")
    async def poll_weixin_qr_login(flow_id: str) -> dict[str, object]:
        flow = _weixin_qr_flows.get(flow_id)
        if flow is None:
            raise HTTPException(status_code=404, detail="二维码会话不存在")
        if time.monotonic() - float(flow["created_at"]) > 600:
            _weixin_qr_flows.pop(flow_id, None)
            return {"status": "expired"}
        payload = await _weixin_get(
            str(flow["base_url"]),
            f"ilink/bot/get_qrcode_status?qrcode={quote(str(flow['qrcode']), safe='')}",
        )
        status = str(payload.get("status") or "wait")
        if status == "scaned_but_redirect":
            redirected = _trusted_weixin_base(str(payload.get("redirect_host") or ""))
            if redirected:
                flow["base_url"] = redirected
            return {"status": "scaned"}
        if status != "confirmed":
            return {"status": status}
        account_id = str(payload.get("ilink_bot_id") or "").strip()
        token = str(payload.get("bot_token") or "").strip()
        owner_user_id = str(payload.get("ilink_user_id") or "").strip()
        base_url = _trusted_weixin_base(str(payload.get("baseurl") or "")) or str(
            flow["base_url"]
        )
        if not account_id or not token:
            raise HTTPException(status_code=502, detail="微信确认结果缺少账号凭据")
        await LocalSecretStore(services.workspace).set_bundle(
            "weixin",
            {"account_id": account_id, "token": token, "base_url": base_url},
        )
        config_data = load_config(services.config_path)
        plugins = cast(dict[str, Any], config_data.setdefault("plugins", {}))
        target = cast(dict[str, Any], plugins.setdefault("weixin", {}))
        target.update({"enabled": True, "account_id": account_id, "base_url": base_url})
        if owner_user_id and not target.get("allow_from"):
            target["allow_from"] = [owner_user_id]
        save_config(services.config_path, config_data)
        services.config.plugins.setdefault("weixin", {}).update(target)
        _weixin_qr_flows.pop(flow_id, None)
        return {
            "status": "confirmed",
            "account_id": account_id,
            "restart_required": True,
        }

    @app.get("/api/dashboard/control/tools")
    def list_tools() -> list[dict[str, Any]]:
        return [
            {
                "name": document.name,
                "description": document.description,
                "risk": document.risk,
                "always_on": document.always_on,
                "source_type": document.source_type,
                "source_name": document.source_name,
            }
            for document in services.tools.get_documents()
        ]

    @app.get("/api/dashboard/control/skills")
    def list_skills() -> list[dict[str, Any]]:
        return skill_rows(services)

    @app.get("/api/dashboard/control/marketplace")
    async def search_marketplace(
        kind: str = Query(pattern=r"^(skill|mcp)$"),
        query: str = Query(default="", max_length=200),
        q: str = Query(default="", max_length=200),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, object]:
        search_query = query or q
        try:
            items = await asyncio.to_thread(
                marketplace.search, kind, search_query, limit
            )
        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503, detail=f"扩展市场暂时不可用: {exc}"
            ) from exc
        return {
            "items": [item.public() for item in items],
            "kind": kind,
            "query": search_query,
        }

    @app.get("/api/dashboard/control/conversation-styles")
    def get_conversation_styles() -> dict[str, Any]:
        return services.conversation_styles.snapshot()

    @app.patch("/api/dashboard/control/conversation-styles")
    def update_conversation_style(
        payload: ConversationStyleUpdatePayload,
    ) -> dict[str, Any]:
        try:
            selected = services.conversation_styles.set_active(payload.style_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            **services.conversation_styles.snapshot(),
            "selected": selected.public(),
            "applies_from": "next_reply",
        }

    @app.get("/api/dashboard/control/marketplace/items/{item_id:path}")
    async def get_marketplace_item(
        item_id: str,
        kind: str = Query(pattern=r"^(skill|mcp)$"),
    ) -> dict[str, object]:
        if len(item_id) > 500:
            raise HTTPException(status_code=422, detail="市场条目 ID 过长")
        try:
            item = await asyncio.to_thread(marketplace.get, kind, item_id)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503, detail=f"扩展市场暂时不可用: {exc}"
            ) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="市场中不存在该扩展能力")
        return item.public()

    @app.post("/api/dashboard/control/marketplace/install")
    async def install_marketplace_item(
        payload: MarketplaceInstallPayload,
    ) -> dict[str, str]:
        try:
            result = await marketplace_installer.install(
                payload.kind,  # type: ignore[arg-type]
                payload.item_id,
                payload.configuration,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503, detail=f"安装扩展能力失败: {exc}"
            ) from exc
        return result.public()

    @app.post("/api/dashboard/control/marketplace/refresh")
    async def refresh_marketplace(
        kind: str | None = Query(default=None, pattern=r"^(skill|mcp)$"),
    ) -> dict[str, object]:
        try:
            await asyncio.to_thread(marketplace.refresh, kind)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503, detail=f"刷新扩展市场失败: {exc}"
            ) from exc
        return {"refreshed": True, "kind": kind}

    @app.post("/api/dashboard/control/skills/install")
    def install_skill(payload: SkillInstallPayload) -> dict[str, object]:
        try:
            installed = install_skill_from_git(
                source=payload.source,
                ref_name=payload.ref,
                source_subdir=payload.subdir,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"安装 Skill 失败: {exc}"
            ) from exc
        return {
            "name": installed.name,
            "source": installed.source,
            "revision": installed.revision,
            "restart_required": False,
        }

    @app.delete("/api/dashboard/control/skills/{name}")
    def remove_skill(name: str) -> dict[str, object]:
        try:
            uninstall_skill(name)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"卸载 Skill 失败: {exc}"
            ) from exc
        return {"name": name, "removed": True, "restart_required": False}

    @app.get("/api/dashboard/control/mcp")
    def list_mcp_servers() -> list[dict[str, Any]]:
        return services.mcp_registry.snapshot()

    @app.get("/api/dashboard/control/mcp/catalog")
    def get_mcp_catalog() -> list[dict[str, object]]:
        installed = {str(row["name"]) for row in services.mcp_registry.snapshot()}
        return list_catalog(installed)

    @app.post("/api/dashboard/control/mcp/catalog/{entry_id}")
    async def install_mcp_from_catalog(entry_id: str) -> dict[str, object]:
        try:
            message = await install_catalog_server(entry_id, services.mcp_registry)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"安装 MCP 失败: {exc}"
            ) from exc
        return {"message": message, "servers": services.mcp_registry.snapshot()}

    @app.post("/api/dashboard/control/mcp")
    async def add_mcp_server(payload: McpCreatePayload) -> dict[str, Any]:
        if payload.transport == "stdio":
            result = await services.mcp_registry.add(
                payload.name,
                payload.command,
                payload.env,
                payload.cwd,
            )
        else:
            result = await services.mcp_registry.add_remote(
                payload.name,
                url=payload.url,
                transport=payload.transport,
                auth_type=payload.auth_type,
                scopes=payload.scopes,
                bearer_token=payload.bearer_token,
                headers=payload.headers,
                oauth_client_id=payload.oauth_client_id,
                oauth_client_secret=payload.oauth_client_secret,
            )
        if result.startswith("连接 MCP"):
            raise HTTPException(status_code=400, detail=result)
        return {"message": result, "servers": services.mcp_registry.snapshot()}

    @app.post("/api/dashboard/control/mcp/{name}/authorize")
    async def authorize_mcp_server(name: str) -> dict[str, str]:
        try:
            authorization_url = await services.mcp_registry.begin_oauth(name)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"authorization_url": authorization_url}

    @app.get(
        "/api/dashboard/control/mcp/oauth/callback/{name}",
        response_class=HTMLResponse,
    )
    async def complete_mcp_oauth(
        name: str,
        code: str = "",
        state: str | None = None,
        error: str = "",
    ) -> HTMLResponse:
        if error or not code:
            return HTMLResponse(
                _oauth_result_page(False, error or "远程服务未返回授权码"),
                status_code=400,
            )
        try:
            services.mcp_registry.complete_oauth_callback(
                name,
                code=code,
                state=state,
            )
        except (RuntimeError, ValueError) as exc:
            return HTMLResponse(_oauth_result_page(False, str(exc)), status_code=400)
        return HTMLResponse(_oauth_result_page(True, "授权已接收，正在连接工具"))

    @app.delete("/api/dashboard/control/mcp/{name}")
    async def remove_mcp_server(name: str) -> dict[str, Any]:
        result = await services.mcp_registry.remove(name)
        if "不存在" in result:
            raise HTTPException(status_code=404, detail=result)
        return {
            "message": result,
            "runtime_removed": uninstall_catalog_runtime(name),
        }

    @app.get("/api/dashboard/control/schedules")
    def list_schedules() -> list[dict[str, Any]]:
        return [schedule_row(job) for job in services.scheduler.list_jobs()]

    @app.post("/api/dashboard/control/schedules")
    async def create_schedule(payload: ScheduleCreatePayload) -> dict[str, Any]:
        result = await ScheduleTool(
            services.scheduler, default_tz=payload.timezone
        ).execute(**payload.model_dump())
        if result.startswith("错误"):
            raise HTTPException(status_code=400, detail=result)
        return {"message": result}

    @app.patch("/api/dashboard/control/schedules/{job_id}")
    def update_schedule(job_id: str, payload: ScheduleCreatePayload) -> dict[str, Any]:
        matches = [
            job.id
            for job in services.scheduler.list_jobs()
            if job.id.startswith(job_id)
        ]
        if len(matches) != 1:
            raise HTTPException(status_code=404, detail="未找到唯一的定时任务")
        tool = ScheduleTool(services.scheduler, default_tz=payload.timezone)
        try:
            replacement = tool.build_job(**payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"错误：{exc}") from exc
        if not services.scheduler.replace_job(matches[0], replacement):
            raise HTTPException(status_code=404, detail="定时任务已不存在")
        return {"updated": True, "job": schedule_row(replacement)}

    @app.delete("/api/dashboard/control/schedules/{job_id}")
    def cancel_schedule(job_id: str) -> dict[str, Any]:
        matches = [
            job.id
            for job in services.scheduler.list_jobs()
            if job.id.startswith(job_id)
        ]
        if len(matches) != 1:
            raise HTTPException(status_code=404, detail="未找到唯一的定时任务")
        services.scheduler.cancel_job(matches[0])
        return {"cancelled": True}


async def _weixin_get(base_url: str, endpoint: str) -> dict[str, Any]:
    response = await get_default_http_requester("external_default").get(
        f"{base_url.rstrip('/')}/{endpoint}",
        headers={
            "iLink-App-Id": "bot",
            "iLink-App-ClientVersion": _WEIXIN_CLIENT_VERSION,
        },
        timeout_s=35,
        budget=RequestBudget(total_timeout_s=40),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="微信返回了无效响应")
    return cast(dict[str, Any], payload)


def _trusted_weixin_base(value: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlsplit(candidate)
    host = str(parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "ilinkai.weixin.qq.com" or host.endswith(".weixin.qq.com")
    ):
        return ""
    return f"https://{parsed.netloc}"


def _qr_data_url(content: str) -> str:
    import qrcode

    image = qrcode.make(content)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _oauth_result_page(success: bool, message: str) -> str:
    title = "授权成功" if success else "授权失败"
    tone = "#166534" if success else "#b91c1c"
    safe_message = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>{title}</title>
<body style="font-family:system-ui;padding:48px;color:#1f2937">
<h1 style="color:{tone}">{title}</h1><p>{safe_message}</p>
<p>可以关闭此页面并返回小满。</p>
<script>window.opener?.postMessage({{type:'xiaoman-mcp-oauth'}}, window.location.origin);</script>
</body></html>"""
