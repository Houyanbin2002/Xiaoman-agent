from __future__ import annotations

import ipaddress
import logging

logger = logging.getLogger(__name__)


def _is_loopback_dashboard_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_dashboard_binding(host: str, *, allow_remote: bool) -> None:
    if _is_loopback_dashboard_host(host):
        return
    if not allow_remote:
        raise ValueError(
            "Dashboard 包含未鉴权的管理接口，只允许监听回环地址；"
            "如已通过防火墙或反向代理完成隔离，可显式启用 unsafe remote bind。"
        )
    logger.warning(
        "Dashboard 正在监听非回环地址 %s；管理接口未内置鉴权，请确保网络侧已隔离",
        host,
    )
