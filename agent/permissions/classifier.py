from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.permissions.models import PermissionClassification

_NETWORK_TOOLS = {"web_search", "web_fetch"}
_MESSAGE_TOOLS = {"message_push"}
_DESTRUCTIVE_TOOLS = {"forget_memory", "process_stop"}
_FILE_WRITE_TOOLS = {"write_file", "edit_file"}

_DELETE_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:rm|del|erase|rmdir|rd)\b|"
    r"\bremove-item\b|\b(?:os\.remove|os\.unlink|shutil\.rmtree|path\.unlink)\b|"
    r"\bgit\s+(?:reset\s+--hard|clean\b)|\b(?:format(?:\.com)?|diskpart)\b",
    re.IGNORECASE,
)
_PROCESS_STOP_COMMAND = re.compile(
    r"\b(?:taskkill|stop-process|kill|pkill|killall)\b", re.IGNORECASE
)
_INSTALL_COMMAND = re.compile(
    r"\b(?:pip|pip3|uv|npm|pnpm|yarn|winget|choco|scoop|apt|apt-get|brew)\s+"
    r"(?:install|add|remove|uninstall|upgrade|update)\b",
    re.IGNORECASE,
)
_NETWORK_COMMAND = re.compile(
    r"\b(?:curl|wget|httpie|xh|invoke-webrequest|invoke-restmethod|iwr|irm)\b",
    re.IGNORECASE,
)
_MUTATION_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:mv|move|cp|copy|mkdir|md|touch|ren|rename)\b|"
    r"\b(?:set-content|add-content|new-item|copy-item|move-item|rename-item)\b|"
    r"\bgit\s+(?:add|commit|push|merge|rebase|checkout|switch|restore)\b|"
    r"(?:^|\s)(?:>|>>)(?:\s|$)",
    re.IGNORECASE,
)
_READ_ONLY_COMMAND = re.compile(
    r"^\s*(?:pwd|ls|dir|rg|grep|find|where|which|whoami|hostname|"
    r"get-childitem|get-content|select-string|"
    r"git\s+(?:status|diff|log|show|branch)|"
    r"python\s+--version|node\s+--version)(?:\s|$)",
    re.IGNORECASE,
)


def _preview(value: object, *, limit: int = 280) -> str:
    text = " ".join(str(value or "").split())
    return f"{text[:limit].rstrip()}…" if len(text) > limit else text


class PermissionClassifier:
    """Turn tool metadata and concrete arguments into a user-facing risk decision."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.expanduser().resolve(strict=False)

    def classify(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        declared_risk: str,
    ) -> PermissionClassification:
        name = tool_name.strip().lower()
        if name == "shell":
            return self._classify_shell(arguments)
        if name in _NETWORK_TOOLS:
            return PermissionClassification(
                category="network",
                risk="medium",
                title="允许访问互联网？",
                description="小满将使用互联网获取这次任务所需的信息。",
                preview=_preview(arguments.get("url") or arguments.get("query")),
            )
        if name in _MESSAGE_TOOLS:
            return PermissionClassification(
                category="external_message",
                risk="high",
                title="允许发送外部消息？",
                description="这项操作会代表你向外部会话发送内容。",
                preview=_preview(arguments.get("message") or arguments.get("content")),
            )
        if name in _FILE_WRITE_TOOLS:
            return self._classify_file_write(name, arguments)
        if name in _DESTRUCTIVE_TOOLS or self._looks_destructive(name):
            return PermissionClassification(
                category="destructive",
                risk="high",
                title="允许执行删除操作？",
                description="这项操作可能删除数据或停止正在运行的任务。",
                preview=_preview(arguments),
            )
        if self._looks_install_or_configuration(name):
            return PermissionClassification(
                category="system_change",
                risk="high",
                title="允许更改系统配置？",
                description="这项操作会安装、卸载或修改连接与系统配置。",
                preview=_preview(arguments),
            )
        if declared_risk == "external-side-effect":
            return PermissionClassification(
                category="external_action",
                risk="high",
                title="允许执行外部操作？",
                description="这项工具调用会改变对话之外的状态。",
                preview=_preview(arguments),
            )
        if declared_risk not in {"", "read-only"}:
            return PermissionClassification(
                category="write",
                risk="medium",
                title="允许保存更改？",
                description="这项操作会写入或更新数据。",
                preview=_preview(arguments),
            )
        return PermissionClassification(
            category="local_read",
            risk="low",
            title="读取信息",
            description="只读取当前任务所需的信息，不会产生更改。",
            preview=_preview(arguments),
        )

    def _classify_file_write(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PermissionClassification:
        raw_path = str(arguments.get("path") or "").strip()
        path = Path(raw_path).expanduser()
        resolved = (
            path.resolve(strict=False)
            if path.is_absolute()
            else (self._workspace / path).resolve(strict=False)
        )
        external = not resolved.is_relative_to(self._workspace)
        action = "编辑" if tool_name == "edit_file" else "写入"
        return PermissionClassification(
            category="external_file_write" if external else "workspace_write",
            risk="high" if external else "medium",
            title=f"允许{action}{'外部' if external else '项目'}文件？",
            description=(
                "目标位于项目工作区之外，请确认路径后再继续。"
                if external
                else "小满将修改当前项目工作区中的文件。"
            ),
            preview=str(resolved),
        )

    def _classify_shell(
        self,
        arguments: dict[str, Any],
    ) -> PermissionClassification:
        command = str(arguments.get("command") or "").strip()
        description = str(arguments.get("description") or "").strip()
        preview = _preview(command)
        if _DELETE_COMMAND.search(command):
            return PermissionClassification(
                category="delete",
                risk="high",
                title="允许删除文件或目录？",
                description=description or "命令包含删除操作，执行后可能无法恢复。",
                preview=preview,
            )
        if _PROCESS_STOP_COMMAND.search(command):
            return PermissionClassification(
                category="process_control",
                risk="high",
                title="允许停止进程？",
                description=description or "命令会终止一个或多个正在运行的进程。",
                preview=preview,
            )
        if _INSTALL_COMMAND.search(command):
            return PermissionClassification(
                category="installation",
                risk="high",
                title="允许安装或卸载软件？",
                description=description or "命令会改变当前环境中的软件或依赖。",
                preview=preview,
            )
        if _MUTATION_COMMAND.search(command):
            return PermissionClassification(
                category="shell_write",
                risk="high",
                title="允许执行更改命令？",
                description=description or "命令会修改文件、版本库或系统状态。",
                preview=preview,
            )
        if _NETWORK_COMMAND.search(command):
            return PermissionClassification(
                category="network",
                risk="medium",
                title="允许通过命令访问互联网？",
                description=description or "命令会连接外部网络。",
                preview=preview,
            )
        if _READ_ONLY_COMMAND.search(command) and not re.search(r"[;&|]", command):
            return PermissionClassification(
                category="local_read",
                risk="low",
                title="运行只读命令",
                description=description or "命令只读取本机信息。",
                preview=preview,
            )
        return PermissionClassification(
            category="shell_unknown",
            risk="high",
            title="允许运行命令？",
            description=description or "无法确认这条命令只会读取信息，请检查后再继续。",
            preview=preview,
        )

    @staticmethod
    def _looks_destructive(name: str) -> bool:
        return any(token in name for token in ("delete", "remove", "forget", "stop"))

    @staticmethod
    def _looks_install_or_configuration(name: str) -> bool:
        return any(
            token in name
            for token in ("install", "uninstall", "mcp_add", "mcp_remove", "config")
        )
