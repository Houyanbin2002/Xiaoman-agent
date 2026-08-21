from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .attachments import AttachmentError, DashboardAttachment


_TEXT_EXTENSIONS = frozenset(
    {".csv", ".htm", ".html", ".json", ".md", ".tsv", ".txt", ".xml"}
)
_IMAGE_EXTENSIONS = frozenset(
    {".gif", ".jpeg", ".jpg", ".png", ".webp"}
)
_DOCUMENT_EXTENSIONS = frozenset(
    {".docx", ".epub", ".pdf", ".pptx", ".xls", ".xlsx"}
)
_MARKITDOWN_TOOL = "mcp_markitdown__convert_to_markdown"
_MAX_PARSED_CHARACTERS = 16 * 1024 * 1024


class DashboardDocumentParser:
    """Prepare uploads for deterministic consumption by the Agent.

    Text and images are already readable by the regular context pipeline.
    Binary office documents are converted once through the installed
    MarkItDown MCP and represented by a colocated Markdown sidecar.
    """

    def __init__(self, tools: Any) -> None:
        self._tools = tools

    async def prepare(self, attachment: DashboardAttachment) -> Path:
        extension = attachment.path.suffix.lower()
        if extension in _TEXT_EXTENSIONS or extension in _IMAGE_EXTENSIONS:
            return attachment.path.resolve()
        if extension not in _DOCUMENT_EXTENSIONS:
            raise AttachmentError(f"暂不支持 {extension or '无扩展名'} 文件")

        getter = getattr(self._tools, "get_tool", None)
        tool = getter(_MARKITDOWN_TOOL) if callable(getter) else None
        if tool is None:
            raise AttachmentError(
                "文档解析服务未连接，请先在 MCP 工具中安装并启用“文档解析”",
                status_code=503,
            )

        try:
            converted = await tool.execute(uri=attachment.path.resolve().as_uri())
        except Exception as exc:
            raise AttachmentError(
                f"{attachment.name} 解析失败：{exc}",
                status_code=422,
            ) from exc

        markdown = str(converted or "").strip()
        if not markdown or markdown.startswith("MCP error"):
            detail = markdown.removeprefix("MCP error").strip(" :()")
            raise AttachmentError(
                f"{attachment.name} 解析失败{f'：{detail}' if detail else ''}",
                status_code=422,
            )
        if len(markdown) > _MAX_PARSED_CHARACTERS:
            markdown = (
                markdown[:_MAX_PARSED_CHARACTERS]
                + "\n\n[文档内容过长，已在安全上限处截断]"
            )

        parsed_path = attachment.path.with_name(f"{attachment.path.name}.md")
        await asyncio.to_thread(parsed_path.write_text, markdown + "\n", encoding="utf-8")
        return parsed_path.resolve()
