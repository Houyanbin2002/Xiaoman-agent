from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import AsyncIterable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from uuid import uuid4


SUPPORTED_ATTACHMENT_EXTENSIONS = frozenset(
    {
        ".csv",
        ".docx",
        ".epub",
        ".gif",
        ".htm",
        ".html",
        ".jpeg",
        ".jpg",
        ".json",
        ".md",
        ".pdf",
        ".png",
        ".pptx",
        ".tsv",
        ".txt",
        ".webp",
        ".xls",
        ".xlsx",
        ".xml",
    }
)
MAX_ATTACHMENT_BYTES = 128 * 1024 * 1024
MAX_ATTACHMENTS_PER_MESSAGE = 8


class AttachmentError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class DashboardAttachment:
    id: str
    chat_id: str
    name: str
    path: Path
    size: int
    mime_type: str
    content_path: Path | None = None

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "mime_type": self.mime_type,
            "parsed": self.content_path is not None and self.content_path != self.path,
        }


class DashboardAttachmentStore:
    """Own dashboard uploads and expose only opaque IDs to the browser."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._records: dict[str, DashboardAttachment] = {}
        self._lock = RLock()

    async def save_stream(
        self,
        *,
        chat_id: str,
        filename: str,
        mime_type: str,
        chunks: AsyncIterable[bytes],
    ) -> DashboardAttachment:
        safe_name = _safe_filename(filename)
        extension = Path(safe_name).suffix.lower()
        if extension not in SUPPORTED_ATTACHMENT_EXTENSIONS:
            raise AttachmentError(
                f"暂不支持 {extension or '无扩展名'} 文件",
            )

        attachment_id = uuid4().hex
        chat_dir = self.root / _chat_directory(chat_id) / attachment_id
        chat_dir.mkdir(parents=True, exist_ok=False)
        temporary = chat_dir / f".{safe_name}.part"
        final_path = chat_dir / safe_name
        size = 0
        try:
            with temporary.open("xb") as target:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_ATTACHMENT_BYTES:
                        raise AttachmentError(
                            "文件超过 128 MB 上限",
                            status_code=413,
                        )
                    target.write(chunk)
            if size == 0:
                raise AttachmentError("不能上传空文件")
            temporary.replace(final_path)
        except Exception:
            shutil.rmtree(chat_dir, ignore_errors=True)
            raise

        record = DashboardAttachment(
            id=attachment_id,
            chat_id=chat_id,
            name=safe_name,
            path=final_path,
            size=size,
            mime_type=(mime_type or "application/octet-stream").strip(),
        )
        with self._lock:
            self._records[attachment_id] = record
        return record

    def set_content_path(
        self,
        chat_id: str,
        attachment_id: str,
        content_path: Path,
    ) -> DashboardAttachment:
        """Bind the readable representation produced by the document parser."""

        resolved = content_path.resolve()
        with self._lock:
            record = self._records.get(attachment_id)
            if record is None or record.chat_id != chat_id:
                raise AttachmentError("附件已失效，请重新添加")
            if not resolved.is_file() or resolved.parent != record.path.parent.resolve():
                raise AttachmentError("文档解析结果无效")
            updated = replace(record, content_path=resolved)
            self._records[attachment_id] = updated
            return updated

    def resolve_many(
        self,
        chat_id: str,
        attachment_ids: list[str],
    ) -> list[DashboardAttachment]:
        ids = list(dict.fromkeys(value.strip() for value in attachment_ids if value.strip()))
        if len(ids) > MAX_ATTACHMENTS_PER_MESSAGE:
            raise AttachmentError(f"每条消息最多添加 {MAX_ATTACHMENTS_PER_MESSAGE} 个文件")
        records: list[DashboardAttachment] = []
        with self._lock:
            for attachment_id in ids:
                record = self._records.get(attachment_id)
                readable_path = record.content_path if record is not None else None
                if (
                    record is None
                    or record.chat_id != chat_id
                    or not record.path.is_file()
                    or (readable_path is not None and not readable_path.is_file())
                ):
                    raise AttachmentError("附件已失效，请重新添加")
                records.append(record)
        return records

    def remove(self, chat_id: str, attachment_id: str) -> bool:
        with self._lock:
            record = self._records.get(attachment_id)
            if record is None or record.chat_id != chat_id:
                return False
            self._records.pop(attachment_id, None)
        shutil.rmtree(record.path.parent, ignore_errors=True)
        return True


def _safe_filename(filename: str) -> str:
    name = Path(str(filename or "").replace("\\", "/")).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(". ")
    if not name or name in {".", ".."}:
        raise AttachmentError("文件名无效")
    if len(name) > 180:
        suffix = Path(name).suffix
        name = f"{Path(name).stem[: max(1, 180 - len(suffix))]}{suffix}"
    return name


def _chat_directory(chat_id: str) -> str:
    return hashlib.sha256(chat_id.encode("utf-8")).hexdigest()[:24]
