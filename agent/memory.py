import logging
import re
import sqlite3
import threading
from pathlib import Path

from utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

_CONSOLIDATION_MARKER_PREFIX = "<!-- consolidation:"
_CONSOLIDATION_MARKER_SUFFIX = " -->"
_CONSOLIDATION_TAIL_BYTES = 1024 * 1024
_JOURNAL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEFAULT_SELF_MD = """# Xiaoman 的自我认知

## 人格与形象
- 我是 Xiaoman，一个直接、温暖、主动参与思考的长期协作伙伴。
- 我优先给出结论，再补充必要细节；不把自己伪装成没有立场的工具。

## 我对当前用户的理解
- 我会从长期记忆中逐步形成对当前用户的理解，不在缺少证据时编造画像。

## 我们关系的定义
- 我与当前用户的关系以透明、尊重边界和持续协作为基础。
"""


class MemoryStore:
    """Markdown support files around the canonical governed memory store.

    - SELF.md     : Xiaoman self-model and relationship definition
    - HISTORY.md  : grep-searchable event log, permanent append
    - RECENT_CONTEXT.md : compacted recent context snapshot for proactive/drift
    - journal/    : per-day event timeline, append-only YYYY-MM-DD.md
    """

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.journal_dir = ensure_dir(self.memory_dir / "journal")
        self.history_file = self.memory_dir / "HISTORY.md"
        self.recent_context_file = self.memory_dir / "RECENT_CONTEXT.md"
        self.self_file = self.memory_dir / "SELF.md"
        self._consolidation_db = self.memory_dir / "consolidation_writes.db"
        self._consolidation_lock = threading.Lock()
        self._init_consolidation_db()

    def append_history_once(
        self,
        entry: str,
        *,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool:
        """按 source_ref 幂等追加 HISTORY，避免重启后重复 consolidation。"""
        text = (entry or "").strip()
        if not text:
            return False
        return self._append_once_with_index(
            target_file=self.history_file,
            text=text,
            source_ref=source_ref,
            kind=kind,
            trailing_blank_line=True,
        )

    def read_history(self, max_chars: int = 0) -> str:
        """读取 HISTORY.md，并过滤 consolidation 标记行。"""
        if not self.history_file.exists():
            return ""
        text = self.history_file.read_text(encoding="utf-8")
        text = self._strip_consolidation_markers(text)
        if max_chars > 0 and len(text) > max_chars:
            return text[-max_chars:]
        return text

    # ── journal/ (per-day event timeline) ───────────────────────────

    def append_journal(
        self,
        date_str: str,
        entry: str,
        *,
        source_ref: str = "",
        kind: str = "journal",
    ) -> bool:
        date_str = date_str.strip()
        text = (entry or "").strip()
        if not _JOURNAL_DATE_RE.fullmatch(date_str) or not text:
            return False
        journal_file = self.journal_dir / f"{date_str}.md"
        if not journal_file.exists():
            journal_file.write_text(f"# {date_str}\n\n", encoding="utf-8")
        if source_ref:
            return self._append_once_with_index(
                target_file=journal_file,
                text=text,
                source_ref=source_ref,
                kind=kind,
                trailing_blank_line=True,
            )
        with open(journal_file, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")
        return True

    # ── RECENT_CONTEXT.md (compacted recent context) ──────────────

    def read_recent_context(self) -> str:
        if self.recent_context_file.exists():
            return self.recent_context_file.read_text(encoding="utf-8")
        return ""

    def write_recent_context(self, content: str) -> None:
        self.recent_context_file.write_text(content, encoding="utf-8")

    def build_recent_activity_context(
        self,
        *,
        max_entries: int = 18,
        max_chars: int = 4500,
    ) -> str:
        """Render a bounded, deduplicated cross-session activity digest.

        HISTORY remains the append-only audit log. RECENT_CONTEXT is only an
        awareness view and must never be treated as a source of stable user
        preferences or execution rules.
        """

        lines = [
            line.strip()
            for line in self.read_history().splitlines()
            if line.strip().startswith("- ")
        ]
        selected: list[str] = []
        seen: set[str] = set()
        used = 0
        for line in reversed(lines):
            normalized = re.sub(r"^[-\s]*(?:\[[^]]+\]\s*)+", "", line).strip().lower()
            if not normalized or normalized in seen:
                continue
            if used + len(line) > max(500, int(max_chars)):
                continue
            seen.add(normalized)
            selected.append(line)
            used += len(line)
            if len(selected) >= max(1, int(max_entries)):
                break
        if not selected:
            return ""
        selected.reverse()
        return (
            "# 近期活动摘要\n\n"
            "> 这是跨会话的近期动态，只用于了解用户最近在忙什么；"
            "不得据此推断长期偏好或覆盖用户当前指令。\n\n" + "\n".join(selected) + "\n"
        )

    # ── SELF.md (Xiaoman self-model) ──────────────────────────────

    def read_self(self) -> str:
        if self.self_file.exists():
            return self.self_file.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _consolidation_marker(source_ref: str, kind: str) -> str:
        src = (source_ref or "").replace("\n", " ").strip()
        kd = (kind or "").replace("\n", " ").strip()
        return f"{_CONSOLIDATION_MARKER_PREFIX}{src}:{kd}{_CONSOLIDATION_MARKER_SUFFIX}"

    @staticmethod
    def _strip_consolidation_markers(text: str) -> str:
        lines = text.splitlines()
        kept = [
            line
            for line in lines
            if not (
                line.startswith(_CONSOLIDATION_MARKER_PREFIX)
                and line.endswith(_CONSOLIDATION_MARKER_SUFFIX)
            )
        ]
        return "\n".join(kept).strip()

    def _init_consolidation_db(self) -> None:
        conn = sqlite3.connect(str(self._consolidation_db))
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_writes (
                    source_ref TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT,
                    trailing_blank_line INTEGER NOT NULL DEFAULT 0,
                    done_at TEXT NOT NULL,
                    PRIMARY KEY (source_ref, kind)
                )""")
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(consolidation_writes)"
                ).fetchall()
            }
            if "payload" not in cols:
                conn.execute("ALTER TABLE consolidation_writes ADD COLUMN payload TEXT")
            if "trailing_blank_line" not in cols:
                conn.execute(
                    "ALTER TABLE consolidation_writes ADD COLUMN trailing_blank_line INTEGER NOT NULL DEFAULT 0"
                )
            conn.commit()
        finally:
            conn.close()

    def _append_once_with_index(
        self,
        *,
        target_file: Path,
        text: str,
        source_ref: str,
        kind: str,
        trailing_blank_line: bool,
    ) -> bool:
        marker = self._consolidation_marker(source_ref, kind)
        src = (source_ref or "").strip()
        kd = (kind or "").strip()
        if not src or not kd or not text:
            return False

        with self._consolidation_lock:
            conn = sqlite3.connect(str(self._consolidation_db), timeout=30.0)
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT payload, trailing_blank_line FROM consolidation_writes WHERE source_ref=? AND kind=?",
                    (src, kd),
                ).fetchone()
                if row is not None:
                    existing_payload = row[0] or ""
                    existing_trailing = bool(int(row[1] or 0))
                    if not self._file_contains_marker(target_file, marker):
                        if existing_payload:
                            with open(target_file, "a", encoding="utf-8") as f:
                                f.write(marker + "\n")
                                f.write(existing_payload.rstrip() + "\n")
                                if existing_trailing:
                                    f.write("\n")
                    conn.execute("COMMIT")
                    return False

                # 恢复路径：若历史崩溃发生在“文件已写，索引未写”，用尾部扫描补索引并跳过重复写。
                if self._tail_contains_marker(target_file, marker):
                    conn.execute(
                        "INSERT OR REPLACE INTO consolidation_writes(source_ref, kind, payload, trailing_blank_line, done_at) VALUES (?, ?, ?, ?, datetime('now'))",
                        (src, kd, text, 1 if trailing_blank_line else 0),
                    )
                    conn.execute("COMMIT")
                    return False

                with open(target_file, "a", encoding="utf-8") as f:
                    f.write(marker + "\n")
                    f.write(text.rstrip() + "\n")
                    if trailing_blank_line:
                        f.write("\n")

                conn.execute(
                    "INSERT OR REPLACE INTO consolidation_writes(source_ref, kind, payload, trailing_blank_line, done_at) VALUES (?, ?, ?, ?, datetime('now'))",
                    (src, kd, text, 1 if trailing_blank_line else 0),
                )
                conn.execute("COMMIT")
                return True
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    @staticmethod
    def _tail_contains_marker(path: Path, marker: str) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                take = min(size, _CONSOLIDATION_TAIL_BYTES)
                if take <= 0:
                    return False
                f.seek(size - take)
                tail = f.read(take).decode("utf-8", errors="ignore")
                return marker in tail
        except Exception:
            return False

    @staticmethod
    def _file_contains_marker(path: Path, marker: str) -> bool:
        if not path.exists():
            return False
        needle = marker.encode("utf-8")
        if not needle:
            return False
        carry = b""
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    data = carry + chunk
                    if needle in data:
                        return True
                    if len(needle) > 1:
                        carry = data[-(len(needle) - 1) :]
                    else:
                        carry = b""
        except Exception:
            return False
        return False
