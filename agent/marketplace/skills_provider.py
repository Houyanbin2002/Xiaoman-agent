from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from .models import MarketplaceItem

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_URL_RESULT = re.compile(
    r"https?://skills\.sh/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.:-]+)"
)
_ITEM_ID = re.compile(
    r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.:-]+)$"
)


class SkillsCliProvider:
    """Search and download skills through the official skills.sh CLI."""

    def __init__(
        self,
        *,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        npx_command: str | None = None,
        cache_path: Path | None = None,
        cache_ttl_seconds: int = 86400,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._run_command = run_command
        self._npx_command = npx_command
        self._cache_path = cache_path or (
            Path.home() / ".xiaoman" / "marketplace" / "skills-search.json"
        )
        self._cache_ttl_seconds = max(0, cache_ttl_seconds)
        self._now = now

    def search(self, query: str, limit: int = 20) -> list[MarketplaceItem]:
        clean_query = query.strip()
        if not clean_query:
            return []
        cached, fresh = self._cached_ids(clean_query)
        if fresh:
            return [_market_item(item_id) for item_id in cached[:limit]]
        try:
            result = self._run(
                [self._npx(), "-y", "skills", "find", clean_query],
                cwd=Path.cwd(),
                timeout=30,
            )
        except (OSError, RuntimeError):
            if cached:
                return [_market_item(item_id) for item_id in cached[:limit]]
            raise
        output = _ANSI_ESCAPE.sub("", result.stdout or "")
        item_ids = _result_ids(output)
        self._store_query(clean_query, item_ids)
        return [_market_item(item_id) for item_id in item_ids[: max(0, min(limit, 100))]]

    def get(self, item_id: str) -> MarketplaceItem | None:
        parsed = _parse_item_id(item_id)
        if parsed is None:
            return None
        return _market_item(item_id)

    def download(self, item_id: str, destination: Path) -> Path:
        parsed = _parse_item_id(item_id)
        if parsed is None:
            raise ValueError("Skill 市场 ID 格式无效")
        owner, repository, slug = parsed
        destination.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                self._npx(),
                "-y",
                "skills",
                "add",
                f"{owner}/{repository}",
                "--skill",
                slug,
                "--agent",
                "universal",
                "--copy",
                "-y",
            ],
            cwd=destination,
            timeout=600,
        )
        found = [path.parent for path in destination.rglob("SKILL.md")]
        exact = [path for path in found if path.name == slug]
        matches = exact or found if len(found) == 1 else exact
        if len(matches) != 1:
            raise RuntimeError(
                f"skills.sh 下载结果无效：预期 1 个 {slug}，实际 {len(matches)} 个"
            )
        return matches[0]

    def refresh(self) -> None:
        try:
            self._cache_path.unlink()
        except FileNotFoundError:
            pass

    def _run(
        self, command: list[str], *, cwd: Path, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update({"CI": "1", "NO_COLOR": "1"})
        result = self._run_command(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(detail or "skills.sh 命令执行失败")
        return result

    def _npx(self) -> str:
        command = self._npx_command or shutil.which("npx.cmd") or shutil.which("npx")
        if not command:
            raise RuntimeError("未检测到 Node.js / npx，暂时无法访问 Skill 市场")
        return command

    def _cached_ids(self, query: str) -> tuple[list[str], bool]:
        document = self._read_cache()
        queries = document.get("queries")
        if not isinstance(queries, dict):
            return [], False
        raw = queries.get(query.casefold())
        if not isinstance(raw, dict):
            return [], False
        ids = raw.get("ids")
        fetched_at = raw.get("fetched_at")
        clean_ids = [item for item in ids if isinstance(item, str)] if isinstance(ids, list) else []
        fresh = isinstance(fetched_at, (int, float)) and (
            self._now() - float(fetched_at) < self._cache_ttl_seconds
        )
        return clean_ids, fresh

    def _store_query(self, query: str, item_ids: list[str]) -> None:
        document = self._read_cache()
        queries = document.get("queries")
        if not isinstance(queries, dict):
            queries = {}
            document["queries"] = queries
        queries[query.casefold()] = {
            "fetched_at": self._now(),
            "ids": item_ids,
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._cache_path)

    def _read_cache(self) -> dict[str, object]:
        try:
            loaded = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}


def _parse_item_id(item_id: str) -> tuple[str, str, str] | None:
    match = _ITEM_ID.fullmatch(item_id.strip())
    return match.groups() if match else None


def _result_ids(output: str) -> list[str]:
    item_ids: list[str] = []
    seen: set[str] = set()
    for match in _URL_RESULT.finditer(output):
        item_id = "/".join(match.groups())
        if item_id in seen:
            continue
        seen.add(item_id)
        item_ids.append(item_id)
    return item_ids


def _market_item(item_id: str) -> MarketplaceItem:
    parsed = _parse_item_id(item_id)
    if parsed is None:
        raise ValueError("Skill 市场 ID 格式无效")
    owner, repository, slug = parsed
    return MarketplaceItem(
        id=item_id,
        kind="skill",
        name=slug,
        description=f"来自 skills.sh 的 {slug} 技能",
        provider=owner,
        source_url=f"https://skills.sh/{item_id}",
        verified=True,
        install_mode="direct",
        unsupported_reason="",
        install_spec={"source": f"{owner}/{repository}", "skill": slug},
    )
