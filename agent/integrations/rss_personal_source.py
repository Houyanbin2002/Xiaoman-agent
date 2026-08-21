from __future__ import annotations

import asyncio
import hashlib
import html
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from core.personal.sources.models import ExternalSourceItem, ExternalSourceSubscription

_MAX_FEED_BYTES = 2 * 1024 * 1024
_TAG_RE = re.compile(r"<[^>]+>")


class RssPersonalSourceAdapter:
    """Read a user-selected RSS/Atom feed as an external attention source."""

    def __init__(self, fetcher: Callable[[str], bytes | str] | None = None) -> None:
        self._fetcher = fetcher or _download

    async def fetch(
        self,
        subscription: ExternalSourceSubscription,
    ) -> list[ExternalSourceItem]:
        url = _validated_url(subscription.resource_url)
        raw = await asyncio.to_thread(self._fetcher, url)
        document = raw.encode("utf-8") if isinstance(raw, str) else raw
        if len(document) > _MAX_FEED_BYTES:
            raise ValueError("RSS 内容超过 2 MB，已停止读取")
        entries = _parse_feed(document)
        mapping = subscription.mapping
        max_items = _bounded_int(mapping.get("max_items"), default=50, maximum=200)
        return [
            _to_source_item(entry, subscription)
            for entry in entries[:max_items]
        ]


def _download(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "Xiaoman/1.0 RSS reader (+local personal assistant)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.2",
        },
    )
    with urlopen(request, timeout=20) as response:
        content = response.read(_MAX_FEED_BYTES + 1)
    if len(content) > _MAX_FEED_BYTES:
        raise ValueError("RSS 内容超过 2 MB，已停止读取")
    return content


def _validated_url(value: str) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("RSS 地址必须是有效的 http/https URL")
    return url


def _parse_feed(document: bytes) -> list[dict[str, str | list[str]]]:
    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError as exc:
        raise ValueError("该地址没有返回有效的 RSS/Atom XML") from exc

    root_name = _local_name(root.tag)
    if root_name == "rss" or _first_child(root, "channel") is not None:
        channel = _first_child(root, "channel")
        if channel is None:
            channel = root
        channel_title = _child_text(channel, "title")
        if "rss reader not yet whitelisted" in channel_title.lower():
            raise ValueError(
                "该 RSS 服务要求先将小满加入白名单；请更换可公开访问的 RSS 地址"
            )
        nodes = [child for child in channel if _local_name(child.tag) == "item"]
    elif root_name == "feed":
        nodes = [child for child in root if _local_name(child.tag) == "entry"]
    else:
        raise ValueError("该地址返回的 XML 不是 RSS 或 Atom feed")

    entries: list[dict[str, str | list[str]]] = []
    for node in nodes:
        title = _clean_text(_child_text(node, "title"))
        summary = _clean_text(
            _child_text(node, "description")
            or _child_text(node, "summary")
            or _child_text(node, "encoded")
            or _child_text(node, "content")
        )
        link = _entry_link(node)
        identity = _child_text(node, "guid") or _child_text(node, "id") or link
        published = _normalise_date(
            _child_text(node, "pubDate")
            or _child_text(node, "published")
            or _child_text(node, "updated")
            or _child_text(node, "date")
        )
        author = _child_text(node, "creator") or _author_text(node)
        categories = [
            _clean_text(child.text or child.attrib.get("term", ""))
            for child in node.iter()
            if _local_name(child.tag) == "category"
        ]
        categories = [item for item in categories if item]
        if not identity:
            seed = "|".join((title, summary, published, author))
            identity = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        if not title:
            title = summary[:120] or "订阅源有新内容"
        entries.append(
            {
                "id": identity,
                "title": title,
                "summary": summary or title,
                "link": link,
                "published_at": published,
                "author": _clean_text(author),
                "categories": categories,
            }
        )
    return entries


def _to_source_item(
    entry: dict[str, str | list[str]],
    subscription: ExternalSourceSubscription,
) -> ExternalSourceItem:
    mapping = subscription.mapping
    published = str(entry.get("published_at") or "")
    observed_at = published or subscription.created_at
    notify_initial = bool(mapping.get("notify_initial", False))
    is_new = _is_at_or_after(published, subscription.created_at)
    title = str(entry["title"])
    summary = str(entry["summary"])
    link = str(entry.get("link") or subscription.resource_url)
    signal_enabled = notify_initial or is_new
    data = {
        "source_type": "rss",
        "feed_url": subscription.resource_url,
        "url": link,
        "author": str(entry.get("author") or ""),
        "categories": list(entry.get("categories") or []),
        "published_at": published,
        "observed_at": observed_at,
        "state": "open",
        "attention_signal": {
            "enabled": signal_enabled,
            "kind": "content.feed_update",
            "domain": str(mapping.get("domain") or "interest"),
            "summary": title,
            "content": summary,
            "occurred_at": observed_at,
            "valid_for_minutes": _bounded_int(
                mapping.get("valid_for_minutes"), default=1440, maximum=10080
            ),
            "severity": _bounded_float(mapping.get("severity"), 0.35),
            "urgency": _bounded_float(mapping.get("urgency"), 0.25),
            "actionability": _bounded_float(mapping.get("actionability"), 0.45),
            "confidence": 0.9,
            "suggested_capabilities": ["message.notify"],
            "reason": "用户明确订阅的信息源出现了新内容",
            "evidence": [{"url": link, "published_at": published}],
        },
    }
    return ExternalSourceItem.build(
        external_id=str(entry["id"]),
        title=title,
        summary=summary,
        data=data,
        source_ref=link,
    )


def _first_child(node: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return next(
        (child for child in node if _local_name(child.tag) == name.lower()),
        None,
    )


def _child_text(node: ElementTree.Element, name: str) -> str:
    wanted = name.lower()
    for child in node.iter():
        if child is node or _local_name(child.tag) != wanted:
            continue
        return "".join(child.itertext()).strip()
    return ""


def _author_text(node: ElementTree.Element) -> str:
    for child in node:
        if _local_name(child.tag) != "author":
            continue
        return _child_text(child, "name") or "".join(child.itertext()).strip()
    return ""


def _entry_link(node: ElementTree.Element) -> str:
    for child in node.iter():
        if _local_name(child.tag) != "link":
            continue
        href = str(child.attrib.get("href") or "").strip()
        relation = str(child.attrib.get("rel") or "alternate").strip()
        if href and relation in {"", "alternate"}:
            return href
        text = "".join(child.itertext()).strip()
        if text:
            return text
    return ""


def _clean_text(value: str) -> str:
    text = html.unescape(_TAG_RE.sub(" ", str(value or "")))
    return " ".join(text.split())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1].lower()


def _normalise_date(value: str) -> str:
    parsed = _parse_date(value)
    return parsed.isoformat() if parsed is not None else ""


def _parse_date(value: str) -> datetime | None:
    candidate = str(value or "").strip()
    if not candidate:
        return None
    try:
        parsed = parsedate_to_datetime(candidate)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_at_or_after(value: str, threshold: str) -> bool:
    current = _parse_date(value)
    boundary = _parse_date(threshold)
    return current is not None and boundary is not None and current >= boundary


def _bounded_int(value: object, *, default: int, maximum: int) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = default
    return max(1, min(number, maximum))


def _bounded_float(value: object, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(number, 1.0))


__all__ = ["RssPersonalSourceAdapter"]
