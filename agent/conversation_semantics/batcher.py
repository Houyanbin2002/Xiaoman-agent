from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from bus.events_lifecycle import TurnCommitted, TurnStarted
from core.conversation_semantics.events import ConversationSemanticBatchCommitted
from core.conversation_semantics.evidence import build_semantic_evidence
from core.conversation_semantics.models import SemanticBatchPayload
from core.conversation_semantics.store import (
    ConversationSemanticStore,
    PreparedSemanticBatch,
)

logger = logging.getLogger(__name__)

_HIGH_VALUE_USER_SIGNAL_RE = re.compile(
    # Explicit directives should be consolidated promptly.  The previous
    # trigger only covered "以后/记住" wording, so equally authoritative
    # forms such as "我喜欢…", "默认…", and corrections could wait for the
    # idle timer and appear to be missed by the caller.
    r"(以后|今后|每次|一直|记住|我喜欢|我的默认|默认|优先|不要再|别再|不要打扰|"
    r"免打扰|提醒我|纠正|作废|取消|改为|改成|已经结束|当前主要关注|不是.+而是)",
    re.IGNORECASE,
)


def _sequence_number(value: object, *, default: int = -1) -> int:
    if not isinstance(value, (str, int, float)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class SemanticAnalyzer(Protocol):
    ANALYSIS_VERSION: str

    async def analyze(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> SemanticBatchPayload: ...


class SessionMessageSource(Protocol):
    def fetch_session_messages(self, session_key: str) -> list[dict[str, Any]]: ...

    def list_sessions(self) -> list[dict[str, Any]]: ...


class SemanticEventPublisher(Protocol):
    async def emit(
        self,
        event: ConversationSemanticBatchCommitted,
    ) -> object: ...


class ConversationSemanticBatcher:
    """Collect committed turns and analyze them outside the reply path."""

    def __init__(
        self,
        *,
        message_source: SessionMessageSource,
        store: ConversationSemanticStore,
        analyzer: SemanticAnalyzer,
        event_bus: SemanticEventPublisher,
        idle_seconds: int = 480,
        max_turns: int = 8,
    ) -> None:
        self._message_source = message_source
        self._store = store
        self._analyzer = analyzer
        self._event_bus = event_bus
        self._idle_seconds = max(1, int(idle_seconds))
        self._max_turns = max(1, int(max_turns))
        self._locks: dict[str, asyncio.Lock] = {}
        self._idle_tasks: dict[str, asyncio.Task[None]] = {}
        self._flush_tasks: set[asyncio.Task[None]] = set()
        self._session_routes: dict[str, tuple[str, str]] = {}
        self._active_session_by_channel: dict[str, str] = {}
        self._closed = False

    async def start(self) -> None:
        for prepared in self._store.list_undelivered():
            try:
                await self._deliver(prepared)
            except Exception:
                # A prepared batch is the recovery checkpoint. Keep startup
                # available even when one downstream domain is temporarily
                # unhealthy; the next flush/restart retries only its receipt.
                logger.exception(
                    "semantic startup redelivery failed batch=%s",
                    prepared.batch_id,
                )
        for row in self._message_source.list_sessions():
            session_key = str(row.get("key") or "").strip()
            if not session_key or not self._pending_messages(session_key):
                continue
            self._session_routes[session_key] = self._route_from_session_key(
                session_key
            )
            self._schedule_flush(session_key, reason="startup_recovery")

    async def on_turn_committed(self, event: TurnCommitted) -> None:
        if self._closed or bool((event.extra or {}).get("skip_post_memory")):
            return
        self._session_routes[event.session_key] = (event.channel, event.chat_id)
        self._active_session_by_channel[event.channel] = event.session_key
        pending = self._pending_messages(event.session_key)
        if self._has_high_value_signal(event, pending):
            self._cancel_idle(event.session_key)
            self._schedule_flush(event.session_key, reason="high_value_signal")
            return
        if self._turn_count(pending) >= self._max_turns:
            self._cancel_idle(event.session_key)
            self._schedule_flush(event.session_key, reason="threshold")
            return
        self._arm_idle(event.session_key)

    async def on_turn_started(self, event: TurnStarted) -> None:
        if self._closed:
            return
        previous = self._active_session_by_channel.get(event.channel)
        self._active_session_by_channel[event.channel] = event.session_key
        self._session_routes[event.session_key] = (event.channel, event.chat_id)
        if previous and previous != event.session_key:
            self._cancel_idle(previous)
            self._schedule_flush(previous, reason="session_switch")

    async def flush(self, session_key: str, *, reason: str) -> None:
        if self._closed:
            return
        lock = self._locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            for prepared in self._store.list_undelivered():
                if prepared.session_key == session_key:
                    await self._deliver(prepared)
            messages = self._pending_messages(session_key)
            if not messages:
                return
            payload = await self._analyzer.analyze(messages)
            evidence = build_semantic_evidence(messages)
            channel, chat_id = self._session_routes.get(
                session_key,
                self._route_from_session_key(session_key),
            )
            end_seq = max(_sequence_number(message.get("seq")) for message in messages)
            prepared = self._store.prepare(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                analysis_version=self._analyzer.ANALYSIS_VERSION,
                message_ids=[str(message["id"]) for message in messages],
                user_message_ids=[
                    str(message["id"])
                    for message in messages
                    if message.get("role") == "user"
                ],
                end_seq=end_seq,
                # Semantic extraction and model-context compaction have
                # independent cursors.  Only the cache-aware summarizer may
                # advance the session context cursor after its summary commits.
                context_consolidate_through=-1,
                payload=payload,
                execution_episode_ids=list(evidence.episode_ids),
                execution_tool_names=list(evidence.execution_tool_names),
            )
            logger.info(
                "conversation semantic batch prepared session=%s batch=%s "
                "messages=%d reason=%s",
                session_key,
                prepared.batch_id,
                len(messages),
                reason,
            )
            await self._deliver(prepared)

    async def drain(self) -> None:
        while self._flush_tasks:
            await asyncio.gather(*tuple(self._flush_tasks))

    async def aclose(self) -> None:
        if self._closed:
            return
        for task in self._idle_tasks.values():
            task.cancel()
        self._idle_tasks.clear()
        for session_key in self._known_session_keys():
            if self._pending_messages(session_key):
                self._schedule_flush(session_key, reason="shutdown")
        try:
            await asyncio.wait_for(self.drain(), timeout=25.0)
        except TimeoutError:
            logger.warning("semantic shutdown flush timed out; startup will resume")
        except Exception as exc:
            # Prepared batches and successful consumer receipts are already
            # durable. A downstream failure must not make application shutdown
            # fail; startup will retry only the missing consumers.
            logger.warning(
                "semantic shutdown flush failed; startup will resume: %s",
                exc,
            )
        finally:
            self._closed = True
            for task in tuple(self._flush_tasks):
                task.cancel()
            if self._flush_tasks:
                await asyncio.gather(*tuple(self._flush_tasks), return_exceptions=True)
            self._store.close()

    def _pending_messages(self, session_key: str) -> list[dict[str, object]]:
        cursor = self._store.pending_cursor(session_key)
        rows = self._message_source.fetch_session_messages(session_key)
        pending: list[dict[str, object]] = []
        for row in rows:
            seq = _sequence_number(row.get("seq"))
            if seq < 0:
                continue
            role = str(row.get("role") or "")
            message_id = str(row.get("id") or "")
            if seq <= cursor or role not in {"user", "assistant"} or not message_id:
                continue
            row_extra = row.get("extra")
            extra = dict(row_extra) if isinstance(row_extra, Mapping) else {}
            if isinstance(row.get("memory_retrieval"), Mapping):
                extra["memory_retrieval"] = dict(
                    row["memory_retrieval"]  # type: ignore[arg-type]
                )
            pending.append(
                {
                    "id": message_id,
                    "seq": seq,
                    "role": role,
                    "content": str(row.get("content") or ""),
                    "tool_chain": row.get("tool_chain") or [],
                    "extra": extra,
                    "timestamp": str(row.get("timestamp") or ""),
                }
            )
        pending.sort(key=lambda item: _sequence_number(item.get("seq")))
        return pending

    @staticmethod
    def _turn_count(messages: Sequence[Mapping[str, object]]) -> int:
        return sum(1 for message in messages if message.get("role") == "user")

    async def _deliver(self, prepared: PreparedSemanticBatch) -> None:
        durable = getattr(self._event_bus, "emit_durable", None)
        if callable(durable):
            durable_result = durable(
                prepared.to_event(),
                delivered=self._store.delivered_consumers(prepared.batch_id),
                on_success=lambda consumer_id: self._store.mark_consumer_delivered(
                    prepared.batch_id, consumer_id
                ),
            )
            if inspect.isawaitable(durable_result):
                await durable_result
        else:
            await self._event_bus.emit(prepared.to_event())
        self._store.mark_delivered(prepared.batch_id)

    @staticmethod
    def _has_high_value_signal(
        event: TurnCommitted,
        pending: Sequence[Mapping[str, object]],
    ) -> bool:
        if _HIGH_VALUE_USER_SIGNAL_RE.search(event.input_message or ""):
            return True
        return bool(build_semantic_evidence(pending).execution_episodes)

    def _known_session_keys(self) -> set[str]:
        keys = set(self._session_routes)
        keys.update(
            str(row.get("key") or "").strip()
            for row in self._message_source.list_sessions()
            if str(row.get("key") or "").strip()
        )
        return keys

    def _arm_idle(self, session_key: str) -> None:
        self._cancel_idle(session_key)
        task = asyncio.create_task(
            self._flush_after_idle(session_key),
            name=f"conversation-semantic-idle:{session_key}",
        )
        self._idle_tasks[session_key] = task
        task.add_done_callback(
            lambda completed, key=session_key: self._idle_done(key, completed)
        )

    async def _flush_after_idle(self, session_key: str) -> None:
        await asyncio.sleep(self._idle_seconds)
        await self.flush(session_key, reason="idle")

    def _idle_done(self, session_key: str, task: asyncio.Task[None]) -> None:
        if self._idle_tasks.get(session_key) is task:
            self._idle_tasks.pop(session_key, None)
        self._log_task_error(task)

    def _cancel_idle(self, session_key: str) -> None:
        task = self._idle_tasks.pop(session_key, None)
        if task is not None:
            task.cancel()

    def _schedule_flush(self, session_key: str, *, reason: str) -> None:
        if self._closed:
            return
        task = asyncio.create_task(
            self.flush(session_key, reason=reason),
            name=f"conversation-semantic-flush:{session_key}",
        )
        self._flush_tasks.add(task)
        task.add_done_callback(self._flush_done)

    def _flush_done(self, task: asyncio.Task[None]) -> None:
        self._flush_tasks.discard(task)
        self._log_task_error(task)

    @staticmethod
    def _log_task_error(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            logger.warning("conversation semantic batch failed: %s", error)

    @staticmethod
    def _route_from_session_key(session_key: str) -> tuple[str, str]:
        channel, separator, chat_id = session_key.partition(":")
        if not separator:
            return "unknown", session_key
        return channel, chat_id
