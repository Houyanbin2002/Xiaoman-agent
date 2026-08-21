from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from core.personal.models import (
    ContextStateData,
    FollowUpTrigger,
    PeriodicReportData,
    PersonalEntityType,
    PersonalRecord,
    ProactiveIntentData,
    RecordSource,
    SceneMode,
)
from core.personal.rhythm.models import (
    DeliveryPolicy,
    PeriodicReport,
    PersonalContextSnapshot,
    RecommendationProvider,
    ReportContributor,
    TaskRecommendation,
)
from core.personal.service import PersonalDataService


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _parse_due_window(value: dict[str, Any], timezone_name: str) -> datetime | None:
    raw_date = str(value.get("due_date") or "").strip()
    if not raw_date:
        return None
    try:
        local_due = datetime.strptime(raw_date, "%Y-%m-%d")
        zone = ZoneInfo(timezone_name)
    except (ValueError, KeyError):
        return None
    period = str(value.get("due_period") or "").lower()
    hour, minute = {
        "morning": (12, 0),
        "noon": (14, 0),
        "afternoon": (18, 0),
        "evening": (23, 59),
    }.get(period, (23, 59))
    return local_due.replace(hour=hour, minute=minute, tzinfo=zone).astimezone(
        timezone.utc
    )


class PersonalRhythmService:
    """Context and opportunity layer over the governed personal fact store."""

    _SCENE_DURATIONS = {
        SceneMode.LEAVING: 8 * 60,
        SceneMode.HOME: 12 * 60,
        SceneMode.BEDTIME: 10 * 60,
        SceneMode.TRAVEL: 24 * 60,
        SceneMode.NEUTRAL: 12 * 60,
    }

    def __init__(
        self,
        personal_data: PersonalDataService,
        *,
        timezone_name: str = "Asia/Shanghai",
    ) -> None:
        self.personal_data = personal_data
        self.timezone_name = timezone_name
        self._recommendation_providers: list[RecommendationProvider] = [
            self._commitment_candidates,
            self._daily_plan_candidates,
            self._trip_candidates,
        ]
        self._report_contributors: list[ReportContributor] = []

    def register_recommendation_provider(
        self,
        provider: RecommendationProvider,
    ) -> None:
        self._recommendation_providers.append(provider)

    def register_report_contributor(
        self,
        contributor: ReportContributor,
    ) -> None:
        self._report_contributors.append(contributor)

    def snapshot(self, *, now: datetime | None = None) -> PersonalContextSnapshot:
        current = (now or _utc_now()).astimezone(timezone.utc)
        self.personal_data.expire_due(now=current.isoformat())
        states = self.personal_data.list(
            entity_type=PersonalEntityType.CONTEXT_STATE,
            limit=500,
        )
        scene_record = next(
            (
                item
                for item in states
                if item.record_key == "scene:current"
                and str(item.data.get("context_type") or "") == "scene"
            ),
            None,
        )
        scene_value = (
            str(scene_record.data.get("mode") or "neutral")
            if scene_record
            else "neutral"
        )
        try:
            scene = SceneMode(scene_value)
        except ValueError:
            scene = SceneMode.NEUTRAL
        focus_records = [
            item
            for item in states
            if str(item.data.get("context_type") or "") == "focus"
            and str(item.data.get("state") or "active") == "active"
            and (_parse_dt(item.data.get("ends_at")) or current) > current
        ]
        focus = max(focus_records, key=lambda item: item.created_at, default=None)
        focus_active = focus is not None
        focus_dnd = bool(focus.data.get("do_not_disturb", True)) if focus else False
        scene_dnd = (
            bool(scene_record.data.get("do_not_disturb", False))
            if scene_record
            else False
        )
        allow_high = True
        if scene_record is not None:
            allow_high = bool(scene_record.data.get("allow_high_priority", True))
        if focus is not None:
            allow_high = allow_high and bool(
                focus.data.get("allow_high_priority", True)
            )
        return PersonalContextSnapshot(
            observed_at=current.isoformat(),
            timezone=self.timezone_name,
            scene=scene,
            scene_ends_at=scene_record.expires_at if scene_record else None,
            focus_active=focus_active,
            focus_label=str(focus.data.get("label") or focus.title) if focus else "",
            focus_ends_at=str(focus.data.get("ends_at") or focus.expires_at)
            if focus
            else None,
            do_not_disturb=focus_dnd or scene_dnd,
            allow_high_priority=allow_high,
            energy=self._current_energy(current),
        )

    def set_scene(
        self,
        scene: SceneMode,
        *,
        duration_minutes: int | None = None,
        now: datetime | None = None,
        source: RecordSource | None = None,
    ) -> PersonalRecord:
        current = (now or _utc_now()).astimezone(timezone.utc)
        duration = max(15, duration_minutes or self._SCENE_DURATIONS[scene])
        ends = current + timedelta(minutes=duration)
        do_not_disturb = scene == SceneMode.BEDTIME
        payload = ContextStateData(
            context_type="scene",
            mode=scene.value,
            started_at=current.isoformat(),
            ends_at=ends.isoformat(),
            label=self._scene_label(scene),
            do_not_disturb=do_not_disturb,
            allow_high_priority=True,
        )
        existing = self.personal_data.find_active_by_key(
            PersonalEntityType.CONTEXT_STATE,
            "scene:current",
        )
        if existing is not None:
            return self.personal_data.update(
                existing.id,
                {
                    "title": self._scene_label(scene),
                    "summary": f"当前场景：{self._scene_label(scene)}",
                    "data": payload,
                    "expires_at": ends.isoformat(),
                },
                actor="user",
                reason="scene changed",
            )
        return self.personal_data.create(
            entity_type=PersonalEntityType.CONTEXT_STATE,
            record_key="scene:current",
            title=self._scene_label(scene),
            summary=f"当前场景：{self._scene_label(scene)}",
            data=payload,
            source=source or RecordSource("dashboard", "rhythm:scene"),
            expires_at=ends.isoformat(),
            actor="user",
        )

    def start_focus(
        self,
        *,
        minutes: int,
        label: str = "专注",
        allow_high_priority: bool = True,
        now: datetime | None = None,
        source: RecordSource | None = None,
    ) -> PersonalRecord:
        current = (now or _utc_now()).astimezone(timezone.utc)
        duration = max(5, min(int(minutes), 12 * 60))
        self.stop_focus(now=current)
        ends = current + timedelta(minutes=duration)
        payload = {
            **ContextStateData(
                context_type="focus",
                mode="focus",
                started_at=current.isoformat(),
                ends_at=ends.isoformat(),
                label=label.strip() or "专注",
                do_not_disturb=True,
                allow_high_priority=allow_high_priority,
            ).__dict__,
            "state": "active",
        }
        return self.personal_data.create(
            entity_type=PersonalEntityType.CONTEXT_STATE,
            record_key=f"focus:{current.isoformat()}",
            title=label.strip() or "专注",
            summary=f"专注至 {ends.astimezone(ZoneInfo(self.timezone_name)).strftime('%H:%M')}",
            data=payload,
            source=source or RecordSource("dashboard", "rhythm:focus"),
            expires_at=ends.isoformat(),
            actor="user",
        )

    def stop_focus(self, *, now: datetime | None = None) -> int:
        current = (now or _utc_now()).astimezone(timezone.utc)
        states = self.personal_data.list(
            entity_type=PersonalEntityType.CONTEXT_STATE,
            limit=500,
        )
        stopped = 0
        for record in states:
            if str(record.data.get("context_type") or "") != "focus":
                continue
            if str(record.data.get("state") or "active") != "active":
                continue
            data = dict(record.data)
            data.update({"state": "completed", "ended_at": current.isoformat()})
            self.personal_data.update(
                record.id,
                {"data": data, "expires_at": current.isoformat()},
                actor="user",
                reason="focus stopped",
            )
            stopped += 1
        self.personal_data.expire_due(now=current.isoformat())
        return stopped

    def delivery_policy(
        self,
        *,
        priority: str,
        now: datetime | None = None,
    ) -> DeliveryPolicy:
        context = self.snapshot(now=now)
        if not context.do_not_disturb:
            return DeliveryPolicy(True, "normal")
        if priority == "high" and context.allow_high_priority:
            return DeliveryPolicy(True, "high_priority_override")
        if context.focus_active:
            return DeliveryPolicy(False, "focus_mode")
        if context.scene == SceneMode.BEDTIME:
            return DeliveryPolicy(False, "bedtime_mode")
        return DeliveryPolicy(False, "do_not_disturb")

    def recommend(
        self,
        *,
        minutes: int = 30,
        now: datetime | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        window = max(5, min(int(minutes), 8 * 60))
        current = (now or _utc_now()).astimezone(timezone.utc)
        context = self.snapshot(now=current)
        candidates: list[TaskRecommendation] = []
        for provider in self._recommendation_providers:
            candidates.extend(provider(window, context, current))
        candidates.sort(
            key=lambda item: (-item.score, item.estimated_minutes, item.title)
        )
        return {
            "available_minutes": window,
            "context": context.to_dict(),
            "recommendations": [item.to_dict() for item in candidates[: max(1, limit)]],
        }

    def create_domain_record(
        self,
        *,
        entity_type: PersonalEntityType,
        title: str,
        summary: str,
        data: dict[str, Any],
        record_key: str = "",
        source: RecordSource | None = None,
    ) -> PersonalRecord:
        allowed = {
            PersonalEntityType.RELATIONSHIP,
            PersonalEntityType.IMPORTANT_DATE,
            PersonalEntityType.FINANCIAL_OBLIGATION,
            PersonalEntityType.TRIP,
            PersonalEntityType.GOAL,
            PersonalEntityType.COMMITMENT,
        }
        if entity_type not in allowed:
            raise ValueError("unsupported rhythm record type")
        return self.personal_data.create(
            entity_type=entity_type,
            record_key=record_key,
            title=title,
            summary=summary,
            data=data,
            source=source or RecordSource("dashboard", "rhythm:record"),
            actor="user",
        )

    def create_follow_up(
        self,
        *,
        title: str,
        message: str,
        reason: str,
        trigger_type: FollowUpTrigger,
        next_trigger_at: str | None = None,
        interval_minutes: int | None = None,
        target_entity_type: str = "",
        target_record_key: str = "",
        inactivity_days: int | None = None,
        condition: dict[str, Any] | None = None,
        cooldown_minutes: int = 60,
        user_confirmed: bool = False,
        source: RecordSource | None = None,
    ) -> PersonalRecord:
        if trigger_type == FollowUpTrigger.INTERVAL and not interval_minutes:
            raise ValueError("interval follow-up requires interval_minutes")
        if (
            trigger_type == FollowUpTrigger.AT_TIME
            and _parse_dt(next_trigger_at) is None
        ):
            raise ValueError(
                "at_time follow-up requires timezone-aware next_trigger_at"
            )
        if trigger_type == FollowUpTrigger.INACTIVITY and not inactivity_days:
            raise ValueError("inactivity follow-up requires inactivity_days")
        now = _utc_now()
        next_at = _parse_dt(next_trigger_at)
        if trigger_type == FollowUpTrigger.INTERVAL and next_at is None:
            next_at = now + timedelta(minutes=max(5, int(interval_minutes or 60)))
        enabled = bool(user_confirmed)
        payload = ProactiveIntentData(
            trigger_type=trigger_type,
            message=message.strip(),
            reason=reason.strip(),
            next_trigger_at=next_at.isoformat() if next_at else None,
            interval_minutes=interval_minutes,
            target_entity_type=target_entity_type,
            target_record_key=target_record_key,
            inactivity_days=inactivity_days,
            condition=dict(condition or {}),
            enabled=enabled,
            status="active" if enabled else "proposed",
            cooldown_minutes=max(5, int(cooldown_minutes)),
        )
        digest = hashlib.sha256(
            f"{title.strip().casefold()}|{message.strip().casefold()}".encode("utf-8")
        ).hexdigest()[:16]
        return self.personal_data.create(
            entity_type=PersonalEntityType.PROACTIVE_INTENT,
            record_key=f"intent:{digest}",
            title=title,
            summary=reason,
            data=payload,
            source=source or RecordSource("assistant", "personal-rhythm"),
            actor="user" if user_confirmed else "assistant",
        )

    def generate_report(
        self,
        *,
        period: str,
        now: datetime | None = None,
        persist: bool = True,
    ) -> PeriodicReport:
        current = (now or _utc_now()).astimezone(timezone.utc)
        local_now = current.astimezone(ZoneInfo(self.timezone_name))
        if period == "week":
            local_start = local_now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=local_now.weekday())
        elif period == "month":
            local_start = local_now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
        else:
            raise ValueError("period must be week or month")
        start = local_start.astimezone(timezone.utc)
        records = self.personal_data.list(limit=10000)
        commitments = [
            item
            for item in records
            if item.entity_type == PersonalEntityType.COMMITMENT
        ]
        completed = [
            item
            for item in commitments
            if str(item.data.get("state") or "") == "completed"
            and (_parse_dt(item.data.get("completed_at")) or current) >= start
        ]
        open_items = [
            item
            for item in commitments
            if str(item.data.get("state") or "open") == "open"
        ]
        overdue = [
            item
            for item in open_items
            if (_parse_dt(item.data.get("due_at")) or current + timedelta(days=1))
            < current
        ]
        progress_values = [
            _number(item.data.get("progress"), 0.0) for item in open_items
        ]
        sleep_values: list[float] = []
        mood_values: list[float] = []
        for item in records:
            observed = _parse_dt(item.data.get("observed_at"))
            if observed is None or observed < start:
                continue
            if item.entity_type == PersonalEntityType.HEALTH_OBSERVATION and str(
                item.data.get("metric") or ""
            ) in {"sleep_hours", "sleep_duration"}:
                value = _number(item.data.get("value"), -1)
                if value >= 0:
                    sleep_values.append(value)
            if (
                item.entity_type == PersonalEntityType.CHECK_IN
                and str(item.data.get("check_in_type") or "") == "mood"
            ):
                value = _number(item.data.get("rating"), -1)
                if value >= 0:
                    mood_values.append(value)
        deviations = self._goal_deviations(records, current)
        metrics = {
            "commitments_completed": len(completed),
            "commitments_open": len(open_items),
            "commitments_overdue": len(overdue),
            "average_open_progress": round(
                sum(progress_values) / len(progress_values), 3
            )
            if progress_values
            else None,
            "average_sleep_hours": round(sum(sleep_values) / len(sleep_values), 2)
            if sleep_values
            else None,
            "average_mood": round(sum(mood_values) / len(mood_values), 2)
            if mood_values
            else None,
            "mood_trend": round(mood_values[-1] - mood_values[0], 2)
            if len(mood_values) >= 2
            else None,
        }
        recommendations: list[str] = []
        if overdue:
            recommendations.append(f"先处理或重新评估 {len(overdue)} 项逾期待办。")
        if deviations:
            recommendations.append(
                f"有 {len(deviations)} 个目标落后于时间进度，建议缩小下一步。"
            )
        if (
            metrics["average_sleep_hours"] is not None
            and metrics["average_sleep_hours"] < 6.5
        ):
            recommendations.append("本周期平均睡眠偏低，安排计划时保留恢复空间。")
        for contributor in self._report_contributors:
            contribution = contributor(records, start, current)
            metrics.update(contribution.metrics)
            deviations.extend(contribution.deviations)
            recommendations.extend(contribution.recommendations)
        if not recommendations:
            recommendations.append("当前没有明显偏差，继续维持现有节奏并记录关键变化。")
        report_record: PersonalRecord | None = None
        end = current
        if persist:
            key = f"report:{period}:{start.date().isoformat()}"
            payload = PeriodicReportData(
                period=period,
                period_start=start.isoformat(),
                period_end=end.isoformat(),
                metrics=metrics,
                deviations=deviations,
                recommendations=recommendations,
            )
            existing = self.personal_data.find_active_by_key(
                PersonalEntityType.PERIODIC_REPORT,
                key,
            )
            if existing is None:
                report_record = self.personal_data.create(
                    entity_type=PersonalEntityType.PERIODIC_REPORT,
                    record_key=key,
                    title="周度回顾" if period == "week" else "月度回顾",
                    summary="个人节奏与目标偏差分析",
                    data=payload,
                    source=RecordSource("system", "personal-rhythm-report"),
                    actor="system",
                )
            else:
                report_record = self.personal_data.update(
                    existing.id,
                    {"data": payload},
                    actor="system",
                    reason="periodic report refreshed",
                    automatic=True,
                )
        return PeriodicReport(
            period=period,
            period_start=start.isoformat(),
            period_end=end.isoformat(),
            metrics=metrics,
            deviations=deviations,
            recommendations=recommendations,
            record_id=report_record.id if report_record else None,
        )

    def overview(self, *, now: datetime | None = None) -> dict[str, Any]:
        records = self.personal_data.list(limit=10000)
        tracked = {
            entity.value: sum(item.entity_type == entity for item in records)
            for entity in (
                PersonalEntityType.RELATIONSHIP,
                PersonalEntityType.IMPORTANT_DATE,
                PersonalEntityType.FINANCIAL_OBLIGATION,
                PersonalEntityType.TRIP,
                PersonalEntityType.GOAL,
                PersonalEntityType.PROACTIVE_INTENT,
            )
        }
        return {"context": self.snapshot(now=now).to_dict(), "counts": tracked}

    def _current_energy(self, now: datetime) -> str:
        check_ins = self.personal_data.list(
            entity_type=PersonalEntityType.CHECK_IN,
            limit=100,
        )
        rated: list[tuple[datetime, float]] = []
        for record in check_ins:
            observed = _parse_dt(record.data.get("observed_at"))
            if observed is None or now - observed > timedelta(days=2):
                continue
            raw = record.data.get("dimensions", {})
            energy = raw.get("energy") if isinstance(raw, dict) else None
            if (
                energy is None
                and str(record.data.get("check_in_type") or "") == "energy"
            ):
                energy = record.data.get("rating")
            value = _number(energy, -1)
            if value >= 0:
                rated.append((observed, value))
        if not rated:
            return "medium"
        latest = max(rated, key=lambda item: item[0])[1]
        if latest <= 3:
            return "low"
        if latest >= 7:
            return "high"
        return "medium"

    def _commitment_candidate(
        self,
        record: PersonalRecord,
        window: int,
        context: PersonalContextSnapshot,
        now: datetime,
    ) -> TaskRecommendation | None:
        if str(record.data.get("state") or "open") != "open":
            return None
        estimated = int(_number(record.data.get("estimated_minutes"), window))
        if estimated <= 0 or estimated > window:
            return None
        contexts = [
            str(item).lower() for item in record.data.get("contexts", []) if str(item)
        ]
        scene = context.scene.value
        if contexts and scene not in contexts and "any" not in contexts:
            return None
        energy = str(record.data.get("energy") or "medium").lower()
        if context.energy == "low" and energy == "high":
            return None
        precise_due = _parse_dt(record.data.get("due_at"))
        due = precise_due or _parse_due_window(record.data, self.timezone_name)
        due_text = str(record.data.get("due_text") or "").strip()
        priority = str(record.data.get("priority") or "normal").lower()
        score = 1.0 - estimated / max(window * 2, 1)
        score += {"urgent": 0.8, "high": 0.55, "normal": 0.25, "low": 0.0}.get(
            priority, 0.2
        )
        reasons = [f"预计 {estimated} 分钟，可在当前时间窗内完成"]
        if due_text:
            reasons.append(f"截止范围：{due_text}")
        if due is not None:
            hours = (due - now).total_seconds() / 3600
            if hours <= 0:
                score += 1.0
                reasons.append("已经逾期")
            elif hours <= 24:
                score += 0.75
                reasons.append("24 小时内截止")
            elif hours <= 72:
                score += 0.4
                reasons.append("临近截止")
        if contexts:
            score += 0.25
            reasons.append("与当前场景匹配")
        if energy == context.energy:
            score += 0.15
            reasons.append("与当前精力匹配")
        next_action = str(
            record.data.get("next_action") or record.summary or record.title
        )
        return TaskRecommendation(
            candidate_id=record.id,
            source_type=record.entity_type.value,
            title=record.title,
            next_action=next_action,
            estimated_minutes=estimated,
            score=score,
            reason="；".join(reasons),
            due_at=precise_due.isoformat() if precise_due else None,
            due_text=due_text,
            context=scene,
            energy=energy,
        )

    def _commitment_candidates(
        self,
        window: int,
        context: PersonalContextSnapshot,
        now: datetime,
    ) -> list[TaskRecommendation]:
        result: list[TaskRecommendation] = []
        for record in self.personal_data.list(
            entity_type=PersonalEntityType.COMMITMENT,
            limit=5000,
        ):
            candidate = self._commitment_candidate(record, window, context, now)
            if candidate is not None:
                result.append(candidate)
        return result

    def _daily_plan_candidates(
        self,
        window: int,
        context: PersonalContextSnapshot,
        now: datetime,
    ) -> list[TaskRecommendation]:
        local_date = now.astimezone(ZoneInfo(self.timezone_name)).date().isoformat()
        result: list[TaskRecommendation] = []
        for plan in self.personal_data.list(
            entity_type=PersonalEntityType.DAILY_PLAN, limit=100
        ):
            if str(plan.data.get("plan_date") or "") != local_date:
                continue
            for index, item in enumerate(plan.data.get("items", [])):
                if not isinstance(item, dict) or bool(item.get("done", False)):
                    continue
                estimated = int(_number(item.get("estimated_minutes"), window))
                if estimated <= 0 or estimated > window:
                    continue
                energy = str(item.get("energy") or "medium")
                if context.energy == "low" and energy == "high":
                    continue
                result.append(
                    TaskRecommendation(
                        candidate_id=f"{plan.id}:{index}",
                        source_type="daily_plan_item",
                        title=str(
                            item.get("title") or item.get("action") or "计划事项"
                        ),
                        next_action=str(
                            item.get("next_action")
                            or item.get("action")
                            or item.get("title")
                            or ""
                        ),
                        estimated_minutes=estimated,
                        score=1.15 - estimated / max(window * 2, 1),
                        reason=f"今天的计划事项，预计 {estimated} 分钟",
                        context=context.scene.value,
                        energy=energy,
                    )
                )
        return result

    def _trip_candidates(
        self,
        window: int,
        context: PersonalContextSnapshot,
        now: datetime,
    ) -> list[TaskRecommendation]:
        result: list[TaskRecommendation] = []
        for trip in self.personal_data.list(
            entity_type=PersonalEntityType.TRIP, limit=100
        ):
            depart = _parse_dt(trip.data.get("depart_at"))
            if depart is None or depart < now or depart - now > timedelta(days=14):
                continue
            for index, item in enumerate(trip.data.get("checklist", [])):
                if not isinstance(item, dict) or bool(item.get("done", False)):
                    continue
                estimated = int(_number(item.get("estimated_minutes"), 10))
                if estimated > window:
                    continue
                result.append(
                    TaskRecommendation(
                        candidate_id=f"{trip.id}:{index}",
                        source_type="trip_checklist_item",
                        title=str(item.get("title") or "旅行准备事项"),
                        next_action=str(item.get("action") or item.get("title") or ""),
                        estimated_minutes=max(1, estimated),
                        score=1.35 + max(0.0, 14 - (depart - now).days) / 20,
                        reason=f"距出发还有 {(depart - now).days} 天，清单仍未完成",
                        due_at=depart.isoformat(),
                        context=context.scene.value,
                        energy="low",
                    )
                )
        return result

    @staticmethod
    def _goal_deviations(
        records: list[PersonalRecord], now: datetime
    ) -> list[dict[str, Any]]:
        deviations: list[dict[str, Any]] = []
        for goal in records:
            if goal.entity_type != PersonalEntityType.GOAL:
                continue
            if str(goal.data.get("state") or "active") != "active":
                continue
            start = _parse_dt(goal.data.get("start_at"))
            due = _parse_dt(goal.data.get("due_at"))
            if start is None or due is None or due <= start:
                continue
            elapsed = min(
                1.0,
                max(0.0, (now - start).total_seconds() / (due - start).total_seconds()),
            )
            target = _number(goal.data.get("target"), 0.0)
            current = _number(goal.data.get("current"), 0.0)
            direction = str(goal.data.get("direction") or "increase")
            actual = current / target if direction == "increase" and target else 0.0
            if direction == "decrease" and current:
                actual = min(1.0, target / current) if target else 0.0
            gap = elapsed - actual
            if elapsed >= 0.25 and gap >= 0.15:
                deviations.append(
                    {
                        "record_id": goal.id,
                        "title": goal.title,
                        "expected_progress": round(elapsed, 3),
                        "actual_progress": round(actual, 3),
                        "gap": round(gap, 3),
                        "due_at": due.isoformat(),
                    }
                )
        return deviations

    @staticmethod
    def _scene_label(scene: SceneMode) -> str:
        return {
            SceneMode.NEUTRAL: "日常",
            SceneMode.LEAVING: "出门",
            SceneMode.HOME: "回家",
            SceneMode.BEDTIME: "睡前",
            SceneMode.TRAVEL: "旅行中",
        }[scene]


__all__ = ["PersonalRhythmService"]
