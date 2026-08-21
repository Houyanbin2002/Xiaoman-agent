from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.tools.personal import PersonalGuidanceTool, PersonalRhythmControlTool
from bootstrap.attention import build_attention_runtime
from core.personal.models import PersonalEntityType, RecordSource, SceneMode
from core.personal.rhythm import (
    PersonalRhythmService,
    ReportContribution,
    TaskRecommendation,
)
from core.personal.service import PersonalDataService
from infra.persistence.personal_store import PersonalStore

NOW = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)


def _services(tmp_path: Path) -> tuple[PersonalDataService, PersonalRhythmService]:
    data = PersonalDataService(PersonalStore(tmp_path / "personal.db"))
    return data, PersonalRhythmService(data)


def _create(
    data: PersonalDataService,
    entity_type: PersonalEntityType,
    title: str,
    payload: dict,
):
    return data.create(
        entity_type=entity_type,
        title=title,
        summary=title,
        data=payload,
        source=RecordSource("test", title),
    )


def test_time_window_recommendation_uses_scene_energy_and_deadline(tmp_path: Path):
    data, rhythm = _services(tmp_path)
    rhythm.set_scene(SceneMode.HOME, now=NOW)
    _create(
        data,
        PersonalEntityType.CHECK_IN,
        "Low energy",
        {"check_in_type": "energy", "rating": 2, "observed_at": NOW.isoformat()},
    )
    _create(
        data,
        PersonalEntityType.COMMITMENT,
        "Quick report edit",
        {
            "state": "open",
            "estimated_minutes": 25,
            "energy": "low",
            "contexts": ["home"],
            "priority": "high",
            "due_at": (NOW + timedelta(hours=5)).isoformat(),
            "next_action": "修完报告第一节",
        },
    )
    _create(
        data,
        PersonalEntityType.COMMITMENT,
        "Hard workout",
        {
            "state": "open",
            "estimated_minutes": 30,
            "energy": "high",
            "contexts": ["home"],
        },
    )
    _create(
        data,
        PersonalEntityType.COMMITMENT,
        "Long cleanup",
        {"state": "open", "estimated_minutes": 90, "energy": "low"},
    )

    result = rhythm.recommend(minutes=30, now=NOW)

    assert result["context"]["scene"] == "home"
    assert result["context"]["energy"] == "low"
    assert [item["title"] for item in result["recommendations"]] == [
        "Quick report edit"
    ]
    assert "24 小时内截止" in result["recommendations"][0]["reason"]
    data.close()


def test_focus_and_bedtime_share_delivery_policy(tmp_path: Path):
    data, rhythm = _services(tmp_path)
    rhythm.start_focus(minutes=50, label="写方案", now=NOW)
    snapshot = rhythm.snapshot(now=NOW + timedelta(minutes=1))
    assert snapshot.focus_active is True
    assert snapshot.do_not_disturb is True
    assert rhythm.delivery_policy(priority="warning", now=NOW).allowed is False
    assert rhythm.delivery_policy(priority="high", now=NOW).allowed is True

    assert rhythm.stop_focus(now=NOW + timedelta(minutes=2)) == 1
    rhythm.set_scene(SceneMode.BEDTIME, now=NOW + timedelta(minutes=3))
    bedtime = rhythm.snapshot(now=NOW + timedelta(minutes=4))
    assert bedtime.scene == SceneMode.BEDTIME
    assert bedtime.do_not_disturb is True
    assert (
        rhythm.delivery_policy(priority="info", now=NOW + timedelta(minutes=4)).reason
        == "bedtime_mode"
    )
    data.close()


def test_date_window_is_ranked_without_inventing_an_exact_due_time(tmp_path: Path):
    data, rhythm = _services(tmp_path)
    _create(
        data,
        PersonalEntityType.COMMITMENT,
        "Morning planning",
        {"estimated_minutes": 15, "deadline": "2026-07-12 上午"},
    )

    result = rhythm.recommend(minutes=30, now=NOW)

    assert result["recommendations"][0]["due_at"] is None
    assert result["recommendations"][0]["due_text"] == "2026-07-12 上午"
    assert "截止范围：2026-07-12 上午" in result["recommendations"][0]["reason"]
    data.close()


@pytest.mark.asyncio
async def test_agent_tools_keep_inferred_followup_disabled_until_confirmed(
    tmp_path: Path,
):
    data, rhythm = _services(tmp_path)
    attention = build_attention_runtime(tmp_path / "personal.db")
    guidance = PersonalGuidanceTool(rhythm)
    control = PersonalRhythmControlTool(rhythm, attention.feedback)
    snapshot = json.loads(await guidance.execute(action="snapshot"))
    assert snapshot["scene"] == "neutral"

    proposed = json.loads(
        await control.execute(
            action="create_follow_up",
            title="问问项目进展",
            message="最近推进得怎么样？",
            reason="项目可能需要持续跟进",
            trigger_type="interval",
            interval_minutes=1440,
            user_confirmed=False,
            channel="dashboard",
            chat_id="owner",
        )
    )
    assert proposed["enabled"] is False
    assert proposed["requires_confirmation"] is True

    confirmed = json.loads(
        await control.execute(
            action="create_follow_up",
            title="每日项目回顾",
            message="今天项目推进了什么？",
            reason="用户明确要求每天询问",
            trigger_type="interval",
            interval_minutes=1440,
            user_confirmed=True,
            channel="dashboard",
            chat_id="owner",
        )
    )
    assert confirmed["enabled"] is True
    attention.close()
    data.close()


def test_plugins_can_extend_recommendations_and_reports_without_new_flow(
    tmp_path: Path,
):
    data, rhythm = _services(tmp_path)
    rhythm.register_recommendation_provider(
        lambda _minutes, context, _now: [
            TaskRecommendation(
                candidate_id="plugin:stretch",
                source_type="plugin",
                title="做一次肩颈拉伸",
                next_action="完成五分钟拉伸",
                estimated_minutes=5,
                score=3.0,
                reason="健身插件根据久坐数据提供",
                context=context.scene.value,
                energy="low",
            )
        ]
    )
    rhythm.register_report_contributor(
        lambda _records, _start, _end: ReportContribution(
            metrics={"training_sessions": 3},
            recommendations=["健身插件建议下周保留一次恢复训练。"],
        )
    )

    recommendations = rhythm.recommend(minutes=30, now=NOW)
    report = rhythm.generate_report(period="week", now=NOW, persist=False)

    assert recommendations["recommendations"][0]["candidate_id"] == "plugin:stretch"
    assert report.metrics["training_sessions"] == 3
    assert "恢复训练" in report.recommendations[-1]
    data.close()
