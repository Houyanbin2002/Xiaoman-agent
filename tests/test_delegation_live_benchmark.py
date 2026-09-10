import pytest
from core.llm import LLMResponse
from eval.delegation_live_benchmark import Meter, grade


def test_rubric_requires_all_numeric_constraints():
    good = '{"totals":{"林岸":8560,"云谷":8300,"沐光":7980},"recommended":"林岸","reasons":"云谷没有素食，沐光只能16人"}'
    assert grade("independent", good)["passed"]
    assert not grade("independent", good.replace("7980", "7880"))["passed"]
    assert not grade("independent", "已创建任务，稍后完成")["passed"]


def test_dependent_rubric_requires_boolean_and_total():
    assert grade("dependent", '{"arrival":"20:15","total":450,"can_check_in":true}')[
        "passed"
    ]
    assert not grade(
        "dependent", '{"arrival":"20:15","total":420,"can_check_in":"true"}'
    )["passed"]


@pytest.mark.asyncio
async def test_meter_uses_actual_provider_content_delta_field():
    class Provider:
        async def chat(self, **kwargs):
            await kwargs["on_content_delta"]({"thinking_delta": "x"})
            await kwargs["on_content_delta"]({"content_delta": "done"})
            return LLMResponse(content="done", total_tokens=10)

    async def receive(_delta):
        return None

    meter = Meter(
        Provider(), {"calls": 0, "tokens": 0, "max_calls": 1, "max_tokens": 100}
    )
    await meter.chat(model="fake", on_content_delta=receive)
    assert meter.calls[0]["first_content_s"] is not None
    assert meter.calls[0]["first_content_s"] >= meter.calls[0]["first_delta_s"]
