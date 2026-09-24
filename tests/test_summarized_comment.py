import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from forecasting_tools import ForecastBot

from main import SummerTemplateBot2026, cap_words

RATIONALES = "# FORECASTS\n## R1: Forecaster 1 Reasoning\nBase rate is low."


class FakeReport:
    """Stands in for a ForecastReport: only what the comment path touches."""

    def __init__(self, explanation: str = "# SUMMARY\nfull report") -> None:
        self.explanation = explanation
        self.question = SimpleNamespace(question_text="Will it happen?")
        self.prediction = 0.3
        self.forecast_rationales = RATIONALES
        self.first_rationale = "## R1: Forecaster 1 Reasoning\n" + "word " * 150
        self.published: list[str] = []

    @staticmethod
    def make_readable_prediction(prediction: float) -> str:
        return f"{prediction:.0%}"

    def model_copy(self, update: dict[str, Any]) -> "FakeReport":
        copy = FakeReport(update["explanation"])
        copy.published = self.published
        return copy

    async def publish_report_to_metaculus(self, metaculus_client: object) -> None:
        self.published.append(self.explanation)


def make_bot(
    monkeypatch: pytest.MonkeyPatch, report: FakeReport, llm_invoke: AsyncMock
) -> SummerTemplateBot2026:
    async def parent_run(self: ForecastBot, question: object) -> FakeReport:
        return report

    monkeypatch.setattr(ForecastBot, "_run_individual_question", parent_run)
    bot = SummerTemplateBot2026(
        publish_reports_to_metaculus=False,
        publish_summarized_comments=True,
        llms={"default": "openrouter/google/gemma-4-31b-it:free"},
    )
    monkeypatch.setattr(
        bot, "get_llm", lambda purpose, kind: SimpleNamespace(invoke=llm_invoke)
    )
    return bot


def test_cap_words_keeps_short_text() -> None:
    assert cap_words("  one two three  ", 5) == "one two three"


def test_cap_words_trims_long_text() -> None:
    assert cap_words("a b c d e f", 3) == "a b c ..."


GOOD_SUMMARY = (
    "The status quo favours No: the rule has held for a decade and no bill is "
    "scheduled. Recent statements lean slightly toward change, so the forecast "
    "sits a little above the base rate."
)


def test_posts_summary_and_returns_full_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = FakeReport()
    invoke = AsyncMock(return_value=GOOD_SUMMARY)
    bot = make_bot(monkeypatch, report, invoke)

    returned = asyncio.run(bot._run_individual_question(object()))

    assert returned is report
    assert report.explanation == "# SUMMARY\nfull report"
    assert report.published == [f"# Rationale\n{GOOD_SUMMARY}"]
    prompt = invoke.await_args.args[0]
    assert "Will it happen?" in prompt and "30%" in prompt and RATIONALES in prompt


def test_long_summary_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    report = FakeReport()
    bot = make_bot(monkeypatch, report, AsyncMock(return_value="x " * 300))

    asyncio.run(bot._run_individual_question(object()))

    body = report.published[0].split("\n", 1)[1]
    assert len(body.split()) == SummerTemplateBot2026.COMMENT_MAX_WORDS + 1
    assert body.endswith(" ...")


@pytest.mark.parametrize(
    "error",
    [
        ValueError("empty answer"),
        openai.APIError(
            "provider 500", request=httpx.Request("POST", "https://x"), body=None
        ),
    ],
)
def test_failed_summary_falls_back_to_capped_first_rationale(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    report = FakeReport()
    bot = make_bot(monkeypatch, report, AsyncMock(side_effect=error))

    asyncio.run(bot._run_individual_question(object()))

    comment = report.published[0]
    assert comment.startswith("# Rationale\nword word")
    assert "Forecaster 1 Reasoning" not in comment
    assert comment.endswith(" ...")


def test_nothing_posted_when_publishing_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    report = FakeReport()
    invoke = AsyncMock()
    bot = make_bot(monkeypatch, report, invoke)
    bot.publish_summarized_comments = False

    asyncio.run(bot._run_individual_question(object()))

    assert report.published == []
    invoke.assert_not_awaited()


def test_truncated_summary_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    report = FakeReport()
    invoke = AsyncMock(side_effect=["The forecast is", GOOD_SUMMARY])
    bot = make_bot(monkeypatch, report, invoke)

    asyncio.run(bot._run_individual_question(object()))

    assert report.published == [f"# Rationale\n{GOOD_SUMMARY}"]
    assert invoke.await_count == 2


def test_truncated_twice_falls_back_to_first_rationale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = FakeReport()
    bot = make_bot(monkeypatch, report, AsyncMock(return_value="The forecast is"))

    asyncio.run(bot._run_individual_question(object()))

    assert report.published[0].startswith("# Rationale\nword word")
