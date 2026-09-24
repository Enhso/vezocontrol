import asyncio
from collections import deque

import pytest

from forecasting_tools import GeneralLlm

from main import FreeTierPacedLlm

WINDOW = FreeTierPacedLlm.WINDOW_SECONDS
BUDGET = FreeTierPacedLlm.TOKENS_PER_MINUTE


@pytest.fixture(autouse=True)
def empty_bookings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FreeTierPacedLlm, "_bookings", deque())


def test_first_call_starts_now() -> None:
    assert FreeTierPacedLlm.book(8_000, now=100.0) == 100.0


def test_calls_within_budget_share_the_window() -> None:
    first = FreeTierPacedLlm.book(BUDGET // 2, now=100.0)
    second = FreeTierPacedLlm.book(BUDGET // 2, now=101.0)

    assert (first, second) == (100.0, 101.0)


def test_template_burst_of_five_forecasts_is_spread_one_per_window() -> None:
    starts = [FreeTierPacedLlm.book(8_000, now=0.0) for _ in range(5)]

    assert starts == [0.0, WINDOW, 2 * WINDOW, 3 * WINDOW, 4 * WINDOW]


def test_overflow_waits_for_oldest_booking_to_leave_window() -> None:
    FreeTierPacedLlm.book(8_000, now=0.0)
    FreeTierPacedLlm.book(2_000, now=10.0)

    assert FreeTierPacedLlm.book(4_000, now=20.0) == WINDOW


def test_call_larger_than_budget_goes_alone() -> None:
    FreeTierPacedLlm.book(1_000, now=0.0)

    assert FreeTierPacedLlm.book(BUDGET * 2, now=5.0) == WINDOW
    assert FreeTierPacedLlm.book(1_000, now=5.0) == 2 * WINDOW


def test_expired_bookings_are_pruned() -> None:
    FreeTierPacedLlm.book(BUDGET, now=0.0)

    assert FreeTierPacedLlm.book(BUDGET, now=WINDOW + 1) == WINDOW + 1
    assert len(FreeTierPacedLlm._bookings) == 1


def test_call_sleeps_until_booked_start_then_calls_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    sent: list[str] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def fake_call(self: GeneralLlm, prompt: str) -> str:
        sent.append(prompt)
        return "response"

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", fake_call)
    monkeypatch.setattr(FreeTierPacedLlm, "input_to_tokens", lambda self, p: BUDGET)
    monkeypatch.setattr("main.time.monotonic", lambda: 0.0)
    llm = FreeTierPacedLlm(model="openrouter/google/gemma-4-31b-it:free")

    async def two_calls() -> list[str]:
        return [
            await llm._mockable_direct_call_to_model("first"),
            await llm._mockable_direct_call_to_model("second"),
        ]

    assert asyncio.run(two_calls()) == ["response", "response"]
    assert sleeps == [0.0, WINDOW]
    assert sent == ["first", "second"]
