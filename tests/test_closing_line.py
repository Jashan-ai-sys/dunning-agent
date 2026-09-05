"""The call must not hang up mid-sentence.

Regression for a live call: the customer agreed to pay, the payment link went
out, and the line went dead about a second into the closing sentence. The graph
sets `finished` the instant it reaches a terminal node, but the closing line is
generated *after* that -- a Vertex round trip, then TTS, then several seconds of
Hindi. The old `sleep(2.0)` expired in the middle of it.
"""

import asyncio
import types

import pytest

from app.voice.pipecat_agent import (
    CLOSING_GRACE,
    CLOSING_START_TIMEOUT,
    CLOSING_TOTAL_TIMEOUT,
    _let_the_closing_line_finish,
)


def _session() -> types.SimpleNamespace:
    return types.SimpleNamespace(speaking=asyncio.Event(), stopped_speaking=asyncio.Event())


async def test_it_waits_for_a_closing_line_longer_than_the_old_two_seconds():
    """The exact failure: a closing line that outlasts the old fixed sleep."""
    session = _session()

    async def speak_for(seconds: float) -> None:
        session.speaking.set()
        await asyncio.sleep(seconds)
        session.speaking.clear()
        session.stopped_speaking.set()

    speech = asyncio.create_task(speak_for(2.5))
    started = asyncio.get_running_loop().time()
    await _let_the_closing_line_finish(session)
    elapsed = asyncio.get_running_loop().time() - started
    await speech

    assert elapsed > 2.5, "hung up before the closing line finished"
    assert elapsed < 2.5 + CLOSING_GRACE + 0.5


async def test_it_does_not_hang_on_a_closing_line_that_never_starts():
    """A TTS failure must not hold a customer on a finished call."""
    session = _session()
    started = asyncio.get_running_loop().time()
    await _let_the_closing_line_finish(session)
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed >= CLOSING_START_TIMEOUT
    assert elapsed < CLOSING_START_TIMEOUT + 1.0


async def test_it_gives_up_on_a_closing_line_that_never_ends():
    """Bounded at the top end too -- a stuck TTS cannot keep the line open."""
    session = _session()
    session.speaking.set()  # started, and never stops

    async def unstick() -> None:
        await asyncio.sleep(0.2)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.voice.pipecat_agent.CLOSING_TOTAL_TIMEOUT", 0.3)
        started = asyncio.get_running_loop().time()
        await _let_the_closing_line_finish(session)
        elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 1.0, "did not give up on a closing line that never ended"


def test_the_ceiling_is_above_a_realistic_closing_line():
    """Guards the numbers themselves. The pay_now line is one sentence with an
    amount in it -- comfortably under ten seconds of speech."""
    assert CLOSING_TOTAL_TIMEOUT >= 10.0
    assert CLOSING_START_TIMEOUT >= 2.0
