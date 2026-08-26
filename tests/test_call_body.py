"""Building what the agent is told, from a case in the database.

The test that earns its place here is the unglamorous one: this code runs
*before* the transport exists, so an exception does not degrade a call, it
prevents one. A customer who picked up hears silence.
"""


from app.voice.call_body import load_call_body


async def test_a_dead_database_does_not_silence_the_agent(monkeypatch):
    """Docker down, Cloud SQL unreachable, credentials rotated -- the agent has
    lost the ability to personalise a call, not to make one."""

    class Refused:
        async def __aenter__(self):
            raise ConnectionRefusedError("[WinError 1225] refused")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("app.voice.call_body.SessionLocal", lambda: Refused())
    assert await load_call_body(1) is None


async def test_an_unknown_case_returns_none(session):
    assert await load_call_body(999_999) is None
