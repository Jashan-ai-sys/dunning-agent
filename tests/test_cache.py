"""The shared cache, and the limits on what it may decide.

Redis is a fast path, never the authority. Every test here is really asking
the same question: when the cache is absent, wrong, or on fire, does the
service still behave exactly as it did before Redis existed?
"""

import pytest

from app import cache


@pytest.fixture(autouse=True)
def _reset():
    cache.reset_for_tests()
    yield
    cache.reset_for_tests()


class Boom:
    """A Redis that is up enough to be called and fails at everything."""

    async def set(self, *a, **k):
        raise ConnectionError("redis is on fire")

    async def exists(self, *a, **k):
        raise ConnectionError("redis is on fire")

    async def delete(self, *a, **k):
        raise ConnectionError("redis is on fire")


class Fake:
    def __init__(self):
        self.keys: dict[str, str] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True

    async def exists(self, key):
        return 1 if key in self.keys else 0

    async def delete(self, key):
        self.keys.pop(key, None)


# --- unconfigured is a supported configuration -----------------------------


async def test_without_redis_every_claim_succeeds(monkeypatch):
    """No Redis is the default. A claim must not become a silent no-op that
    stops work the durable rule would have allowed."""
    monkeypatch.setattr(cache, "_redis", lambda: None)
    assert await cache.claim("anything", 60) is True
    assert await cache.seen("anything", 60) is False


async def test_without_redis_nothing_is_ever_seen(monkeypatch):
    monkeypatch.setattr(cache, "_redis", lambda: None)
    assert await cache.seen_only("cooldown:cust_1") is False


# --- a broken Redis must not break the service -----------------------------


async def test_a_failing_claim_falls_through(monkeypatch):
    """Failing open means the database still decides. Failing closed would
    drop a payment event because a cache was unreachable."""
    monkeypatch.setattr(cache, "_redis", lambda: Boom())
    assert await cache.claim("webhook:evt_1", 60) is True
    assert await cache.seen("webhook:evt_1", 60) is False


async def test_a_failing_read_reports_nothing_seen(monkeypatch):
    """An unreadable cooldown must not suppress a call that policy allows."""
    monkeypatch.setattr(cache, "_redis", lambda: Boom())
    assert await cache.seen_only("cooldown:cust_1") is False


async def test_a_failing_delete_is_swallowed(monkeypatch):
    monkeypatch.setattr(cache, "_redis", lambda: Boom())
    await cache.forget("webhook:evt_1")  # must not raise


# --- the behaviour it exists for -------------------------------------------


async def test_the_first_claim_wins_and_the_second_does_not(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(cache, "_redis", lambda: fake)
    assert await cache.claim("webhook:evt_1", 60) is True
    assert await cache.claim("webhook:evt_1", 60) is False


async def test_seen_is_the_inverse_of_claim(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(cache, "_redis", lambda: fake)
    assert await cache.seen("webhook:evt_9", 60) is False   # first sighting
    assert await cache.seen("webhook:evt_9", 60) is True    # redelivery


async def test_reading_a_cooldown_does_not_create_it(monkeypatch):
    """Checking whether someone was recently called must not itself count as
    calling them -- that would spend a quiet period nobody used."""
    fake = Fake()
    monkeypatch.setattr(cache, "_redis", lambda: fake)
    assert await cache.seen_only("cooldown:cust_1") is False
    assert fake.keys == {}


async def test_forgetting_a_claim_lets_it_be_taken_again(monkeypatch):
    """A claim whose durable write failed has to be released, or a retry of a
    failed insert looks like a duplicate and the event is dropped."""
    fake = Fake()
    monkeypatch.setattr(cache, "_redis", lambda: fake)
    await cache.claim("webhook:evt_2", 60)
    await cache.forget("webhook:evt_2")
    assert await cache.claim("webhook:evt_2", 60) is True
