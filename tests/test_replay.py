"""The replay sweep -- the safety net Razorpay's own redelivery does not provide.

Once we return 200, Razorpay considers the event delivered. If our handler then
dies, only this sweep recovers it.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.config import Settings
from app.models import RecoveryCase, WebhookEvent
from app.razorpay.client import RazorpayError
from app.webhooks.processor import is_retryable, replay_unprocessed, requeue_dead
from tests.payloads import payment_failed_event


async def add_unprocessed(session, *, event_id: str, age_minutes: int, payload=None):
    event = WebhookEvent(
        razorpay_event_id=event_id,
        event="payment.failed",
        payload=payload or payment_failed_event(),
        signature_verified=True,
        created_at=datetime.now(UTC) - timedelta(minutes=age_minutes),
    )
    session.add(event)
    await session.commit()
    await session.refresh(event)
    return event


async def test_stale_unprocessed_event_is_replayed(session, fake_client, monkeypatch):
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", lambda: fake_client)
    event = await add_unprocessed(session, event_id="evt_stale", age_minutes=10)

    replayed = await replay_unprocessed()

    assert replayed == 1
    await session.refresh(event)
    assert event.processed_at is not None
    assert event.processing_error is None

    case = (await session.execute(select(RecoveryCase))).scalar_one()
    assert case.razorpay_payment_id == "pay_FAIL1"


async def test_in_flight_event_is_left_alone(session, fake_client, monkeypatch):
    """A just-arrived event is still being handled by its background task;
    replaying it now would double-process it."""
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", lambda: fake_client)
    event = await add_unprocessed(session, event_id="evt_fresh", age_minutes=0)

    assert await replay_unprocessed() == 0

    await session.refresh(event)
    assert event.processed_at is None


async def test_already_processed_events_are_not_replayed(session, fake_client, monkeypatch):
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", lambda: fake_client)
    event = await add_unprocessed(session, event_id="evt_done", age_minutes=10)
    event.processed_at = datetime.now(UTC)
    await session.commit()

    assert await replay_unprocessed() == 0


async def test_replay_is_idempotent(session, fake_client, monkeypatch):
    """Replaying twice must not open a second case for the same payment."""
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", lambda: fake_client)
    await add_unprocessed(session, event_id="evt_stale", age_minutes=10)

    assert await replay_unprocessed() == 1
    assert await replay_unprocessed() == 0

    cases = (await session.execute(select(RecoveryCase))).scalars().all()
    assert len(cases) == 1


async def test_failing_handler_records_the_error_and_stays_queued(session, monkeypatch):
    """A handler that keeps failing must leave a diagnosable trail, not vanish."""

    class BrokenClient:
        async def fetch_invoice(self, invoice_id: str) -> dict:
            raise RuntimeError("razorpay api down")

    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", BrokenClient)
    event = await add_unprocessed(session, event_id="evt_broken", age_minutes=10)

    assert await replay_unprocessed() == 1

    await session.refresh(event)
    assert event.processed_at is None
    assert "razorpay api down" in event.processing_error
    assert (await session.execute(select(RecoveryCase))).scalars().all() == []


# --- Dead letters ---------------------------------------------------------
#
# A queue that only ever retries does not become resilient, it jams. The sweep
# reads the oldest unprocessed envelopes first, so one that fails
# deterministically sits at the front of every pass -- and because record_event
# dedupes on the event id, Razorpay redelivering it changes nothing. The sweep
# is the only retry path there is.


class PoisonClient:
    """Razorpay refusing the request itself. Replaying produces the same 400."""

    async def fetch_invoice(self, invoice_id: str) -> dict:
        raise RazorpayError(400, '{"error": {"description": "invoice is malformed"}}')


class FlakyClient:
    """Razorpay having a bad minute. Worth trying again."""

    async def fetch_invoice(self, invoice_id: str) -> dict:
        raise RazorpayError(503, "service unavailable")


def with_max_attempts(monkeypatch, attempts: int) -> None:
    monkeypatch.setattr(
        "app.webhooks.processor.get_settings",
        lambda: Settings(webhook_max_attempts=attempts),
    )


async def bury(session, *, event_id: str, age_minutes: int = 10) -> WebhookEvent:
    event = await add_unprocessed(session, event_id=event_id, age_minutes=age_minutes)
    event.dead_at = datetime.now(UTC)
    event.attempt_count = 5
    await session.commit()
    await session.refresh(event)
    return event


@pytest.mark.parametrize(
    "exc,expected",
    [
        (RazorpayError(400, "bad request"), False),
        (RazorpayError(401, "unauthorised"), False),
        (RazorpayError(404, "no such invoice"), False),
        (RazorpayError(408, "request timeout"), True),
        (RazorpayError(429, "rate limited"), True),
        (RazorpayError(500, "server error"), True),
        (RazorpayError(503, "unavailable"), True),
        (RuntimeError("something else entirely"), True),
    ],
)
def test_retryability_is_decided_by_what_razorpay_said(exc, expected):
    """A 4xx is a statement about the request, not about the moment -- except
    for rate limiting and timeouts, which are 4xx by number and transient in
    fact. Anything unclassifiable is retryable, because dead-lettering a
    recoverable payment event is the worse mistake of the two."""
    assert is_retryable(exc) is expected


async def test_a_poison_event_is_buried_on_the_first_attempt(session, monkeypatch):
    """There is nothing to learn from repeating a request Razorpay rejected."""
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", PoisonClient)
    with_max_attempts(monkeypatch, 5)
    event = await add_unprocessed(session, event_id="evt_poison", age_minutes=10)

    await replay_unprocessed()

    await session.refresh(event)
    assert event.dead_at is not None
    assert event.attempt_count == 1
    assert "invoice is malformed" in event.processing_error


async def test_a_transient_failure_is_retried_until_its_budget_runs_out(session, monkeypatch):
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", FlakyClient)
    with_max_attempts(monkeypatch, 3)
    event = await add_unprocessed(session, event_id="evt_flaky", age_minutes=10)

    for _ in range(2):
        await replay_unprocessed()
    await session.refresh(event)
    assert event.dead_at is None, "still inside its budget"
    assert event.attempt_count == 2

    await replay_unprocessed()
    await session.refresh(event)
    assert event.dead_at is not None
    assert event.attempt_count == 3


async def test_a_dead_event_is_not_replayed_again(session, monkeypatch):
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", PoisonClient)
    event = await bury(session, event_id="evt_buried")

    assert await replay_unprocessed() == 0

    await session.refresh(event)
    assert event.attempt_count == 5, "not retried, so the count did not move"


async def test_dead_events_do_not_starve_the_queue(session, fake_client, monkeypatch):
    """The regression the dead_at filter exists for.

    Dead envelopes are the oldest unprocessed rows, so they sort to the front
    of every sweep. Once there are more of them than the sweep's limit, no
    newer event is ever replayed again -- and nothing else retries it.
    """
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", lambda: fake_client)
    for i in range(3):
        await bury(session, event_id=f"evt_dead_{i}", age_minutes=60)
    live = await add_unprocessed(session, event_id="evt_live", age_minutes=10)

    # A limit the dead events alone would fill.
    assert await replay_unprocessed(limit=3) == 1

    await session.refresh(live)
    assert live.processed_at is not None
    assert (await session.execute(select(RecoveryCase))).scalar_one() is not None


async def test_requeueing_gives_a_dead_event_a_full_budget_again(session, fake_client, monkeypatch):
    """The other half of a dead-letter queue: burying is not dropping. Once the
    bug that killed them is fixed, the events are still here."""
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", lambda: fake_client)
    event = await bury(session, event_id="evt_revive")

    assert await requeue_dead() == 1

    await session.refresh(event)
    assert event.dead_at is None
    assert event.attempt_count == 0

    assert await replay_unprocessed() == 1
    await session.refresh(event)
    assert event.processed_at is not None


async def test_requeueing_leaves_healthy_events_alone(session):
    live = await add_unprocessed(session, event_id="evt_untouched", age_minutes=10)

    assert await requeue_dead() == 0

    await session.refresh(live)
    assert live.attempt_count == 0
