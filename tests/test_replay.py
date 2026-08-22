"""The replay sweep -- the safety net Razorpay's own redelivery does not provide.

Once we return 200, Razorpay considers the event delivered. If our handler then
dies, only this sweep recovers it.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.models import RecoveryCase, WebhookEvent
from app.webhooks.processor import replay_unprocessed
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
