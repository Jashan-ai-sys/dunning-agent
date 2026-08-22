"""End-to-end tests through the HTTP endpoint, including signature enforcement
and Razorpay's at-least-once redelivery."""

import json

import httpx
import pytest
from sqlalchemy import func, select

from app.main import app
from app.models import RecoveryCase, WebhookEvent
from app.razorpay.signature import compute_signature
from tests.conftest import TEST_WEBHOOK_SECRET
from tests.payloads import payment_failed_event


@pytest.fixture
async def client(session, fake_client, monkeypatch):
    # The background processor builds its own client; keep it off the network.
    monkeypatch.setattr("app.webhooks.processor.RazorpayClient", lambda: fake_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _post_args(payload: dict, *, secret: str = TEST_WEBHOOK_SECRET, event_id: str = "evt_1"):
    body = json.dumps(payload).encode()
    return {
        "content": body,
        "headers": {
            "X-Razorpay-Signature": compute_signature(body, secret),
            "X-Razorpay-Event-Id": event_id,
            "Content-Type": "application/json",
        },
    }


async def test_valid_event_is_accepted_and_processed(client, session):
    response = await client.post("/webhooks/razorpay", **_post_args(payment_failed_event()))

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}

    stored = (await session.execute(select(WebhookEvent))).scalar_one()
    assert stored.event == "payment.failed"
    assert stored.processed_at is not None
    assert stored.processing_error is None

    case = (await session.execute(select(RecoveryCase))).scalar_one()
    assert case.razorpay_payment_id == "pay_FAIL1"


async def test_bad_signature_is_rejected_and_stores_nothing(client, session):
    response = await client.post(
        "/webhooks/razorpay", **_post_args(payment_failed_event(), secret="wrong_secret")
    )

    assert response.status_code == 401
    count = (await session.execute(select(func.count(WebhookEvent.id)))).scalar_one()
    assert count == 0


async def test_missing_signature_is_rejected(client, session):
    body = json.dumps(payment_failed_event()).encode()
    response = await client.post("/webhooks/razorpay", content=body)

    assert response.status_code == 401
    count = (await session.execute(select(func.count(WebhookEvent.id)))).scalar_one()
    assert count == 0


async def test_redelivered_event_is_deduplicated(client, session):
    args = _post_args(payment_failed_event())
    first = await client.post("/webhooks/razorpay", **args)
    second = await client.post("/webhooks/razorpay", **args)

    assert first.json() == {"status": "accepted"}
    assert second.json() == {"status": "duplicate"}

    events = (await session.execute(select(func.count(WebhookEvent.id)))).scalar_one()
    cases = (await session.execute(select(func.count(RecoveryCase.id)))).scalar_one()
    assert events == 1
    assert cases == 1


async def test_malformed_body_is_rejected(client, session):
    body = b"not json"
    response = await client.post(
        "/webhooks/razorpay",
        content=body,
        headers={"X-Razorpay-Signature": compute_signature(body, TEST_WEBHOOK_SECRET)},
    )
    assert response.status_code == 400


async def test_unhandled_event_is_stored_and_retired(client, session):
    """We subscribe to more events than we act on; those must not pile up on
    the replay queue."""
    payload = {"entity": "event", "event": "subscription.activated", "payload": {}}
    response = await client.post("/webhooks/razorpay", **_post_args(payload))

    assert response.status_code == 200
    stored = (await session.execute(select(WebhookEvent))).scalar_one()
    assert stored.event == "subscription.activated"
    assert stored.processed_at is not None

    cases = (await session.execute(select(func.count(RecoveryCase.id)))).scalar_one()
    assert cases == 0
