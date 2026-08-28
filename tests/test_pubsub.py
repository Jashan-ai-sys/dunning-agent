"""Push delivery, and the guarantees it must not weaken.

Push is an accelerator on the cron sweep, not a replacement. Every test here
asks whether losing it, or being lied to by it, costs anything more than
latency.
"""

import base64

import pytest
from fastapi.testclient import TestClient

from app import events
from app.config import Settings
from app.main import app

PUSH_URL = "/webhooks/pubsub/event"


def push_body(payload: str) -> dict:
    return {"message": {"data": base64.b64encode(payload.encode()).decode()}}


@pytest.fixture(autouse=True)
def _reset():
    events.reset_for_tests()
    yield
    events.reset_for_tests()


# --- the publisher ---------------------------------------------------------


async def test_no_topic_means_no_publish_and_no_error(monkeypatch):
    """Cron-only is the default and a supported configuration."""
    monkeypatch.setattr(events, "_client", lambda: None)
    assert await events.announce_event(7, "payment.failed") is False


async def test_a_publish_failure_is_swallowed(monkeypatch):
    """The sweep still has the row. Failing the request because the
    accelerator broke would trade a slow path for no path."""

    class Broken:
        def publish(self, *a, **k):
            raise ConnectionError("pubsub unreachable")

    monkeypatch.setattr(events, "_client", lambda: Broken())
    monkeypatch.setattr(
        events, "get_settings", lambda: Settings(pubsub_topic="projects/p/topics/t")
    )
    assert await events.announce_event(7, "payment.failed") is False


async def test_a_successful_publish_reports_true(monkeypatch):
    published = {}

    class Future:
        def result(self, timeout=None):
            return "msg-1"

    class Ok:
        def publish(self, topic, data, **attrs):
            published["topic"] = topic
            published["data"] = data
            published["attrs"] = attrs
            return Future()

    monkeypatch.setattr(events, "_client", lambda: Ok())
    monkeypatch.setattr(
        events, "get_settings", lambda: Settings(pubsub_topic="projects/p/topics/t")
    )

    assert await events.announce_event(42, "payment.failed") is True
    assert published["data"] == b"42", "the id travels, not the payload"
    assert published["attrs"]["event"] == "payment.failed"


# --- the push endpoint -----------------------------------------------------


def test_push_is_refused_when_not_configured():
    """An endpoint that dispatches handlers must not run unauthenticated
    because a setting was forgotten."""
    with TestClient(app) as client:
        r = client.post(PUSH_URL, json=push_body("1"))
    assert r.status_code == 503


def test_push_without_a_token_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "app.webhooks.pubsub_router.get_settings",
        lambda: Settings(pubsub_push_service_account="pusher@example.iam.gserviceaccount.com"),
    )
    with TestClient(app) as client:
        r = client.post(PUSH_URL, json=push_body("1"))
    assert r.status_code == 401


def test_a_token_that_does_not_verify_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "app.webhooks.pubsub_router.get_settings",
        lambda: Settings(pubsub_push_service_account="pusher@example.iam.gserviceaccount.com"),
    )
    with TestClient(app) as client:
        r = client.post(
            PUSH_URL, json=push_body("1"), headers={"Authorization": "Bearer not-a-jwt"}
        )
    assert r.status_code == 401


def test_a_token_from_the_wrong_principal_is_rejected(monkeypatch):
    """Verifying the signature is not enough -- any Google account can mint a
    valid token. It has to be *our* subscription."""
    monkeypatch.setattr(
        "app.webhooks.pubsub_router.get_settings",
        lambda: Settings(pubsub_push_service_account="pusher@example.iam.gserviceaccount.com"),
    )
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda *a, **k: {"email": "someone-else@evil.example", "email_verified": True},
    )
    with TestClient(app) as client:
        r = client.post(
            PUSH_URL, json=push_body("1"), headers={"Authorization": "Bearer x"}
        )
    assert r.status_code == 403


def test_an_authorised_push_dispatches_the_envelope(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "app.webhooks.pubsub_router.get_settings",
        lambda: Settings(pubsub_push_service_account="pusher@example.iam.gserviceaccount.com"),
    )
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda *a, **k: {"email": "pusher@example.iam.gserviceaccount.com", "email_verified": True},
    )

    async def fake_process(pk):
        seen.append(pk)

    monkeypatch.setattr("app.webhooks.pubsub_router.process_event", fake_process)

    with TestClient(app) as client:
        r = client.post(
            PUSH_URL, json=push_body("99"), headers={"Authorization": "Bearer x"}
        )
    assert r.status_code == 204
    assert seen == [99]


@pytest.mark.parametrize("payload", ["not-a-number", ""])
def test_an_unreadable_message_is_acknowledged_not_retried(monkeypatch, payload):
    """Pub/Sub redelivers until acknowledged. A permanently malformed message
    would otherwise come back forever."""
    monkeypatch.setattr(
        "app.webhooks.pubsub_router.get_settings",
        lambda: Settings(pubsub_push_service_account="pusher@example.iam.gserviceaccount.com"),
    )
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda *a, **k: {"email": "pusher@example.iam.gserviceaccount.com", "email_verified": True},
    )
    body = push_body(payload) if payload else {"message": {}}
    with TestClient(app) as client:
        r = client.post(PUSH_URL, json=body, headers={"Authorization": "Bearer x"})
    assert r.status_code == 204
