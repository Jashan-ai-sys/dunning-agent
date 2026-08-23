"""LiveKitChannel guards and metadata. No network involved."""

import json

import pytest

from app.config import Settings
from app.models import Customer, RecoveryCase
from app.voice.dispatch import LiveKitChannel

CONFIGURED = {
    "livekit_url": "wss://example.livekit.cloud",
    "livekit_api_key": "APIkey",
    "livekit_api_secret": "secret",
    "livekit_sip_trunk_id": "ST_trunk123",
    "company_name": "Acme",
}


def configure(monkeypatch, **overrides) -> None:
    settings = Settings(**{**CONFIGURED, **overrides})
    monkeypatch.setattr("app.voice.dispatch.get_settings", lambda: settings)


def make_case(**kwargs) -> RecoveryCase:
    defaults = {
        "id": 7,
        "razorpay_payment_id": "pay_1",
        "original_amount": 49_900,
        "failure_reason": "insufficient funds",
    }
    return RecoveryCase(**{**defaults, **kwargs})


def make_customer(**kwargs) -> Customer:
    defaults = {
        "razorpay_customer_id": "cust_1",
        "name": "Asha Rao",
        "phone": "+919000000000",
        "preferred_language": "hinglish",
    }
    return Customer(**{**defaults, **kwargs})


def test_refuses_to_construct_without_livekit_credentials(monkeypatch):
    configure(monkeypatch, livekit_url="", livekit_api_key="", livekit_api_secret="")
    with pytest.raises(RuntimeError, match="LiveKit is not configured"):
        LiveKitChannel()


def test_refuses_to_construct_without_a_sip_trunk(monkeypatch):
    """LiveKit Cloud alone cannot reach the phone network. Failing loudly here
    beats dispatching an agent into a room nobody ever joins."""
    configure(monkeypatch, livekit_sip_trunk_id="")
    with pytest.raises(RuntimeError, match="SIP_TRUNK_ID"):
        LiveKitChannel()


def test_room_name_is_derived_from_the_case(monkeypatch):
    configure(monkeypatch)
    assert LiveKitChannel.room_name(make_case(id=42)) == "recovery-42"


def test_metadata_carries_a_spoken_amount_not_a_raw_numeral(monkeypatch):
    """The agent reads this aloud. 49900 spoken as digits is a digit crawl, and
    a Latin "Rs" inside a Devanagari sentence makes Hindi voices stumble."""
    configure(monkeypatch)
    metadata = json.loads(LiveKitChannel()._metadata(make_case(), make_customer()))
    # The fixture customer prefers Hinglish, so the unit word follows suit.
    assert metadata["amount_spoken"] == "499 rupaye"
    assert metadata["amount_paise"] == 49_900


def test_spoken_amount_follows_the_customers_language(monkeypatch):
    configure(monkeypatch)
    hindi = json.loads(
        LiveKitChannel()._metadata(make_case(), make_customer(preferred_language="hi"))
    )
    assert hindi["amount_spoken"] == "499 रुपये"


def test_large_amounts_are_collapsed_before_they_reach_the_voice(monkeypatch):
    """500000 read as digits is the failure the Blostem backend wrote a
    normalizer to stop."""
    configure(monkeypatch)
    metadata = json.loads(
        LiveKitChannel()._metadata(
            make_case(original_amount=500_000_00), make_customer(preferred_language="hi")
        )
    )
    assert metadata["amount_spoken"] == "5 लाख रुपये"


def test_metadata_carries_everything_the_flow_needs(monkeypatch):
    configure(monkeypatch)
    metadata = json.loads(LiveKitChannel()._metadata(make_case(), make_customer()))
    assert metadata["recovery_case_id"] == 7
    assert metadata["phone"] == "+919000000000"
    assert metadata["customer_name"] == "Asha Rao"
    assert metadata["preferred_language"] == "hinglish"
    assert metadata["company_name"] == "Acme"
    assert metadata["failure_reason"] == "insufficient funds"


def test_metadata_falls_back_when_the_customer_has_no_name(monkeypatch):
    """A missing name must not become the literal string 'None' on a call."""
    configure(monkeypatch)
    metadata = json.loads(LiveKitChannel()._metadata(make_case(), make_customer(name=None)))
    assert metadata["customer_name"] == "there"


def test_metadata_falls_back_when_the_failure_reason_is_unknown(monkeypatch):
    configure(monkeypatch)
    metadata = json.loads(
        LiveKitChannel()._metadata(make_case(failure_reason=None), make_customer())
    )
    assert metadata["failure_reason"] == "the bank declined it"
