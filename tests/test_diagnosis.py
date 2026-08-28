"""Root-cause classification. Pure unit tests -- no database, no network."""

import pytest

from app.diagnosis import RootCause, classify, diagnose
from app.models import RecoveryCase


def case(**overrides) -> RecoveryCase:
    defaults = {
        "razorpay_payment_id": "pay_1",
        "original_amount": 49_900,
        "failure_source": None,
        "failure_reason_code": None,
    }
    return RecoveryCase(**{**defaults, **overrides})


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("insufficient_funds", RootCause.CUSTOMER_FUNDS),
        ("card_expired", RootCause.CUSTOMER_INSTRUMENT),
        ("mandate_revoked", RootCause.CUSTOMER_INSTRUMENT),
        ("invalid_otp", RootCause.CUSTOMER_ACTION),
        ("gateway_timeout", RootCause.TRANSIENT),
        ("international_transaction_not_allowed", RootCause.BANK_DECLINE),
    ],
)
def test_a_known_reason_classifies_precisely(reason, expected):
    assert classify(None, reason) is expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("customer", RootCause.CUSTOMER_ACTION),
        ("issuer", RootCause.BANK_DECLINE),
        ("bank", RootCause.BANK_DECLINE),
        ("gateway", RootCause.TRANSIENT),
        ("internal", RootCause.TRANSIENT),
        ("business", RootCause.CONFIGURATION),
    ],
)
def test_source_classifies_when_the_reason_is_unknown(source, expected):
    assert classify(source, "some_reason_we_have_never_seen") is expected


def test_a_known_reason_outranks_its_source():
    """insufficient_funds arrives with source 'issuer', but the issuer did not
    decline anything -- there was no money. The reason is the truer answer."""
    assert classify("issuer", "insufficient_funds") is RootCause.CUSTOMER_FUNDS


def test_razorpay_casing_does_not_change_the_answer():
    assert classify("  ISSUER  ", "  Insufficient_Funds  ") is RootCause.CUSTOMER_FUNDS


@pytest.mark.parametrize("source,reason", [(None, None), ("", ""), ("moon", "cheese")])
def test_nothing_usable_is_unknown_rather_than_a_guess(source, reason):
    assert classify(source, reason) is RootCause.UNKNOWN


def test_diagnose_reads_the_case_fields():
    assert diagnose(case(failure_source="business")) is RootCause.CONFIGURATION
