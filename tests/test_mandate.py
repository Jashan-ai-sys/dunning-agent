"""Choosing a mandate to charge. Pure unit tests -- no database, no network.

Every rule here is a way a real charge would fail at Razorpay, and one of them
(``max_amount``) would fail after the customer had been told we were taking the
money.
"""

import pytest

from app.mandate import usable_token

NOW_EPOCH = 1_800_000_000


def token(**overrides) -> dict:
    defaults = {
        "id": "token_1",
        "recurring": True,
        "method": "card",
        "used_at": NOW_EPOCH - 86_400,
        "expired_at": NOW_EPOCH + 86_400,
    }
    return {**defaults, **overrides}


def test_a_healthy_token_is_chargeable():
    assert usable_token([token()], amount=49_900, now_epoch=NOW_EPOCH)["id"] == "token_1"


def test_no_tokens_is_not_an_error():
    """A customer with no mandate on file is ordinary; the caller sends a link."""
    assert usable_token([], amount=49_900, now_epoch=NOW_EPOCH) is None


def test_a_non_recurring_token_is_never_charged():
    """A card saved for convenience is not permission to bill it."""
    assert usable_token([token(recurring=False)], amount=1, now_epoch=NOW_EPOCH) is None


def test_an_expired_token_is_skipped():
    assert usable_token(
        [token(expired_at=NOW_EPOCH - 1)], amount=1, now_epoch=NOW_EPOCH
    ) is None


def test_a_token_expiring_exactly_now_is_skipped():
    assert usable_token(
        [token(expired_at=NOW_EPOCH)], amount=1, now_epoch=NOW_EPOCH
    ) is None


def test_a_token_with_no_expiry_is_allowed():
    """E-mandates often omit it; absent is not the same as expired."""
    result = usable_token([token(expired_at=None)], amount=1, now_epoch=NOW_EPOCH)
    assert result is not None


@pytest.mark.parametrize("status", ["initiated", "cancelled", "rejected"])
def test_an_unconfirmed_mandate_is_skipped(status):
    assert usable_token(
        [token(recurring_details={"status": status})], amount=1, now_epoch=NOW_EPOCH
    ) is None


def test_a_confirmed_mandate_is_allowed():
    result = usable_token(
        [token(recurring_details={"status": "confirmed"})], amount=1, now_epoch=NOW_EPOCH
    )
    assert result is not None


def test_a_charge_above_the_mandate_ceiling_is_not_attempted():
    """E-mandates carry a limit agreed at authorisation. A plan upgraded since
    can sit above it, and Razorpay would reject the charge."""
    assert usable_token(
        [token(max_amount=10_000)], amount=49_900, now_epoch=NOW_EPOCH
    ) is None


def test_a_charge_exactly_at_the_ceiling_is_allowed():
    result = usable_token([token(max_amount=49_900)], amount=49_900, now_epoch=NOW_EPOCH)
    assert result is not None


def test_the_most_recently_used_mandate_wins():
    """The instrument they actually transact on beats one they abandoned."""
    stale = token(id="token_old", used_at=NOW_EPOCH - 900_000)
    fresh = token(id="token_new", used_at=NOW_EPOCH - 3_600)
    assert usable_token([stale, fresh], amount=1, now_epoch=NOW_EPOCH)["id"] == "token_new"


def test_a_token_that_has_never_been_used_still_qualifies():
    result = usable_token([token(used_at=None)], amount=1, now_epoch=NOW_EPOCH)
    assert result is not None
