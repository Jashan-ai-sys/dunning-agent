"""Calling priority: tiering, then scoring.

The tier rules are pure. The parity test at the bottom needs Postgres, because
its whole point is that the SQL rendering agrees with the Python one.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from app.models import RecoveryCase
from app.priority import (
    HALTED_BOOST,
    Priority,
    score,
    score_expression,
    tier_for,
    tier_from,
)


def case(**overrides) -> RecoveryCase:
    defaults = {
        "razorpay_payment_id": "pay_1",
        "original_amount": 49_900,
        "failure_source": None,
        "failure_reason_code": None,
        "failure_step": None,
    }
    return RecoveryCase(**{**defaults, **overrides})


@pytest.mark.parametrize("reason", ["mandate_revoked", "card_expired", "card_blocked"])
def test_a_broken_mandate_is_the_most_urgent_thing_in_the_queue(reason):
    """Every future charge fails too, not just this one, and no payment link
    fixes that."""
    assert tier_from("customer", reason, None) is Priority.MANDATE_BROKEN


@pytest.mark.parametrize(
    "step", ["payment_authentication", "payment_authorization", "PAYMENT_AUTHORIZATION"]
)
def test_a_customer_who_was_in_the_flow_comes_second(step):
    """They were sitting there trying to pay. Recent, demonstrated intent."""
    assert tier_from("gateway", None, step) is Priority.PAYMENT_ATTEMPTED


def test_a_charge_made_on_their_behalf_is_not_an_attempt_by_them():
    """payment_initiation is the recurring charge: nobody was watching."""
    assert tier_from("gateway", None, "payment_initiation") is Priority.BACKGROUND


def test_the_step_outranks_the_reason_code():
    """A bank-side reason still means they were present if the step says so."""
    assert tier_from("bank", None, "payment_authentication") is Priority.PAYMENT_ATTEMPTED


def test_a_broken_mandate_outranks_the_step():
    """A revoked mandate is tier 1 even though they were standing right there:
    a link would settle today and leave the subscription dead."""
    assert (
        tier_from("customer", "mandate_revoked", "payment_authentication")
        is Priority.MANDATE_BROKEN
    )


def test_missing_funds_counts_as_an_attempted_payment():
    assert tier_from("issuer", "insufficient_funds", None) is Priority.PAYMENT_ATTEMPTED


@pytest.mark.parametrize("source", ["gateway", "internal", "bank", "issuer"])
def test_bank_and_gateway_failures_sit_at_the_back(source):
    assert tier_from(source, None, None) is Priority.BACKGROUND


def test_an_undiagnosable_case_is_background_not_urgent():
    """Absence of information is not a reason to jump the queue."""
    assert tier_from(None, None, None) is Priority.BACKGROUND


def test_tier_for_reads_the_case_columns():
    assert tier_for(case(failure_reason_code="card_expired")) is Priority.MANDATE_BROKEN


def test_tiers_sort_in_the_intended_order():
    """Lower is more urgent -- the claim query orders ascending."""
    assert (
        Priority.MANDATE_BROKEN
        < Priority.PAYMENT_ATTEMPTED
        < Priority.CHECKOUT_ABANDONED
        < Priority.BACKGROUND
    )


# --- Scoring --------------------------------------------------------------


def test_a_big_enough_debt_outranks_a_higher_tier():
    """The behaviour strict tiering could not express."""
    assert score(Priority.PAYMENT_ATTEMPTED, 499_900) > score(Priority.MANDATE_BROKEN, 20_000)


def test_a_comparable_debt_does_not():
    """If any amount could jump a tier, the tiers would mean nothing."""
    assert score(Priority.MANDATE_BROKEN, 49_900) > score(Priority.PAYMENT_ATTEMPTED, 49_900)


def test_the_weights_mean_what_the_comment_says():
    """It should take roughly a 25x debt to climb one tier -- not 2x.

    Pinned because the tier gaps are the whole policy: widen them and money
    never wins, narrow them and the tiers stop mattering.
    """
    base = 20_000  # Rs 200
    assert score(Priority.PAYMENT_ATTEMPTED, base * 2) < score(Priority.MANDATE_BROKEN, base)
    assert score(Priority.PAYMENT_ATTEMPTED, base * 25) > score(Priority.MANDATE_BROKEN, base)


def test_within_a_tier_the_larger_debt_wins():
    assert score(Priority.BACKGROUND, 100_000) > score(Priority.BACKGROUND, 99_999)


def test_a_halted_subscription_is_worth_more_than_the_same_case_unhalted():
    """Razorpay has stopped retrying, so we are the only route left."""
    assert score(Priority.BACKGROUND, 49_900, halted=True) == pytest.approx(
        score(Priority.BACKGROUND, 49_900) * HALTED_BOOST
    )


def test_a_zero_or_negative_amount_does_not_blow_up():
    """log(0) is undefined and a negative amount should never reach here, but
    the queue must not be something a bad row can crash."""
    assert score(Priority.BACKGROUND, 0) == 0.0
    assert score(Priority.BACKGROUND, -500) == 0.0


def test_an_unknown_tier_falls_back_to_background():
    assert score(99, 49_900) == score(Priority.BACKGROUND, 49_900)


@pytest.mark.parametrize(
    "tier,amount,halted",
    [
        (1, 49_900, False),
        (2, 499_900, False),
        (3, 9_900, True),
        (4, 1, False),
        (4, 0, True),
    ],
)
async def test_the_sql_score_agrees_with_the_python_one(session, tier, amount, halted):
    """The guard against drift.

    The claim query orders in SQL; everything that explains or tests the order
    reasons in Python. Two renderings of one formula is a standing invitation
    for them to disagree about which customer gets called first.
    """
    case = RecoveryCase(
        razorpay_payment_id=f"pay_{tier}_{amount}_{halted}",
        original_amount=amount,
        priority_tier=tier,
        halted_at=datetime(2026, 8, 1, tzinfo=UTC) if halted else None,
    )
    session.add(case)
    await session.commit()

    from_sql = (
        await session.execute(
            select(score_expression()).where(RecoveryCase.id == case.id)
        )
    ).scalar_one()

    assert float(from_sql) == pytest.approx(score(tier, amount, halted=halted))
