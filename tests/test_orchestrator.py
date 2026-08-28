"""Orchestrator behaviour against a real schema."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.channels import ContactResult
from app.config import Settings
from app.constants import ActionType, CaseStatus
from app.models import Customer, RecoveryAction, RecoveryCase
from app.orchestrator import run_once

NOW = datetime(2026, 8, 20, 8, 30, tzinfo=UTC)  # 14:00 IST, inside the window


def settings(**overrides) -> Settings:
    base = {
        "min_recoverable_amount_paise": 5_000,
        "retry_backoff_hours": 24,
        "contact_window_start_hour": 9,
        "contact_window_end_hour": 21,
        "contact_timezone": "Asia/Kolkata",
        "worker_batch_size": 50,
    }
    return Settings(**{**base, **overrides})


class SpyChannel:
    """Records who it was asked to contact."""

    name = "spy"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.contacted: list[int] = []

    async def initiate(self, case: RecoveryCase, customer: Customer) -> ContactResult:
        if self.fail:
            raise RuntimeError("telephony provider unavailable")
        self.contacted.append(case.id)
        return ContactResult(channel=self.name, reference="room_1")


class SpyRazorpay:
    """Stands in for the payment-link API.

    The link intervention is the cheap half of the ladder, so most ticks now
    touch Razorpay. A test of the loop must not leave the machine.
    """

    def __init__(self, fail: bool = False, tokens: list[dict] | None = None) -> None:
        self.fail = fail
        self.created: list[dict] = []
        self.tokens = tokens if tokens is not None else []
        self.charges: list[dict] = []

    async def fetch_customer_tokens(self, customer_id: str) -> dict:
        return {"items": self.tokens}

    async def create_order(self, payload: dict) -> dict:
        return {"id": f"order_{len(self.charges) + 1}", **payload}

    async def create_recurring_payment(self, payload: dict) -> dict:
        self.charges.append(payload)
        return {"razorpay_payment_id": f"pay_recurring_{len(self.charges)}"}

    async def create_payment_link(self, payload: dict) -> dict:
        if self.fail:
            raise RuntimeError("razorpay unavailable")
        self.created.append(payload)
        return {
            "id": f"plink_{len(self.created)}",
            "short_url": f"https://rzp.io/i/{len(self.created)}",
            "reference_id": payload["reference_id"],
        }


async def seed(session, *, case_kwargs=None, customer_kwargs=None) -> RecoveryCase:
    customer = Customer(
        **{
            "razorpay_customer_id": "cust_1",
            "phone": "+919000000000",
            **(customer_kwargs or {}),
        }
    )
    session.add(customer)
    case = RecoveryCase(
        **{
            "razorpay_payment_id": "pay_1",
            "razorpay_customer_id": customer.razorpay_customer_id,
            "razorpay_subscription_id": "sub_1",
            "original_amount": 49_900,
            "status": CaseStatus.OPEN,
            **(case_kwargs or {}),
        }
    )
    session.add(case)
    await session.commit()
    await session.refresh(case)
    return case


async def actions_for(session, case_id: int) -> list[str]:
    rows = await session.execute(
        select(RecoveryAction.action_type)
        .where(RecoveryAction.recovery_case_id == case_id)
        .order_by(RecoveryAction.id)
    )
    return list(rows.scalars())


async def test_a_fresh_case_is_sent_a_link_rather_than_called(session):
    """The cheap intervention opens the ladder."""
    case = await seed(session)
    channel, razorpay = SpyChannel(), SpyRazorpay()

    result = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert result.as_dict()["linked"] == 1
    assert result.as_dict()["contacted"] == 0
    assert channel.contacted == []
    assert len(razorpay.created) == 1

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS
    assert case.attempt_count == 1
    assert case.last_attempt_at is not None
    assert case.payment_link_url == "https://rzp.io/i/1"
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.PAYMENT_LINK_CREATED,
    ]


async def test_a_second_attempt_is_escalated_to_a_call(session):
    """A link has already gone out and been ignored, so this one costs a call."""
    case = await seed(session, case_kwargs={"attempt_count": 1})
    channel, razorpay = SpyChannel(), SpyRazorpay()

    result = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert result.as_dict()["contacted"] == 1
    assert channel.contacted == [case.id]
    assert razorpay.created == []

    await session.refresh(case)
    assert case.attempt_count == 2
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.VOICE_CALL,
    ]


async def test_a_dead_instrument_is_called_without_wasting_a_link(session):
    """Root cause overrules the ladder: an expired card needs a conversation,
    not a link that leaves the mandate broken."""
    case = await seed(
        session,
        case_kwargs={"failure_source": "customer", "failure_reason_code": "card_expired"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert channel.contacted == [case.id]
    assert razorpay.created == []


async def test_our_own_outage_contacts_nobody(session):
    """A gateway failure is ours. The customer hears nothing about it."""
    case = await seed(session, case_kwargs={"failure_source": "gateway"})
    channel, razorpay = SpyChannel(), SpyRazorpay()

    result = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert result.as_dict()["waiting"] == 1
    assert channel.contacted == []
    assert razorpay.created == []
    await session.refresh(case)
    assert case.status == CaseStatus.OPEN


async def test_suppressed_customer_is_stopped_not_contacted(session):
    """The compliance rule, end to end through the loop."""
    case = await seed(session, customer_kwargs={"do_not_contact": True})
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED
    assert channel.contacted == []
    assert razorpay.created == []


async def test_backoff_prevents_a_second_contact_in_the_same_day(session):
    """The stopping-rule guarantee: one tick per minute must not mean one
    contact per minute."""
    case = await seed(session)
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)
    await run_once(
        session, channel, now=NOW + timedelta(minutes=1), settings=settings(), client=razorpay
    )

    await session.refresh(case)
    assert case.attempt_count == 1
    assert len(razorpay.created) == 1
    assert channel.contacted == []


async def test_case_is_stopped_once_attempts_are_exhausted(session):
    case = await seed(session, case_kwargs={"attempt_count": 3, "max_attempts": 3})
    channel, razorpay = SpyChannel(), SpyRazorpay()

    result = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert result.as_dict()["stopped"] == 1
    assert channel.contacted == []
    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.STOPPED,
    ]


async def test_stopped_case_is_not_picked_up_again(session):
    """STOP is permanent -- the bounded-workflow requirement."""
    await seed(session, case_kwargs={"attempt_count": 3, "max_attempts": 3})
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)
    second = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert second.considered == 0


async def test_waiting_case_writes_no_audit_noise(session):
    """A case outside the contact window is re-evaluated every tick; it must not
    append a row each time."""
    case = await seed(session)
    channel, razorpay = SpyChannel(), SpyRazorpay()
    midnight_ist = datetime(2026, 8, 20, 20, 0, tzinfo=UTC)

    for _ in range(3):
        result = await run_once(
            session, channel, now=midnight_ist, settings=settings(), client=razorpay
        )

    assert result.as_dict()["waiting"] == 1
    assert channel.contacted == []
    assert await actions_for(session, case.id) == []
    await session.refresh(case)
    assert case.status == CaseStatus.OPEN


async def test_channel_failure_does_not_burn_an_attempt(session):
    """An outage on our side must not consume the customer's attempt budget,
    but must still back off rather than hot-loop."""
    case = await seed(session, case_kwargs={"attempt_count": 1})
    channel, razorpay = SpyChannel(fail=True), SpyRazorpay()

    result = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert result.as_dict()["failed"] == 1
    await session.refresh(case)
    assert case.attempt_count == 1
    assert case.last_attempt_at is not None
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.VOICE_CALL,
    ]


async def test_a_failed_link_does_not_burn_an_attempt_either(session):
    """Razorpay being down is our problem, not the customer's budget."""
    case = await seed(session)
    channel, razorpay = SpyChannel(), SpyRazorpay(fail=True)

    result = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert result.as_dict()["failed"] == 1
    await session.refresh(case)
    assert case.attempt_count == 0
    assert case.last_attempt_at is not None
    assert case.payment_link_id is None
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.PAYMENT_LINK_CREATED,
    ]


async def test_channel_failure_does_not_abort_the_batch(session):
    """One bad case must not stop the other cases in the tick."""
    await seed(session, case_kwargs={"attempt_count": 1})
    await seed(
        session,
        case_kwargs={"razorpay_payment_id": "pay_2", "attempt_count": 1},
        customer_kwargs={"razorpay_customer_id": "cust_2"},
    )
    channel, razorpay = SpyChannel(fail=True), SpyRazorpay()

    result = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert result.considered == 2
    assert result.as_dict()["failed"] == 2


async def test_case_without_a_reachable_customer_is_stopped(session):
    case = await seed(session, customer_kwargs={"phone": None})
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED
    assert channel.contacted == []


async def test_halted_cases_are_worked_first(session):
    """Razorpay has given up on a halted subscription, so it is the most urgent
    thing in the queue."""
    await seed(session, case_kwargs={"razorpay_payment_id": "pay_old", "attempt_count": 1})
    halted = await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_halted",
            "halted_at": NOW - timedelta(hours=1),
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_2"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(
        session, channel, now=NOW, settings=settings(worker_batch_size=50), client=razorpay
    )

    assert channel.contacted[0] == halted.id


async def test_batch_size_bounds_the_tick(session):
    for i in range(3):
        await seed(
            session,
            case_kwargs={"razorpay_payment_id": f"pay_{i}"},
            customer_kwargs={"razorpay_customer_id": f"cust_{i}"},
        )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    result = await run_once(
        session, channel, now=NOW, settings=settings(worker_batch_size=2), client=razorpay
    )

    assert result.considered == 2


@pytest.mark.parametrize("amount", [4_999, 1])
async def test_small_debts_are_stopped_not_worked(session, amount):
    case = await seed(session, case_kwargs={"original_amount": amount})
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    await session.refresh(case)
    assert case.status == CaseStatus.STOPPED
    assert channel.contacted == []


# --- Parked cases must not own the queue ----------------------------------


async def test_cases_waiting_on_the_bank_do_not_crowd_out_the_batch(session):
    """The starvation regression.

    A case deferring to Razorpay's retry stays OPEN, and is among the oldest
    rows, so it sorts to the front of every batch. Once there are more of them
    than ``worker_batch_size``, a newer case is never claimed at all -- it is
    not delayed, it is never worked. Parking them with ``next_eligible_at``
    keeps them out of the claim until their grace window is up.
    """
    # Inside their grace window, so the policy parks them -- and they are
    # older than the fresh case, so they sort ahead of it.
    parked_since = NOW - timedelta(hours=2)
    for i in range(5):
        await seed(
            session,
            case_kwargs={
                "razorpay_payment_id": f"pay_stalled_{i}",
                "failure_source": "bank",
                "created_at": parked_since,
            },
            customer_kwargs={"razorpay_customer_id": f"cust_stalled_{i}"},
        )
    fresh = await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_fresh",
            "failure_source": "issuer",
            "failure_reason_code": "insufficient_funds",
            "created_at": NOW - timedelta(minutes=5),
        },
        customer_kwargs={"razorpay_customer_id": "cust_fresh"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    # First tick: the stalled cases fill the batch and get parked.
    await run_once(
        session, channel, now=NOW, settings=settings(worker_batch_size=5), client=razorpay
    )
    # Second tick: they are no longer eligible, so the fresh case is reachable.
    await run_once(
        session, channel, now=NOW, settings=settings(worker_batch_size=5), client=razorpay
    )

    await session.refresh(fresh)
    assert len(razorpay.created) == 1, "the fresh case never got worked"
    assert fresh.attempt_count == 1


async def test_a_parked_case_is_worked_once_its_grace_expires(session):
    """Parking must not become a second permanent terminal state."""
    case = await seed(
        session,
        case_kwargs={"failure_source": "bank", "created_at": NOW - timedelta(hours=1)},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)
    await session.refresh(case)
    assert case.next_eligible_at is not None
    assert razorpay.created == []

    # 72h on, which is 14:00 IST again -- inside the contact window, so the
    # only thing that could still hold the case is the grace period itself.
    later = NOW + timedelta(hours=72)
    await run_once(session, channel, now=later, settings=settings(), client=razorpay)

    await session.refresh(case)
    assert len(razorpay.created) == 1
    assert case.next_eligible_at is None


async def test_acting_on_a_case_clears_its_parking(session):
    case = await seed(session, case_kwargs={"next_eligible_at": NOW - timedelta(hours=1)})
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    await session.refresh(case)
    assert case.next_eligible_at is None
    assert case.attempt_count == 1


async def test_an_email_only_customer_is_recovered_by_link(session):
    """No phone is no longer the end of the case."""
    case = await seed(
        session, customer_kwargs={"phone": None, "email": "a@example.com"}
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS
    assert len(razorpay.created) == 1
    assert channel.contacted == []


# --- Mandate retry --------------------------------------------------------

LIVE_TOKEN = {
    "id": "token_live",
    "recurring": True,
    "method": "emandate",
    "used_at": 1_700_000_000,
    "expired_at": None,
    "max_amount": 999_900,
}


async def test_a_missing_funds_case_re_charges_the_mandate(session):
    case = await seed(
        session,
        case_kwargs={"failure_source": "issuer", "failure_reason_code": "insufficient_funds"},
        customer_kwargs={"email": "a@example.com"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay(tokens=[LIVE_TOKEN])

    result = await run_once(
        session,
        channel,
        now=NOW,
        settings=settings(mandate_retry_enabled=True),
        client=razorpay,
    )

    assert result.as_dict()["charged"] == 1
    assert len(razorpay.charges) == 1
    assert razorpay.charges[0]["token"] == "token_live"
    assert razorpay.charges[0]["amount"] == case.original_amount
    # Never both: a link as well would be two live ways to settle one debt.
    assert razorpay.created == []
    assert channel.contacted == []

    await session.refresh(case)
    assert case.attempt_count == 1
    assert await actions_for(session, case.id) == [
        ActionType.POLICY_DECISION,
        ActionType.MANDATE_RETRIED,
    ]


async def test_no_chargeable_mandate_does_not_burn_an_attempt(session):
    """Nothing reached the customer, so the next tick can still send a link."""
    case = await seed(
        session,
        case_kwargs={"failure_source": "issuer", "failure_reason_code": "insufficient_funds"},
        customer_kwargs={"email": "a@example.com"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay(tokens=[])

    result = await run_once(
        session,
        channel,
        now=NOW,
        settings=settings(mandate_retry_enabled=True),
        client=razorpay,
    )

    assert result.as_dict()["charged"] == 0
    assert razorpay.charges == []
    await session.refresh(case)
    assert case.attempt_count == 0
    assert case.last_attempt_at is not None


async def test_a_customer_without_an_email_is_not_charged(session):
    """Razorpay rejects a recurring charge with no email, so do not attempt it."""
    await seed(
        session,
        case_kwargs={"failure_source": "issuer", "failure_reason_code": "insufficient_funds"},
        customer_kwargs={"email": None},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay(tokens=[LIVE_TOKEN])

    await run_once(
        session,
        channel,
        now=NOW,
        settings=settings(mandate_retry_enabled=True),
        client=razorpay,
    )

    assert razorpay.charges == []


async def test_mandate_retry_stays_off_by_default(session):
    case = await seed(
        session,
        case_kwargs={"failure_source": "issuer", "failure_reason_code": "insufficient_funds"},
        customer_kwargs={"email": "a@example.com"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay(tokens=[LIVE_TOKEN])

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert razorpay.charges == []
    assert len(razorpay.created) == 1
    await session.refresh(case)
    assert case.attempt_count == 1


# --- Calling priority -----------------------------------------------------


async def test_a_big_enough_debt_outranks_a_higher_tier(session):
    """The point of scoring rather than strict tiering.

    A Rs 200 broken mandate is more urgent than a Rs 200 bank decline. It is
    not more urgent than a Rs 4,999 payment the customer was in the middle of
    making, and strict tiering would have called it first forever.
    """
    big = await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_big",
            "original_amount": 499_900,
            "created_at": NOW - timedelta(days=5),
            "failure_source": "issuer",
            "failure_reason_code": "insufficient_funds",
            "priority_tier": 2,
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_big"},
    )
    await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_small_mandate",
            "original_amount": 9_900,
            "created_at": NOW - timedelta(hours=1),
            "failure_source": "customer",
            "failure_reason_code": "mandate_revoked",
            "priority_tier": 1,
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_mandate"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert channel.contacted[0] == big.id


async def test_a_comparable_debt_does_not_outrank_a_higher_tier(session):
    """The other half: money has to be *decisively* larger to jump a tier, or
    the tiers would stop meaning anything. Same amount, tier still decides."""
    await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_attempted",
            "original_amount": 49_900,
            "created_at": NOW - timedelta(days=5),
            "priority_tier": 2,
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_attempted"},
    )
    mandate = await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_mandate",
            "original_amount": 49_900,
            "created_at": NOW - timedelta(hours=1),
            "priority_tier": 1,
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_mandate"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert channel.contacted[0] == mandate.id


async def test_the_bigger_debt_wins_inside_a_tier(session):
    """Nothing distinguishes them but the money, so take the money."""
    small = await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_small",
            "original_amount": 9_900,
            "created_at": NOW - timedelta(days=3),
            "priority_tier": 2,
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_small"},
    )
    big = await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_big",
            "original_amount": 249_900,
            "created_at": NOW - timedelta(hours=2),
            "priority_tier": 2,
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_big"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert channel.contacted == [big.id, small.id]


async def test_nothing_starves_inside_its_tier(session):
    """Same tier, same amount -- the older case still goes first."""
    older = await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_older",
            "created_at": NOW - timedelta(days=4),
            "priority_tier": 2,
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_older"},
    )
    await seed(
        session,
        case_kwargs={
            "razorpay_payment_id": "pay_newer",
            "created_at": NOW - timedelta(hours=3),
            "priority_tier": 2,
            "attempt_count": 1,
        },
        customer_kwargs={"razorpay_customer_id": "cust_newer"},
    )
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert channel.contacted[0] == older.id


# --- One person, several debts, one tick -----------------------------------


async def test_one_customer_with_several_debts_is_contacted_once(session):
    """The bug this exists for, end to end through the loop.

    Four cases opened against one subscription in two hours from repeated
    authorisation attempts. Each carried its own untouched attempt budget, so a
    single tick would have placed four calls to one person.
    """
    customer = Customer(
        razorpay_customer_id="cust_many", phone="+919000000000", email="m@example.com"
    )
    session.add(customer)
    for i in range(4):
        session.add(
            RecoveryCase(
                razorpay_payment_id=f"pay_attempt_{i}",
                razorpay_customer_id="cust_many",
                razorpay_invoice_id="inv_same",
                original_amount=49_900,
                status=CaseStatus.OPEN,
                attempt_count=1,
                max_attempts=3,
                created_at=NOW - timedelta(hours=4 - i),
            )
        )
    await session.commit()
    channel, razorpay = SpyChannel(), SpyRazorpay()

    result = await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)

    assert result.considered == 4
    assert len(channel.contacted) == 1, "the same person must not be rung once per debt"
    assert result.as_dict()["waiting"] == 3

    await session.refresh(customer)
    assert customer.last_contacted_at is not None


async def test_the_other_debts_are_worked_once_the_cooldown_lifts(session):
    """Held back, not abandoned."""
    customer = Customer(razorpay_customer_id="cust_two", phone="+919000000000")
    session.add(customer)
    for i in range(2):
        session.add(
            RecoveryCase(
                razorpay_payment_id=f"pay_two_{i}",
                razorpay_customer_id="cust_two",
                original_amount=49_900,
                status=CaseStatus.OPEN,
                attempt_count=1,
                max_attempts=3,
                created_at=NOW - timedelta(hours=2 - i),
            )
        )
    await session.commit()
    channel, razorpay = SpyChannel(), SpyRazorpay()

    await run_once(session, channel, now=NOW, settings=settings(), client=razorpay)
    assert len(channel.contacted) == 1

    await run_once(
        session, channel, now=NOW + timedelta(hours=25), settings=settings(), client=razorpay
    )
    assert len(channel.contacted) == 2
