"""Behaviour of the event handlers against a real Postgres schema."""

from sqlalchemy import func, select

from app.constants import ActionType, CaseSource, CaseStatus
from app.models import Customer, Payment, RecoveryAction, RecoveryCase, Subscription
from app.priority import Priority
from app.webhooks.handlers import (
    handle_order_paid,
    handle_payment_captured,
    handle_payment_failed,
    handle_subscription_charged,
    handle_subscription_halted,
    handle_subscription_pending,
)
from tests.payloads import (
    invoice_entity,
    order_paid_event,
    payment_captured_event,
    payment_failed_event,
    subscription_charged_event,
    subscription_halted_event,
    subscription_pending_event,
)


async def _cases(session) -> list[RecoveryCase]:
    return list((await session.execute(select(RecoveryCase))).scalars())


async def _actions(session, case_id: int) -> list[str]:
    rows = await session.execute(
        select(RecoveryAction.action_type)
        .where(RecoveryAction.recovery_case_id == case_id)
        .order_by(RecoveryAction.id)
    )
    return list(rows.scalars())


async def test_failed_subscription_payment_opens_a_case(session, fake_client):
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()

    cases = await _cases(session)
    assert len(cases) == 1
    case = cases[0]
    assert case.status == CaseStatus.OPEN
    assert case.razorpay_payment_id == "pay_FAIL1"
    assert case.razorpay_subscription_id == "sub_1"
    assert case.razorpay_customer_id == "cust_1"
    assert case.original_amount == 49900
    assert case.failure_code == "BAD_REQUEST_ERROR"
    assert await _actions(session, case.id) == [ActionType.CASE_OPENED]


async def test_failure_hydrates_customer_and_subscription(session, fake_client):
    """The payment entity has no customer_id, so both rows must come from the
    invoice lookup plus follow-up fetches."""
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()

    customer = (await session.execute(select(Customer))).scalar_one()
    assert customer.razorpay_customer_id == "cust_1"
    assert customer.phone == "+919000000000"
    assert customer.preferred_language == "hi"

    subscription = (await session.execute(select(Subscription))).scalar_one()
    assert subscription.razorpay_subscription_id == "sub_1"
    assert ("invoice", "inv_1") in fake_client.calls


async def test_replayed_failure_does_not_duplicate_the_case(session, fake_client):
    """Razorpay delivers at-least-once; a second delivery must be a no-op."""
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()

    assert len(await _cases(session)) == 1
    total_actions = (await session.execute(select(func.count(RecoveryAction.id)))).scalar_one()
    assert total_actions == 1


async def test_one_off_failure_records_payment_but_opens_no_case(session, fake_client):
    """A checkout failure with no invoice is not a subscription recovery case."""
    await handle_payment_failed(
        session, payment_failed_event(payment_id="pay_ONEOFF", invoice_id=None), fake_client
    )
    await session.commit()

    assert await _cases(session) == []
    payment = (await session.execute(select(Payment))).scalar_one()
    assert payment.razorpay_payment_id == "pay_ONEOFF"
    assert payment.status == "failed"
    assert payment.razorpay_subscription_id is None


async def test_pending_appends_to_an_open_case(session, fake_client):
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    await handle_subscription_pending(session, subscription_pending_event(), fake_client)
    await session.commit()

    case = (await _cases(session))[0]
    assert await _actions(session, case.id) == [
        ActionType.CASE_OPENED,
        ActionType.SUBSCRIPTION_PENDING,
    ]


async def test_halted_stamps_the_escalation_signal(session, fake_client):
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    await handle_subscription_halted(session, subscription_halted_event(), fake_client)
    await session.commit()

    case = (await _cases(session))[0]
    assert case.halted_at is not None
    assert ActionType.SUBSCRIPTION_HALTED in await _actions(session, case.id)


async def test_pending_without_a_case_is_tolerated(session, fake_client):
    """We may see the subscription go pending before the failed payment."""
    await handle_subscription_pending(session, subscription_pending_event(), fake_client)
    await session.commit()

    assert await _cases(session) == []
    subscription = (await session.execute(select(Subscription))).scalar_one()
    assert subscription.status == "pending"


async def test_subscription_charged_recovers_the_open_case(session, fake_client):
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    await handle_subscription_charged(session, subscription_charged_event(), fake_client)
    await session.commit()

    case = (await _cases(session))[0]
    assert case.status == CaseStatus.RECOVERED
    assert case.recovered_payment_id == "pay_OK1"
    assert case.recovered_amount == 49900
    assert case.recovered_at is not None
    assert ActionType.PAYMENT_CAPTURED in await _actions(session, case.id)


async def test_recovery_is_counted_once_across_both_success_events(session, fake_client):
    """subscription.charged and payment.captured both fire for one recovery.
    Double-counting here would inflate the headline metric."""
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    await handle_subscription_charged(session, subscription_charged_event(), fake_client)
    await session.commit()
    await handle_payment_captured(session, payment_captured_event(), fake_client)
    await session.commit()

    case = (await _cases(session))[0]
    assert case.status == CaseStatus.RECOVERED
    captured = [a for a in await _actions(session, case.id) if a == ActionType.PAYMENT_CAPTURED]
    assert len(captured) == 1


async def test_payment_link_note_attributes_recovery_to_its_case(session, fake_client):
    """The Phase 4 attribution path: a link payment carries the case id and is
    not tied to the subscription invoice."""
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    case_id = (await _cases(session))[0].id

    await handle_payment_captured(
        session,
        payment_captured_event(
            payment_id="pay_LINK1",
            invoice_id=None,
            notes={"recovery_case_id": str(case_id)},
        ),
        fake_client,
    )
    await session.commit()

    case = (await _cases(session))[0]
    assert case.status == CaseStatus.RECOVERED
    assert case.recovered_payment_id == "pay_LINK1"


async def test_unrelated_capture_does_not_close_a_case(session, fake_client):
    """A successful payment on a different subscription must leave ours open."""
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()

    fake_client.invoices["inv_other"] = invoice_entity(
        "inv_other", subscription_id="sub_other", customer_id="cust_2"
    )
    await handle_payment_captured(
        session,
        payment_captured_event(payment_id="pay_OTHER", invoice_id="inv_other"),
        fake_client,
    )
    await session.commit()

    case = (await _cases(session))[0]
    assert case.status == CaseStatus.OPEN


# --- Abandoned checkout ---------------------------------------------------


async def _known_customer(session, phone="+919000000000", email="a@example.com"):
    customer = Customer(razorpay_customer_id="cust_known", phone=phone, email=email)
    session.add(customer)
    await session.commit()
    return customer


async def test_a_failed_checkout_by_a_known_customer_opens_a_case(session, fake_client):
    """Somebody reached the payment page, tried, and it broke under them. Until
    now this was logged and dropped for having no subscription behind it."""
    await _known_customer(session)

    await handle_payment_failed(
        session, payment_failed_event(invoice_id=None), fake_client
    )
    await session.commit()

    cases = await _cases(session)
    assert len(cases) == 1
    assert cases[0].source == CaseSource.CHECKOUT
    assert cases[0].razorpay_order_id == "order_1"
    assert cases[0].priority_tier == Priority.CHECKOUT_ABANDONED
    assert ActionType.CHECKOUT_ABANDONED in await _actions(session, cases[0].id)


async def test_a_new_checkout_case_is_parked_through_its_grace_window(session, fake_client):
    """A customer who retries with another card two minutes later has not
    abandoned anything. Chasing them mid-purchase would be absurd."""
    await _known_customer(session)

    await handle_payment_failed(
        session, payment_failed_event(invoice_id=None), fake_client
    )
    await session.commit()

    case = (await _cases(session))[0]
    assert case.next_eligible_at is not None
    assert case.next_eligible_at > case.created_at


async def test_a_stranger_who_abandons_a_checkout_is_not_chased(session, fake_client):
    """The order carries no customer and the payment only a bare email and
    phone. Manufacturing a customer record for someone we have no relationship
    with is a consent decision, not a technical one."""
    await handle_payment_failed(
        session, payment_failed_event(invoice_id=None), fake_client
    )
    await session.commit()

    assert await _cases(session) == []
    # The payment is still recorded -- detected, just not actionable.
    assert (await session.execute(select(func.count(Payment.id)))).scalar_one() == 1


async def test_paying_the_order_later_closes_the_checkout_case(session, fake_client):
    """The desirable outcome: they retried on their own inside the grace
    window and the agent never had to ring."""
    await _known_customer(session)
    await handle_payment_failed(
        session, payment_failed_event(invoice_id=None), fake_client
    )
    await session.commit()
    case = (await _cases(session))[0]

    await handle_order_paid(session, order_paid_event(), fake_client)
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.RECOVERED
    assert case.recovered_amount == 49900
    assert ActionType.PAYMENT_CAPTURED in await _actions(session, case.id)


async def test_an_order_we_have_no_case_for_is_ignored(session, fake_client):
    """Most paid orders are ordinary business, not recoveries."""
    await handle_order_paid(session, order_paid_event(order_id="order_unknown"), fake_client)
    await session.commit()

    assert await _cases(session) == []


async def test_a_replayed_checkout_failure_does_not_open_a_second_case(session, fake_client):
    await _known_customer(session)
    for _ in range(2):
        await handle_payment_failed(
            session, payment_failed_event(invoice_id=None), fake_client
        )
        await session.commit()

    assert len(await _cases(session)) == 1


# --- One debt, one case ----------------------------------------------------


async def test_a_second_failure_on_one_invoice_does_not_open_a_second_case(session, fake_client):
    """Every retry against an invoice arrives with a fresh payment id.

    Keying only on that opened four cases for one Rs 499 debt here, each with
    its own untouched contact budget. A repeat failure is more evidence about a
    debt we are already chasing.
    """
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    first = (await _cases(session))[0]

    await handle_payment_failed(
        session, payment_failed_event(payment_id="pay_RETRY_2"), fake_client
    )
    await session.commit()

    assert len(await _cases(session)) == 1
    assert ActionType.REPEAT_FAILURE in await _actions(session, first.id)


async def test_the_repeat_records_what_actually_failed(session, fake_client):
    """The second attempt can fail for a different reason than the first, and
    the trail has to show that even though no new case opens."""
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    case = (await _cases(session))[0]

    await handle_payment_failed(
        session, payment_failed_event(payment_id="pay_RETRY_3"), fake_client
    )
    await session.commit()

    rows = await session.execute(
        select(RecoveryAction.metadata_json)
        .where(RecoveryAction.recovery_case_id == case.id)
        .where(RecoveryAction.action_type == ActionType.REPEAT_FAILURE)
    )
    meta = rows.scalar_one()
    assert meta["razorpay_payment_id"] == "pay_RETRY_3"
    assert meta["error_reason"] == "insufficient_funds"


async def test_a_closed_case_does_not_swallow_a_new_failure(session, fake_client):
    """Dedupe applies to debts still being chased. Once a case is closed, a
    fresh failure on that invoice is genuinely new work."""
    await handle_payment_failed(session, payment_failed_event(), fake_client)
    await session.commit()
    case = (await _cases(session))[0]
    case.status = CaseStatus.RECOVERED
    await session.commit()

    await handle_payment_failed(
        session, payment_failed_event(payment_id="pay_AFTER_CLOSE"), fake_client
    )
    await session.commit()

    assert len(await _cases(session)) == 2
