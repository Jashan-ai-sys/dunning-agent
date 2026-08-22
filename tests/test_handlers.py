"""Behaviour of the event handlers against a real Postgres schema."""

from sqlalchemy import func, select

from app.constants import ActionType, CaseStatus
from app.models import Customer, Payment, RecoveryAction, RecoveryCase, Subscription
from app.webhooks.handlers import (
    handle_payment_captured,
    handle_payment_failed,
    handle_subscription_charged,
    handle_subscription_halted,
    handle_subscription_pending,
)
from tests.payloads import (
    invoice_entity,
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
