"""The two events that make a silent mandate failure detectable.

A mandate registration authorises as a zero-amount eMandate payment. If it
worked, `subscription.authenticated` follows. If it did not, nothing follows --
Razorpay has no "mandate failed" webhook, and signals it by omission.

That combination is the most valuable signal in dunning and the easiest to
miss: money arrived, every dashboard looks healthy, and every future charge on
that subscription is already broken.

These tests pin the two halves. Noticing the *absence* is a sweep, not a
handler, and is not built yet -- see `Known gaps`.
"""


from sqlalchemy import select

from app.models import RecoveryCase, Subscription
from app.webhooks.handlers import EVENT_HANDLERS


def _authorized_emandate(payment_id: str = "pay_mandate_1") -> dict:
    """A real shape: the zero-amount registration Razorpay authorises."""
    return {
        "event": "payment.authorized",
        "razorpay_event_id": "evt_auth_1",
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "amount": 0,
                    "currency": "INR",
                    "status": "authorized",
                    "method": "emandate",
                    "token_id": "token_abc",
                    "order_id": "order_abc",
                    "invoice_id": None,
                    "contact": "+919000000000",
                    "bank": "HDFC",
                }
            }
        },
    }


def test_both_events_are_registered():
    assert "payment.authorized" in EVENT_HANDLERS
    assert "subscription.authenticated" in EVENT_HANDLERS


async def test_a_zero_amount_mandate_registration_opens_no_case(session):
    """Nothing is owed. A registration is not a debt, and a case for one would
    be chased by a system whose whole job is chasing debts."""
    event = _authorized_emandate()
    await EVENT_HANDLERS["payment.authorized"](session, event, None)
    await session.commit()

    cases = (
        await session.execute(
            select(RecoveryCase).where(RecoveryCase.razorpay_payment_id == "pay_mandate_1")
        )
    ).scalars().all()
    assert cases == [], "a zero-amount mandate registration must not open a case"


async def test_an_ordinary_authorized_payment_opens_no_case(session):
    """`payment.authorized` fires for every uncaptured payment. None of them is
    a recovery case -- only a *failure* is."""
    event = _authorized_emandate("pay_ordinary_1")
    event["payload"]["payment"]["entity"].update(amount=49900, method="card", token_id=None)
    await EVENT_HANDLERS["payment.authorized"](session, event, None)
    await session.commit()

    cases = (
        await session.execute(
            select(RecoveryCase).where(RecoveryCase.razorpay_payment_id == "pay_ordinary_1")
        )
    ).scalars().all()
    assert cases == []


async def test_subscription_authenticated_records_the_live_mandate(session):
    """This event's presence is what distinguishes a mandate that worked from
    one that failed silently, so it has to be durably recorded."""
    event = {
        "event": "subscription.authenticated",
        "razorpay_event_id": "evt_authn_1",
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_live_1",
                    "customer_id": None,      # no client call in this test
                    "plan_id": "plan_1",
                    "status": "authenticated",
                }
            }
        },
    }
    await EVENT_HANDLERS["subscription.authenticated"](session, event, None)
    await session.commit()

    row = (
        await session.execute(
            select(Subscription).where(Subscription.razorpay_subscription_id == "sub_live_1")
        )
    ).scalar_one_or_none()
    assert row is not None, "the authenticated subscription must be recorded"
    assert row.status == "authenticated"
