"""Payment links: the recovery instrument, and how a payment gets attributed."""

import pytest
from sqlalchemy import select

from app.config import Settings
from app.constants import ActionType, CaseStatus
from app.models import Customer, RecoveryAction, RecoveryCase
from app.payment_links import (
    build_payload,
    case_id_from_reference,
    create_recovery_link,
    reference_id_for,
)
from app.webhooks.handlers import handle_payment_captured, handle_payment_link_paid
from tests.payloads import event, payment_entity

SETTINGS = Settings(company_name="Acme", payment_link_expiry_hours=48)


class FakeLinkClient:
    """Records what would have been sent to Razorpay."""

    def __init__(self, link_id: str = "plink_1", notify_fails: bool = False) -> None:
        self.link_id = link_id
        self.payloads: list[dict] = []
        self.notified: list[tuple[str, str]] = []
        self.notify_fails = notify_fails

    async def create_payment_link(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {
            "id": self.link_id,
            "short_url": f"https://rzp.io/i/{self.link_id}",
            "reference_id": payload["reference_id"],
            "status": "created",
        }

    async def notify_payment_link(self, link_id: str, medium: str) -> dict:
        if self.notify_fails:
            raise RuntimeError("razorpay is having a day")
        self.notified.append((link_id, medium))
        return {"success": True}


async def seed(session, **case_kwargs) -> tuple[RecoveryCase, Customer]:
    customer = Customer(
        razorpay_customer_id="cust_1",
        name="Asha Rao",
        phone="+919000000000",
        email="asha@example.com",
    )
    session.add(customer)
    case = RecoveryCase(
        **{
            "razorpay_payment_id": "pay_FAIL1",
            "razorpay_customer_id": "cust_1",
            "razorpay_subscription_id": "sub_1",
            "original_amount": 49_900,
            "status": CaseStatus.IN_PROGRESS,
            **case_kwargs,
        }
    )
    session.add(case)
    await session.commit()
    await session.refresh(case)
    return case, customer


# --- reference ids -----------------------------------------------------


def test_reference_id_round_trips():
    case = RecoveryCase(id=42, razorpay_payment_id="pay_1", original_amount=1)
    assert case_id_from_reference(reference_id_for(case)) == 42


def test_reference_id_fits_razorpay_limit():
    """Razorpay caps reference_id at 40 characters."""
    case = RecoveryCase(id=9_999_999_999, razorpay_payment_id="pay_1", original_amount=1)
    assert len(reference_id_for(case)) <= 40


@pytest.mark.parametrize("value", [None, "", "order_123", "recovery-", "recovery-abc"])
def test_foreign_reference_ids_are_ignored(value):
    """Someone else's payment link must never credit one of our cases."""
    assert case_id_from_reference(value) is None


# --- payload -----------------------------------------------------------


def test_payload_carries_both_attribution_keys(session=None):
    case = RecoveryCase(id=7, razorpay_payment_id="pay_1", original_amount=49_900)
    customer = Customer(razorpay_customer_id="c", name="A", phone="+91900", email="a@b.c")
    payload = build_payload(case, customer, settings=SETTINGS, expire_at=123)

    # Attempt-suffixed: Razorpay reserves a reference_id account-wide forever,
    # so keying on the case alone made the second link a case ever needs a 400.
    assert payload["reference_id"] == "recovery-7-0"
    assert payload["notes"]["recovery_case_id"] == "7"


def test_payload_amount_is_paise_not_rupees():
    """Razorpay takes the smallest currency unit. Converting here would charge
    a hundredth of the debt."""
    case = RecoveryCase(id=1, razorpay_payment_id="pay_1", original_amount=49_900)
    payload = build_payload(
        case, Customer(razorpay_customer_id="c"), settings=SETTINGS, expire_at=1
    )
    assert payload["amount"] == 49_900


def test_payload_omits_customer_block_when_nothing_is_known():
    """Razorpay rejects an empty customer object."""
    case = RecoveryCase(id=1, razorpay_payment_id="pay_1", original_amount=100)
    payload = build_payload(
        case, Customer(razorpay_customer_id="c"), settings=SETTINGS, expire_at=1
    )
    assert "customer" not in payload
    assert payload["notify"] == {"sms": False, "email": False}


# --- creation ----------------------------------------------------------


async def test_creating_a_link_records_it_on_the_case(session):
    case, customer = await seed(session)
    client = FakeLinkClient()

    link = await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()

    assert link.short_url.startswith("https://rzp.io/i/")
    await session.refresh(case)
    assert case.payment_link_id == "plink_1"
    assert case.payment_link_url == link.short_url
    assert case.payment_link_sent_at is not None

    actions = await session.execute(
        select(RecoveryAction.action_type).where(RecoveryAction.recovery_case_id == case.id)
    )
    assert list(actions.scalars()) == [ActionType.PAYMENT_LINK_CREATED]


async def test_a_second_request_reuses_the_existing_link(session):
    """Two links for one debt risks charging the customer twice."""
    case, customer = await seed(session)
    client = FakeLinkClient()

    first = await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()
    second = await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()

    assert first.id == second.id
    assert len(client.payloads) == 1

    # One row per call, but only the first one actually created anything. The
    # second is a resend -- the orchestrator burns an attempt on the strength
    # of that row, so it must exist and must not claim a link was created.
    rows = await session.execute(
        select(RecoveryAction.metadata_json).order_by(RecoveryAction.id)
    )
    created, resent = list(rows.scalars())
    assert created.get("created") is not False
    assert resent["created"] is False
    assert resent["resent_via"] == ["sms", "email"]


async def test_reusing_a_link_still_delivers_it(session):
    """Razorpay notifies once, at creation.

    Without a resend, a customer called a second time hears the agent say it is
    sending a link and receives nothing -- which is what happened on a real
    call. A second *link* would risk charging them twice; a second SMS only
    repeats what they just asked to be told.
    """
    case, customer = await seed(session)
    client = FakeLinkClient()

    await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()
    assert client.notified == []  # creation notifies on its own

    await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()

    assert client.notified == [("plink_1", "sms"), ("plink_1", "email")]


async def test_a_resend_is_attempted_only_where_we_can_reach_them(session):
    case, customer = await seed(session)
    customer.email = None
    await session.commit()
    client = FakeLinkClient()

    await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()
    await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()

    assert client.notified == [("plink_1", "sms")]


async def test_a_failed_resend_does_not_break_the_call(session):
    """They still have the first message, and the agent reads the URL aloud."""
    case, customer = await seed(session)
    client = FakeLinkClient()

    await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()

    client.notify_fails = True
    link = await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()

    assert link is not None
    assert link.id == "plink_1"


async def test_link_creation_does_not_close_the_case(session):
    """Sending a link is not recovering money."""
    case, customer = await seed(session)
    await create_recovery_link(session, case, customer, FakeLinkClient(), settings=SETTINGS)
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS
    assert case.recovered_amount is None


# --- attribution -------------------------------------------------------


def link_paid_event(reference_id="recovery-1", notes=None, payment_id="pay_LINK1"):
    return event(
        "payment_link.paid",
        {
            "payment_link": {
                "entity": {
                    "id": "plink_1",
                    "reference_id": reference_id,
                    "notes": notes if notes is not None else {},
                    "status": "paid",
                }
            },
            "payment": {
                "entity": payment_entity(payment_id, status="captured", invoice_id=None)
            },
        },
    )


async def test_link_payment_is_attributed_by_reference_id(session, fake_client):
    case, _ = await seed(session)
    await handle_payment_link_paid(
        session, link_paid_event(reference_id=f"recovery-{case.id}"), fake_client
    )
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.RECOVERED
    assert case.recovered_payment_id == "pay_LINK1"
    assert case.recovered_amount == 49_900


async def test_link_payment_falls_back_to_notes(session, fake_client):
    """Belt and braces: if reference_id is absent, notes still credit the case."""
    case, _ = await seed(session)
    await handle_payment_link_paid(
        session,
        link_paid_event(reference_id=None, notes={"recovery_case_id": str(case.id)}),
        fake_client,
    )
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.RECOVERED


async def test_notes_as_empty_list_does_not_crash(session, fake_client):
    """Razorpay's own docs show notes arriving as [] rather than {}."""
    case, _ = await seed(session)
    await handle_payment_link_paid(
        session, link_paid_event(reference_id=None, notes=[]), fake_client
    )
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS  # unattributed, but no crash


async def test_someone_elses_payment_link_is_ignored(session, fake_client):
    case, _ = await seed(session)
    await handle_payment_link_paid(session, link_paid_event(reference_id="order_999"), fake_client)
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS


async def test_payment_captured_with_list_notes_does_not_crash(session, fake_client):
    """The same hazard on the payment.captured path."""
    case, _ = await seed(session)
    payload = event(
        "payment.captured",
        {"payment": {"entity": payment_entity("pay_X", status="captured", invoice_id=None)}},
    )
    payload["payload"]["payment"]["entity"]["notes"] = []

    await handle_payment_captured(session, payload, fake_client)
    await session.commit()

    await session.refresh(case)
    assert case.status == CaseStatus.IN_PROGRESS


def test_a_second_attempt_gets_a_fresh_reference():
    """Razorpay reserves reference_id account-wide and forever. Keyed on the
    case alone, a case needing a second link is refused outright:

        400 payment link with given reference_id: recovery-1 already exists
    """
    case = RecoveryCase(id=1, razorpay_payment_id="p", original_amount=1000)
    case.attempt_count = 0
    first = reference_id_for(case)
    case.attempt_count = 1
    assert reference_id_for(case) != first


def test_an_unflushed_case_does_not_send_the_string_None():
    """attempt_count is None until the row is flushed, and 'recovery-7-None'
    would be a real reference Razorpay reserves permanently."""
    case = RecoveryCase(id=7, razorpay_payment_id="p", original_amount=1000)
    assert "None" not in reference_id_for(case)


def test_links_written_before_the_suffix_still_attribute():
    """Those payments are real money; reconciliation cannot stop working
    because the reference format moved on."""
    assert case_id_from_reference("recovery-1") == 1
    assert case_id_from_reference("recovery-1-0") == 1
    assert case_id_from_reference("recovery-42-7") == 42


async def test_a_resend_that_reached_nobody_says_so_in_the_trail(session):
    """An audit entry claiming a link was resent when nothing left the building
    is worse than no entry: the orchestrator burns an attempt on it, and
    reconciliation later would be reading a lie."""
    case, customer = await seed(session)
    client = FakeLinkClient()

    await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()

    client.notify_fails = True
    await create_recovery_link(session, case, customer, client, settings=SETTINGS)
    await session.commit()

    rows = await session.execute(
        select(RecoveryAction.metadata_json).order_by(RecoveryAction.id)
    )
    _, resent = list(rows.scalars())
    assert resent["resent_via"] == []
